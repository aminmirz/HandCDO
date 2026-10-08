"""
Grasp TPE Optimizer

This module provides a TPE (Tree-structured Parzen Estimator) optimizer for grasp parameters.
Each hand-object combination has its own separate optimizer that optimizes:
- obj_dist_palm: 3D offset from palm to object
- obj_angle_palm: 3D Euler angles for object orientation relative to palm
- finger_spread: finger spreading angle

Note: finger_m_rand (finger actuation speed multiplier) uses a fixed value of 0.6.

The optimizer is designed to be used inside the simulation script to avoid
reloading Isaac Sim for each grasp iteration.
"""

from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
import numpy as np
import optuna
from optuna.samplers import TPESampler
import torch
import logging

# Suppress Optuna logging
optuna.logging.set_verbosity(optuna.logging.WARNING)

logger = logging.getLogger(__name__)


@dataclass
class GraspParameterRanges:
    """Parameter ranges for grasp optimization.
    
    All ranges are specified per-task. If a single tuple is provided,
    it's used for all tasks. If a list of tuples is provided, each
    tuple corresponds to a specific task index.
    """
    # obj_dist_palm ranges: (min, max) for each of [x, y, z]
    # Shape: [3] tuples or per-task [num_tasks, 3] tuples
    obj_dist_palm_x: Tuple[float, float] = (-0.01, 0.01)  # Reduced by half
    obj_dist_palm_y: Tuple[float, float] = (-0.02, 0.02)  # Reduced by half
    obj_dist_palm_z: Tuple[float, float] = (-0.02, 0.02)  # Reduced by half (halved again)
    
    # obj_angle_palm ranges: (min, max) for each of [roll, pitch, yaw] in radians
    obj_angle_palm_roll: Tuple[float, float] = (-np.pi/12, np.pi/12)  # Reduced by half
    obj_angle_palm_pitch: Tuple[float, float] = (-np.pi/18, np.pi/18)  # Reduced by half
    obj_angle_palm_yaw: Tuple[float, float] = (-np.pi/18, np.pi/18)  # Reduced by half
    
    # finger_spread range: (min, max) in radians
    finger_spread: Tuple[float, float] = (-np.pi/12, np.pi/12)

    # finger_m_rand: fixed value (not sampled)
    finger_m_rand: float = 0.6


@dataclass
class GraspConfig:
    """A single grasp configuration."""
    obj_dist_palm: np.ndarray  # [3] - x, y, z offset
    obj_angle_palm: np.ndarray  # [3] - roll, pitch, yaw in radians
    finger_spread: float
    finger_m_rand: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'obj_dist_palm_x': float(self.obj_dist_palm[0]),
            'obj_dist_palm_y': float(self.obj_dist_palm[1]),
            'obj_dist_palm_z': float(self.obj_dist_palm[2]),
            'obj_angle_palm_roll': float(self.obj_angle_palm[0]),
            'obj_angle_palm_pitch': float(self.obj_angle_palm[1]),
            'obj_angle_palm_yaw': float(self.obj_angle_palm[2]),
            'finger_spread': float(self.finger_spread),
            'finger_m_rand': float(self.finger_m_rand),
        }
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'GraspConfig':
        """Create from dictionary."""
        return cls(
            obj_dist_palm=np.array([
                d['obj_dist_palm_x'],
                d['obj_dist_palm_y'],
                d['obj_dist_palm_z']
            ]),
            obj_angle_palm=np.array([
                d['obj_angle_palm_roll'],
                d['obj_angle_palm_pitch'],
                d['obj_angle_palm_yaw']
            ]),
            finger_spread=d['finger_spread'],
            finger_m_rand=d['finger_m_rand'],
        )
    
    def to_tensor(self, device: str = 'cpu') -> Dict[str, torch.Tensor]:
        """Convert to tensors for simulation."""
        return {
            'obj_dist_palm': torch.tensor(self.obj_dist_palm, dtype=torch.float32, device=device),
            'obj_angle_palm': torch.tensor(self.obj_angle_palm, dtype=torch.float32, device=device),
            'finger_spread': torch.tensor(self.finger_spread, dtype=torch.float32, device=device),
            'finger_m_rand': torch.tensor(self.finger_m_rand, dtype=torch.float32, device=device),
        }


class GraspOptimizer:
    """
    TPE optimizer for grasp parameters.
    
    Manages separate Optuna studies for each hand-object combination.
    Designed to be used inside the simulation script to avoid reloading Isaac Sim.
    """
    
    def __init__(
        self,
        num_hands: int,
        num_objects: int,
        param_ranges: Optional[GraspParameterRanges] = None,
        seed: int = 42,
        n_startup_trials: Optional[int] = None,
        n_ei_candidates: Optional[int] = None,
        multivariate: bool = False,
        gamma: Optional[float] = None,
    ):
        """
        Initialize the grasp optimizer.
        
        Args:
            num_hands: Number of hands to optimize
            num_objects: Number of objects (tasks) to optimize for
            param_ranges: Parameter ranges for optimization. If None, uses defaults.
            seed: Random seed for reproducibility
        """
        self.num_hands = num_hands
        self.num_objects = num_objects
        self.param_ranges = param_ranges or GraspParameterRanges()
        self.seed = seed
        self.n_startup_trials = n_startup_trials
        self.n_ei_candidates = n_ei_candidates
        self.multivariate = multivariate
        self.gamma = gamma
        
        # Create a study for each hand-object combination
        # Key: (hand_idx, object_idx) -> Optuna study
        self.studies: Dict[Tuple[int, int], optuna.Study] = {}
        
        # Track all trials and scores for each combination
        # Key: (hand_idx, object_idx) -> list of (GraspConfig, score) tuples
        self.history: Dict[Tuple[int, int], List[Tuple[GraspConfig, float]]] = {}
        
        # Initialize studies
        for hand_idx in range(num_hands):
            for obj_idx in range(num_objects):
                key = (hand_idx, obj_idx)
                # Derive TPE hyperparameters for this combo if not explicitly set
                sampler_seed = seed + hand_idx * num_objects + obj_idx
                sampler_kwargs: Dict[str, Any] = {
                    "seed": sampler_seed,
                }
                if self.n_startup_trials is not None:
                    sampler_kwargs["n_startup_trials"] = self.n_startup_trials
                if self.n_ei_candidates is not None:
                    sampler_kwargs["n_ei_candidates"] = self.n_ei_candidates
                if self.multivariate:
                    sampler_kwargs["multivariate"] = True

                sampler = TPESampler(**sampler_kwargs)
                # Optional: override gamma (good/bad split fraction). Uses internal Optuna API.
                if self.gamma is not None:
                    try:
                        gamma_value = float(self.gamma)
                        if 0.0 < gamma_value <= 0.5:
                            sampler._gamma = lambda _n_trials: gamma_value  # type: ignore[attr-defined]
                            logger.info(
                                f"Using custom grasp-level TPE gamma={gamma_value} "
                                f"for hand {hand_idx}, obj {obj_idx}"
                            )
                        else:
                            logger.warning(
                                f"Grasp-level gamma={gamma_value} is outside (0, 0.5]; "
                                f"ignoring custom gamma for hand {hand_idx}, obj {obj_idx}."
                            )
                    except Exception as e:  # pragma: no cover - defensive
                        logger.warning(
                            f"Failed to set custom grasp-level gamma for hand {hand_idx}, "
                            f"obj {obj_idx}: {e}"
                        )
                self.studies[key] = optuna.create_study(
                    direction='maximize',
                    sampler=sampler,
                )
                self.history[key] = []
    
    def _sample_params(self, trial: optuna.Trial) -> Dict[str, float]:
        """Sample parameters for a trial."""
        params = {}
        
        # Object distance from palm
        params['obj_dist_palm_x'] = trial.suggest_float(
            'obj_dist_palm_x',
            self.param_ranges.obj_dist_palm_x[0],
            self.param_ranges.obj_dist_palm_x[1]
        )
        params['obj_dist_palm_y'] = trial.suggest_float(
            'obj_dist_palm_y',
            self.param_ranges.obj_dist_palm_y[0],
            self.param_ranges.obj_dist_palm_y[1]
        )
        params['obj_dist_palm_z'] = trial.suggest_float(
            'obj_dist_palm_z',
            self.param_ranges.obj_dist_palm_z[0],
            self.param_ranges.obj_dist_palm_z[1]
        )
        
        # Object angle relative to palm
        params['obj_angle_palm_roll'] = trial.suggest_float(
            'obj_angle_palm_roll',
            self.param_ranges.obj_angle_palm_roll[0],
            self.param_ranges.obj_angle_palm_roll[1]
        )
        params['obj_angle_palm_pitch'] = trial.suggest_float(
            'obj_angle_palm_pitch',
            self.param_ranges.obj_angle_palm_pitch[0],
            self.param_ranges.obj_angle_palm_pitch[1]
        )
        params['obj_angle_palm_yaw'] = trial.suggest_float(
            'obj_angle_palm_yaw',
            self.param_ranges.obj_angle_palm_yaw[0],
            self.param_ranges.obj_angle_palm_yaw[1]
        )
        
        # Finger spread
        params['finger_spread'] = trial.suggest_float(
            'finger_spread',
            self.param_ranges.finger_spread[0],
            self.param_ranges.finger_spread[1]
        )

        # Finger actuation multiplier (fixed, not sampled)
        params['finger_m_rand'] = self.param_ranges.finger_m_rand

        return params
    
    def sample_grasp_configs(
        self,
        hand_idx: int,
        object_idx: int,
        n_samples: int,
    ) -> List[GraspConfig]:
        """
        Sample n grasp configurations for a hand-object combination.
        
        Uses the TPE sampler to propose new configurations based on
        previous trial results.
        
        Args:
            hand_idx: Index of the hand
            object_idx: Index of the object (task)
            n_samples: Number of configurations to sample
            
        Returns:
            List of GraspConfig objects
        """
        key = (hand_idx, object_idx)
        study = self.studies[key]
        
        configs = []
        for _ in range(n_samples):
            # Ask for a new trial
            trial = study.ask()
            
            # Sample parameters
            params = self._sample_params(trial)
            
            # Create GraspConfig
            config = GraspConfig.from_dict(params)
            
            # Store the trial for later update
            # We store the trial number so we can report the score later
            config._trial = trial
            
            configs.append(config)
        
        return configs
    
    def update_with_results(
        self,
        hand_idx: int,
        object_idx: int,
        configs: List[GraspConfig],
        scores: List[float],
    ):
        """
        Update the TPE optimizer with trial results.
        
        Args:
            hand_idx: Index of the hand
            object_idx: Index of the object (task)
            configs: List of GraspConfig objects that were evaluated
            scores: List of corresponding scores
        """
        key = (hand_idx, object_idx)
        study = self.studies[key]
        
        for config, score in zip(configs, scores):
            # Report the score to Optuna
            if hasattr(config, '_trial'):
                study.tell(config._trial, score)
            
            # Add to history
            self.history[key].append((config, score))
    
    def get_best_config(
        self,
        hand_idx: int,
        object_idx: int,
    ) -> Optional[GraspConfig]:
        """
        Get the best configuration found so far for a hand-object combination.
        
        Args:
            hand_idx: Index of the hand
            object_idx: Index of the object (task)
            
        Returns:
            Best GraspConfig or None if no trials have been completed
        """
        key = (hand_idx, object_idx)
        study = self.studies[key]
        
        if study.best_trial is None:
            return None
        
        return GraspConfig.from_dict(study.best_params)
    
    def get_best_score(
        self,
        hand_idx: int,
        object_idx: int,
    ) -> Optional[float]:
        """
        Get the best score found so far for a hand-object combination.
        
        Args:
            hand_idx: Index of the hand
            object_idx: Index of the object (task)
            
        Returns:
            Best score or None if no trials have been completed
        """
        key = (hand_idx, object_idx)
        study = self.studies[key]
        
        if study.best_trial is None:
            return None
        
        return study.best_value
    
    def get_top_k_scores(
        self,
        hand_idx: int,
        object_idx: int,
        k: int,
    ) -> List[float]:
        """
        Get the top-k scores for a hand-object combination.
        
        Args:
            hand_idx: Index of the hand
            object_idx: Index of the object (task)
            k: Number of top scores to return
            
        Returns:
            List of top-k scores (sorted descending)
        """
        key = (hand_idx, object_idx)
        history = self.history[key]
        
        if not history:
            return []
        
        # Extract scores and sort descending
        scores = [score for _, score in history]
        scores.sort(reverse=True)
        
        return scores[:k]
    
    def compute_grasp_score(
        self,
        hand_idx: int,
        object_idx: int,
        top_k: int,
    ) -> float:
        """
        Compute the grasp score for a hand-object combination.
        
        The grasp score is the average of the top-k wrench test scores.
        
        Args:
            hand_idx: Index of the hand
            object_idx: Index of the object (task)
            top_k: Number of top scores to average
            
        Returns:
            Grasp score (average of top-k scores)
        """
        top_scores = self.get_top_k_scores(hand_idx, object_idx, top_k)
        
        if not top_scores:
            return 0.0
        
        return sum(top_scores) / len(top_scores)
    
    def compute_hand_score(
        self,
        hand_idx: int,
        top_k: int,
    ) -> float:
        """
        Compute the hand score by averaging grasp scores across all objects.
        
        Args:
            hand_idx: Index of the hand
            top_k: Number of top scores to average for each grasp score
            
        Returns:
            Hand score (average of grasp scores across all objects)
        """
        grasp_scores = []
        for obj_idx in range(self.num_objects):
            grasp_score = self.compute_grasp_score(hand_idx, obj_idx, top_k)
            grasp_scores.append(grasp_score)
        
        if not grasp_scores:
            return 0.0
        
        return sum(grasp_scores) / len(grasp_scores)
    
    def get_all_hand_scores(self, top_k: int) -> Dict[int, float]:
        """
        Get scores for all hands.
        
        Args:
            top_k: Number of top scores to average for each grasp score
            
        Returns:
            Dictionary mapping hand_idx to hand_score
        """
        return {
            hand_idx: self.compute_hand_score(hand_idx, top_k)
            for hand_idx in range(self.num_hands)
        }
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the optimization state.
        
        Returns:
            Dictionary with optimization summary
        """
        summary = {
            'num_hands': self.num_hands,
            'num_objects': self.num_objects,
            'combinations': {},
        }
        
        for hand_idx in range(self.num_hands):
            for obj_idx in range(self.num_objects):
                key = (hand_idx, obj_idx)
                study = self.studies[key]
                history = self.history[key]
                
                combo_summary = {
                    'num_trials': len(history),
                    'best_score': study.best_value if study.best_trial else None,
                    'best_params': study.best_params if study.best_trial else None,
                }
                summary['combinations'][f'hand_{hand_idx}_obj_{obj_idx}'] = combo_summary
        
        return summary

    def export_history(self) -> Dict[str, Any]:
        """Export the full (config, score) history for all hand-object combos as a JSON-safe dict."""
        result = {}
        for (hand_idx, obj_idx), entries in self.history.items():
            key = f"hand_{hand_idx}_obj_{obj_idx}"
            result[key] = [
                {"config": config.to_dict(), "score": score}
                for config, score in entries
            ]
        return result

    def export_top_k_configs(self, top_k: int) -> Dict[str, Any]:
        """For each (hand, object) combo, return the top-k (config, score) entries sorted by score descending."""
        result = {}
        for (hand_idx, obj_idx), entries in self.history.items():
            key = f"hand_{hand_idx}_obj_{obj_idx}"
            sorted_entries = sorted(entries, key=lambda x: x[1], reverse=True)[:top_k]
            result[key] = [
                {"config": config.to_dict(), "score": score, "rank": i}
                for i, (config, score) in enumerate(sorted_entries)
            ]
        return result


class BatchGraspOptimizer:
    """
    Batch optimizer that samples grasp configs for all hand-object combinations at once.
    
    This is optimized for simulation where we want to sample configs for all combinations
    and run them in parallel across environments.
    """
    
    def __init__(
        self,
        num_hands: int,
        num_objects: int,
        n_samples_per_combo: int,
        param_ranges: Optional[GraspParameterRanges] = None,
        seed: int = 42,
        n_startup_trials: Optional[int] = None,
        n_ei_candidates: Optional[int] = None,
        multivariate: bool = False,
        gamma: Optional[float] = None,
    ):
        """
        Initialize the batch grasp optimizer.
        
        Args:
            num_hands: Number of hands
            num_objects: Number of objects (tasks)
            n_samples_per_combo: Number of samples per hand-object combination
            param_ranges: Parameter ranges for optimization
            seed: Random seed
        """
        self.num_hands = num_hands
        self.num_objects = num_objects
        self.n_samples_per_combo = n_samples_per_combo
        self.num_combos = num_hands * num_objects
        self.total_envs = self.num_combos * n_samples_per_combo
        
        # Create underlying optimizer
        self.optimizer = GraspOptimizer(
            num_hands=num_hands,
            num_objects=num_objects,
            param_ranges=param_ranges,
            seed=seed,
            n_startup_trials=n_startup_trials,
            n_ei_candidates=n_ei_candidates,
            multivariate=multivariate,
            gamma=gamma,
        )
        
        # Current batch of configs (for tracking)
        self._current_configs: Optional[Dict[Tuple[int, int], List[GraspConfig]]] = None
    
    def sample_all_configs(self) -> Dict[Tuple[int, int], List[GraspConfig]]:
        """
        Sample grasp configs for all hand-object combinations.
        
        Returns:
            Dictionary mapping (hand_idx, obj_idx) to list of GraspConfig
        """
        configs = {}
        for hand_idx in range(self.num_hands):
            for obj_idx in range(self.num_objects):
                key = (hand_idx, obj_idx)
                configs[key] = self.optimizer.sample_grasp_configs(
                    hand_idx, obj_idx, self.n_samples_per_combo
                )
        
        self._current_configs = configs
        return configs
    
    def configs_to_tensors(
        self,
        configs: Dict[Tuple[int, int], List[GraspConfig]],
        device: str = 'cpu',
    ) -> Dict[str, torch.Tensor]:
        """
        Convert all configs to tensors for simulation.
        
        The tensors are organized by environment index, where environments
        are ordered as: (hand_0, obj_0, sample_0), (hand_0, obj_0, sample_1), ...
        
        Args:
            configs: Dictionary from sample_all_configs()
            device: Device for tensors
            
        Returns:
            Dictionary with tensors:
            - obj_dist_palm: [total_envs, 3]
            - obj_angle_palm: [total_envs, 3]
            - finger_spread: [total_envs]
            - finger_m_rand: [total_envs]
        """
        obj_dist_palm = torch.zeros((self.total_envs, 3), device=device)
        obj_angle_palm = torch.zeros((self.total_envs, 3), device=device)
        finger_spread = torch.zeros(self.total_envs, device=device)
        finger_m_rand = torch.zeros(self.total_envs, device=device)
        
        env_idx = 0
        for hand_idx in range(self.num_hands):
            for obj_idx in range(self.num_objects):
                key = (hand_idx, obj_idx)
                for config in configs[key]:
                    obj_dist_palm[env_idx] = torch.tensor(
                        config.obj_dist_palm, device=device
                    )
                    obj_angle_palm[env_idx] = torch.tensor(
                        config.obj_angle_palm, device=device
                    )
                    finger_spread[env_idx] = config.finger_spread
                    finger_m_rand[env_idx] = config.finger_m_rand
                    env_idx += 1
        
        return {
            'obj_dist_palm': obj_dist_palm,
            'obj_angle_palm': obj_angle_palm,
            'finger_spread': finger_spread,
            'finger_m_rand': finger_m_rand,
        }
    
    def update_all_with_results(
        self,
        configs: Dict[Tuple[int, int], List[GraspConfig]],
        scores: torch.Tensor,
    ):
        """
        Update all optimizers with results.
        
        Args:
            configs: Dictionary from sample_all_configs()
            scores: Tensor of scores [total_envs] in same order as configs_to_tensors()
        """
        env_idx = 0
        for hand_idx in range(self.num_hands):
            for obj_idx in range(self.num_objects):
                key = (hand_idx, obj_idx)
                combo_configs = configs[key]
                combo_scores = []
                for _ in range(len(combo_configs)):
                    combo_scores.append(scores[env_idx].item())
                    env_idx += 1
                
                self.optimizer.update_with_results(
                    hand_idx, obj_idx, combo_configs, combo_scores
                )
    
    def compute_all_grasp_scores(self, top_k: int) -> Dict[Tuple[int, int], float]:
        """
        Compute grasp scores for all hand-object combinations.
        
        Args:
            top_k: Number of top scores to average
            
        Returns:
            Dictionary mapping (hand_idx, obj_idx) to grasp score
        """
        scores = {}
        for hand_idx in range(self.num_hands):
            for obj_idx in range(self.num_objects):
                key = (hand_idx, obj_idx)
                scores[key] = self.optimizer.compute_grasp_score(
                    hand_idx, obj_idx, top_k
                )
        return scores
    
    def compute_all_hand_scores(self, top_k: int) -> Dict[int, float]:
        """
        Compute hand scores for all hands.
        
        Args:
            top_k: Number of top scores to average for each grasp score
            
        Returns:
            Dictionary mapping hand_idx to hand score
        """
        return self.optimizer.get_all_hand_scores(top_k)
