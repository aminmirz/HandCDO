"""Parameter generation (sampling) layer for hand generation.

This module owns ALL randomized parameter sampling. `Config.py` holds only concrete
values, geometry, and save/load; `ParamGen.py` defines the sampling ranges and draws
concrete values from them, then builds a populated config object.

Design (per project decision):
  * `set_ranges()`  -- defines the `initial_*` RANGE variables ONLY.
  * `sample*()`     -- draws concrete values FROM those ranges and writes them into a config.
  * `build()`       -- creates a config, samples into it, returns it.

So "range definition" and "sample drawing" live in separate methods.

Finger-code grammar produced here (see FingerClass for decoding):
    'R-BEFORE-AFTER'  (thumb: 'R--AFTER', no before segment)
  R          single digit rotation mode (finger 0..4, thumb 0..1)
  BEFORE/AFTER  digit string, one digit per add-joint; count = number of digits
                each digit: '1' = short joint, '2' = long joint
  A segment equal to 'x' (or R == 'x') is sampled.

R -> legacy 3-bit code (selects the kinematic chain in FingerClass); also used for the
joint-count adjustment ported from the original implementation.
"""

import random
from typing import Union, Tuple

import numpy as np

from Config import PalmConfig, FingerConfig


# Authoritative rotation-mode map: new single digit -> legacy 3-bit code.
R_TO_OLDCODE = {0: '000', 1: '101', 2: '110', 3: '100', 4: '111'}


# ----------------------- module-level sampling helpers -----------------------
def _samp_float(spec) -> float:
    """Uniform draw if spec is a (lo, hi) tuple, else the fixed scalar (as float)."""
    if isinstance(spec, tuple):
        return float(random.uniform(spec[0], spec[1]))
    return float(spec)


def _samp_int(spec) -> int:
    """randint draw if spec is a (lo, hi) tuple, else the fixed scalar (as int)."""
    if isinstance(spec, tuple):
        return random.randint(spec[0], spec[1])
    return int(spec)


def _samp_choice(spec):
    """random.choice if spec is a tuple, else the fixed scalar."""
    if isinstance(spec, tuple):
        return random.choice(spec)
    return spec


def _fill_float_list(lst: list, n: int, spec) -> None:
    """Append floats sampled from spec until len(lst) == n."""
    while len(lst) < n:
        lst.append(_samp_float(spec))


def _sample_locations(spec, n: int) -> list:
    """Per-digit t-locations in [0,1]: tuple range -> uniform samples; None -> evenly spaced."""
    if isinstance(spec, tuple):
        return [float(random.uniform(spec[0], spec[1])) for _ in range(n)]
    return [(i + 1) / (n + 1) for i in range(n)]


# ================================ PalmParams ================================
class PalmParams:
    """Sampling builder for PalmConfig."""

    def __init__(self, fix: bool = False, thumb_side: str = 'random', generation_mode: str = 'standard', **range_overrides):
        self.fix = bool(fix)
        self.thumb_side = thumb_side
        self.generation_mode = generation_mode
        self.set_ranges()
        self._apply_overrides(range_overrides)

    # ---------- ranges ONLY ----------
    def set_ranges(self) -> None:
        f = self.fix  # index 1 = fixed scalar, index 0 = random range
        # outline
        self.initial_palm_size_mm: Union[float, Tuple[float, float]] = [(70.0, 150.0), 150.0][f]
        self.initial_resolution_mm: Union[float, Tuple[float, float]] = [(3.0, 10.0), 4.0][f]
        self.initial_outline_sides: Union[int, Tuple[int, int]] = [(4, 10), 8][f]
        self.initial_outline_aspect_ratio: Union[float, Tuple[float, float]] = [(1.0, 1.5), 1.55][f]
        self.initial_outline_longer_axis: Union[str, Tuple[str, str]] = [('x', 'y'), 'x'][f]
        self.initial_outline_rotation_deg: Union[float, Tuple[float, float]] = [(0.0, 180.0), 0.0][f]
        self.initial_smoothing_iters: Union[int, Tuple[int, int]] = [(1, 5), 5][f]
        self.initial_smoothing_t: Union[float, Tuple[float, float]] = [(0.0, 0.5), 0.25][f]
        self.initial_palm_wall_thickness_mm: Union[float, Tuple[float, float]] = [4.0, 4.0][f]
        # finger / thumb counts and per-digit params
        self.initial_finger_number: Union[int, Tuple[int, int]] = [(1, 8), 3][f] if self.generation_mode == 'finger_only' else [(1, 5), 3][f]
        self.initial_thumb_number: Union[int, Tuple[int, int]] = [0, 0][f] if self.generation_mode == 'finger_only' else [(1, 3), 1][f]
        self.initial_finger_angle_deg: Union[float, Tuple[float, float]] = [(-5, 5), 0][f]
        self.initial_thumb_angle_deg: Union[float, Tuple[float, float]] = [(-5, 5), 0][f]
        self.initial_finger_base_normal_offset_mm: Union[float, Tuple[float, float]] = [(-5, 5), 0][f]
        self.initial_thumb_base_normal_offset_mm: Union[float, Tuple[float, float]] = [(-5, 0), 0][f]
        self.initial_finger_base_side_offset_mm: Union[float, Tuple[float, float]] = [(-1, 1), 0][f]
        self.initial_thumb_base_side_offset_mm: Union[float, Tuple[float, float]] = [(-1, 1), 0][f]
        # per-digit location along the valid outline curve: fractional arc-length t in [0,1]
        # (0=rightmost, 1=leftmost). Tuple range -> sample uniformly; None -> evenly spaced.
        self.initial_finger_location: Union[Tuple[float, float], None] = [(0.0, 1.0), None][f]
        self.initial_thumb_location: Union[Tuple[float, float], None] = [(0.0, 1.0), None][f]
        # palm pad bumps
        self.initial_bump_type: Union[str, Tuple[str, ...]] = [('gaussian', 'conic', 'spherical'), 'gaussian'][f]
        self.initial_bump_number: Union[int, Tuple[int, int]] = [(0, 3), 1][f]
        self.initial_bump_max_height_intensity_mm: Union[float, Tuple[float, float]] = [(0.0, 15.0), 0.0][f]
        self.initial_bump_height_intensity: Union[float, Tuple[float, float]] = [(0.0, 1.0), 1][f]
        self.initial_bumps_spread: Union[float, Tuple[float, float]] = [(0.1, 0.5), 0.25][f]
        self.initial_bumps_aspect_ratio: Union[float, Tuple[float, float]] = [(1.0, 3.0), 1.0][f]
        self.initial_bump_rotation_deg: Union[float, Tuple[float, float]] = [(0, 180), 0][f]
        self.initial_bump_center_angle_deg: Union[float, Tuple[float, float]] = [(0.0, 360.0), 0.0][f]
        self.initial_bump_center_offset: Union[float, Tuple[float, float]] = [(0.0, 1.0), 0.0][f]

    def _apply_overrides(self, overrides: dict) -> None:
        for k, v in overrides.items():
            if not hasattr(self, k):
                raise AttributeError(f"PalmParams has no range parameter '{k}'")
            setattr(self, k, v)

    # ---------- sampling: write concrete values into a PalmConfig ----------
    def sample(self, cfg: PalmConfig) -> None:
        cfg.generation_mode = self.generation_mode
        # counts first (they size the per-digit lists)
        cfg.finger_number = _samp_int(self.initial_finger_number)
        cfg.thumb_number = 0 if self.generation_mode == 'finger_only' else _samp_int(self.initial_thumb_number)

        # per-digit lists (reset then fill to the sampled counts)
        cfg.thumb_angle_deg_list = []
        cfg.finger_angle_deg_list = []
        cfg.thumb_base_normal_offset_mm_list = []
        cfg.finger_base_normal_offset_mm_list = []
        cfg.finger_base_side_offset_mm_list = []
        cfg.thumb_base_side_offset_mm_list = []
        _fill_float_list(cfg.thumb_angle_deg_list, cfg.thumb_number, self.initial_thumb_angle_deg)
        _fill_float_list(cfg.finger_angle_deg_list, cfg.finger_number, self.initial_finger_angle_deg)
        _fill_float_list(cfg.thumb_base_normal_offset_mm_list, cfg.thumb_number, self.initial_thumb_base_normal_offset_mm)
        _fill_float_list(cfg.finger_base_normal_offset_mm_list, cfg.finger_number, self.initial_finger_base_normal_offset_mm)
        _fill_float_list(cfg.finger_base_side_offset_mm_list, cfg.finger_number, self.initial_finger_base_side_offset_mm)
        _fill_float_list(cfg.thumb_base_side_offset_mm_list, cfg.thumb_number, self.initial_thumb_base_side_offset_mm)

        # per-digit location along the valid outline curve (t in [0,1])
        cfg.finger_location_list = _sample_locations(self.initial_finger_location, cfg.finger_number)
        cfg.thumb_location_list = _sample_locations(self.initial_thumb_location, cfg.thumb_number)

        # palm size, then enforce minimum based on segment geometry + counts (from cfg.segments())
        cfg.palm_size_mm = _samp_float(self.initial_palm_size_mm)
        if self.generation_mode != 'finger_only':
            cfg.palm_size_mm = max(
                cfg.palm_size_mm,
                float(cfg.thumb_number * (cfg.thumb_corner_seg1_mm ** 2 + cfg.thumb_seg1_margin_offset_mm ** 2) ** 0.5),
            )
        finger_size_factor = 0.65 if self.generation_mode == 'finger_only' else 1.4
        cfg.palm_size_mm = max(cfg.palm_size_mm, finger_size_factor * float(cfg.finger_number * cfg.finger_min_spacing_mm))
        cfg.palm_wall_thickness_mm = _samp_float(self.initial_palm_wall_thickness_mm)

        # outline
        cfg.resolution_mm = _samp_float(self.initial_resolution_mm)
        cfg.outline_sides = _samp_int(self.initial_outline_sides)
        cfg.outline_aspect_ratio = _samp_float(self.initial_outline_aspect_ratio)
        cfg.outline_longer_axis = _samp_choice(self.initial_outline_longer_axis)
        cfg.outline_rotation_deg = _samp_float(self.initial_outline_rotation_deg)
        cfg.smoothing_iters = _samp_int(self.initial_smoothing_iters)
        cfg.smoothing_t = _samp_float(self.initial_smoothing_t)

        # palm pad bumps
        cfg.bump_type = _samp_choice(self.initial_bump_type)
        cfg.bumps_number = _samp_int(self.initial_bump_number)
        cfg.bump_max_height_intensity_mm = _samp_float(self.initial_bump_max_height_intensity_mm)
        cfg.bump_height_intensity_list = []
        cfg.bumps_spread_list = []
        cfg.bumps_aspect_ratio_list = []
        cfg.bump_rotation_deg_list = []
        cfg.bump_center_angle_deg_list = []
        cfg.bump_center_offset_list = []
        _fill_float_list(cfg.bump_height_intensity_list, cfg.bumps_number, self.initial_bump_height_intensity)
        _fill_float_list(cfg.bumps_spread_list, cfg.bumps_number, self.initial_bumps_spread)
        _fill_float_list(cfg.bumps_aspect_ratio_list, cfg.bumps_number, self.initial_bumps_aspect_ratio)
        _fill_float_list(cfg.bump_rotation_deg_list, cfg.bumps_number, self.initial_bump_rotation_deg)
        _fill_float_list(cfg.bump_center_angle_deg_list, cfg.bumps_number, self.initial_bump_center_angle_deg)
        _fill_float_list(cfg.bump_center_offset_list, cfg.bumps_number, self.initial_bump_center_offset)

        # reset stored geometry arrays so cfg.update() regenerates from the sampled values
        cfg.finger_points = np.empty((0, 2), dtype=float)
        cfg.thumb_points = np.empty((0, 2), dtype=float)
        cfg.finger_bases = np.empty((0, 2), dtype=float)
        cfg.thumb_bases = np.empty((0, 2), dtype=float)
        cfg.finger_bases_normal_vectors = np.empty((0, 2), dtype=float)
        cfg.thumb_bases_normal_vectors = np.empty((0, 2), dtype=float)

    def build(self, max_location_attempts: int = 100) -> PalmConfig:
        cfg = PalmConfig(data='fixed')   # deterministic constructor; randomness lives here
        cfg.generation_mode = self.generation_mode
        cfg.thumb_side = self.thumb_side
        last_error = None
        for _ in range(max_location_attempts):
            self.sample(cfg)
            try:
                cfg.update()             # rebuild outline + select finger/thumb points
                return cfg
            except ValueError as e:
                last_error = e
        raise ValueError(f"Could not sample valid finger/thumb locations after {max_location_attempts} attempts: {last_error}")


# =============================== FingerParams ===============================
class FingerParams:
    """Sampling builder for FingerConfig (fingers and thumbs)."""

    def __init__(self, type: str = 'finger', fix: bool = False, **range_overrides):
        self.type = type
        self.fix = bool(fix)
        self.set_ranges()
        self._apply_overrides(range_overrides)

    @property
    def is_finger(self) -> bool:
        return self.type == 'finger'

    # ---------- ranges ONLY ----------
    def set_ranges(self) -> None:
        f = self.fix
        # link length
        self.initial_link_added_length_mm: Union[float, Tuple[float, float]] = [(0.0, 10.0), 0.0][f]
        # finger pad bump ranges (note: 'cubic' option and max 4.0, per original finger pad())
        self.initial_bump_type = [('gaussian', 'conic', 'cubic', 'spherical'), 'gaussian'][f]
        self.initial_bump_number: Union[int, Tuple[int, int]] = [(0, 3), 0][f]
        self.initial_bump_max_height_intensity_mm: Union[float, Tuple[float, float]] = [(0.0, 4.0), 0.0][f]
        self.initial_bump_height_intensity: Union[float, Tuple[float, float]] = [(0.0, 1.0), 1][f]
        self.initial_bumps_spread: Union[float, Tuple[float, float]] = [(0.1, 0.5), 0.25][f]
        self.initial_bumps_aspect_ratio: Union[float, Tuple[float, float]] = [(1.0, 3.0), 1.0][f]
        self.initial_bump_rotation_deg: Union[float, Tuple[float, float]] = [(0, 180), 0][f]
        self.initial_bump_center_angle_deg: Union[float, Tuple[float, float]] = [(0.0, 360.0), 0.0][f]
        self.initial_bump_center_offset: Union[float, Tuple[float, float]] = [(0.0, 1.0), 0.0][f]
        # code-generation ranges
        self.initial_min_add_joint: int = 2
        self.initial_max_add_joint: int = 5 if self.is_finger else 4
        # rotation-mode digit range
        self.initial_R: Union[int, Tuple[int, int]] = ([(0, 4), 0][f] if self.is_finger else [(0, 1), 1][f])
        # joint-type digit choices ('1' short, '2' long)
        self.initial_type_digits: Tuple[str, str] = ('1', '2')

    def _apply_overrides(self, overrides: dict) -> None:
        for k, v in overrides.items():
            if not hasattr(self, k):
                raise AttributeError(f"FingerParams has no range parameter '{k}'")
            setattr(self, k, v)

    # ---------- code-generation sampling helpers ----------
    @staticmethod
    def _check_minmax(value: int, lo: int, hi: int) -> int:
        """-1 if value > hi, +1 if value < lo, else 0 (ported from original)."""
        if value > hi:
            return -1
        if value < lo:
            return 1
        return 0

    def _resolve_digits(self, seg_in: str, is_x: bool, count: int) -> str:
        """Produce a `count`-long joint-type digit string. If `is_x`, sample every digit;
        otherwise use the explicit digits, padding (sampled) / truncating to `count`."""
        if is_x:
            return ''.join(random.choice(self.initial_type_digits) for _ in range(count))
        digits = list(seg_in)
        while len(digits) < count:
            digits.append(random.choice(self.initial_type_digits))
        return ''.join(digits[:count])

    def _adjust_bounds(self, R: int) -> Tuple[int, int]:
        """Apply the rotation-mode joint-count adjustment (ported, via R->old-code)."""
        lo = self.initial_min_add_joint
        hi = self.initial_max_add_joint
        if self.is_finger:
            old = R_TO_OLDCODE[R]
            if old != '000':
                hi -= 1
            if old != '100':
                hi -= 1
            if old in ('101', '110'):
                lo -= 1
        else:
            hi -= 1
            if R == 1:
                hi -= 1
        return lo, hi

    # ---------- code generation: resolve wildcards -> explicit new-format code ----------
    def sample_code(self, cfg: FingerConfig, code: str) -> None:
        parts = str(code).split('-')
        while len(parts) < 3:
            parts.append('')
        R_seg, before_seg_in, after_seg_in = parts[0], parts[1], parts[2]

        # --- rotation mode ---
        if R_seg == 'x':
            R = _samp_int(self.initial_R)
        else:
            R = int(R_seg)

        lo, hi = self._adjust_bounds(R)

        if self.is_finger:
            before_is_x = (before_seg_in == 'x')
            after_is_x = (after_seg_in == 'x')
            nb = None if before_is_x else len(before_seg_in)
            na = None if after_is_x else len(after_seg_in)

            if nb is not None and na is not None:
                # both explicit -> clamp total into [lo, hi] (ported behavior)
                while self._check_minmax(nb + na, lo, hi) != 0:
                    adj = self._check_minmax(nb + na, lo, hi)
                    if adj == -1:
                        if nb > 0:
                            nb += adj
                        else:
                            na += adj
                    else:
                        na += adj
            elif nb is not None:
                total = random.randint(lo, hi)
                while self._check_minmax(nb, lo, hi) != 0:
                    nb += self._check_minmax(nb, lo, hi)
                na = max(0, total - nb)
            elif na is not None:
                total = random.randint(lo, hi)
                while self._check_minmax(na, lo, hi) != 0:
                    na += self._check_minmax(na, lo, hi)
                nb = max(0, total - na)
            else:
                total = random.randint(lo, hi)
                nb = random.randint(0, total)
                na = total - nb

            before_digits = self._resolve_digits(before_seg_in, before_is_x, nb)
            after_digits = self._resolve_digits(after_seg_in, after_is_x, na)
            new_code = f"{R}-{before_digits}-{after_digits}"
        else:
            # thumb: no before segment; after segment length is the joint count
            after_is_x = (after_seg_in == 'x' or after_seg_in == '')
            if not after_is_x:
                na = len(after_seg_in)
                while self._check_minmax(na, lo, hi) != 0:
                    na += self._check_minmax(na, lo, hi)
            else:
                na = random.randint(lo, hi)
            after_digits = self._resolve_digits(after_seg_in, after_is_x, na)
            new_code = f"{R}--{after_digits}"

        cfg.code = new_code
        cfg.code_data()  # parse explicit code -> rotation_mode and max_add_joint_slots

        # links (ported generate_links): one slot per add-joint slot
        cfg.link_added_length_mm_list = []
        _fill_float_list(cfg.link_added_length_mm_list, self.initial_max_add_joint, self.initial_link_added_length_mm)

    # ---------- pad bump sampling (ported generate_pads) ----------
    def sample_pads(self, cfg: FingerConfig) -> None:
        n = self.initial_max_add_joint
        cfg.bump_type_list = []
        cfg.bump_number_list = []
        cfg.bump_max_height_intensity_mm_list = []
        cfg.bump_height_intensity_list = []
        cfg.bumps_spread_list = []
        cfg.bumps_aspect_ratio_list = []
        cfg.bump_rotation_deg_list = []
        cfg.bump_center_angle_deg_list = []
        cfg.bump_center_offset_list = []
        for _ in range(n):
            cfg.bump_type_list.append(_samp_choice(self.initial_bump_type))
            nb = _samp_int(self.initial_bump_number)
            cfg.bump_number_list.append(nb)
            cfg.bump_max_height_intensity_mm_list.append(_samp_float(self.initial_bump_max_height_intensity_mm))
            cfg.bump_height_intensity_list.append([_samp_float(self.initial_bump_height_intensity) for _ in range(nb)])
            cfg.bumps_spread_list.append([_samp_float(self.initial_bumps_spread) for _ in range(nb)])
            cfg.bumps_aspect_ratio_list.append([_samp_float(self.initial_bumps_aspect_ratio) for _ in range(nb)])
            cfg.bump_rotation_deg_list.append([_samp_float(self.initial_bump_rotation_deg) for _ in range(nb)])
            cfg.bump_center_angle_deg_list.append([_samp_float(self.initial_bump_center_angle_deg) for _ in range(nb)])
            cfg.bump_center_offset_list.append([_samp_float(self.initial_bump_center_offset) for _ in range(nb)])

    def sample(self, cfg: FingerConfig, code: str) -> None:
        self.sample_code(cfg, code)
        self.sample_pads(cfg)

    def build(self, code: Union[str, None] = None, id: Union[int, None] = None) -> FingerConfig:
        code = code if code is not None else ('x-x-x' if self.is_finger else 'x--x')
        # construct with an explicit type-aware default code (NOT the wildcard), then sample
        cfg = FingerConfig(type=self.type, data='fixed', id=id)
        self.sample(cfg, code)
        return cfg
