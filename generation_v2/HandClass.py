from Config import *
import os
import subprocess
import numpy as np
from PalmClass import PalmOutline
from FingerClass import Finger
from utils import make_inner_outline_from_outer



class Hand:

    def __init__(
        self,
        palm_cfg: PalmConfig,
        finger_cfgs: list[FingerConfig],
        thumb_cfgs: list[FingerConfig],
        hand_cfg: HandConfig = HandConfig(),
        root_dir = 'output'
    ):
        self.root_dir = root_dir
        self.plots = Plots()
        self.hand_cfg = hand_cfg
        self.palm_cfg = palm_cfg
        self.hand_cfg.palm_wall_thickness_mm = float(getattr(self.palm_cfg, "palm_wall_thickness_mm", self.hand_cfg.palm_wall_thickness_mm))
        # if the number of configs is ore than the palm config number of fingers/thumbs, then drop the extra configs
        index2drop = self.palm_cfg.finger_index2drop
        while len(finger_cfgs) > self.palm_cfg.finger_number:
            finger_cfgs.pop(index2drop)

            # remove the extra finger bases and normal vectors and points with the index2drop
            self.palm_cfg.finger_bases = np.delete(self.palm_cfg.finger_bases, index2drop, axis=0)
            self.palm_cfg.finger_bases_normal_vectors = np.delete(self.palm_cfg.finger_bases_normal_vectors, index2drop, axis=0)
            self.palm_cfg.finger_points = np.delete(self.palm_cfg.finger_points, index2drop, axis=0)
            index2drop += 1
            
        index2drop = self.palm_cfg.thumb_index2drop
        while len(thumb_cfgs) > self.palm_cfg.thumb_number:
            thumb_cfgs.pop(index2drop)
            # remove the extra thumb bases and normal vectors and points with the index2drop
            self.palm_cfg.thumb_bases = np.delete(self.palm_cfg.thumb_bases, index2drop, axis=0)
            self.palm_cfg.thumb_bases_normal_vectors = np.delete(self.palm_cfg.thumb_bases_normal_vectors, index2drop, axis=0)
            self.palm_cfg.thumb_points = np.delete(self.palm_cfg.thumb_points, index2drop, axis=0)
            index2drop += 1


        self.palm = PalmOutline(self.palm_cfg)
        self.finger_cfgs = finger_cfgs
        self.thumb_cfgs = thumb_cfgs

        # create the root directory if it doesn't exist
        if not os.path.exists(self.root_dir):
            os.makedirs(self.root_dir)


        # finger config
        outbound_ids = 0
        if isinstance(self.palm_cfg.finger_number, tuple):
            print("Warning: palm config finger number is not initialized correctly")
            return
            
        valid_ids = list(range(self.palm_cfg.finger_number))
        for i, finger_cfg in enumerate(self.finger_cfgs):
            if finger_cfg.id is not None:
                if finger_cfg.id in valid_ids:
                    valid_ids.remove(finger_cfg.id)
                else:
                    print(f"Warning: Invalid finger id {finger_cfg.id}")
                    outbound_ids += 1

        for i, finger_cfg in enumerate(self.finger_cfgs):
            if finger_cfg.id is None:
                print("The finger id is not initialized, assigning a ids randomly")
                try:
                    finger_cfg.id = valid_ids.pop(0)
                except Exception as e:
                    print(f"Error: {e}")
                    print("Not enough valid finger ids to assign to the fingers")
                    return
        
        if len(valid_ids) - outbound_ids > 0:
            print("Error: finger numbers and finger configs do not match")
            return

        # thumb config
        outbound_ids = 0
        if isinstance(self.palm_cfg.thumb_number, tuple):
            print("Warning: palm config thumb number is not initialized correctly")
            return
            
        valid_ids = list(range(self.palm_cfg.thumb_number))
        for i, thumb_cfg in enumerate(self.thumb_cfgs):
            if thumb_cfg.id is not None:
                if thumb_cfg.id in valid_ids:
                    valid_ids.remove(thumb_cfg.id)
                else:
                    print(f"Warning: Invalid thumb id {thumb_cfg.id}")
                    outbound_ids += 1

        for i, thumb_cfg in enumerate(self.thumb_cfgs):
            if thumb_cfg.id is None:
                print("The thumb id is not initialized, assigning a ids randomly")
                try:
                    thumb_cfg.id = valid_ids.pop(0)
                except Exception as e:
                    print(f"Error: {e}")
                    print("Not enough valid thumb ids to assign to the thumbs")
                    return
        
        if len(valid_ids) - outbound_ids > 0:
            print("Error: thumb numbers and thumb configs do not match")
            return


        # reorder the finger and thumb configs by id
        self.finger_cfgs = sorted(self.finger_cfgs, key=lambda x: x.id)
        self.thumb_cfgs = sorted(self.thumb_cfgs, key=lambda x: x.id)

        def _xy_array(values, n):
            arr = np.asarray(values, dtype=float)
            if n == 0:
                return np.empty((0, 2), dtype=float)
            return arr.reshape((n, 2))

        # add z axis to arrays
        finger_bases_2d = _xy_array(self.palm.cfg.finger_bases, self.palm.cfg.finger_number)
        finger_normals_2d = _xy_array(self.palm.cfg.finger_bases_normal_vectors, self.palm.cfg.finger_number)
        thumb_bases_2d = _xy_array(self.palm.cfg.thumb_bases, self.palm.cfg.thumb_number)
        thumb_normals_2d = _xy_array(self.palm.cfg.thumb_bases_normal_vectors, self.palm.cfg.thumb_number)
        self.finger_bases = np.concatenate((finger_bases_2d, np.zeros((self.palm.cfg.finger_number, 1))), axis=1)
        self.finger_bases_normal_vectors = np.concatenate((finger_normals_2d, np.zeros((self.palm.cfg.finger_number, 1))), axis=1)
        self.thumb_bases = np.concatenate((thumb_bases_2d, np.zeros((self.palm.cfg.thumb_number, 1))), axis=1)
        self.thumb_bases_normal_vectors = np.concatenate((thumb_normals_2d, np.zeros((self.palm.cfg.thumb_number, 1))), axis=1)
        

        self.fingers = []
        self.thumbs = []
        for i in range(self.palm.cfg.finger_number):
            finger_base = {"origin":self.finger_bases[i], "direction":self.finger_bases_normal_vectors[i], "up":np.array([0.0, 0.0, 1.0])}
            self.fingers.append(Finger(self.finger_cfgs[i], base=finger_base))
        for i in range(self.palm_cfg.thumb_number):
            thumb_cfgs[i].type = self.palm_cfg.thumb_side_list[i]
            thumb_base = {"origin":self.thumb_bases[i], "direction":self.thumb_bases_normal_vectors[i], "up":np.array([0.0, 0.0, 1.0])}
            self.thumbs.append(Finger(self.thumb_cfgs[i], base=thumb_base))

    def save_assembly_data(self):
        # save the trasformation data and type of objects so that we later use it inside the blender file
        try:
            components_blend = os.path.join(os.path.dirname(__file__), "blender", "components.blend")
            prefixes = []
            matrices = []
            groups = []
            subgroups = []
            elem_types = []
            # Joint limits (only valid when elem_types[i] == "joint"; links use NaN placeholders)
            joint_lower = []
            joint_upper = []
            joint_velocity = []
            joint_effort = []
            # Joint pad visualization (only for joint elements)
            joint_surface_outline_points = []   # list of (N,3) float arrays in mm (surface local frame)
            joint_surface_local_T = []          # list of (4,4) float arrays (mm-based translation) — LOCAL in GenerateHand scene
            joint_bump_params = []             # list of bump param lists per joint surface (empty if none)
            joint_id = []                      # joint self.id (only for joint elements; NaN placeholder otherwise)
            link_added_length_mm = []          # per-element link added length (0.0 for non-link_0 elements)

            def _append_elem(elem, group_name: str, subgroup_name: str):
                name = getattr(elem, "blender_name", None)
                T = getattr(elem, "transformation", None)
                if name is None or T is None:
                    return

                prefixes.append(str(name))
                matrices.append(np.array(T, dtype=float))
                groups.append(str(group_name))
                subgroups.append(str(subgroup_name))

                et = str(getattr(elem, "type", ""))
                elem_types.append(et)
                link_added_length_mm.append(float(getattr(elem, 'link_added_length_mm', 0.0)))

                if et == "joint":
                    rr = getattr(elem, "rotation_range_rad", None)
                    try:
                        lo = float(rr[0]) if rr is not None else float("nan")
                        hi = float(rr[1]) if rr is not None else float("nan")
                        # URDF expects lower <= upper
                        lo2 = float(min(lo, hi))
                        hi2 = float(max(lo, hi))
                    except Exception:
                        lo2 = float("nan")
                        hi2 = float("nan")

                    joint_lower.append(lo2)
                    joint_upper.append(hi2)
                    joint_velocity.append(float(getattr(elem, "velocity", float("nan"))))
                    joint_effort.append(float(getattr(elem, "effort", float("nan"))))

                    # Pad outline + local transform
                    js = getattr(elem, "joint_surface", None)
                    Tloc = getattr(elem, "joint_surface_local_transform", None)
                    try:
                        if js is not None and hasattr(js, "get_outline"):
                            op = np.asarray(js.get_outline(), dtype=float)
                        else:
                            op = np.zeros((0, 3), dtype=float)
                        if op.ndim != 2 or op.shape[1] != 3:
                            op = np.zeros((0, 3), dtype=float)
                    except Exception:
                        op = np.zeros((0, 3), dtype=float)
                    try:
                        Tloc = np.asarray(Tloc, dtype=float) if Tloc is not None else np.eye(4, dtype=float)
                        if Tloc.shape != (4, 4):
                            Tloc = np.eye(4, dtype=float)
                    except Exception:
                        Tloc = np.eye(4, dtype=float)
                    # Local transform of the joint surface (use downstream; avoids double-applying main object world)
                    try:
                        Tloc = getattr(elem, "joint_surface_local_transform", None)
                        Tloc = np.asarray(Tloc, dtype=float) if Tloc is not None else np.eye(4, dtype=float)
                        if Tloc.shape != (4, 4):
                            Tloc = np.eye(4, dtype=float)
                    except Exception:
                        Tloc = np.eye(4, dtype=float)

                    try:
                        hm = getattr(elem, "heightmap", None)
                        bumps = getattr(hm, "bump_params", []) if hm is not None and hasattr(hm, "bump_params") else []
                        bumps = list(bumps) if bumps is not None else []
                        meta = {
                            "raw_hmin": float(getattr(hm, "bump_raw_hmin", 0.0)) if hm is not None else 0.0,
                            "raw_hmax": float(getattr(hm, "bump_raw_hmax", 0.0)) if hm is not None else 0.0,
                            "scale": float(getattr(hm, "bump_scale", 0.0)) if hm is not None else 0.0,
                        }
                        bp = {"bumps": bumps, "meta": meta}
                    except Exception:
                        bp = {"bumps": [], "meta": {"raw_hmin": 0.0, "raw_hmax": 0.0, "scale": 0.0}}

                    joint_surface_outline_points.append(op)
                    joint_surface_local_T.append(Tloc)
                    joint_bump_params.append(bp)
                    try:
                        jid = getattr(elem, "id", float("nan"))
                        joint_id.append(float(jid) if jid is not None else float("nan"))
                    except Exception:
                        joint_id.append(float("nan"))
                else:
                    joint_lower.append(float("nan"))
                    joint_upper.append(float("nan"))
                    joint_velocity.append(float("nan"))
                    joint_effort.append(float("nan"))
                    joint_surface_outline_points.append(np.zeros((0, 3), dtype=float))
                    joint_surface_local_T.append(np.eye(4, dtype=float))
                    joint_bump_params.append({"bumps": [], "meta": {"raw_hmin": 0.0, "raw_hmax": 0.0, "scale": 0.0}})
                    joint_id.append(float("nan"))
            # fingers
            for idx, finger in enumerate(self.fingers, start=1):
                for elem in getattr(finger, "elements", []):
                    _append_elem(elem, "fingers", f"finger_{idx}")
            # thumbs
            for idx, thumb in enumerate(self.thumbs, start=1):
                for elem in getattr(thumb, "elements", []):
                    _append_elem(elem, "thumbs", f"thumb_{idx}")

            # This is the sequential element index used by the Blender assembly script as part of the suffix:
            # e.g. objects end with "_f{counter}" or "_t{counter}" where counter == elem_counter.
            elem_counter = np.arange(len(prefixes), dtype=int)
            pad_default_cfg = self.finger_cfgs[0] if self.finger_cfgs else (self.thumb_cfgs[0] if self.thumb_cfgs else None)
            finger_pad_resolution_level = int(getattr(pad_default_cfg, "pad_resolution_level", 4))
            finger_bump_edge_zeroing_spread = float(getattr(pad_default_cfg, "bump_edge_zeroing_spread", 0.3))
            finger_pad_thickness_mm = float(getattr(pad_default_cfg, "pad_thickness_mm", 0.5))
            finger_pad_resolution_levels = np.array(
                [int(getattr(cfg, "pad_resolution_level", finger_pad_resolution_level)) for cfg in self.finger_cfgs],
                dtype=int,
            )
            thumb_pad_resolution_levels = np.array(
                [int(getattr(cfg, "pad_resolution_level", finger_pad_resolution_level)) for cfg in self.thumb_cfgs],
                dtype=int,
            )
            finger_pad_thickness_mm_list = np.array(
                [float(getattr(cfg, "pad_thickness_mm", finger_pad_thickness_mm)) for cfg in self.finger_cfgs],
                dtype=float,
            )
            thumb_pad_thickness_mm_list = np.array(
                [float(getattr(cfg, "pad_thickness_mm", finger_pad_thickness_mm)) for cfg in self.thumb_cfgs],
                dtype=float,
            )
            np.savez(
                os.path.join(self.root_dir, "assembly_data.npz"),
                components_blend=np.array(components_blend),
                prefixes=np.array(prefixes, dtype=object),
                matrices=np.array(matrices, dtype=float),
                groups=np.array(groups, dtype=object),
                subgroups=np.array(subgroups, dtype=object),
                elem_counter=np.array(elem_counter, dtype=int),
                elem_types=np.array(elem_types, dtype=object),
                joint_lower=np.array(joint_lower, dtype=float),
                joint_upper=np.array(joint_upper, dtype=float),
                joint_velocity=np.array(joint_velocity, dtype=float),
                joint_effort=np.array(joint_effort, dtype=float),
                joint_surface_outline_points=np.array(joint_surface_outline_points, dtype=object),
                joint_surface_local_T=np.array(joint_surface_local_T, dtype=float),
                joint_bump_params=np.array(joint_bump_params, dtype=object),
                joint_id=np.array(joint_id, dtype=float),
                finger_pad_resolution_level=finger_pad_resolution_level,
                finger_pad_resolution_levels=finger_pad_resolution_levels,
                thumb_pad_resolution_levels=thumb_pad_resolution_levels,
                finger_pad_thickness_mm=finger_pad_thickness_mm,
                finger_pad_thickness_mm_list=finger_pad_thickness_mm_list,
                thumb_pad_thickness_mm_list=thumb_pad_thickness_mm_list,
                finger_bump_edge_zeroing_spread=finger_bump_edge_zeroing_spread,
                # Per-finger pad bump config parameters (indexed by finger, then by pad position)
                finger_bump_type_lists=np.array([cfg.bump_type_list for cfg in self.finger_cfgs], dtype=object),
                finger_bump_number_lists=np.array([cfg.bump_number_list for cfg in self.finger_cfgs], dtype=object),
                finger_bump_max_height_mm_lists=np.array([cfg.bump_max_height_intensity_mm_list for cfg in self.finger_cfgs], dtype=object),
                finger_bump_height_intensity_lists=np.array([cfg.bump_height_intensity_list for cfg in self.finger_cfgs], dtype=object),
                finger_bumps_spread_lists=np.array([cfg.bumps_spread_list for cfg in self.finger_cfgs], dtype=object),
                finger_bumps_aspect_ratio_lists=np.array([cfg.bumps_aspect_ratio_list for cfg in self.finger_cfgs], dtype=object),
                finger_bump_rotation_deg_lists=np.array([cfg.bump_rotation_deg_list for cfg in self.finger_cfgs], dtype=object),
                finger_bump_center_angle_deg_lists=np.array([cfg.bump_center_angle_deg_list for cfg in self.finger_cfgs], dtype=object),
                finger_bump_center_offset_lists=np.array([cfg.bump_center_offset_list for cfg in self.finger_cfgs], dtype=object),
                # Per-thumb pad bump config parameters
                thumb_bump_type_lists=np.array([cfg.bump_type_list for cfg in self.thumb_cfgs], dtype=object),
                thumb_bump_number_lists=np.array([cfg.bump_number_list for cfg in self.thumb_cfgs], dtype=object),
                thumb_bump_max_height_mm_lists=np.array([cfg.bump_max_height_intensity_mm_list for cfg in self.thumb_cfgs], dtype=object),
                thumb_bump_height_intensity_lists=np.array([cfg.bump_height_intensity_list for cfg in self.thumb_cfgs], dtype=object),
                thumb_bumps_spread_lists=np.array([cfg.bumps_spread_list for cfg in self.thumb_cfgs], dtype=object),
                thumb_bumps_aspect_ratio_lists=np.array([cfg.bumps_aspect_ratio_list for cfg in self.thumb_cfgs], dtype=object),
                thumb_bump_rotation_deg_lists=np.array([cfg.bump_rotation_deg_list for cfg in self.thumb_cfgs], dtype=object),
                thumb_bump_center_angle_deg_lists=np.array([cfg.bump_center_angle_deg_list for cfg in self.thumb_cfgs], dtype=object),
                thumb_bump_center_offset_lists=np.array([cfg.bump_center_offset_list for cfg in self.thumb_cfgs], dtype=object),
                # Fingertip scale factors
                finger_fingertip_scale_factors=np.array(
                    [cfg.fingertip_scale_factor for cfg in self.finger_cfgs], dtype=object),
                thumb_fingertip_scale_factors=np.array(
                    [cfg.fingertip_scale_factor for cfg in self.thumb_cfgs], dtype=object),
                link_added_length_mm=np.array(link_added_length_mm, dtype=float),
                collision_mesh=np.array(self.hand_cfg.collision_mesh),
            )
            return os.path.join(self.root_dir, "assembly_data.npz")
        except Exception as e:
            print(f"Failed to save assembly data to {os.path.join(self.root_dir, 'assembly_data.npz')}: {e}")
            return None

    def save_palm_data(self):
        """
        Save the palm outlines (in mm) so that a Blender script can generate the palm main body.
        The outline is stored as Nx2 array of XY points in millimetres plus the target thickness.
        Additionally, store finger/thumb base positions and normals, and thumb
        corner + segment directions for downstream use (e.g., visualization).
        also the inward offset of the palm outline is saved.
        """
        def save_thumb_squares(thumb_corners: np.ndarray, thumb_seg1_dirs: np.ndarray, thumb_seg2_dirs: np.ndarray, inward_offset_mm: float = 5.0):
            squares = []
            if thumb_corners is None or thumb_seg1_dirs is None or thumb_seg2_dirs is None or thumb_corners.size == 0 or thumb_seg1_dirs.size == 0 or thumb_seg2_dirs.size == 0:
                return []
            
            L = 70.0  # mm
            for i in range(min(len(thumb_corners), len(thumb_seg1_dirs), len(thumb_seg2_dirs))):
                c = thumb_corners[i]
                d1 = thumb_seg1_dirs[i]
                d2 = thumb_seg2_dirs[i]
                # move the square approximately inward_offset_mm on the bisector of the two segments
                c_offset = -(inward_offset_mm+3.0) * (d1 + d2)
                # Build square: corner at c, edges of length L along d1 and d2
                p0 = c + c_offset
                p1 = c + L * d1 + c_offset
                p2 = c + L * (d1 + d2) + c_offset
                p3 = c + L * d2 + c_offset
                sq = np.vstack([p0, p1, p2, p3, p0])
                squares.append(sq)
            return squares

        def plot_palm_outlines(outline: np.ndarray,
                               inner_outline: np.ndarray,
                               thumb_squares: list,
                               finger_bases_2d: np.ndarray,
                               finger_normals_2d: np.ndarray,
                               thumb_bases_2d: np.ndarray,
                               thumb_normals_2d: np.ndarray,
                               save_path: str | None = None):
            import matplotlib.pyplot as plt

            def _unit(v: np.ndarray) -> np.ndarray:
                v = np.asarray(v, dtype=float).reshape(2,)
                n = float(np.linalg.norm(v))
                if n <= 0.0:
                    return np.zeros((2,), dtype=float)
                return v / n

            def _thumb_joint_rect(base_xy: np.ndarray, normal_xy: np.ndarray,
                                  long_mm: float = 38.0, short_mm: float = 24.0) -> np.ndarray:
                """
                Rectangle attached to thumb base:
                - longer axis aligned with normal_xy
                - one short side has base_xy as midpoint
                Returns closed polyline (5,2).
                """
                u = _unit(normal_xy)                  # long axis direction
                if float(np.linalg.norm(u)) <= 0.0:
                    return np.zeros((0, 2), dtype=float)
                v = np.array([-u[1], u[0]], dtype=float)  # perpendicular (short axis)
                h = float(short_mm) / 2.0
                L = float(long_mm)

                base_xy = np.asarray(base_xy, dtype=float).reshape(2,)
                p0 = base_xy - h * v
                p1 = base_xy + h * v
                p2 = base_xy + L * u + h * v
                p3 = base_xy + L * u - h * v
                return np.vstack([p0, p1, p2, p3, p0])

            plt.figure()
            plt.plot(outline[:, 0], outline[:, 1], "k-o", label="Outer body outline")
            plt.plot(inner_outline[:, 0], inner_outline[:, 1], "r-o", label="Inner outline")
            # for i, sq in enumerate(thumb_squares):
            #     lbl = "Thumb square" if i == 0 else None
            #     plt.plot(sq[:, 0], sq[:, 1], "b-", linewidth=1.5, label=lbl)
                
            # plot a rectangle attached to the thumb base representing the first joint. Size is 38x24 with longer axis aligned with thumb_base_normal_vector
            # one of the short sides has the thumb base as midpoint
            if thumb_bases_2d is not None and thumb_normals_2d is not None:
                tb = np.asarray(thumb_bases_2d, dtype=float)
                tn = np.asarray(thumb_normals_2d, dtype=float)
                n = min(len(tb), len(tn))
                for i in range(n):
                    rect = _thumb_joint_rect(tb[i], tn[i], long_mm=38.0, short_mm=24.0)
                    if rect.size == 0:
                        continue
                    lbl = "Thumb joint (38x24)" if i == 0 else None
                    plt.plot(rect[:, 0], rect[:, 1], "g-", linewidth=2.0, label=lbl)

            # Plot base locations and normals (finger + thumb)
            arrow_len = 15.0  # mm (visualization only)

            if finger_bases_2d is not None and finger_normals_2d is not None:
                fb = np.asarray(finger_bases_2d, dtype=float)
                fn = np.asarray(finger_normals_2d, dtype=float)
                n = min(len(fb), len(fn))
                if n > 0:
                    plt.scatter(fb[:n, 0], fb[:n, 1], c="tab:orange", s=35, label="Finger bases", zorder=5)
                    fn_u = np.array([_unit(fn[i]) for i in range(n)], dtype=float)
                    plt.quiver(
                        fb[:n, 0], fb[:n, 1],
                        fn_u[:n, 0] * arrow_len, fn_u[:n, 1] * arrow_len,
                        angles="xy", scale_units="xy", scale=1.0,
                        color="tab:orange", width=0.004, zorder=6, label="Finger normals"
                    )

            if thumb_bases_2d is not None and thumb_normals_2d is not None:
                tb = np.asarray(thumb_bases_2d, dtype=float)
                tn = np.asarray(thumb_normals_2d, dtype=float)
                n = min(len(tb), len(tn))
                if n > 0:
                    plt.scatter(tb[:n, 0], tb[:n, 1], c="tab:green", s=45, label="Thumb bases", zorder=5)
                    tn_u = np.array([_unit(tn[i]) for i in range(n)], dtype=float)
                    plt.quiver(
                        tb[:n, 0], tb[:n, 1],
                        tn_u[:n, 0] * arrow_len, tn_u[:n, 1] * arrow_len,
                        angles="xy", scale_units="xy", scale=1.0,
                        color="tab:green", width=0.004, zorder=6, label="Thumb normals"
                    )
            
            plt.gca().set_aspect("equal", adjustable="datalim")
            plt.xlabel("X (mm)")
            plt.ylabel("Y (mm)")
            plt.legend()
            plt.title("Palm outlines")
            plt.grid(False)
            if save_path:
                try:
                    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
                    plt.savefig(save_path, dpi=200, bbox_inches="tight")
                    print(f"Saved palm outline plot: {save_path}")
                except Exception as e:
                    print(f"Failed to save palm outline plot to {save_path}: {e}")
            plt.show()

        try:
            # Thumb corners and segment directions derived from thumb_seg_frames
            frames = np.array(self.palm.thumb_seg_frames, dtype=float)
            if frames.ndim == 3 and frames.shape[0] > 0:
                corners = frames[:, 2, :].copy()   # (N,2)
                seg1_pts = frames[:, 1, :].copy()  # (N,2)
                seg2_pts = frames[:, 3, :].copy()  # (N,2)
            else:
                corners = np.empty((0, 2), dtype=float)
                seg1_pts = np.empty((0, 2), dtype=float)
                seg2_pts = np.empty((0, 2), dtype=float)

            d1 = seg1_pts - corners
            d2 = seg2_pts - corners

            def _unit_rows(v):
                v = np.asarray(v, dtype=float)
                n = np.linalg.norm(v, axis=1, keepdims=True)
                # avoid division by zero
                return np.divide(v, n, out=np.zeros_like(v), where=n > 0.0)

            seg1_dirs = _unit_rows(d1)
            seg2_dirs = _unit_rows(d2)

            thumb_corners_2d = corners
            thumb_seg1_dirs_2d = seg1_dirs
            thumb_seg2_dirs_2d = seg2_dirs


            outer_outline = np.asarray(self.palm.get_outline(), dtype=float)
            clean_inner_outline = np.asarray(getattr(self.palm.cfg, "inner_outline_points", np.empty((0, 2))), dtype=float)

            # The palm body outline has already been modified by finger/thumb
            # attachment segments and smoothing. Build the cavity outline from
            # that final body outline so the wall follows those same local
            # deformations instead of cutting with the original clean polygon.
            inner_outline = make_inner_outline_from_outer(
                outer_outline,
                wall_mm=float(self.hand_cfg.palm_wall_thickness_mm),
                steps=10,
            )
            thumb_squares = save_thumb_squares(thumb_corners_2d, thumb_seg1_dirs_2d, thumb_seg2_dirs_2d, inward_offset_mm=5.0)
            if Plots().palm_outline_plot_2d:
                plot_palm_outlines(
                    outer_outline, inner_outline, thumb_squares,
                    self.palm.cfg.finger_bases, self.palm.cfg.finger_bases_normal_vectors,
                    self.palm.cfg.thumb_bases, self.palm.cfg.thumb_bases_normal_vectors,
                    save_path=os.path.join(self.root_dir, "palm_outlines_data.png"),
                )

            np.savez(
                os.path.join(self.root_dir, "palm_outlines_data.npz"),
                outline=outer_outline,
                outer_outline=outer_outline,
                inner_outline=inner_outline,
                inward_offset_outline=inner_outline,
                clean_inner_outline=clean_inner_outline,
                outline_semantics_version=3,
                wall_generation_method="pyclipper_final_outline_inward_offset",
                thumb_squares=thumb_squares,
                palm_thickness_mm=float(self.hand_cfg.palm_thickness_mm),
                palm_wall_thickness_mm=float(self.hand_cfg.palm_wall_thickness_mm),
                palm_extrude_height_mm=float(self.hand_cfg.palm_extrude_height_mm),
                thumb_mount_size_mm=self.hand_cfg.thumb_mount_size_mm,
                thumb_mount_normal_offset_mm=float(self.hand_cfg.thumb_mount_normal_offset_mm),
                thumb_mount_height_mm=float(self.hand_cfg.thumb_mount_height_mm),
                finger_palm_base_offset_mm=float(self.hand_cfg.finger_palm_base_offset_mm),
                origin_offset_mm=np.array(self.hand_cfg.origin_offset_mm, dtype=float),
                finger_bases_2d=self.palm.cfg.finger_bases,
                finger_normals_2d=self.palm.cfg.finger_bases_normal_vectors,
                thumb_bases_2d=self.palm.cfg.thumb_bases,
                thumb_normals_2d=self.palm.cfg.thumb_bases_normal_vectors,
                thumb_corners_2d=thumb_corners_2d,
                thumb_seg1_dirs_2d=thumb_seg1_dirs_2d,
                thumb_seg2_dirs_2d=thumb_seg2_dirs_2d,
                palm_pad_resolution_level=int(self.palm_cfg.pad_resolution_level),
                palm_pad_thickness_mm=float(self.palm_cfg.pad_thickness_mm),
                palm_bump_edge_zeroing_spread=float(self.palm_cfg.bump_edge_zeroing_spread),
                # Palm bump parameters
                palm_bump_type=str(self.palm_cfg.bump_type),
                palm_bumps_number=int(self.palm_cfg.bumps_number),
                palm_bump_max_height_intensity_mm=float(self.palm_cfg.bump_max_height_intensity_mm),
                palm_bump_height_intensity_list=np.array(self.palm_cfg.bump_height_intensity_list, dtype=float),
                palm_bumps_spread_list=np.array(self.palm_cfg.bumps_spread_list, dtype=float),
                palm_bumps_aspect_ratio_list=np.array(self.palm_cfg.bumps_aspect_ratio_list, dtype=float),
                palm_bump_rotation_deg_list=np.array(self.palm_cfg.bump_rotation_deg_list, dtype=float),
                palm_bump_center_angle_deg_list=np.array(self.palm_cfg.bump_center_angle_deg_list, dtype=float),
                palm_bump_center_offset_list=np.array(self.palm_cfg.bump_center_offset_list, dtype=float),
                # Detailed viz flag (controls whether screw holes, mounts, etc. are added)
                detailed_viz=bool(self.palm_cfg.detailed_viz),
                collision_mesh=np.array(self.hand_cfg.collision_mesh),
            )
            return os.path.join(self.root_dir, "palm_outlines_data.npz")
        except Exception as e:
            print(f"Failed to save palm outline data: {e}")
            return None

    def blender_full_assembly(self):
        script_path = os.path.join(os.path.dirname(__file__), "blender_full_assembly.py")
        # here we just run a blender script --blender -b <file_path> -python <script_path>
        data_path = self.save_assembly_data()
        data_path_palm = self.save_palm_data()
        if data_path is None:
            print("No data to assemble.")
            return
        # try to spawn Blender in background
        blender_exe = os.environ.get("BLENDER_EXE", "blender")
        out_root = self.root_dir if self.root_dir else os.path.dirname(__file__)
        cmd = [
            blender_exe,
            "--factory-startup",
            "-noaudio",
            "-b",
            "-P", script_path,
            "--",
            data_path,
            out_root,
        ]
        try:
            print("Running Blender assembly")#:", " ".join(cmd))
            subprocess.run(cmd, check=True)
        except Exception as e:
            print("Failed to launch Blender for assembly. Set BLENDER_EXE env var to your Blender executable.")
            print(e)

        # save all the configs to the root directory
        self.palm_cfg.save_config(os.path.join(self.root_dir, "palm_cfg.py"))
        for finger_cfg in self.finger_cfgs:
            finger_cfg.save_config(os.path.join(self.root_dir, f"finger_cfg_{finger_cfg.id}.py"))
        for thumb_cfg in self.thumb_cfgs:
            thumb_cfg.save_config(os.path.join(self.root_dir, f"thumb_cfg_{thumb_cfg.id}.py"))
        self.hand_cfg.save_config(os.path.join(self.root_dir, "hand_cfg.py"))
