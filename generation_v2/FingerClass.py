from typing import Optional, Union
from Config import * 
import numpy as np
from utils import *
import matplotlib.pyplot as plt
from matplotlib import patches 
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib as mpl
import matplotlib.colors as mcolors

class joint_surface:
    def __init__(self, surface_size: str = 'small'):
        self.config = FingerConfig()
        self.surface_size = surface_size
        # make the ouline of a rectangle with the dimensions of the config.motor_small_surface_dim_mm or config.motor_large_surface_dim_mm
        if self.surface_size == 'small':
            self.outline_points = np.array([[-self.config.motor_small_surface_dim_mm[0]/2, -self.config.motor_small_surface_dim_mm[1]/2, 0.0],
                                    [self.config.motor_small_surface_dim_mm[0]/2, -self.config.motor_small_surface_dim_mm[1]/2, 0.0],
                                    [self.config.motor_small_surface_dim_mm[0]/2, self.config.motor_small_surface_dim_mm[1]/2, 0.0],
                                    [-self.config.motor_small_surface_dim_mm[0]/2, self.config.motor_small_surface_dim_mm[1]/2, 0.0]])
        elif self.surface_size == 'large':
            self.outline_points = np.array([[-self.config.motor_large_surface_dim_mm[0]/2, -self.config.motor_large_surface_dim_mm[1]/2, 0.0],
                                    [self.config.motor_large_surface_dim_mm[0]/2, -self.config.motor_large_surface_dim_mm[1]/2, 0.0],
                                    [self.config.motor_large_surface_dim_mm[0]/2, self.config.motor_large_surface_dim_mm[1]/2, 0.0],
                                    [-self.config.motor_large_surface_dim_mm[0]/2, self.config.motor_large_surface_dim_mm[1]/2, 0.0]])
        else:
            raise ValueError(f"Invalid surface size: {self.surface_size}")

    def plot_outline(self):
        fig, ax = plt.subplots()
        ax.fill(self.outline_points[:, 0], self.outline_points[:, 1], color='gray', alpha=0.2, linewidth=0.0)
        ax.scatter(self.outline_points[:, 0], self.outline_points[:, 1], color='dimgray', marker='o', alpha=0.25, s=10)
        ax.set_xlabel('x (mm)')
        ax.set_ylabel('y (mm)')
        ax.grid(False)
        ax.axis('equal')
        return fig, ax

    def get_outline(self):
        return self.outline_points

    def get_surface_radius(self):
        return np.max(np.linalg.norm(self.outline_points, axis=1))

class joint:
    def __init__(self, id: Union[int, str], attachment_index: int = 0, attachment_angle_deg: float = 0.0, touple_idx: int = 0, rotation_range_rad: Optional[tuple[float, float]] = None):
        self.type = "joint"
        if isinstance(id, str):
            blender_name_list = [
                "n_l_n",
                "n_l_p",  
                "n_s_n",  
                "n_s_p",  
                "ws_l_p", 
                "ws_s_p", 
                "wt_l_n", 
                "wt_s_p", 
                ]
            self.id = blender_name_list.index(id)
        else:
            self.id = id
        self.blender_name = [
            "n_l_n",
            "n_l_p",  
            "n_s_n",  
            "n_s_p",  
            "ws_l_p", 
            "ws_s_p", 
            "wt_l_n", 
            "wt_s_p", 
            ][self.id]

        self.transformation = np.eye(4) # Global transform (homogeneous 4x4) for the joint
        self.attachments = self.get_attachments()
        self.attachment_index = attachment_index
        self.attachment_angle_deg = attachment_angle_deg
        self.touple_idx = touple_idx
        self.attach_geom = {"size_mm": (15.0,15.0,5.0), "color": "red"}
        # joint axis information
        if rotation_range_rad is None:
            self.rotation_range_rad = [
                (np.deg2rad(-180.0), np.deg2rad(180.0)),
                (np.deg2rad(-180.0), np.deg2rad(180.0)),
                (np.deg2rad(-180.0), np.deg2rad(180.0)),
                (np.deg2rad(-180.0), np.deg2rad(180.0)),
                (np.deg2rad(0.0), np.deg2rad(-90.0)),
                (np.deg2rad(0.0), np.deg2rad(-180.0)),
                (np.deg2rad(0.0), np.deg2rad(-90.0)),
                (np.deg2rad(-90.0), np.deg2rad(90.0)),
                ][self.id]
        else:
            self.rotation_range_rad = rotation_range_rad
        self.velocity = 8.0 # rad/s
        self.effort = 0.60 # Nm
        self.joint_axis = {"axis_direction": (0.0, 1.0, 0.0), "viz_diameter":2.0, "viz_length": 30.0, "position": (0,0.0,0.0)}
        self.joint_geom = {"size_mm": (20.0, 23.0, 34.0), "center_location_mm": (0.0, 0.0, 7.5)} # geometry is a cube
        self.joint_surface_normal_list = self.valid_joint_surface_normal()
        self.joint_surface = None
        self.joint_surface_local_transform = None
        self.joint_surface_world_transform = None
    
    def valid_joint_surface_normal(self):
        # return normals that do not match with the normal of the attachments
        normal_list = [(1,0,0), (-1,0,0), (0,0,1), (0,0,-1)]
        att_dicts = []
        for att in self.attachments:
            if isinstance(att, dict):
                att_dicts.append(att)
            elif isinstance(att, tuple):
                att_dicts.append(att[0])
            else:
                raise ValueError(f"Invalid attachment type: {type(att)}")
        # Remove any normals that match an attachment's normal
        for att in att_dicts:
            att_n = get_attachment_local_normal(att)
            normal_list = [n for n in normal_list if not np.allclose(att_n, n)]
        return normal_list

    def set_joint_surface(self, normal: list):
        # z plane surfaces have small joint_surface outline, and x plane surfaces have large joint_surface outline.
        if abs(normal[2]) == 1.0:
            self.joint_surface = joint_surface("small")
        elif abs(normal[0]) == 1.0:
            self.joint_surface = joint_surface("large")
        else:
            raise ValueError(f"Invalid normal: {normal[0]}")
        
    def get_attachments(self):
        # initial attachment vector is (0.0, 0.0, 1.0)
        attachments = []
        if self.blender_name.split("_")[1] == "l":
            if self.blender_name.split("_")[2] == "n":
                attachment_1 = {"name": "attach_1", "position": (-13.0, 0.0, 0.0), "rotation": (0.0, -90.0, 0.0)}
                attachment_2 = {"name": "attach_2", "position": (-13.0, 0.0, 15.0), "rotation": (0.0, -90.0, 0.0)}
            elif self.blender_name.split("_")[2] == "p":
                attachment_1 = {"name": "attach_1", "position": (13.0, 0.0, 0.0), "rotation": (0.0, 90.0, 0.0)}
                attachment_2 = {"name": "attach_2", "position": (13.0, 0.0, 15.0), "rotation": (0.0, 90.0, 0.0)}
            else:
                raise ValueError(f"Invalid link type: {self.blender_name.split('_')[1]}")
            attachments.append((attachment_1, attachment_2))
        elif self.blender_name.split("_")[1] == "s":
            if self.blender_name.split("_")[2] == "n":
                attachment = {"name": "attach_1", "position": (0.0, 0.0, -12.5), "rotation": (0.0, 180.0, 0.0)}
            elif self.blender_name.split("_")[2] == "p":
                attachment = {"name": "attach_1", "position": (0.0, 0.0, 27.5), "rotation": (0.0, 0.0, 0.0)}
            else:
                raise ValueError(f"Invalid link type: {self.blender_name.split('_')[1]}")
            attachments.append(attachment)
        else:
            raise ValueError(f"Invalid link type: {self.blender_name.split('_')[1]}")

        if self.blender_name[0] == "n":
            attachment = {"name": "attach_motor_1", "position": (0.0, 14.5, 0.0), "rotation": (-90.0, 0.0, 0.0)}
            attachments.append(attachment)
        elif self.blender_name[0] == "w":
            if self.blender_name[1] == "s":
                attachment = {"name": "attach_joint_1", "position": (-17.0, 0.0, 0.0), "rotation": (0.0, -90.0, 0.0)}
            elif self.blender_name[1] == "t":
                attachment = {"name": "attach_joint_1", "position": (0.0, 0.0, -17.0), "rotation": (0.0, 180.0, 0.0)}
            else:
                raise ValueError(f"Invalid joint type: {self.blender_name.split('_')[0]}")
            attachments.append(attachment)
        else:
            raise ValueError(f"Invalid joint type: {self.blender_name.split('_')[0]}")
        return attachments
    
class link:
    def __init__(self, id: int, attachment_index: int = 0, attachment_angle_deg: float = 0.0, fin_type: str = 'finger'):
        self.type = "link"
        self.id = id
        self.fin_type = fin_type
        if self.fin_type == 'finger':
            self.blender_name = "link_" + str(id)
        elif self.fin_type == 'thumb':
            self.blender_name = "link_t_" + str(id)
        else:
            raise ValueError(f"Invalid finger type: {self.fin_type}")
        self.transformation = np.eye(4)
        self.attachments = self.get_attachments()
        self.attachment_index = attachment_index
        self.attachment_angle_deg = attachment_angle_deg
        self.attach_geom = {"size_mm": (15.0,15.0,5.0), "color": "blue"}


        self.second_axis = (1.0, 0.0, 0.0)
        self.transformation = np.eye(4)

    def get_attachments(self):
        if self.fin_type =="finger":
            if self.id == 0:
                attachments = [
                    {"name": "attach_1", "position": (0.0, 0.0, 0.0), "rotation": (0.0, 180.0, 0.0)},
                    {"name": "attach_2", "position": (0.0, 0.0, 5), "rotation": (0.0, 0.0, 0.0)},
                ]
            elif self.id == 1:
                attachments = [
                    {"name": "attach_1", "position": (0.0, 0.0, 0.0), "rotation": (0.0, 180.0, 0.0)},
                    {"name": "attach_2", "position": (-26.8461, 0.0, 8.6), "rotation": (0.0, 0.0, 0.0)},
                ]
            elif self.id == 2:
                attachments = [
                    {"name": "attach_1", "position": (0.0, 0.0, 0.0), "rotation": (0.0, 180.0, 0.0)},
                    {"name": "attach_2", "position": (-39.8461, 0.0, 8.6), "rotation": (0.0, 0.0, 0.0)},
                ]
            elif self.id == 3   :
                attachments = [
                    {"name": "attach_1", "position": (0.0, 0.0, 0.0), "rotation": (0.0, 180.0, 0.0)},
                    {"name": "attach_2", "position": (-27.5, 0.0, 20), "rotation": (0.0, 90.0, 0.0)},
                ]
            elif self.id == 4:
                attachments = [
                    {"name": "attach_1", "position": (0.0, 0.0, 0.0), "rotation": (0.0, 180.0, 0.0)},
                ]
            else:
                raise ValueError(f"Invalid link type: {self.blender_name}")

        elif self.fin_type == "thumb":
            if self.id == 1:
                attachments = [
                    {"name": "attach_1", "position": (13.0, 0.0, 0.0), "rotation": (0.0, 90.0, 0.0)},
                    {"name": "attach_2", "position": (0.0, 14.5, 0.0), "rotation": (-90.0, 0.0, 0.0)},
                ]
            else:
                raise ValueError(f"Invalid link type: {self.blender_name}")
        return attachments

class Finger:
    def __init__(self, cfg: FingerConfig, base = None):
        self.base = base if base is not None else {"origin":(0.0, 0.0, 0.0), "direction":(0.0, 1.0, 0.0), "up":(0.0, 0.0, 1.0)}
        self.cfg = cfg

        if self.cfg.type == 'finger':
            self.elements = FingerCode2Chain(self.cfg.code,
                link_added_length_mm_list=self.cfg.link_added_length_mm_list).get_chain()

        elif self.cfg.type in ['thumb_right', 'thumb_left']:
            self.elements = ThumbCode2Chain(self.cfg.code,
                link_added_length_mm_list=self.cfg.link_added_length_mm_list,
                side=self.cfg.type.split('_')[1]).get_chain()
        else:
            raise ValueError(f"Invalid type: {self.cfg.type}")


        self.connect_elements()
        
  
    def connect_elements(self):
        if self.cfg.type == 'finger':
            self.compute_element_transforms_in_chain()
            self.compute_joint_surfaces_finger()
        elif self.cfg.type in ['thumb_right', 'thumb_left']:
            self.compute_element_transforms_in_chain()
            self.compute_joint_surfaces_thumb()
        else:
            raise ValueError(f"Invalid type: {self.cfg.type}")
    
    def compute_joint_surfaces_thumb(self):
        tol_cos: float = 0.99
        # Side-specific target direction
        up = normalize(np.array(self.base["up"], dtype=float))
        dirv = normalize(np.array(self.base["direction"], dtype=float))
        if self.cfg.type == "thumb_right":
            target = normalize(np.cross(up, dirv))   # up × dir
        else:
            target = normalize(np.cross(dirv, up))   # dir × up
        for geom in self.elements:
            if getattr(geom, "type", None) != "joint":
                continue
            normals = getattr(geom, "joint_surface_normal_list", [])
            R_global = geom.transformation[:3, :3]
            for idx, n_local in enumerate(normals):
                n_world = normalize(R_global @ np.array(n_local, dtype=float))
                if float(np.dot(n_world, target)) >= tol_cos:
                    center_local = face_center_from_normal_and_base_geom(n_local, geom.joint_geom)
                    R_surface_local = frame_from_normal(np.array(n_local, dtype=float), np.array([0.0, 0.0, 1.0]))
                    T_surface_local = make_transform(R_surface_local, center_local)
                    T_surface_world = geom.transformation @ T_surface_local
                    geom.set_joint_surface(n_local)
                    geom.joint_surface_local_transform = T_surface_local
                    geom.joint_surface_world_transform = T_surface_world

    def compute_element_transforms_in_chain(self):
        # Base frame for first attachment
        R_base = frame_from_normal(np.array(self.base["direction"], dtype=float), np.array(self.base["up"], dtype=float))
        T_base = make_transform(R_base, np.array(self.base["origin"], dtype=float))

        # Helpers to keep pre-rotation logic clean and consistent
        def pre_rotation(T_prev, R_world_pre, geom, curr_attachment_idx):
            if self.cfg.type == 'finger':
                try:
                    R_prev = T_prev[:3, :3]
                    base_up = normalize(np.array(self.base["up"], dtype=float))
                    base_dir_default = normalize(np.array(self.base["direction"], dtype=float))

                    if getattr(geom, "type", None) == "joint":
                        blender_name = getattr(geom, "blender_name", "")
                        if blender_name == "ws_l_p":
                            v_source_world = normalize(R_world_pre @ np.array([0.0, 0.0, -1.0], dtype=float))
                            target_candidates = [base_up]
                        elif blender_name == "wt_s_p":
                            # align local +X with base up
                            v_source_world = normalize(R_world_pre @ np.array([1.0, 0.0, 0.0], dtype=float))
                            target_candidates = [base_up]
                        else:
                            attachments_curr = getattr(geom, "attachments", [])
                            other_idx_curr = 1 - int(curr_attachment_idx) if isinstance(attachments_curr, list) and len(attachments_curr) >= 2 else int(curr_attachment_idx)
                            T_other_local_curr = get_attachment_local_transform(geom, other_idx_curr)
                            n_other_local_curr = T_other_local_curr[:3, :3] @ np.array([0.0, 0.0, 1.0], dtype=float)
                            v_source_world = normalize(R_world_pre @ n_other_local_curr)
                            # Align to either base up or base direction
                            target_candidates = [base_up, base_dir_default]
                    else:
                        # Link: align local +X with base up
                        v_source_world = normalize(R_world_pre @ np.array([1.0, 0.0, 0.0], dtype=float))
                        target_candidates = [base_up]

                    v_prev_xy = (R_prev.T @ v_source_world)[:2]
                    if np.linalg.norm(v_prev_xy) <= 1e-9:
                        return np.eye(4)
                    v_prev_xy = v_prev_xy / np.linalg.norm(v_prev_xy)
                    best_phi = 0.0
                    best_abs = np.inf
                    for tgt in target_candidates:
                        t_prev_xy = (R_prev.T @ tgt)[:2]
                        if np.linalg.norm(t_prev_xy) < 1e-9:
                            continue
                        t_prev_xy = t_prev_xy / np.linalg.norm(t_prev_xy)
                        sin_val = v_prev_xy[0]*t_prev_xy[1] - v_prev_xy[1]*t_prev_xy[0]
                        cos_val = v_prev_xy[0]*t_prev_xy[0] + v_prev_xy[1]*t_prev_xy[1]
                        phi = float(np.degrees(np.arctan2(sin_val, cos_val)))
                        if abs(phi) < best_abs:
                            best_abs = abs(phi)
                            best_phi = phi
                    Rz_pre = rotation_about_axis(np.array([0.0, 0.0, 1.0]), best_phi)
                    return world_about_frame_rotation(T_prev, Rz_pre)
                except Exception:
                    return np.eye(4)
            elif self.cfg.type in ['thumb_right', 'thumb_left']:
                try:
                    R_prev = T_prev[:3, :3]
                    base_up = normalize(np.array(self.base["up"], dtype=float))
                    base_dir = normalize(np.array(self.base["direction"], dtype=float))
                    base_up_cross_dir = normalize(np.cross(base_up, base_dir))        # up × dir
                    base_dir_cross_up = normalize(np.cross(base_dir, base_up))        # dir × up
                    is_right = (self.cfg.type == 'thumb_right')
                    is_left = (self.cfg.type == 'thumb_left')

                    if getattr(geom, "type", None) == "joint":
                        blender_name = getattr(geom, "blender_name", "")
                        if blender_name == "wt_s_p":
                            # Align local +X with side-specific cross:
                            # right: up × dir, left: dir × up
                            v_source_world = normalize(R_world_pre @ np.array([1.0, 0.0, 0.0], dtype=float))
                            target_candidates = [base_up_cross_dir if is_right else base_dir_cross_up]
                        elif blender_name == "ws_l_p":
                            # Align local -Z with side-specific cross:
                            # right: up × dir, left: dir × up
                            v_source_world = normalize(R_world_pre @ np.array([0.0, 0.0, -1.0], dtype=float))
                            target_candidates = [base_up_cross_dir if is_right else base_dir_cross_up]
                        else:
                            # Use secondary attachment normal (next attachment) as source
                            attachments_curr = getattr(geom, "attachments", [])
                            other_idx_curr = 1 - int(curr_attachment_idx) if isinstance(attachments_curr, list) and len(attachments_curr) >= 2 else int(curr_attachment_idx)
                            T_other_local_curr = get_attachment_local_transform(geom, other_idx_curr)
                            n_other_local_curr = T_other_local_curr[:3, :3] @ np.array([0.0, 0.0, 1.0], dtype=float)
                            v_source_world = normalize(R_world_pre @ n_other_local_curr)
                            # Targets (side-specific):
                            # right: [dir, dir × up], left: [dir, up × dir]
                            target_candidates = [base_dir, (base_dir_cross_up if is_right else base_up_cross_dir)]
                    else:
                        # Link: same rule as "other joints" (side-specific)
                        attachments_curr = getattr(geom, "attachments", [])
                        other_idx_curr = 1 - int(curr_attachment_idx) if isinstance(attachments_curr, list) and len(attachments_curr) >= 2 else int(curr_attachment_idx)
                        T_other_local_curr = get_attachment_local_transform(geom, other_idx_curr)
                        n_other_local_curr = T_other_local_curr[:3, :3] @ np.array([0.0, 0.0, 1.0], dtype=float)
                        v_source_world = normalize(R_world_pre @ n_other_local_curr)
                        target_candidates = [base_dir, (base_dir_cross_up if is_right else base_up_cross_dir)]

                    v_prev_xy = (R_prev.T @ v_source_world)[:2]
                    if np.linalg.norm(v_prev_xy) <= 1e-9:
                        return np.eye(4)
                    v_prev_xy = v_prev_xy / np.linalg.norm(v_prev_xy)
                    best_phi = 0.0
                    best_abs = np.inf
                    for tgt in target_candidates:
                        t_prev_xy = (R_prev.T @ tgt)[:2]
                        if np.linalg.norm(t_prev_xy) < 1e-9:
                            continue
                        t_prev_xy = t_prev_xy / np.linalg.norm(t_prev_xy)
                        sin_val = v_prev_xy[0]*t_prev_xy[1] - v_prev_xy[1]*t_prev_xy[0]
                        cos_val = v_prev_xy[0]*t_prev_xy[0] + v_prev_xy[1]*t_prev_xy[1]
                        phi = float(np.degrees(np.arctan2(sin_val, cos_val)))
                        if abs(phi) < best_abs:
                            best_abs = abs(phi)
                            best_phi = phi
                    Rz_pre = rotation_about_axis(np.array([0.0, 0.0, 1.0]), best_phi)
                    return world_about_frame_rotation(T_prev, Rz_pre)
                except Exception:
                    return np.eye(4)

        # First element: align, flip, apply angle
        first = self.elements[0]
        attachment  = first.attachment_index
        T_local = get_attachment_local_transform(first, attachment)
        T_align = T_base @ invert_transform(T_local)
        Rx180 = rotation_about_axis(np.array([1.0, 0.0, 0.0]), 180.0)
        T_flip = world_about_frame_rotation(T_base, Rx180)
        R_world_pre = (T_flip @ T_align)[:3, :3]
        T_pre_first = pre_rotation(T_base, R_world_pre, first, attachment)
        angle = float(getattr(first, "attachment_angle_deg", 0.0))
        Rz = rotation_about_axis(np.array([0.0, 0.0, 1.0]), angle)
        T_angle = world_about_frame_rotation(T_base, Rz)
        first.transformation = T_angle @ T_pre_first @ T_flip @ T_align

        # Remaining elements: align to previous attachment, flip, apply angle
        for i in range(1, len(self.elements)):
            prev = self.elements[i - 1]
            curr = self.elements[i]
            prev_attachment = 1-prev.attachment_index
            curr_attachment = curr.attachment_index
            T_prev_attach_world = get_attachment_global_transform(prev, prev_attachment)
            T_curr_local = get_attachment_local_transform(curr, curr_attachment)
            T_align = T_prev_attach_world @ invert_transform(T_curr_local)
            Rx180 = rotation_about_axis(np.array([1.0, 0.0, 0.0]), 180.0)
            T_flip = world_about_frame_rotation(T_prev_attach_world, Rx180)
            R_world_pre = (T_flip @ T_align)[:3, :3]
            T_pre = pre_rotation(T_prev_attach_world, R_world_pre, curr, curr_attachment)
            angle = float(getattr(curr, "attachment_angle_deg", 0.0))
            Rz = rotation_about_axis(np.array([0.0, 0.0, 1.0]), angle)
            T_angle = world_about_frame_rotation(T_prev_attach_world, Rz)
            curr.transformation = T_angle @ T_pre @ T_flip @ T_align

    def compute_joint_surfaces_finger(self, tol_cos: float = 0.99):
        target = np.array([0.0, 0.0, 1.0])
        for geom in self.elements:
            if getattr(geom, "type", None) != "joint":
                continue
            normals = getattr(geom, "joint_surface_normal_list", [])
            R_global = geom.transformation[:3, :3]
            for idx, n_local in enumerate(normals):
                n_world = normalize(R_global @ np.array(n_local, dtype=float))
                if float(np.dot(n_world, target)) >= tol_cos:
                    center_local = face_center_from_normal_and_base_geom(n_local, geom.joint_geom)
                    R_surface_local = frame_from_normal(np.array(n_local, dtype=float), np.array([0.0, 0.0, 1.0]))
                    T_surface_local = make_transform(R_surface_local, center_local)
                    T_surface_world = geom.transformation @ T_surface_local
                    geom.set_joint_surface(n_local)
                    geom.joint_surface_local_transform = T_surface_local
                    geom.joint_surface_world_transform = T_surface_world
                      
    def attach(self, new_geometry):
        self.elements.append(new_geometry)
        self.connect_elements()

    def replace(self, index: int, new_geometry):
        if index < 0 or index >= len(self.elements):
            raise IndexError("Index out of range")
        self.elements[index] = new_geometry
        self.connect_elements()

    def insert(self, index: int, new_geometry):
        if index < 0 or index > len(self.elements):
            raise IndexError("Index out of range")
        self.elements.insert(index, new_geometry)
        self.connect_elements()

    def set_attachment_angle_deg(self, index: int, angle: float):
        if index < 0 or index >= len(self.elements):
            raise IndexError("Index out of range")
        self.elements[index].attachment_angle_deg = float(angle)

    def rotate_attachment_angle_deg(self, index: int, delta_angle: float):
        if index < 0 or index >= len(self.elements):
            raise IndexError("Index out of range")
        self.elements[index].attachment_angle_deg = float(self.elements[index].attachment_angle_deg + delta_angle)

    def plot_2d(self, view: str = 'xy'):
        def draw_heightmap_2d(geom, ax):
            # show geom.heightmap after transormation by geom.transformation
            hm = getattr(geom, "heightmap", None)
            Tsurf = getattr(geom, "joint_surface_world_transform", None)
            if hm is None or Tsurf is None:
                return ax

            xs = getattr(hm, "grid_x", None)
            ys = getattr(hm, "grid_y", None)
            H = np.array(getattr(hm, "heightmap", np.empty((0, 0))), dtype=float)
            if xs is None or ys is None or H.size == 0:
                return ax
            # Build world XY for each grid vertex by transforming surface-local (x, y, 0)
            X, Y = np.meshgrid(xs, ys)
            pts_local = np.c_[X.ravel(), Y.ravel(), np.zeros(X.size, dtype=float)]
            pts_world = transform_points(Tsurf, pts_local)
            Xw = pts_world[:, ix].reshape(X.shape)
            Yw = pts_world[:, iy].reshape(Y.shape)
            # Mask invalids and render as a quad mesh; NaNs outside polygon will be transparent
            Zc = np.ma.masked_invalid(H)
            ax.pcolormesh(Xw, Yw, Zc, shading='auto', cmap='inferno', alpha=1, zorder=10)
            return ax

        def draw_joint_surface_2d(geom, ax):
            outline = transform_points(geom.joint_surface_world_transform, geom.joint_surface.outline_points)
            surf_poly = patches.Polygon(proj2d(outline), closed=True, facecolor='lightgray', edgecolor='gray', linewidth=0.8, alpha=1, zorder=9)
            ax.add_patch(surf_poly)
            
            return ax

        def draw_joint_axis_2d(geom, ax, view, ix, iy):
            Rg = geom.transformation[:3, :3]
            pg = geom.transformation[:3, 3]
            axis_local = np.array(geom.joint_axis["axis_direction"], dtype=float)
            axis_world = Rg @ axis_local
            pg2 = np.array([pg[0], pg[1], pg[2]])
            # Determine the normal to the current view plane
            if view == 'xy':
                plane_normal = np.array([0.0, 0.0, 1.0])
            elif view == 'xz':
                plane_normal = np.array([0.0, 1.0, 0.0])
            elif view == 'yz':
                plane_normal = np.array([1.0, 0.0, 0.0])
            else:
                plane_normal = np.array([0.0, 0.0, 1.0])
            # If axis is nearly perpendicular to the plane, draw a circle
            if abs(float(np.dot(axis_world / (np.linalg.norm(axis_world) + 1e-12), plane_normal))) >= 0.99:
                circ = patches.Circle((float(pg2[ix]), float(pg2[iy])), radius=3.0, fill=False, linestyle='-', linewidth=2.0, color='darkorchid', zorder=20)
                ax.add_patch(circ)
                ax.scatter([pg2[ix]], [pg2[iy]], marker='x', color='darkorchid', s=40, linewidths=2.0, zorder=20)
            else:
                dir_vec = np.array([axis_world[ix], axis_world[iy]], dtype=float)
                n = np.linalg.norm(dir_vec)
                if n < 1e-9:
                    ax.scatter([pg2[ix]], [pg2[iy]], marker='x', color='darkorchid', s=40, linewidths=2.0, zorder=20)
                else:
                    dir_vec /= n
                    L = 20.0
                    p1 = (pg2[ix] - 0.5 * L * dir_vec[0], pg2[iy] - 0.5 * L * dir_vec[1])
                    p2 = (pg2[ix] + 0.5 * L * dir_vec[0], pg2[iy] + 0.5 * L * dir_vec[1])        
                    ax.plot([p1[0], p2[0]], [p1[1], p2[1]], linestyle='--', color='darkorchid', linewidth=2.0, zorder=20) 

            return ax
        
        def draw_joint_base_2d(geom, ax):
            for face in get_cube_faces(geom.joint_geom["size_mm"], geom.joint_geom["center_location_mm"]):
                face_world = transform_points(geom.transformation, face)
                face_poly = patches.Polygon(proj2d(face_world), closed=True, facecolor='green', edgecolor='green', linewidth=0.8, alpha=0.35, zorder=2)
                ax.add_patch(face_poly)
            return ax

        def draw_attachments_2d(geom, ax):
            # Draw joint/link attachments (all views)
            attachments = getattr(geom, "attachments", [])
            for idx, att in enumerate(attachments):
                if isinstance(att, tuple):
                    pair_choices = [0, 1]
                else:
                    pair_choices = [0]
                for pc in pair_choices:
                    T_att_world = get_attachment_global_transform(geom, idx, pc)
                    for face in get_cube_faces(geom.attach_geom["size_mm"]):
                        face_world = transform_points(T_att_world, face)
                        face_poly = patches.Polygon(proj2d(face_world), closed=True, facecolor=geom.attach_geom["color"], edgecolor=geom.attach_geom["color"], linewidth=0.7, alpha=0.35, zorder=3)
                        ax.add_patch(face_poly)
            return ax
        
        def proj2d(arr3: np.ndarray) -> np.ndarray:
            return arr3[:, [ix, iy]]
        
        # choose projection indices
        idx_map = {'xy': (0, 1), 'xz': (0, 2), 'yz': (1, 2)}
        if view not in idx_map:
            view = 'xy'
        ix, iy = idx_map[view]


        fig, ax = plt.subplots()
        for geom in self.elements:
            ax = draw_attachments_2d(geom, ax)
            if getattr(geom, "type", None) == "joint":
                ax = draw_joint_base_2d(geom, ax)
                if view == 'xy' and geom.joint_surface is not None:
                    ax = draw_joint_surface_2d(geom, ax)
                    ax = draw_heightmap_2d(geom, ax)
                ax = draw_joint_axis_2d(geom, ax, view, ix, iy)

        ax.set_aspect('equal', adjustable='datalim')
        labels = {'xy': ('X (mm)', 'Y (mm)'), 'xz': ('X (mm)', 'Z (mm)'), 'yz': ('Y (mm)', 'Z (mm)')}
        ax.set_xlabel(labels[view][0])
        ax.set_ylabel(labels[view][1])
        plt.show()
        return fig, ax

    def plot_3d(self):
        def _set_axes_equal(ax3d):
            # Make axes of 3D plot have equal scale
            x_limits = ax3d.get_xlim3d()
            y_limits = ax3d.get_ylim3d()
            z_limits = ax3d.get_zlim3d()
            x_range = abs(x_limits[1] - x_limits[0])
            x_middle = np.mean(x_limits)
            y_range = abs(y_limits[1] - y_limits[0])
            y_middle = np.mean(y_limits)
            z_range = abs(z_limits[1] - z_limits[0])
            z_middle = np.mean(z_limits)
            plot_radius = 0.5 * max([x_range, y_range, z_range, 1.0])
            ax3d.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
            ax3d.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
            ax3d.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])

        def draw_joint_base_3d(geom, ax):
            faces_world = [transform_points(geom.transformation, face) for face in get_cube_faces(geom.joint_geom["size_mm"], geom.joint_geom["center_location_mm"])]
            poly = Poly3DCollection(faces_world, facecolors='green', edgecolors='green', linewidths=0.6, alpha=0.25)
            ax.add_collection3d(poly)
            return [v for f in faces_world for v in f]

        def draw_attachments_3d(geom, ax):
            pts = []
            attachments = getattr(geom, "attachments", [])
            for idx, att in enumerate(attachments):
                pair_choices = [0, 1] if isinstance(att, tuple) else [0]
                for pc in pair_choices:
                    T_att_world = get_attachment_global_transform(geom, idx, pc)
                    faces_world = [transform_points(T_att_world, face) for face in get_cube_faces(geom.attach_geom["size_mm"])]
                    poly = Poly3DCollection(faces_world, facecolors=geom.attach_geom["color"], edgecolors=geom.attach_geom["color"], linewidths=0.6, alpha=0.35)
                    ax.add_collection3d(poly)
                    pts.extend([v for f in faces_world for v in f])
            return pts

        def draw_joint_heightmap_3d(geom, ax):
            pts_out = []
            hmobj = getattr(geom, "heightmap", None)
            Tsurf = getattr(geom, "joint_surface_world_transform", None)
            if hmobj is None or Tsurf is None:
                return pts_out
            xs = getattr(hmobj, "grid_x", None)
            ys = getattr(hmobj, "grid_y", None)
            H = np.array(getattr(hmobj, "heightmap", np.empty((0, 0))), dtype=float)
            if xs is None or ys is None or H.size == 0:
                return pts_out
            # Build surface-local grid
            X, Y = np.meshgrid(xs, ys)
            pts_local = np.c_[X.ravel(), Y.ravel(), np.zeros(X.size, dtype=float)]
            pts_world = transform_points(Tsurf, pts_local)
            Xw = pts_world[:, 0].reshape(X.shape)
            Yw = pts_world[:, 1].reshape(Y.shape)
            Zw = pts_world[:, 2].reshape(X.shape)
            # Raise along surface normal by height H
            n_world = (Tsurf[:3, :3] @ np.array([0.0, 0.0, 1.0], dtype=float))
            nx, ny, nz = float(n_world[0]), float(n_world[1]), float(n_world[2])
            Xw = Xw + nx * H
            Yw = Yw + ny * H
            Zw = Zw + nz * H
            # Adaptive decimation
            ny_g, nx_g = H.shape
            target = 180
            step_y = max(1, int(np.ceil(ny_g / target)))
            step_x = max(1, int(np.ceil(nx_g / target)))
            Xd = Xw[::step_y, ::step_x]
            Yd = Yw[::step_y, ::step_x]
            Zd = Zw[::step_y, ::step_x]
            Hd = H[::step_y, ::step_x]
            valid = np.isfinite(Hd)
            Hmin = float(np.nanmin(Hd)) if valid.any() else 0.0
            Hmax = float(np.nanmax(Hd)) if valid.any() else 1.0
            norm = mcolors.Normalize(vmin=Hmin, vmax=Hmax)
            colors = mpl.colormaps['inferno'](norm(np.where(valid, Hd, Hmin)))
            colors[..., -1] = np.where(valid, 0.9, 0.0)
            ax.plot_surface(
                Xd, Yd, Zd,
                rstride=1, cstride=1,
                facecolors=colors,
                linewidth=0,
                antialiased=False,
                edgecolor='none',
                shade=False
            )
            pts_out.extend(np.c_[Xd.ravel(), Yd.ravel(), Zd.ravel()].tolist())
            return pts_out

        def draw_joint_axis_3d(geom, ax):
            Rg = geom.transformation[:3, :3]
            pg = geom.transformation[:3, 3]
            axis_local = np.array(geom.joint_axis["axis_direction"], dtype=float)
            axis_world = Rg @ axis_local
            axis_world = axis_world / (np.linalg.norm(axis_world) + 1e-12)
            axis_len = float(geom.joint_axis["viz_length"])
            p1 = pg - 0.5 * axis_len * axis_world
            p2 = pg + 0.5 * axis_len * axis_world
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], color='darkorchid', linewidth=2.0)
            return [p1, p2]

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        # Accumulate points for autoscaling
        all_pts = []

        for geom in self.elements:
            if getattr(geom, "type", None) == "joint":
                all_pts.extend(draw_joint_base_3d(geom, ax))
                all_pts.extend(draw_attachments_3d(geom, ax))
                all_pts.extend(draw_joint_heightmap_3d(geom, ax))
                all_pts.extend(draw_joint_axis_3d(geom, ax))
            else:
                all_pts.extend(draw_attachments_3d(geom, ax))

        # Axes labels and aspect
        ax.set_xlabel('X (mm)')
        ax.set_ylabel('Y (mm)')
        ax.set_zlabel('Z (mm)')

        # Autoscale based on collected points
        if all_pts:
            all_pts_arr = np.array(all_pts)
            xmin, ymin, zmin = np.min(all_pts_arr, axis=0)
            xmax, ymax, zmax = np.max(all_pts_arr, axis=0)
            ax.set_xlim([xmin, xmax])
            ax.set_ylim([ymin, ymax])
            ax.set_zlim([zmin, zmax])
            _set_axes_equal(ax)
        plt.show()
        return fig, ax
            
class FingerCode2Chain:
    def __init__(self, finger_code: str, link_added_length_mm_list: list = None):
        # finger codes are in the format 'R-BEFORE-AFTER' (split on '-')
        '''
        R is the rotation mode (single digit). Legacy 3-bit code in parentheses
        selects the kinematic chain (unchanged):
          0 -> no side/axial        (legacy 000)
          1 -> side + grasp.long    (legacy 101)
          2 -> axial                (legacy 110)
          3 -> side + grasp.short   (legacy 100)
          4 -> side + axial         (legacy 111)

        BEFORE / AFTER are joint-type digit strings for the add-joints before/after
        the rotation block. One digit per joint, so the count = number of digits:
          1: short joint
          2: long joint
        An empty segment means zero joints, e.g. '4--112' (0 before, 3 after),
        '0-11-' (2 before, 0 after), '2-1-1' (1 short before, 1 short after).
        '''
        self.link_added_length_mm_list = link_added_length_mm_list or []
        self._add_link_counter = 0
        self.chain_elements = []
        self.finger_code = finger_code
        # print(self.finger_code)

        parts = str(finger_code).split('-')
        self._R = int(parts[0])
        self._before = parts[1] if len(parts) > 1 else ''
        self._after = parts[2] if len(parts) > 2 else ''

        range_list = {2:(-0.5,0.5), 3:(-1.0,1.0), 5:(-0.5,2.0), 6:(-0.5,2.0)}
        # keyed by R digit; bodies preserved verbatim from the legacy string-keyed map
        self.rotation_chain = {
            0:[],                                                                                       # legacy 000
            1:[link(2), joint(2,rotation_range_rad=range_list[2]), joint(5,rotation_range_rad=range_list[5])],            # legacy 101
            2:[link(1), joint(2,rotation_range_rad=range_list[2]), joint(6,touple_idx=1,rotation_range_rad=range_list[6])],# legacy 110
            3:[link(3), joint(3,rotation_range_rad=range_list[3])],                                       # legacy 100
            4:[link(2), joint(2,rotation_range_rad=range_list[2]), joint(3,rotation_range_rad=range_list[3])],            # legacy 111
            }[self._R]

        self.assemble_joints()
        self.add_fingertip()

    def add_joint(self, type_digit: str):
        range_list = {4:(-0.2,1.7), 7:(-0.5,1.7)}
        return {
            '1':[joint(4,rotation_range_rad=range_list[4])],   # short
            '2':[joint(7,rotation_range_rad=range_list[7])],   # long
            }[type_digit]

    def add_link(self):
        idx = self._add_link_counter
        self._add_link_counter += 1
        if idx < len(self.link_added_length_mm_list) and self.link_added_length_mm_list[idx] > 0:
            l = link(0)
            length_mm = self.link_added_length_mm_list[idx]
            l.attachments[1]["position"] = (0.0, 0.0, length_mm)
            l.link_added_length_mm = length_mm
            return [l]
        return []

    def assemble_joints(self):
        if self.finger_code is None: return None
        chain = []
        for d in self._before:                 # one digit per before-joint
            chain.extend(self.add_link())
            chain.extend(self.add_joint(d))
        if self.rotation_chain != []:
            chain.extend(self.add_link())
            chain.extend(self.rotation_chain)
        for d in self._after:                  # one digit per after-joint
            chain.extend(self.add_link())
            chain.extend(self.add_joint(d))
        self.chain_elements = chain
        # print(self.chain_elements)

    def add_fingertip(self):
        self.chain_elements.append(link(4))
        
    def get_chain(self):
        return self.chain_elements

class ThumbCode2Chain:
    def __init__(self, thumb_code: str, link_added_length_mm_list: list = None, side: str = 'left'):
        # New thumb-code grammar: 'R--AFTER' (no before segment; split on '-').
        #   R in {0,1}: '0' primary rotation + link, '1' primary + secondary rotation joint.
        #   AFTER is a joint-type digit string (1=short, 2=long); count = number of digits.
        self.link_added_length_mm_list = link_added_length_mm_list or []
        self._add_link_counter = 0
        self.side = side
        self.chain_elements = []
        self.thumb_code = thumb_code
        # print(self.thumb_code)

        parts = str(thumb_code).split('-')
        self._R = parts[0]
        self._after = parts[-1] if len(parts) >= 2 else ''

        secondary_joint = joint(1, rotation_range_rad=(-2.0,0.45)) if self.side == 'right' else joint(0, rotation_range_rad=(-0.45,2.0))
        primary_joint = joint(3, rotation_range_rad=(-1.6,0.0)) if self.side == 'right' else joint(3, rotation_range_rad=(0.0,1.6))
        self.rotation_chain = {
            '0': [primary_joint,link(1,fin_type='thumb')],
            '1': [primary_joint, secondary_joint]
        }[self._R]

        self.assemble_joints()
        self.add_fingertip()

    def add_joint(self, type_digit: str):
        range_list = {4:(-0.2,1.7), 7:(-1.0,1.7)}
        return {
            '1': [joint(4, rotation_range_rad=range_list[4])],   # short
            '2': [joint(7, rotation_range_rad=range_list[7])],   # long
        }[type_digit]

    def add_link(self):
        idx = self._add_link_counter
        self._add_link_counter += 1
        if idx < len(self.link_added_length_mm_list) and self.link_added_length_mm_list[idx] > 0:
            l = link(0)
            length_mm = self.link_added_length_mm_list[idx]
            l.attachments[1]["position"] = (0.0, 0.0, length_mm)
            l.link_added_length_mm = length_mm
            return [l]
        return []

    def assemble_joints(self):
        # New grammar 'R--AFTER': one digit per after-joint.
        chain = []
        chain.extend(self.rotation_chain)
        for d in self._after:
            chain.extend(self.add_link())
            chain.extend(self.add_joint(d))
        self.chain_elements = chain
        # print(self.chain_elements)
    
    def add_fingertip(self):
        self.chain_elements.append(link(4))

    def get_chain(self):
        return self.chain_elements
