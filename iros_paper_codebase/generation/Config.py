import numpy as np
from typing import Union, Tuple
import importlib.util
from pathlib import Path
import random

def warning(message, default=None):
    print(f"Warning: {message}")
    return default

class PalmConfig:
   """Default configurations"""
   def __init__(self, data: str = "fixed", detailed_viz: bool = False):
      self.detailed_viz = detailed_viz
      self.fix = 'fixed' == data
      self.base_outline()
      self.finger_thumb()
      self.pad()
      self.segments()
      self.generate_values()

      if data in ["fixed", "random", "draw"]:
         self.mode = data
      else:
         self.mode = "fixed"
         try:
            self.import_config(data)
         except:
            raise ValueError(f"invalid palm config data type: {data}")


      self.initialize_outline()
      self.finger_thumb_points_selection()
      self.finger_index2drop = 0
      self.thumb_index2drop = 0

   def import_config(self, path):
      path = Path(path)
      spec = importlib.util.spec_from_file_location(path.stem, path)
      if spec is None or spec.loader is None:
         raise ValueError(f"Could not import module from path: {path}")
      module = importlib.util.module_from_spec(spec)
      spec.loader.exec_module(module)

      for k, _ in vars(self).items():
         if k == "mode":
            continue
         v = getattr(module, k)
         if k in ["outline_points", "finger_points", "thumb_points", "finger_bases", "thumb_bases", "finger_bases_normal_vectors", "thumb_bases_normal_vectors"]:
            v = np.array(v)
         setattr(self, k, v)

   def save_config(self, path):
      with open(path, "w") as f:
         for k, v in vars(self).items():
            if k in ["outline_points", "finger_points", "thumb_points", "finger_bases", "thumb_bases", "finger_bases_normal_vectors", "thumb_bases_normal_vectors"]:
               v = v.tolist()
            f.write(f"{k} = {repr(v)}\n")

   def base_outline(self): # Palm base outline geometry        
      self.initial_palm_size_mm: Union[float, Tuple[float, float]] = [(80.0, 140.0), 170.0][self.fix]
      self.initial_resolution_mm: Union[float, Tuple[float, float]] = [(3.0, 10.0), 4.0][self.fix] # The resolution determines the number of points in the outline. The distance between the points is resolution units (mm).
      self.initial_outline_sides: Union[int, Tuple[int, int]] = [(4, 10), 8][self.fix] 
      self.initial_outline_aspect_ratio: Union[float, Tuple[float, float]] = [(1.0, 1.5), 1.55][self.fix]
      self.initial_outline_longer_axis: Union[str, Tuple[str, str]] = [('x', 'y'), 'x'][self.fix] 
      self.initial_outline_rotation_deg: Union[float, Tuple[float, float]] = [(0.0, 180.0), 0.0][self.fix]
      self.initial_smoothing_iters: Union[int, Tuple[int, int]] = [(1, 5), 5][self.fix]
      self.initial_smoothing_t: Union[float, Tuple[float, float]] = [(0.0, 0.5), 0.25][self.fix] 

      
      self.outline_points = np.empty((0, 2), dtype=float)

      # fixed values
      self.cut_sharp_outline_threshold_deg: float = 135

   def finger_thumb(self): # Finger and thumb geometry
      self.thumb_side : str = 'random' # 'random', 'left', 'right'
      self.initial_finger_number: Union[int, Tuple[int, int]] = [(1, 5), 3][self.fix] # Based on the finger and thumb number, the size of the palm is affected.
      self.initial_thumb_number: Union[int, Tuple[int, int]] = [(1, 3), 1][self.fix]
      self.initial_finger_angle_deg: Union[float, Tuple[float, float]] = [(-5, 5), 0][self.fix]
      self.initial_thumb_angle_deg: Union[float, Tuple[float, float]] = [(-5, 5), 0][self.fix]
      self.initial_finger_base_normal_offset_mm: Union[float, Tuple[float, float]] = [(-5, 5), 0][self.fix]
      self.initial_thumb_base_normal_offset_mm: Union[float, Tuple[float, float]] = [(-5, 0), 0][self.fix]
      self.initial_finger_base_side_offset_mm: Union[float, Tuple[float, float]] = [(-1, 1), 0][self.fix]
      self.initial_thumb_base_side_offset_mm: Union[float, Tuple[float, float]] = [(-1, 1), 0][self.fix]

      self.finger_angle_deg_list: list[float] = []
      self.finger_base_normal_offset_mm_list: list[float] = []  
      self.thumb_angle_deg_list: list[float] = []
      self.thumb_base_normal_offset_mm_list: list[float] = []
      self.finger_base_side_offset_mm_list: list[float] = []
      self.thumb_base_side_offset_mm_list: list[float] = []

      # fixed values
      self.finger_ignore_y_threshold_mm: float = 35.0
      self.thumb_ignore_y_threshold_mm: float = 30.0

      # stored values
      self.finger_points: np.ndarray = np.empty((0, 2), dtype=float)  
      self.thumb_points: np.ndarray = np.empty((0, 2), dtype=float)
      self.finger_bases: np.ndarray = np.empty((0, 2), dtype=float)
      self.thumb_bases: np.ndarray = np.empty((0, 2), dtype=float)
      self.finger_bases_normal_vectors: np.ndarray = np.empty((0, 2), dtype=float)
      self.thumb_bases_normal_vectors: np.ndarray = np.empty((0, 2), dtype=float)
      self.thumb_side_list: list[str] = []
      self.finger_points_outline_index_list: list[int] = []
      self.thumb_points_outline_index_list: list[int] = []

   def pad(self): # Surface pad geometry
      self.pad_resolution_level = 3 #7
      self.pad_thickness_mm = 1.0
      self.bump_edge_zeroing_spread = 0.3
      
      self.initial_bump_type: Union[str, Tuple[str, ...]] = [('gaussian', 'conic', 'spherical'), 'gaussian'][self.fix]
      self.initial_bump_number: Union[int, Tuple[int, int]] = [(0, 3), 1][self.fix]
      self.initial_bump_max_height_intensity_mm: Union[float, Tuple[float, float]] = [(0.0, 15.0), 0.0][self.fix]
      self.initial_bump_height_intensity: Union[float, Tuple[float, float]] = [(0.0, 1.0), 1][self.fix]
      self.initial_bumps_spread: Union[float, Tuple[float, float]] = [(0.1, 0.5), 0.25][self.fix] # sigma scaled to the surface size
      self.initial_bumps_aspect_ratio: Union[float, Tuple[float, float]] = [(1.0, 3.0), 1.0][self.fix]
      self.initial_bump_rotation_deg: Union[float, Tuple[float, float]] = [(0, 180), 0][self.fix]
      self.initial_bump_center_angle_deg: Union[float, Tuple[float, float]] = [(0.0, 360.0), 0.0][self.fix] # bump center angle to x-axis
      self.initial_bump_center_offset: Union[float, Tuple[float, float]] = [(0.0, 1.0), 0.0][self.fix] # bump center offset from the origin of the surface outline scaled to the surface size
      
      self.bump_height_intensity_list: list[float] = []
      self.bumps_spread_list: list[float] = []
      self.bumps_aspect_ratio_list: list[float] = []
      self.bump_rotation_deg_list: list[float] = []
      self.bump_center_angle_deg_list: list[float] = []
      self.bump_center_offset_list: list[float] = []

   def segments(self): # Fixed finger and thumb segments geometry
      self.finger_base_seg_mm = 32.0
      self.finger_base_seg_margin_mm = 8.0
      self.thumb_corner_seg1_mm = 40.0
      self.thumb_corner_seg2_mm = 25.0
      self.thumb_seg1_margin_offset_mm = 25.0
      self.thumb_seg2_margin_offset_mm = 5.0
      self.thumb_base_corner_offset_mm = (10.0, 10.0)
      self.thumb_corner_outline_offset_mm = float(np.sqrt((self.thumb_corner_seg1_mm**2 + self.thumb_corner_seg2_mm**2))/2)
      self.finger_min_spacing_mm = self.finger_base_seg_mm + self.finger_base_seg_margin_mm#*2 + 3
      self.thumb_min_spacing_mm = (self.thumb_corner_seg1_mm**2 + self.thumb_seg1_margin_offset_mm**2)**0.5 + 30

   def generate_values(self):  # check if the config items are tuples or single values, then assign to the variables
      # finger and thumb number
      self.thumb_number = random.randint(self.initial_thumb_number[0], self.initial_thumb_number[1]) if isinstance(self.initial_thumb_number, tuple) else self.initial_thumb_number
      self.finger_number = random.randint(self.initial_finger_number[0], self.initial_finger_number[1]) if isinstance(self.initial_finger_number, tuple) else self.initial_finger_number

      # finger and thumb angles
      while len(self.thumb_angle_deg_list) < self.thumb_number:
         if isinstance(self.initial_thumb_angle_deg, tuple):
               self.thumb_angle_deg_list.append(float(random.uniform(self.initial_thumb_angle_deg[0], self.initial_thumb_angle_deg[1])))
         else:
               self.thumb_angle_deg_list.append(float(self.initial_thumb_angle_deg))

      while len(self.finger_angle_deg_list) < self.finger_number:
         if isinstance(self.initial_finger_angle_deg, tuple):
               self.finger_angle_deg_list.append(float(random.uniform(self.initial_finger_angle_deg[0], self.initial_finger_angle_deg[1])))
         else:
               self.finger_angle_deg_list.append(float(self.initial_finger_angle_deg))

      # finger and thumb base offsets
      while len(self.thumb_base_normal_offset_mm_list) < self.thumb_number:
         if isinstance(self.initial_thumb_base_normal_offset_mm, tuple):
               self.thumb_base_normal_offset_mm_list.append(float(random.uniform(self.initial_thumb_base_normal_offset_mm[0], self.initial_thumb_base_normal_offset_mm[1])))
         else:
               self.thumb_base_normal_offset_mm_list.append(float(self.initial_thumb_base_normal_offset_mm))

      while len(self.finger_base_normal_offset_mm_list) < self.finger_number:
         if isinstance(self.initial_finger_base_normal_offset_mm, tuple):
               self.finger_base_normal_offset_mm_list.append(float(random.uniform(self.initial_finger_base_normal_offset_mm[0], self.initial_finger_base_normal_offset_mm[1])))
         else:
               self.finger_base_normal_offset_mm_list.append(float(self.initial_finger_base_normal_offset_mm))



      while len(self.finger_base_side_offset_mm_list) < self.finger_number:
         if isinstance(self.initial_finger_base_side_offset_mm, tuple):
               self.finger_base_side_offset_mm_list.append(float(random.uniform(self.initial_finger_base_side_offset_mm[0], self.initial_finger_base_side_offset_mm[1])))
         else:
               self.finger_base_side_offset_mm_list.append(float(self.initial_finger_base_side_offset_mm))

      while len(self.thumb_base_side_offset_mm_list) < self.thumb_number:
         if isinstance(self.initial_thumb_base_side_offset_mm, tuple):
               self.thumb_base_side_offset_mm_list.append(float(random.uniform(self.initial_thumb_base_side_offset_mm[0], self.initial_thumb_base_side_offset_mm[1])))
         else:
               self.thumb_base_side_offset_mm_list.append(float(self.initial_thumb_base_side_offset_mm))


      # palm size
      if isinstance(self.initial_palm_size_mm, tuple):
         self.palm_size_mm = float(random.uniform(self.initial_palm_size_mm[0], self.initial_palm_size_mm[1]))
      else:
         self.palm_size_mm = float(self.initial_palm_size_mm)
      self.palm_size_mm = max(self.palm_size_mm, float(self.thumb_number * (self.thumb_corner_seg1_mm**2 + self.thumb_seg1_margin_offset_mm**2)**0.5))
      self.palm_size_mm = max(self.palm_size_mm, 1.4*float(self.finger_number * self.finger_min_spacing_mm))

      # outline
      self.resolution_mm = float(random.uniform(self.initial_resolution_mm[0], self.initial_resolution_mm[1])) if isinstance(self.initial_resolution_mm, tuple) else self.initial_resolution_mm
      self.outline_sides = random.randint(self.initial_outline_sides[0], self.initial_outline_sides[1]) if isinstance(self.initial_outline_sides, tuple) else self.initial_outline_sides
      self.outline_aspect_ratio = float(random.uniform(self.initial_outline_aspect_ratio[0], self.initial_outline_aspect_ratio[1])) if isinstance(self.initial_outline_aspect_ratio, tuple) else self.initial_outline_aspect_ratio
      self.outline_longer_axis = random.choice(self.initial_outline_longer_axis) if isinstance(self.initial_outline_longer_axis, tuple) else self.initial_outline_longer_axis
      self.outline_rotation_deg = float(random.uniform(self.initial_outline_rotation_deg[0], self.initial_outline_rotation_deg[1])) if isinstance(self.initial_outline_rotation_deg, tuple) else self.initial_outline_rotation_deg
      self.smoothing_iters = random.randint(self.initial_smoothing_iters[0], self.initial_smoothing_iters[1]) if isinstance(self.initial_smoothing_iters, tuple) else self.initial_smoothing_iters
      self.smoothing_t = random.uniform(self.initial_smoothing_t[0], self.initial_smoothing_t[1]) if isinstance(self.initial_smoothing_t, tuple) else self.initial_smoothing_t

      # pad bumps
      self.bump_type = random.choice(self.initial_bump_type) if isinstance(self.initial_bump_type, tuple) else self.initial_bump_type
      self.bumps_number = random.randint(self.initial_bump_number[0], self.initial_bump_number[1]) if isinstance(self.initial_bump_number, tuple) else self.initial_bump_number
      self.bump_max_height_intensity_mm = float(random.uniform(self.initial_bump_max_height_intensity_mm[0], self.initial_bump_max_height_intensity_mm[1])) if isinstance(self.initial_bump_max_height_intensity_mm, tuple) else self.initial_bump_max_height_intensity_mm

      # pad bump height intensity
      while len(self.bump_height_intensity_list) < self.bumps_number:
         if isinstance(self.initial_bump_height_intensity, tuple):
               self.bump_height_intensity_list.append(float(random.uniform(self.initial_bump_height_intensity[0], self.initial_bump_height_intensity[1])))
         else:
               self.bump_height_intensity_list.append(float(self.initial_bump_height_intensity))

      # pad bumps spread
      while len(self.bumps_spread_list) < self.bumps_number:
         if isinstance(self.initial_bumps_spread, tuple):
               self.bumps_spread_list.append(float(random.uniform(self.initial_bumps_spread[0], self.initial_bumps_spread[1])))
         else:
               self.bumps_spread_list.append(float(self.initial_bumps_spread))

      # pad bumps aspect ratio
      while len(self.bumps_aspect_ratio_list) < self.bumps_number:
         if isinstance(self.initial_bumps_aspect_ratio, tuple):
               self.bumps_aspect_ratio_list.append(float(random.uniform(self.initial_bumps_aspect_ratio[0], self.initial_bumps_aspect_ratio[1])))
         else:
               self.bumps_aspect_ratio_list.append(float(self.initial_bumps_aspect_ratio))

      # pad bump rotation degree
      while len(self.bump_rotation_deg_list) < self.bumps_number:
         if isinstance(self.initial_bump_rotation_deg, tuple):
               self.bump_rotation_deg_list.append(float(random.uniform(self.initial_bump_rotation_deg[0], self.initial_bump_rotation_deg[1])))
         else:
               self.bump_rotation_deg_list.append(float(self.initial_bump_rotation_deg))

      # pad bump center angle degree
      while len(self.bump_center_angle_deg_list) < self.bumps_number:
         if isinstance(self.initial_bump_center_angle_deg, tuple):
               self.bump_center_angle_deg_list.append(float(random.uniform(self.initial_bump_center_angle_deg[0], self.initial_bump_center_angle_deg[1])))
         else:
               self.bump_center_angle_deg_list.append(float(self.initial_bump_center_angle_deg))

      # pad bump center offset
      while len(self.bump_center_offset_list) < self.bumps_number:
         if isinstance(self.initial_bump_center_offset, tuple):
               self.bump_center_offset_list.append(float(random.uniform(self.initial_bump_center_offset[0], self.initial_bump_center_offset[1])))
         else:
               self.bump_center_offset_list.append(float(self.initial_bump_center_offset))




      # initialize stored values
      self.finger_points = np.empty((0, 2), dtype=float)  
      self.thumb_points = np.empty((0, 2), dtype=float)
      self.finger_bases = np.empty((0, 2), dtype=float)
      self.thumb_bases = np.empty((0, 2), dtype=float)
      self.finger_bases_normal_vectors = np.empty((0, 2), dtype=float)
      self.thumb_bases_normal_vectors = np.empty((0, 2), dtype=float)

   def initialize_outline(self):
      if self.mode == "draw":
         self.draw_outline()
      else:
         print("Initializing outline from config values")
         self.outline_polygon()
         # self.cut_sharp_outline()
      
   def outline_polygon(self):#
      # make the polygone with the sides, palm size, rotation, and aspect ratio. Then divide the polygone sides into segments based on the resolution.
      N = self.outline_sides if isinstance(self.outline_sides, int) else warning("Outline sides is not initialized correctly", 8)
      r = self.palm_size_mm / 2.0 if isinstance(self.palm_size_mm, float) else warning("Palm size is not initialized correctly", 50.0)
      rot_rad = np.deg2rad(self.outline_rotation_deg) if isinstance(self.outline_rotation_deg, float) else warning("Outline rotation degree is not initialized correctly", 0.0)

      # Create the N polygon points
      angles = np.linspace(0, 2*np.pi, N, endpoint=False)
      poly = np.vstack([r * np.cos(angles), r * np.sin(angles)]).T

      # rotate the polygon
      R = np.array([[np.cos(rot_rad), -np.sin(rot_rad)], [np.sin(rot_rad), np.cos(rot_rad)]]) # Rotate polygon
      poly = poly @ R.T

      
      # stretch x axis
      if self.outline_longer_axis == 'x':
            poly[:, 1] = poly[:, 1] / self.outline_aspect_ratio
      else:
            poly[:, 0] = poly[:, 0] / self.outline_aspect_ratio



      # interpolate points on each edge
      outline_points = []
      poly_copy = poly.copy()
      for i in range(N):
            p1 = poly_copy[i]
            p2 = poly_copy[(i + 1) % N]
            edge_length = np.linalg.norm(p2 - p1)
            num_points = max(int(np.floor(edge_length / self.resolution_mm)), 1) if isinstance(self.resolution_mm, float) else warning("Resolution is not initialized correctly",1)
            for j in range(num_points):
               t = j / num_points
               pt = (1 - t) * p1 + t * p2
               outline_points.append(pt)
      outline_points = np.array(outline_points)

      # # reverse the order of the points
      # outline_points = outline_points[::-1]
      # roll the points so that the one with the largest x and closest y to zero is the first one
      self.outline_points = np.roll(outline_points, -np.argmin(-outline_points[:, 0] + np.abs(outline_points[:, 1])), axis=0)
      self.cut_sharp_outline()

      # import matplotlib.pyplot as plt
      # print(f"N: {N}")
      # plt.plot(self.outline_points[:, 0], self.outline_points[:, 1], 'o')
      # plt.axis('equal')
      # plt.show()

   def cut_sharp_outline(self):#
      # cut the sharp corners of the outline.
      # the sharp corners are the corners with an angle smaller than cut_sharp_outline_threshold_deg.
      if self.outline_points is None or self.outline_points.shape[0] < 3:
         print("Error: Outline points are not initialized correctly")
         return

      temp_points = self.outline_points.copy()
      n = self.outline_points.shape[0]
      count = 0
      for i in range(n):
         prev_idx = (i - 1) % n
         next_idx = (i + 1) % n
         p1 = self.outline_points[prev_idx]
         p2 = self.outline_points[i]
         p3 = self.outline_points[next_idx]
         v1 = p1 - p2
         v2 = p3 - p2
         norm1 = np.linalg.norm(v1)
         norm2 = np.linalg.norm(v2)

         cosang = np.dot(v1, v2) / (norm1 * norm2)
         cosang = np.clip(cosang, -1.0, 1.0)  # numerical safety
         angle = np.arccos(cosang)
         if angle < np.deg2rad(self.cut_sharp_outline_threshold_deg):
               temp_points = np.delete(temp_points, i-count, axis=0)
               count += 1

      self.outline_points = temp_points

   def draw_outline(self):#
      # open up a new window and user can draw the outline lefclicking and hoevering the mouse.
      # Hold left mouse button and drag to draw; press Enter/Esc to finish.
      points = []
      min_dist = float(self.resolution_mm) if isinstance(self.resolution_mm, float) else warning("Resolution is not initialized correctly", 5.0)
      drawing = {'active': False}

      import matplotlib.pyplot as plt
      fig, ax = plt.subplots()
      ax.set_title('Draw outline: hold Left Mouse and drag. Press Enter/Esc to finish.')
      ax.set_xlim(-200, 200)
      ax.set_ylim(-200, 200)
      ax.axis('equal')
      ax.grid(False)

      # If an outline already exists, show it lightly for reference
      # if hasattr(self, 'outline_points') and isinstance(self.outline_points, np.ndarray) and self.outline_points.shape[0] > 0:
      #     ax.plot(self.outline_points[:, 0], self.outline_points[:, 1], color='lightgray', linewidth=1, alpha=0.5)

      line, = ax.plot([], [], color='black', linewidth=2)

      def add_point(x, y):
         if x is None or y is None:
               return
         p = np.array([x, y], dtype=float)
         if len(points) == 0 or np.linalg.norm(p - points[-1]) >= min_dist * 0.25:
               points.append(p)
               xs = [pt[0] for pt in points]
               ys = [pt[1] for pt in points]
               line.set_data(xs, ys)
               fig.canvas.draw_idle()

      def on_press(event):
         if event.button == 1 and event.inaxes == ax:
               drawing['active'] = True
               add_point(event.xdata, event.ydata)

      def on_release(event):
         if event.button == 1:
               drawing['active'] = False

      def on_move(event):
         if drawing['active'] and event.inaxes == ax:
               add_point(event.xdata, event.ydata)

      def on_key(event):
         if event.key in ('enter', 'return', 'escape'):
               plt.close(fig)

      fig.canvas.mpl_connect('button_press_event', on_press)
      fig.canvas.mpl_connect('button_release_event', on_release)
      fig.canvas.mpl_connect('motion_notify_event', on_move)
      fig.canvas.mpl_connect('key_press_event', on_key)

      plt.show()

      # Post-process points
      if len(points) >= 3:
         pts = np.array(points, dtype=float)

         # Downsample to approximately resolution spacing
         simplified = [pts[0]]
         for p in pts[1:]:
               if np.linalg.norm(p - simplified[-1]) >= min_dist:
                  simplified.append(p)
         if len(simplified) >= 3:
               pts = np.array(simplified, dtype=float)

         # Close polyline if not already close
         if np.linalg.norm(pts[0] - pts[-1]) > min_dist * 0.5:
               pts = np.vstack([pts, pts[0]])

         # Roll start to the point with largest x and smallest |y| (like outline_polygon)
         start_idx = int(np.argmin(-pts[:, 0] + np.abs(pts[:, 1])))
         pts = np.roll(pts, -start_idx, axis=0)

         self.outline_points = pts
      else:
         print("Error: less than 3 points are drawn")
         return

      self.palm_size_mm = np.max(np.linalg.norm(self.outline_points, axis=1)) * 2.0
      # center the outline points around the origin and reorder the points so that the one with the largest x and closest y to zero is the first one.
      self.outline_points = self.outline_points - np.mean(self.outline_points, axis=0)
      self.outline_points = self.outline_points[np.argsort(np.arctan2(self.outline_points[:, 1], self.outline_points[:, 0]))]

   def finger_thumb_points_selection(self):
      # select the finger and thumb bases from the outline points (finger_number + thumb_number). 
      # Finger chosen from the points with y>finger_ignore_y_threshold_mm and thumb from y<-thumb_ignore_y_threshold_mm.
      # Distance between the selected points should be at least config.finger_min_spacing_mm and config.thumb_min_spacing_mm.
      if self.finger_number == 0 and self.thumb_number == 0:
         print("Error: Finger and thumb number are not initialized correctly, is zero")
         return
      elif isinstance(self.finger_number, tuple) or isinstance(self.thumb_number, tuple):
         print("Error: Finger and thumb number are not initialized correctly, is tuple")
         return

      if len(self.finger_points) == self.finger_number:
         print('Using the config finger points locations')
         finger_points_selected = True
         finger_points = self.finger_points.copy()
      else:
         print('Generating new finger points locations')
         finger_points_selected = False

      if len(self.thumb_points) == self.thumb_number:
         print('Using the config thumb points locations')
         thumb_points_selected = True
         thumb_points = self.thumb_points.copy()
      else:
         print('Generating new thumb points locations')
         thumb_points_selected = False

      if finger_points_selected and thumb_points_selected:
         return

      rearrange = True
      count = 0
      while rearrange:
         rearrange = False
         count += 1
         if count > 300:
               if self.mode != "draw":
                  print("Reinitializing the outline and finger and thumb points")
                  self.generate_values()
                  self.initialize_outline()
                  finger_points_selected = False
                  thumb_points_selected = False
                  count = 0
               else:
                  #scale up the outline points by 1.5
                  self.outline_points = self.outline_points * 1.5
         

         if not finger_points_selected:
               finger_outline_points = self.outline_points[self.outline_points[:, 1] > self.finger_ignore_y_threshold_mm].copy()
               finger_points = []
               for i in range(self.finger_number):
                  for iter in range(100):
                     if len(finger_outline_points) == 0:
                           rearrange = True
                           break
                     idx = random.randrange(len(finger_outline_points))
                     finger_point = finger_outline_points[idx]
                     idx_outline = int(np.argmin(np.linalg.norm(self.outline_points - finger_point, axis=1)))
                     if i == 0:
                           finger_points.append(finger_point)
                           finger_outline_points = np.delete(finger_outline_points, idx, axis=0)
                           self.finger_points_outline_index_list.append(idx_outline)
                           break
                     else: # check the distance with all the previous finger bases
                           if all(np.linalg.norm(finger_point - fp) >= self.finger_min_spacing_mm for fp in finger_points):
                              finger_points.append(finger_point)
                              finger_outline_points = np.delete(finger_outline_points, idx, axis=0)
                              self.finger_points_outline_index_list.append(idx_outline)
                              break
                           elif iter == 99:
                              # print(f"Finger base #{i} is too close to the previous finger bases. Reinitializing the finger bases.")
                              rearrange = True
                              break
               if len(finger_points) == self.finger_number:
                  finger_points_selected = True
               else:
                  finger_points_selected = False
                  rearrange = True

         if finger_points_selected and not thumb_points_selected:
               thumb_outline_points = self.outline_points[self.outline_points[:, 1] < -self.thumb_ignore_y_threshold_mm].copy()
               # select the thumb points with negative x value if thumb_side is left, and the thumb points with positive x value if thumb_side is right.
               if self.thumb_side == 'left':
                  thumb_outline_points = thumb_outline_points[thumb_outline_points[:, 0] < 0]
               elif self.thumb_side == 'right':
                  thumb_outline_points = thumb_outline_points[thumb_outline_points[:, 0] > 0]

               thumb_points = []
               for i in range(self.thumb_number):
                  for iter in range(100):
                     if len(thumb_outline_points) == 0:
                           rearrange = True
                           break
                     idx = random.randrange(len(thumb_outline_points))
                     thumb_point = thumb_outline_points[idx]
                     idx_outline = int(np.argmin(np.linalg.norm(self.outline_points - thumb_point, axis=1)))
                     if i == 0:
                           thumb_points.append(thumb_point)
                           thumb_outline_points = np.delete(thumb_outline_points, idx, axis=0)
                           self.thumb_points_outline_index_list.append(idx_outline)
                           break
                     else: # check the distance with all the previous thumb bases
                           if all(np.linalg.norm(thumb_point - tp) >= self.thumb_min_spacing_mm for tp in thumb_points):
                              thumb_points.append(thumb_point)
                              thumb_outline_points = np.delete(thumb_outline_points, idx, axis=0)
                              self.thumb_points_outline_index_list.append(idx_outline)
                              break
                           elif iter == 99:
                              # print(f"Thumb base #{i} is too close to the previous thumb bases. Reinitializing the thumb bases.")
                              rearrange = True
                              break
               if len(thumb_points) == self.thumb_number:
                  thumb_points_selected = True
               else:
                  thumb_points_selected = False
                  rearrange = True

      self.finger_points = np.array(finger_points)
      self.thumb_points = np.array(thumb_points)

      # order the finger and thumb points by the angle of the vector connecting the origin to the point with x axis (low to high).
      self.finger_points = self.finger_points[np.argsort(np.arctan2(self.finger_points[:, 1], self.finger_points[:, 0]))]
      self.thumb_points = self.thumb_points[np.argsort(np.arctan2(self.thumb_points[:, 1], -self.thumb_points[:, 0]))]

   def update(self):
      if self.mode == "fixed":
         self.initialize_outline()
      self.finger_thumb_points_selection()

class FingerConfig:
   def __init__(self, type: str = 'finger', data: str = "fixed", code: str = 'xxx-xx-xxxxxxxxxx', id: Union[int, None] = None):
      '''
      type: str = 'finger' or 'thumb_right' or 'thumb_left' or 'thumb'
      data: str = 'fixed' or 'random'
      code: str = 'xxx-xx-xxxxxxxxxx' or 'x-x-xxxxxxxx'
      '''
      self.id = id
      self.type = type
      self.code = code
      self.fingertip_scale_factor = (1.0, 1.0, 1.0)

      self.fix = 'fixed' == data
      self.geometry()
      self.pad()
      self.code_data()
      self.generate()
      self.generate_pads()


      if data in ["fixed", "random"]:
         self.mode = data
      else:
         self.mode = "fixed"
         try:
            self.import_config(data)
         except:
            raise ValueError(f"invalid finger config data type: {data}")

   def import_config(self, path):
      path = Path(path)
      spec = importlib.util.spec_from_file_location(path.stem, path)
      if spec is None or spec.loader is None:
         raise ValueError(f"Could not import module from path: {path}")
      module = importlib.util.module_from_spec(spec)
      spec.loader.exec_module(module)

      for k, _ in vars(self).items():
         if k == "mode":
            continue
         setattr(self, k, getattr(module, k))

   def save_config(self, path):
      with open(path, "w") as f:
         for k, v in vars(self).items():
            f.write(f"{k} = {repr(v)}\n")

   def geometry(self):
      self.initial_link_added_length_mm: Union[float, Tuple[float, float]] = [(0.0, 10.0), 0.0][self.fix]
      # self.initial_link_added_width_mm: Union[float, Tuple[float, float]] = [(-4.0, 10.0), 0.0][self.fix]

      self.link_added_length_mm_list: list[float] = []
      # self.link_added_width_mm_list: list[float] = []

   def pad(self):
      self.pad_resolution_level = 2 #4
      self.pad_thickness_mm = 1.0
      self.bump_edge_zeroing_spread = 0.3
      
      self.initial_bump_type: Union[str, Tuple[str, ...]] = [('gaussian', 'conic', 'cubic', 'spherical'), 'gaussian'][self.fix]
      self.initial_bump_number: Union[int, Tuple[int, int]] = [(0, 3), 0][self.fix]
      self.initial_bump_max_height_intensity_mm: Union[float, Tuple[float, float]] = [(0.0, 4.0), 0.0][self.fix]
      self.initial_bump_height_intensity: Union[float, Tuple[float, float]] = [(0.0, 1.0), 1][self.fix]
      self.initial_bumps_spread: Union[float, Tuple[float, float]] = [(0.1, 0.5), 0.25][self.fix] # sigma scaled to the surface size
      self.initial_bumps_aspect_ratio: Union[float, Tuple[float, float]] = [(1.0, 3.0), 1.0][self.fix]
      self.initial_bump_rotation_deg: Union[float, Tuple[float, float]] = [(0, 180), 0][self.fix]
      self.initial_bump_center_angle_deg: Union[float, Tuple[float, float]] = [(0.0, 360.0), 0.0][self.fix] # bump center angle to x-axis
      self.initial_bump_center_offset: Union[float, Tuple[float, float]] = [(0.0, 1.0), 0.0][self.fix] # bump center offset from the origin of the surface outline scaled to the surface size
   
      self.bump_type_list: list[str] = []
      self.bump_number_list: list[int] = []
      self.bump_max_height_intensity_mm_list: list[float] = []

      self.bump_height_intensity_list: list[list[float]] = []
      self.bumps_spread_list: list[list[float]] = []
      self.bumps_aspect_ratio_list: list[list[float]] = []
      self.bump_rotation_deg_list: list[list[float]] = []
      self.bump_center_angle_deg_list: list[list[float]] = []
      self.bump_center_offset_list: list[list[float]] = []

      # Fixed Surface values
      self.motor_small_surface_dim_mm = (18, 18)
      self.motor_large_surface_dim_mm = (18, 28)

   def code_data(self):
      # Code Generation Parameters
      self.initial_min_add_joint = 2
      self.initial_max_add_joint = 5 if self.type == 'finger' else 4
      self.initial_rotation_mode = ('000', '100', '101', '110', '111') if self.type == 'finger' else ('0', '1')
      self.initial_add_joint_type = ('01', '10')

      self.add_joint_type: Union[str, Tuple[str, str]] = [self.initial_add_joint_type, '01'][self.fix]
      self.rotation_mode: Union[str, Tuple[str, str]] = [self.initial_rotation_mode, '000'][self.fix] if self.type == 'finger' else [self.initial_rotation_mode, '1'][self.fix]
      self.min_add_joint = self.initial_min_add_joint
      self.max_add_joint = self.initial_max_add_joint
      self.add_joint = [None, self.min_add_joint][self.fix]
      self.before_joints = [None, 0][self.fix] if self.type == 'finger' else 0
      self.after_joints = [None, self.min_add_joint][self.fix]
      self.add_joint_type_str = ''

   def generate_finger_code(self):
      # assign rotation mode
      if self.code.startswith('x'):
         if isinstance(self.rotation_mode, tuple):
            self.rotation_mode = random.choice(self.rotation_mode)
      elif self.code.startswith('1') and 'x' in self.code.split('-')[0]:
         if isinstance(self.rotation_mode, tuple):
            self.rotation_mode = random.choice([x for x in self.rotation_mode if x != '000'])
      else:
         self.rotation_mode = self.code.split('-')[0]
        
      # assign before and after joints
      if self.rotation_mode != '000':
         self.max_add_joint -= 1
      if self.rotation_mode != '100':
         self.max_add_joint -= 1
      if self.rotation_mode == '101' or self.rotation_mode == '110':
         self.min_add_joint -= 1

      if self.code.split('-')[1][0] != 'x':
         self.before_joints = int(self.code.split('-')[1][0])
      if self.code.split('-')[1][1] != 'x':
         self.after_joints = int(self.code.split('-')[1][1])

      if self.before_joints is not None and self.after_joints is not None:
         while self.check_minmax(self.before_joints + self.after_joints) != 0:
            adjust = self.check_minmax(self.before_joints + self.after_joints)
            if adjust == -1:
               if self.before_joints > 0: self.before_joints += adjust
               else: self.after_joints += adjust
            elif adjust == 1:
               self.after_joints += adjust
         self.add_joint = self.before_joints + self.after_joints

      elif self.before_joints is not None:
         self.add_joint = random.randint(self.min_add_joint, self.max_add_joint)
         while self.check_minmax(self.before_joints) != 0:
            self.before_joints += self.check_minmax(self.before_joints)
         self.after_joints = self.add_joint - self.before_joints

      elif self.after_joints is not None:
         self.add_joint = random.randint(self.min_add_joint, self.max_add_joint)
         while self.check_minmax(self.after_joints) != 0:
            self.after_joints += self.check_minmax(self.after_joints)
         self.before_joints = self.add_joint - self.after_joints
      
      else:
         self.add_joint = random.randint(self.min_add_joint, self.max_add_joint)
         self.before_joints = random.randint(0, self.add_joint)
         self.after_joints = self.add_joint - self.before_joints  

      self.assign_add_joint_type()
      self.code = f'{self.rotation_mode}-{self.before_joints}{self.after_joints}-{self.add_joint_type_str}'

   def generate_thumb_code(self):
      # assign rotation mode
      if self.code.startswith('x'):
         if isinstance(self.rotation_mode, tuple):
            self.rotation_mode = random.choice(self.rotation_mode)
      else:
         self.rotation_mode = self.code.split('-')[0]
        
      # assign before and after joints
      self.max_add_joint -= 1
      if self.rotation_mode == '1':
         self.max_add_joint -= 1

      if 'x' not in self.code.split('-')[1]:
         self.after_joints = int(self.code.split('-')[1][0])
         self.before_joints = 0

      if self.after_joints is not None:
         while self.check_minmax(self.after_joints) != 0:
            self.after_joints += self.check_minmax(self.after_joints)
         self.add_joint = self.after_joints
      else:
         self.add_joint = random.randint(self.min_add_joint, self.max_add_joint)
         self.after_joints = self.add_joint


      self.assign_add_joint_type()
      self.code = f'{self.rotation_mode}-{self.after_joints}-{self.add_joint_type_str}'

   def assign_add_joint_type(self):
      # assign add joint type
      self.add_joint_type_str = ''
      code_add_joint_type = self.code.split('-')[2]
      for i in range(self.initial_max_add_joint):
         if 'x' in code_add_joint_type[i*2:(i+1)*2]:
            self.add_joint_type_str += random.choice(self.add_joint_type) if isinstance(self.add_joint_type, tuple) else self.add_joint_type
         else:
            if code_add_joint_type[i*2:(i+1)*2] in self.initial_add_joint_type:
               self.add_joint_type_str += code_add_joint_type[i*2:(i+1)*2]
            else:
               print("Warning: Invalid add joint string")
               self.add_joint_type_str += random.choice(self.add_joint_type) if isinstance(self.add_joint_type, tuple) else self.add_joint_type

   def check_minmax(self, value):
      if value > self.max_add_joint:
         print("Warning: number of joints are not in the valid range")
         print(f"value: {value}, max_add_joint: {self.max_add_joint}, min_add_joint: {self.min_add_joint}")
         return -1
      if value < self.min_add_joint:
         print("Warning: number of joints are not in the valid range")
         print(f"value: {value}, max_add_joint: {self.max_add_joint}, min_add_joint: {self.min_add_joint}")
         return 1
      return 0

   def generate_pads(self):
      for i in range(self.initial_max_add_joint):
         # pad bumps
         self.bump_type_list.append(random.choice(self.initial_bump_type) if isinstance(self.initial_bump_type, tuple) else self.initial_bump_type)
         self.bump_number_list.append(random.randint(self.initial_bump_number[0], self.initial_bump_number[1]) if isinstance(self.initial_bump_number, tuple) else self.initial_bump_number)
         self.bump_max_height_intensity_mm_list.append(float(random.uniform(self.initial_bump_max_height_intensity_mm[0], self.initial_bump_max_height_intensity_mm[1])) if isinstance(self.initial_bump_max_height_intensity_mm, tuple) else self.initial_bump_max_height_intensity_mm)

         # pad bump height intensity
         if len(self.bump_height_intensity_list) <= i:
            self.bump_height_intensity_list.append([])
         while len(self.bump_height_intensity_list[i]) < self.bump_number_list[i]:
            if isinstance(self.initial_bump_height_intensity, tuple):
                  self.bump_height_intensity_list[i].append(float(random.uniform(self.initial_bump_height_intensity[0], self.initial_bump_height_intensity[1])))
            else:
                  self.bump_height_intensity_list[i].append(float(self.initial_bump_height_intensity))

         # pad bumps spread
         if len(self.bumps_spread_list) <= i:
            self.bumps_spread_list.append([])
         while len(self.bumps_spread_list[i]) < self.bump_number_list[i]:
            if isinstance(self.initial_bumps_spread, tuple):
                  self.bumps_spread_list[i].append(float(random.uniform(self.initial_bumps_spread[0], self.initial_bumps_spread[1])))
            else:
                  self.bumps_spread_list[i].append(float(self.initial_bumps_spread))

         # pad bumps aspect ratio
         if len(self.bumps_aspect_ratio_list) <= i:
            self.bumps_aspect_ratio_list.append([])
         while len(self.bumps_aspect_ratio_list[i]) < self.bump_number_list[i]:
            if isinstance(self.initial_bumps_aspect_ratio, tuple):
                  self.bumps_aspect_ratio_list[i].append(float(random.uniform(self.initial_bumps_aspect_ratio[0], self.initial_bumps_aspect_ratio[1])))
            else:
                  self.bumps_aspect_ratio_list[i].append(float(self.initial_bumps_aspect_ratio))

         # pad bump rotation degree
         if len(self.bump_rotation_deg_list) <= i:
            self.bump_rotation_deg_list.append([])
         while len(self.bump_rotation_deg_list[i]) < self.bump_number_list[i]:
            if isinstance(self.initial_bump_rotation_deg, tuple):
                  self.bump_rotation_deg_list[i].append(float(random.uniform(self.initial_bump_rotation_deg[0], self.initial_bump_rotation_deg[1])))
            else:
                  self.bump_rotation_deg_list[i].append(float(self.initial_bump_rotation_deg))

         # pad bump center angle degree
         if len(self.bump_center_angle_deg_list) <= i:
            self.bump_center_angle_deg_list.append([])
         while len(self.bump_center_angle_deg_list[i]) < self.bump_number_list[i]:
            if isinstance(self.initial_bump_center_angle_deg, tuple):
                  self.bump_center_angle_deg_list[i].append(float(random.uniform(self.initial_bump_center_angle_deg[0], self.initial_bump_center_angle_deg[1])))
            else:
                  self.bump_center_angle_deg_list[i].append(float(self.initial_bump_center_angle_deg))

         # pad bump center offset
         if len(self.bump_center_offset_list) <= i:
            self.bump_center_offset_list.append([])
         while len(self.bump_center_offset_list[i]) < self.bump_number_list[i]:
            if isinstance(self.initial_bump_center_offset, tuple):
                  self.bump_center_offset_list[i].append(float(random.uniform(self.initial_bump_center_offset[0], self.initial_bump_center_offset[1])))
            else:
                  self.bump_center_offset_list[i].append(float(self.initial_bump_center_offset))

   def generate_links(self):
      while len(self.link_added_length_mm_list) < self.initial_max_add_joint:
         if isinstance(self.initial_link_added_length_mm, tuple):
            self.link_added_length_mm_list.append(float(random.uniform(self.initial_link_added_length_mm[0], self.initial_link_added_length_mm[1])))
         else:
            self.link_added_length_mm_list.append(float(self.initial_link_added_length_mm))

   def generate(self):
      if self.type == 'finger':
         self.generate_finger_code()
      else:
         self.generate_thumb_code()
      self.generate_links()

class FingertipConfig:
   def __init__(self, data: str = "fixed"):
      self.fix = 'fixed' == data

      if data in ["fixed", "random"]:
         self.mode = data
      else:
         self.mode = "fixed"
         try:
            self.import_config(data)
         except:
            raise ValueError(f"invalid fingertip config data type: {data}")

   def import_config(self, path):
      path = Path(path)
      spec = importlib.util.spec_from_file_location(path.stem, path)
      if spec is None or spec.loader is None:
         raise ValueError(f"Could not import module from path: {path}")
      module = importlib.util.module_from_spec(spec)
      spec.loader.exec_module(module)
      for k, _ in vars(self).items():
         if k == "mode":
            continue
         setattr(self, k, getattr(module, k))

   def save_config(self, path):
      with open(path, "w") as f:
         for k, v in vars(self).items():
            f.write(f"{k} = {repr(v)}\n")

   def geometry(self):
      self.initial_outline_sides: Union[int, Tuple[int, int]] = [(3, 10), 20][self.fix]
      self.height_mm: Union[None, float, Tuple[float, float]] = [(10.0, 40.0), 30.0][self.fix]
      self.width_mm: Union[None, float, Tuple[float, float]] = [(10.0, 40.0), 30.0][self.fix]
      self.top_scale_ratio_x: Union[float, Tuple[float, float]] = [(0.0, 1.0), 0.5][self.fix]
      self.top_scale_origin_ratio_x: Union[float, Tuple[float, float]] = [(0.0, 1.0), 1][self.fix]
      self.top_scale_ratio_y: Union[float, Tuple[float, float]] = [(0.0, 1.0), 0.5][self.fix]
      self.top_scale_origin_ratio_y: Union[float, Tuple[float, float]] = [(0.0, 1.0), 0.5][self.fix]
      self.fillet_radius_mm: Union[float, Tuple[float, float]] = [(1.0, 8.0), 0.0][self.fix]
      self.fillet_segments: Union[int, Tuple[int, int]] = [(2, 8), 5][self.fix]

      self.outline_points: np.ndarray = np.empty((0, 3), dtype=float)
      self.vertices: np.ndarray = np.empty((0, 3), dtype=float)
      self.faces: np.ndarray = np.empty((0, 3), dtype=int)

      # fized values
      self.base_size = 20
      self.base_points = [(10.0,-10.0,7.0), (10.0,10.0,7.0)]

   def generate(self):
      self.generate_body_values()
      self.generate_body()


   def generate_body_values(self):
      # --- Step 1: Resolve random/fixed values ---
      self.N = random.randint(self.initial_outline_sides[0], self.initial_outline_sides[1]) if isinstance(self.initial_outline_sides, tuple) else self.initial_outline_sides
      self.height_mm = float(random.uniform(self.height_mm[0], self.height_mm[1])) if isinstance(self.height_mm, tuple) else (float(self.height_mm) if self.height_mm is not None else None)
      self.width_mm = float(random.uniform(self.width_mm[0], self.width_mm[1])) if isinstance(self.width_mm, tuple) else (float(self.width_mm) if self.width_mm is not None else None)
      self.top_scale_ratio_x = float(random.uniform(self.top_scale_ratio_x[0], self.top_scale_ratio_x[1])) if isinstance(self.top_scale_ratio_x, tuple) else float(self.top_scale_ratio_x)
      self.top_scale_origin_ratio_x = float(random.uniform(self.top_scale_origin_ratio_x[0], self.top_scale_origin_ratio_x[1])) if isinstance(self.top_scale_origin_ratio_x, tuple) else float(self.top_scale_origin_ratio_x)
      self.top_scale_ratio_y = float(random.uniform(self.top_scale_ratio_y[0], self.top_scale_ratio_y[1])) if isinstance(self.top_scale_ratio_y, tuple) else float(self.top_scale_ratio_y)
      self.top_scale_origin_ratio_y = float(random.uniform(self.top_scale_origin_ratio_y[0], self.top_scale_origin_ratio_y[1])) if isinstance(self.top_scale_origin_ratio_y, tuple) else float(self.top_scale_origin_ratio_y)
      self.fillet_radius_mm = float(random.uniform(self.fillet_radius_mm[0], self.fillet_radius_mm[1])) if isinstance(self.fillet_radius_mm, tuple) else float(self.fillet_radius_mm)
      self.fillet_segments = random.randint(self.fillet_segments[0], self.fillet_segments[1]) if isinstance(self.fillet_segments, tuple) else int(self.fillet_segments)

   def generate_body(self):
      """
      The fingertip outline is made up of a polygon with a side always at the bottom and with size self.base_size. 
      polygone is on the plane with self.base_size/2 offset from the world x-plane. and the bottom of the polygone is at z = 7.0mm.
      if height_mm is set, the z range is scaled so the top is at z = 7.0 + height_mm. If height_mm is None, the natural polygon height is used (just offset so bottom is at z=7.0).
      if width_mm is set, only non-bottom points are scaled in y by width_mm / (max(y) - min(y)). The bottom edge stays at its original y values (base_size). If width_mm is None, no y scaling is applied.
      the outline is extruded along the x-axis for self.base_size.
      the extruded outline is then modifiled by the self.top_scale_ratio_x, self.top_scale_origin_x, self.top_scale_ratio_y, self.top_scale_origin_y as follows:
      - the top_scale_origin_ratio_x determines a offset x-plane where value zero mean on the original polygone plane (offset of self.base_size/2) and value one mean at the opposite side of the polygone (offset of -self.base_size/2).
      - the top_scale_ratio_x determines the scale ratio of the x-axis for the top vertices. value one mean no scaling and value zero mean scaling to zero.
      - other vertices are gradually scaled toward the bottom vertices which are not scaled based on the z location of the vertices.
      - similar for the y-axis.
      - let t = (z - 7.0) / height_mm  (normalized height: 0 at bottom, 1 at top)
      - let origin_x = base_size/2 - top_scale_origin_ratio_x * base_size
      - let origin_y = base_size/2 - top_scale_origin_ratio_y * base_size
      - the new values for point (x, y, z) is:
        - x = origin_x + (x - origin_x) * (1 + t * (top_scale_ratio_x - 1))
        - y = origin_y + (y - origin_y) * (1 + t * (top_scale_ratio_y - 1))
        - z = z
      """

      # --- Step 2: Create regular polygon (all sides = base_size) with flat bottom ---
      S = self.base_size        # 20 mm
      z_bottom = 7.0
      N = self.N

      # Regular N-gon with side length S → circumradius R = S / (2·sin(π/N))
      # All sides are exactly base_size; the bottom side spans y ∈ [-S/2, S/2].
      R = S / (2.0 * np.sin(np.pi / N))

      # Vertex angles: bottom side midpoint at -π/2, vertex 0 = bottom-left
      angles = np.array([(-np.pi / 2 - np.pi / N) + 2 * np.pi * i / N for i in range(N)])
      y_pts = R * np.cos(angles)
      z_pts = R * np.sin(angles)

      # Scale z so bottom maps to z_bottom (7.0) and top to z_bottom + height_mm.
      # If height_mm is None, just offset so z_min sits at z_bottom (no height scaling).
      z_min = z_pts.min()
      z_max = z_pts.max()
      z_range = z_max - z_min
      if self.height_mm is not None:
         if z_range > 0:
            z_pts = z_bottom + (z_pts - z_min) / z_range * self.height_mm
         else:
            z_pts[:] = z_bottom
      else:
         z_pts = z_pts - z_min + z_bottom  # just shift, keep natural polygon height

      # Scale y for non-bottom points only, preserving the bottom edge at its original y values.
      # If width_mm is None, skip y-scaling entirely (use natural polygon y values).
      if self.width_mm is not None:
         y_min = y_pts.min()
         y_max = y_pts.max()
         y_range = y_max - y_min
         if y_range > 0:
            scale_y = self.width_mm / y_range
            for i in range(N):
               if i not in {0, 1}:  # skip bottom vertices
                  y_pts[i] = y_pts[i] * scale_y

      # Place all points on the polygon x-plane at x = base_size / 2
      x_pts = np.full(N, S / 2.0)

      # --- Step 3: Fillet corners (except bottom edge) ---
      # Replace each non-bottom corner with a circular arc of fillet_segments points.
      # Bottom vertices (indices 0 and 1) are kept sharp.
      yz = np.column_stack([y_pts, z_pts])  # (N, 2) polygon in y-z plane
      bottom_indices = {0, 1}
      radius = float(self.fillet_radius_mm)
      n_seg = int(self.fillet_segments)

      if radius > 0 and n_seg >= 1 and len(yz) >= 3:
         filleted = []
         n = len(yz)
         for i in range(n):
            if i in bottom_indices:
               filleted.append(yz[i])
               continue

            P = yz[i]
            A = yz[(i - 1) % n]
            B = yz[(i + 1) % n]

            # Edge directions from corner toward neighbors
            dA = A - P
            dB = B - P
            lenA = np.linalg.norm(dA)
            lenB = np.linalg.norm(dB)
            if lenA < 1e-9 or lenB < 1e-9:
               filleted.append(P)
               continue

            uA = dA / lenA
            uB = dB / lenB

            # Half-angle at the corner
            cos_half = np.dot(uA, uB)
            cos_half = np.clip(cos_half, -1.0, 1.0)
            half_angle = np.arccos(cos_half) / 2.0

            if abs(np.sin(half_angle)) < 1e-9:
               # Nearly straight or 180° — no fillet needed
               filleted.append(P)
               continue

            # Clamp radius so tangent length doesn't exceed half of either edge
            tan_len = radius / np.tan(half_angle)
            max_tan = min(lenA, lenB) * 0.45  # leave some margin
            r = radius
            if tan_len > max_tan:
               r = max_tan * np.tan(half_angle)
               tan_len = max_tan

            # Tangent points on each edge
            T1 = P + uA * tan_len
            T2 = P + uB * tan_len

            # Fillet center: offset from corner along the bisector
            bisector = uA + uB
            bis_len = np.linalg.norm(bisector)
            if bis_len < 1e-9:
               filleted.append(P)
               continue
            bisector = bisector / bis_len
            center_dist = r / np.sin(half_angle)
            C = P + bisector * center_dist

            # Compute arc angles from center
            angle_start = np.arctan2(T1[1] - C[1], T1[0] - C[0])
            angle_end = np.arctan2(T2[1] - C[1], T2[0] - C[0])

            # Ensure we sweep the shorter arc (the one on the polygon interior)
            diff = angle_end - angle_start
            if diff > np.pi:
               diff -= 2.0 * np.pi
            elif diff < -np.pi:
               diff += 2.0 * np.pi

            # Generate arc points (including both tangent endpoints)
            for k in range(n_seg + 1):
               t = k / n_seg
               a = angle_start + t * diff
               pt = C + r * np.array([np.cos(a), np.sin(a)])
               filleted.append(pt)

         yz = np.array(filleted)

      # --- Step 4: Store the 3D outline (x, y, z) ---
      x_pts = np.full(len(yz), S / 2.0)
      self.outline_points = np.column_stack([x_pts, yz])

   def extrude_and_scale(self):
      """
      Extrude the outline along the -x axis for base_size (from x=base_size/2 to x=-base_size/2),
      then apply the top-scaling transformation to all vertices.
      Produces self.vertices (2N, 3) and self.faces (F, 3).
      """
      S = self.base_size
      N = self.outline_points.shape[0]
      z_bottom = 7.0
      # Use actual z-range for normalization (works whether height_mm was set or None)
      height = self.outline_points[:, 2].max() - z_bottom

      # --- Extrude: front face at x = S/2, back face at x = -S/2 ---
      front = self.outline_points.copy()                 # (N, 3) already at x = S/2
      back = self.outline_points.copy()
      back[:, 0] = -S / 2.0                              # shift x to -S/2

      # Vertices: front = indices 0..N-1, back = indices N..2N-1
      vertices = np.vstack([front, back])

      # --- Build triangle faces ---
      faces = []

      # Front face: fan triangulation (CCW when viewed from +x → outward normal +x)
      for i in range(1, N - 1):
         faces.append([0, i, i + 1])

      # Back face: fan triangulation (reversed winding → outward normal -x)
      for i in range(1, N - 1):
         faces.append([N, N + i + 1, N + i])

      # Side faces: each polygon edge becomes a quad → 2 triangles
      for i in range(N):
         j = (i + 1) % N
         fi, fj, bi, bj = i, j, N + i, N + j
         faces.append([fi, bi, bj])
         faces.append([fi, bj, fj])

      faces = np.array(faces, dtype=int)

      # --- Apply top scaling to all vertices ---
      origin_x = S / 2.0 - self.top_scale_origin_ratio_x * S
      origin_y = S / 2.0 - self.top_scale_origin_ratio_y * S

      for vi in range(vertices.shape[0]):
         x, y, z = vertices[vi]
         t = np.clip((z - z_bottom) / height, 0.0, 1.0) if height > 0 else 0.0
         vertices[vi, 0] = origin_x + (x - origin_x) * (1.0 + t * (self.top_scale_ratio_x - 1.0))
         vertices[vi, 1] = origin_y + (y - origin_y) * (1.0 + t * (self.top_scale_ratio_y - 1.0))
         # z unchanged

      self.vertices = vertices
      self.faces = faces

   def save_mesh(self, directory: str, filename: str = "fingertip.stl"):
      """Save the mesh as a binary STL file to the given directory."""
      import struct

      out_dir = Path(directory)
      out_dir.mkdir(parents=True, exist_ok=True)
      filepath = out_dir / filename

      with open(filepath, "wb") as f:
         header = b"FingertipConfig generated mesh"
         f.write(header + b"\0" * (80 - len(header)))
         f.write(struct.pack("<I", len(self.faces)))

         for face in self.faces:
            v0 = self.vertices[face[0]]
            v1 = self.vertices[face[1]]
            v2 = self.vertices[face[2]]

            # Compute face normal
            e1 = v1 - v0
            e2 = v2 - v0
            normal = np.cross(e1, e2)
            norm_len = np.linalg.norm(normal)
            if norm_len > 0:
               normal = normal / norm_len

            f.write(struct.pack("<3f", float(normal[0]), float(normal[1]), float(normal[2])))
            f.write(struct.pack("<3f", float(v0[0]), float(v0[1]), float(v0[2])))
            f.write(struct.pack("<3f", float(v1[0]), float(v1[1]), float(v1[2])))
            f.write(struct.pack("<3f", float(v2[0]), float(v2[1]), float(v2[2])))
            f.write(struct.pack("<H", 0))

      print(f"Saved fingertip mesh: {filepath}")
      return str(filepath)


class HandConfig:
   def __init__(self, data: str = "fixed", origin_offset_mm: tuple[float, float, float] = (0.0, 0.0, 0.0)):
      self.collision_mesh: bool = True
      self.palm_geometry()
      self.thumb_mount_geometry()
      # self.origin_offset_mm = origin_offset_mm
      self.origin_offset_mm = (6.56, -38, 11.4)
      if data not in ["fixed", "random"]:
         try:
            self.import_config(data)
         except:
            raise ValueError(f"invalid hand config data type: {data}")
   

   def import_config(self, path):
      path = Path(path)
      spec = importlib.util.spec_from_file_location(path.stem, path)
      if spec is None or spec.loader is None:
         raise ValueError(f"Could not import module from path: {path}")
      module = importlib.util.module_from_spec(spec)
      spec.loader.exec_module(module)

      for k, _ in vars(self).items():
         if k == "mode":
            continue
         setattr(self, k, getattr(module, k))

   def save_config(self, path):
      with open(path, "w") as f:
         for k, v in vars(self).items():
            f.write(f"{k} = {repr(v)}\n")
   
   def palm_geometry(self):
      self.palm_extrude_height_mm = 45.0
      self.palm_wall_thickness_mm = 4.0
      self.palm_thickness_mm = 4.0
      self.finger_palm_base_offset_mm = 10.0

   def thumb_mount_geometry(self):
      self.thumb_mount_size_mm = (25, 7)
      self.thumb_mount_normal_offset_mm = self.thumb_mount_size_mm[1]/2.0
      self.thumb_mount_height_mm = self.palm_extrude_height_mm - self.palm_thickness_mm

class Plots:
   def __init__(self):
      default = False
      self.heightmap_plot_2d: bool = default 
      self.heightmap_plot_3d: bool = default 
      self.finger_plot_2d: bool = default 
      self.finger_plot_3d: bool = default 
      self.hand_plot_2d: bool = default
      self.hand_plot_3d: bool = default
      self.palm_outline_plot_2d: bool = default