import numpy as np
from typing import Union
import importlib.util
from pathlib import Path

def warning(message, default=None):
    print(f"Warning: {message}")
    return default

class PalmConfig:
   """Default configurations"""
   def __init__(self, data: str = "fixed", detailed_viz: bool = False):
      self.detailed_viz = detailed_viz
      self._derive_finger_locations_from_points = False
      self._derive_thumb_locations_from_points = False
      self._loaded_from_file = False

      self.outline_points = np.empty((0, 2), dtype=float)
      self.inner_outline_points = np.empty((0, 2), dtype=float)
      self.outline_semantics_version = 3
      self.cut_sharp_outline_threshold_deg: float = 135
      
      self.finger_thumb()
      self.pad()
      self.segments()
      self.generate_values()

      if data == "random":
         raise ValueError("PalmConfig(data='random') is no longer supported. Use ParamGen.PalmParams(...).build() to sample a random palm config.")
      if data in ["fixed", "draw"]:
         self.mode = data
      else:
         self.mode = "fixed"
         try:
            self.import_config(data)
         except:
            raise ValueError(f"invalid palm config data type: {data}")

      if self._loaded_from_file and self.outline_points.shape[0] >= 3:
         self._ensure_outline_semantics()
      else:
         self.initialize_outline()
      self._derive_missing_locations_from_saved_points()
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
      self._loaded_from_file = True
      self._derive_finger_locations_from_points = (not hasattr(module, "finger_location_list")) and hasattr(module, "finger_points")
      self._derive_thumb_locations_from_points = (not hasattr(module, "thumb_location_list")) and hasattr(module, "thumb_points")

      for k, _ in vars(self).items():
         if k == "mode" or k.startswith("_"):
            continue
         # tolerant: keep current default if the saved file lacks this attribute (schema drift)
         v = getattr(module, k, getattr(self, k))
         if k in ["outline_points", "inner_outline_points", "finger_points", "thumb_points", "finger_bases", "thumb_bases", "finger_bases_normal_vectors", "thumb_bases_normal_vectors"]:
            v = np.array(v)
         setattr(self, k, v)

   def save_config(self, path):
      with open(path, "w") as f:
         for k, v in vars(self).items():
            if k.startswith("_"):
               continue
            if k in ["outline_points", "inner_outline_points", "finger_points", "thumb_points", "finger_bases", "thumb_bases", "finger_bases_normal_vectors", "thumb_bases_normal_vectors"]:
               v = v.tolist()
            f.write(f"{k} = {repr(v)}\n")

   def finger_thumb(self): # Finger and thumb geometry
      self.generation_mode: str = 'standard' # 'standard' or 'finger_only'
      self.thumb_side : str = 'right' # 'random', 'left', 'right'

      self.finger_angle_deg_list: list[float] = []
      self.finger_base_normal_offset_mm_list: list[float] = []
      self.thumb_angle_deg_list: list[float] = []
      self.thumb_base_normal_offset_mm_list: list[float] = []
      self.finger_base_side_offset_mm_list: list[float] = []
      self.thumb_base_side_offset_mm_list: list[float] = []
      # location of each finger/thumb base along the valid outline curve, as a
      # fractional arc-length t in [0,1] (0 = rightmost valid point, 1 = leftmost).
      self.finger_location_list: list[float] = []
      self.thumb_location_list: list[float] = []

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

   def pad(self): # Surface pad geometry
      self.pad_resolution_level = 4 #7
      self.pad_thickness_mm = 1.0
      self.bump_edge_zeroing_spread = 0.3
      

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

   def generate_values(self): 
      # finger and thumb number
      self.thumb_number = 1
      self.finger_number = 3
      self.thumb_angle_deg_list = [0 for _ in range(self.thumb_number)]
      self.finger_angle_deg_list = [0 for _ in range(self.finger_number)]
      self.thumb_base_normal_offset_mm_list = [0 for _ in range(self.thumb_number)]
      self.finger_base_normal_offset_mm_list = [0 for _ in range(self.finger_number)]
      self.finger_base_side_offset_mm_list = [0 for _ in range(self.finger_number)]
      self.thumb_base_side_offset_mm_list = [0 for _ in range(self.thumb_number)]
      # fixed-mode default locations: evenly spaced interior points along the valid curve
      self.finger_location_list = [(i + 1) / (self.finger_number + 1) for i in range(self.finger_number)]
      self.thumb_location_list = [(i + 1) / (self.thumb_number + 1) for i in range(self.thumb_number)]

      # palm
      self.palm_size_mm = 140.0
      self.palm_size_mm = max(self.palm_size_mm, float(self.thumb_number * (self.thumb_corner_seg1_mm**2 + self.thumb_seg1_margin_offset_mm**2)**0.5))
      self.palm_size_mm = max(self.palm_size_mm, 1.4*float(self.finger_number * self.finger_min_spacing_mm))
      self.palm_wall_thickness_mm = 4.0

      # outline
      self.resolution_mm = 3.0
      self.outline_sides = 4
      self.outline_aspect_ratio = 1.55
      self.outline_longer_axis = 'x'
      self.outline_rotation_deg = 0.0
      self.smoothing_iters = 6
      self.smoothing_t = 0.25

      # pad bumps
      self.bump_type = 'gaussian'
      self.bumps_number = 0
      self.bump_max_height_intensity_mm = 0.0 

      self.bump_height_intensity_list = [1.0 for _ in range(self.bumps_number)]  # range (0.0, 1.0)
      self.bumps_spread_list = [0.25 for _ in range(self.bumps_number)]  # sigma scaled to the surface size, range (0.1, 0.5)
      self.bumps_aspect_ratio_list = [1.0 for _ in range(self.bumps_number)]
      self.bump_rotation_deg_list = [0.0 for _ in range(self.bumps_number)]
      self.bump_center_angle_deg_list = [0.0 for _ in range(self.bumps_number)]
      self.bump_center_offset_list = [0.0 for _ in range(self.bumps_number)]  # bump center offset from the origin of the surface outline scaled to the surface size, range (0.0, 1.0)

      # initialize stored values
      self.inner_outline_points = np.empty((0, 2), dtype=float)
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
      
   def outline_polygon(self):#
      # Generate the single active attachment outline. The inward wall/cavity
      # outline is computed later from the final PalmClass-modified outline.
      N = self.outline_sides if isinstance(self.outline_sides, int) else warning("Outline sides is not initialized correctly", 8)
      target_width = self.palm_size_mm if isinstance(self.palm_size_mm, float) else warning("Palm size is not initialized correctly", 50.0)
      rot_rad = np.deg2rad(self.outline_rotation_deg) if isinstance(self.outline_rotation_deg, float) else warning("Outline rotation degree is not initialized correctly", 0.0)
      res = float(self.resolution_mm) if isinstance(self.resolution_mm, float) else warning("Resolution is not initialized correctly", 1.0)

      def _width_for_scale(scale):
         outline = self._build_outline_points(N, scale, rot_rad, res)
         if outline.shape[0] < 3:
            return 0.0, outline
         return float(np.max(outline[:, 0]) - np.min(outline[:, 0])), outline

      # Binary search a uniform scale that makes final outline X width match target_width.
      lo = 1e-3
      hi = max(float(target_width), 1.0)
      width, outline = _width_for_scale(hi)
      while width < target_width and hi < target_width * 100.0:
         hi *= 2.0
         width, outline = _width_for_scale(hi)
      for _ in range(40):
         mid = 0.5 * (lo + hi)
         width, outline_mid = _width_for_scale(mid)
         if width < target_width:
            lo = mid
         else:
            hi = mid
            outline = outline_mid

      self.outline_points = self._roll_outline_start(outline)
      self.inner_outline_points = np.empty((0, 2), dtype=float)
      self.palm_size_mm = float(np.max(self.outline_points[:, 0]) - np.min(self.outline_points[:, 0]))
      self.outline_semantics_version = 3

      # import matplotlib.pyplot as plt
      # print(f"N: {N}")
      # plt.plot(self.outline_points[:, 0], self.outline_points[:, 1], 'o')
      # plt.axis('equal')
      # plt.show()

   def _build_outline_points(self, sides: int, scale: float, rot_rad: float, resolution_mm: float):
      phase = (np.pi / 2.0 - np.pi / sides) if sides % 2 == 0 else -np.pi / 2.0
      angles = phase + np.linspace(0, 2*np.pi, sides, endpoint=False)
      poly = np.vstack([scale * np.cos(angles), scale * np.sin(angles)]).T

      if self.outline_longer_axis == 'x':
         poly[:, 1] = poly[:, 1] / self.outline_aspect_ratio
      else:
         poly[:, 0] = poly[:, 0] / self.outline_aspect_ratio

      R = np.array([[np.cos(rot_rad), -np.sin(rot_rad)], [np.sin(rot_rad), np.cos(rot_rad)]])
      poly = poly @ R.T

      outline_points = []
      for i in range(sides):
         p1 = poly[i]
         p2 = poly[(i + 1) % sides]
         edge_length = np.linalg.norm(p2 - p1)
         num_points = max(int(np.floor(edge_length / resolution_mm)), 1)
         for j in range(num_points):
            t = j / num_points
            outline_points.append((1 - t) * p1 + t * p2)
      pts = np.array(outline_points, dtype=float)
      pts = self._cut_sharp_outline_points(pts)
      return self._roll_outline_start(pts)

   @staticmethod
   def _roll_outline_start(points: np.ndarray):
      pts = np.asarray(points, dtype=float)
      if pts.shape[0] == 0:
         return pts
      return np.roll(pts, -np.argmin(-pts[:, 0] + np.abs(pts[:, 1])), axis=0)

   def _cut_sharp_outline_points(self, points: np.ndarray):
      if points is None or points.shape[0] < 3:
         return np.asarray(points, dtype=float)

      temp_points = np.asarray(points, dtype=float).copy()
      n = points.shape[0]
      count = 0
      for i in range(n):
         prev_idx = (i - 1) % n
         next_idx = (i + 1) % n
         p1 = points[prev_idx]
         p2 = points[i]
         p3 = points[next_idx]
         v1 = p1 - p2
         v2 = p3 - p2
         norm1 = np.linalg.norm(v1)
         norm2 = np.linalg.norm(v2)
         if norm1 <= 1e-12 or norm2 <= 1e-12:
               continue

         cosang = np.dot(v1, v2) / (norm1 * norm2)
         cosang = np.clip(cosang, -1.0, 1.0)  # numerical safety
         angle = np.arccos(cosang)
         if angle < np.deg2rad(self.cut_sharp_outline_threshold_deg):
               temp_points = np.delete(temp_points, i-count, axis=0)
               count += 1

      return temp_points

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

      # center the outline points around the origin and reorder the points so that the one with the largest x and closest y to zero is the first one.
      drawn_outline = self.outline_points - np.mean(self.outline_points, axis=0)
      drawn_outline = drawn_outline[np.argsort(np.arctan2(drawn_outline[:, 1], drawn_outline[:, 0]))]
      self.outline_points = self._roll_outline_start(drawn_outline)
      self.inner_outline_points = np.empty((0, 2), dtype=float)
      self.palm_size_mm = float(np.max(self.outline_points[:, 0]) - np.min(self.outline_points[:, 0]))
      self.outline_semantics_version = 3

   def _ensure_outline_semantics(self):
      self.outline_points = np.asarray(self.outline_points, dtype=float)
      # v3 semantics: outline_points is the only attachment/body outline.
      # Older saved configs also keep their visible outline as the attachment
      # outline; the wall cutter is regenerated after PalmClass modifications.
      self.outline_points = self._roll_outline_start(self.outline_points)
      self.inner_outline_points = np.empty((0, 2), dtype=float)
      self.outline_semantics_version = 3

      if self.outline_points.shape[0] >= 3:
         self.palm_size_mm = float(np.max(self.outline_points[:, 0]) - np.min(self.outline_points[:, 0]))

   def _region_curve(self, is_finger):
      """Ordered (M,2) polyline of the valid base region along the outline, oriented
      rightmost (t=0) -> leftmost (t=1). Returns None if fewer than 2 valid points.
      The region is the longest contiguous run (along the closed outline loop) of points
      with y above/below the finger/thumb threshold. A left/right thumb_side constraint is
      only applied when exactly one thumb is requested."""
      pts = self.outline_points
      if len(pts) < 2:
         return None
      if is_finger and self.generation_mode == 'finger_only':
         return np.vstack([pts, pts[0]])
      if is_finger:
         mask = pts[:, 1] > self.finger_ignore_y_threshold_mm
      else:
         mask = pts[:, 1] < -self.thumb_ignore_y_threshold_mm
         if self.thumb_number == 1 and self.thumb_side == 'left':
            mask = mask & (pts[:, 0] < 0)
         elif self.thumb_number == 1 and self.thumb_side == 'right':
            mask = mask & (pts[:, 0] > 0)
      if int(mask.sum()) < 2:
         return None
      run = max(self._contiguous_runs(mask), key=len)
      if len(run) < 2:
         return None
      curve = pts[run]
      if curve[0, 0] < curve[-1, 0]:   # ensure rightmost is t=0
         curve = curve[::-1]
      return curve

   @staticmethod
   def _contiguous_runs(mask):
      """Lists of indices for maximal contiguous True runs on a circular boolean mask."""
      n = len(mask)
      idx = np.arange(n)
      if mask.all():
         return [list(idx)]
      shift = int(np.where(~mask)[0][0])  # rotate to start at a False so a run can't wrap
      rmask = np.roll(mask, -shift)
      ridx = np.roll(idx, -shift)
      runs, cur = [], []
      for m, ix in zip(rmask, ridx):
         if m:
            cur.append(int(ix))
         elif cur:
            runs.append(cur); cur = []
      if cur:
         runs.append(cur)
      return runs

   @staticmethod
   def _curve_point_at(curve, t):
      """2D point at fractional arc-length t in [0,1] along the polyline `curve`."""
      t = float(min(max(t, 0.0), 1.0))
      return PalmConfig._curve_point_at_s(curve, t * PalmConfig._curve_length(curve))

   @staticmethod
   def _curve_length(curve):
      """Arc length of a polyline."""
      seg = np.diff(curve, axis=0)
      seglen = np.linalg.norm(seg, axis=1)
      return float(seglen.sum())

   @staticmethod
   def _curve_point_at_s(curve, s):
      """2D point at arc-length distance s along the polyline `curve`."""
      seg = np.diff(curve, axis=0)
      seglen = np.linalg.norm(seg, axis=1)
      total = float(seglen.sum())
      if total <= 1e-9:
         return curve[0].copy()
      s = float(min(max(s, 0.0), total))
      cum = np.concatenate([[0.0], np.cumsum(seglen)])
      j = int(np.searchsorted(cum, s) - 1)
      j = min(max(j, 0), len(seg) - 1)
      frac = (s - cum[j]) / max(seglen[j], 1e-9)
      return curve[j] + frac * (curve[j + 1] - curve[j])

   @staticmethod
   def _curve_location_of_point(curve, point):
      """Fractional arc-length of the closest projection of `point` onto `curve`."""
      point = np.asarray(point, dtype=float)
      seg = np.diff(curve, axis=0)
      seglen = np.linalg.norm(seg, axis=1)
      total = float(seglen.sum())
      if total <= 1e-9:
         return 0.0

      best_dist = float("inf")
      best_s = 0.0
      acc = 0.0
      for i, v in enumerate(seg):
         denom = float(np.dot(v, v))
         u = 0.0 if denom <= 1e-12 else float(np.dot(point - curve[i], v) / denom)
         u = min(max(u, 0.0), 1.0)
         proj = curve[i] + u * v
         dist = float(np.linalg.norm(point - proj))
         if dist < best_dist:
            best_dist = dist
            best_s = acc + u * seglen[i]
         acc += seglen[i]
      return float(min(max(best_s / total, 0.0), 1.0))

   def _derive_missing_locations_from_saved_points(self):
      """Backfill t-location lists for old saved configs that only stored XY points."""
      if self._derive_finger_locations_from_points and len(self.finger_points) == self.finger_number:
         curve = self._region_curve(is_finger=True)
         if curve is not None:
            self.finger_location_list = [self._curve_location_of_point(curve, p) for p in self.finger_points]
      if self._derive_thumb_locations_from_points and len(self.thumb_points) == self.thumb_number:
         curve = self._region_curve(is_finger=False)
         if curve is not None:
            self.thumb_location_list = [self._curve_location_of_point(curve, p) for p in self.thumb_points]

   def _place_digits(self, locations, side_offsets_mm, n, min_spacing, is_finger):
      """Map per-digit t-locations to 2D points on the valid curve, enforcing 2D
      min_spacing. For fingers, side offset is a signed arc-length shift before
      validation. For thumbs, side offset keeps the original pipeline semantics
      and is applied later in PalmClass as a straight local tangent translation."""
      if n == 0:
         return np.empty((0, 2), dtype=float), [], True
      curve = self._region_curve(is_finger)
      if curve is None:
         return None, list(locations), False
      total = self._curve_length(curve)
      if total <= 1e-9:
         return None, list(locations), False
      locs = [float(min(max(t, 0.0), 1.0)) for t in locations]
      if len(locs) != n:                       # self-heal (e.g. old config lacking t's)
         locs = [(i + 1) / (n + 1) for i in range(n)]
      offsets = [float(v) for v in side_offsets_mm] if is_finger else [0.0 for _ in range(n)]
      if len(offsets) != n:
         offsets = [0.0 for _ in range(n)]
      pairs = sorted(zip(locs, offsets), key=lambda item: item[0])
      locs = [p[0] for p in pairs]
      offsets = [p[1] for p in pairs]
      step = max(min_spacing / (total * 4.0), 2e-3)   # nudge step in t-space
      pts, final, ok = [], [], True
      for i, t in enumerate(locs):
         ti = t if i == 0 else max(t, final[-1])
         si = min(max(ti * total + offsets[i], 0.0), total)
         pi = self._curve_point_at_s(curve, si)
         if i > 0:
            while np.linalg.norm(pi - pts[-1]) < min_spacing and ti < 1.0:
               ti = min(1.0, ti + step)
               si = min(max(ti * total + offsets[i], 0.0), total)
               pi = self._curve_point_at_s(curve, si)
            if np.linalg.norm(pi - pts[-1]) < min_spacing:
               ok = False
         pts.append(pi)
         final.append(ti)
      return np.array(pts, dtype=float), final, ok

   @staticmethod
   def _check_pairwise_spacing(points_a, points_b, min_spacing):
      pa = np.asarray(points_a, dtype=float)
      pb = np.asarray(points_b, dtype=float)
      if pa.size == 0 or pb.size == 0:
         return True
      for a in pa:
         for b in pb:
            if np.linalg.norm(a - b) < min_spacing:
               return False
      return True

   @staticmethod
   def _check_same_set_spacing(points, min_spacing):
      pts = np.asarray(points, dtype=float)
      if pts.shape[0] < 2:
         return True
      for i in range(pts.shape[0]):
         for j in range(i + 1, pts.shape[0]):
            if np.linalg.norm(pts[i] - pts[j]) < min_spacing:
               return False
      return True

   def _sort_digits_by_location(self, is_finger):
      """Co-sort all per-digit lists (location, angle, offsets) ascending by location so
      each digit keeps its (t, angle, offsets) tuple and index 0 = rightmost."""
      if is_finger:
         names = ['finger_location_list', 'finger_angle_deg_list',
                  'finger_base_normal_offset_mm_list', 'finger_base_side_offset_mm_list']
      else:
         names = ['thumb_location_list', 'thumb_angle_deg_list',
                  'thumb_base_normal_offset_mm_list', 'thumb_base_side_offset_mm_list']
      loc = getattr(self, names[0])
      n = len(loc)
      if n < 2:
         return
      order = list(np.argsort(loc))
      for name in names:
         lst = getattr(self, name)
         if len(lst) == n:
            setattr(self, name, [lst[k] for k in order])

   def finger_thumb_points_selection(self):
      # Localize finger/thumb bases by a fractional arc-length parameter t in [0,1] per
      # digit along the valid outline curve (0=rightmost, 1=leftmost). t (finger_location_list
      # / thumb_location_list) is the single source of truth; points are always regenerated
      # from it. Finger side offsets shift along this curve before validation; thumb side
      # offsets are applied later as the original local tangent translation in PalmClass.
      # This method does not mutate palm size; sampled locations that do not fit should be
      # resampled by ParamGen.
      if self.finger_number == 0 and self.thumb_number == 0:
         print("Error: Finger and thumb number are not initialized correctly, is zero")
         return
      if isinstance(self.finger_number, tuple) or isinstance(self.thumb_number, tuple):
         print("Error: Finger and thumb number are not initialized correctly, is tuple")
         return
      if self.generation_mode == 'finger_only':
         self.thumb_number = 0
         self.thumb_angle_deg_list = []
         self.thumb_base_normal_offset_mm_list = []
         self.thumb_base_side_offset_mm_list = []
         self.thumb_location_list = []
         self.thumb_points = np.empty((0, 2), dtype=float)
         self.thumb_bases = np.empty((0, 2), dtype=float)
         self.thumb_bases_normal_vectors = np.empty((0, 2), dtype=float)
         self.thumb_side_list = []
         if self.finger_number == 0:
            raise ValueError("Finger-only generation mode requires at least one finger.")
         self._sort_digits_by_location(is_finger=True)
         fpts, floc, fok = self._place_digits(
            self.finger_location_list,
            self.finger_base_side_offset_mm_list,
            self.finger_number,
            self.finger_min_spacing_mm,
            True,
         )
         if not fok:
            raise ValueError(
               "Finger-only location parameters do not fit the current palm outline. "
               "Resample finger_location_list or increase the requested palm size."
            )
         spacing_ok = self._check_same_set_spacing(fpts, self.finger_min_spacing_mm)
         if not spacing_ok:
            raise ValueError(
               "Finger-only side offsets or locations do not fit the current palm outline. "
               "Reduce side offsets, resample locations, or increase the requested palm size."
            )
         self.finger_points = fpts
         self.finger_location_list = floc
         return

      # co-sort per-digit lists once so location order matches angle/offset order
      self._sort_digits_by_location(is_finger=True)
      self._sort_digits_by_location(is_finger=False)

      fpts, floc, fok = self._place_digits(self.finger_location_list, self.finger_base_side_offset_mm_list, self.finger_number, self.finger_min_spacing_mm, True)
      tpts, tloc, tok = self._place_digits(self.thumb_location_list, self.thumb_base_side_offset_mm_list, self.thumb_number, self.thumb_min_spacing_mm, False)
      cross_min_spacing = 0.5 * (self.finger_min_spacing_mm + self.thumb_min_spacing_mm)
      spacing_ok = (
         self._check_same_set_spacing(fpts, self.finger_min_spacing_mm)
         and self._check_same_set_spacing(tpts, self.thumb_min_spacing_mm)
         and self._check_pairwise_spacing(fpts, tpts, cross_min_spacing)
      )
      if not (fok and tok):
         raise ValueError(
            "Finger/thumb location parameters do not fit the current palm outline. "
            "Resample finger_location_list/thumb_location_list or increase the requested palm size."
         )
      if not spacing_ok:
         raise ValueError(
            "Finger side offsets or finger/thumb locations do not fit the current palm outline. "
            "Reduce side offsets, resample locations, or increase the requested palm size."
         )

      self.finger_points = fpts
      self.thumb_points = tpts
      self.finger_location_list = floc
      self.thumb_location_list = tloc

   def update(self):
      if self.mode == "fixed":
         self.initialize_outline()
      self.finger_thumb_points_selection()

class FingerConfig:
   def __init__(self, type: str = 'finger', data: str = "fixed", code: Union[str, None] = None, id: Union[int, None] = None):
      '''
      type: str = 'finger' or 'thumb_right' or 'thumb_left' or 'thumb'
      data: str = 'fixed' or path to a saved config file
      code: new-grammar explicit code 'R-BEFORE-AFTER' (e.g. '2-1-1', thumb '1--11').
            If None, a type-aware fixed default is used. Wildcard ('x') codes are
            resolved by ParamGen.FingerParams, not here.
      '''
      self.id = id
      self.type = type
      # type-aware explicit default; non-finger types are thumbs
      self.code = code if code is not None else ('0--11' if type == 'finger' else '1--11')
      self.fingertip_scale_factor = (1.0, 1.0, 1.0)

      self.geometry()
      self.pad()
      self.code_data()
      self.generate_links()
      self.generate_pads()


      if data == "random":
         raise ValueError("FingerConfig(data='random') is no longer supported. Use ParamGen.FingerParams(...).build() to sample a random finger/thumb config.")
      if data == "fixed":
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
         # tolerant: keep current default if the saved file lacks this attribute (schema drift)
         setattr(self, k, getattr(module, k, getattr(self, k)))
      self._migrate_legacy_code()
      self.code_data()

   def save_config(self, path):
      with open(path, "w") as f:
         for k, v in vars(self).items():
            f.write(f"{k} = {repr(v)}\n")

   @staticmethod
   def _legacy_pairs_to_digits(pairs: str, n: int) -> str:
      pair_to_digit = {'01': '1', '10': '2'}
      out = []
      for i in range(n):
         pair = pairs[i * 2:(i + 1) * 2]
         out.append(pair_to_digit.get(pair, '1'))
      return ''.join(out)

   def _migrate_legacy_code(self):
      """Convert old codes like '110-11-0101010101' to new '2-1-1' grammar."""
      parts = str(self.code).split('-')
      if self.type == 'finger':
         legacy_to_r = {'000': '0', '101': '1', '110': '2', '100': '3', '111': '4'}
         if len(parts) >= 3 and parts[0] in legacy_to_r and len(parts[1]) >= 2:
            before_n = int(parts[1][0])
            after_n = int(parts[1][1])
            type_pairs = parts[2]
            before = self._legacy_pairs_to_digits(type_pairs[:before_n * 2], before_n)
            after = self._legacy_pairs_to_digits(type_pairs[before_n * 2:], after_n)
            self.code = f"{legacy_to_r[parts[0]]}-{before}-{after}"
      else:
         if len(parts) >= 3 and parts[0] in ('0', '1') and parts[1].isdigit() and len(parts[2]) > 0:
            after_n = int(parts[1])
            after = self._legacy_pairs_to_digits(parts[2], after_n)
            self.code = f"{parts[0]}--{after}"

   def geometry(self):

      self.link_added_length_mm_list: list[float] = []
      # self.link_added_width_mm_list: list[float] = []

   def pad(self):
      self.pad_resolution_level = 2 #4
      self.pad_thickness_mm = 1.0
      self.bump_edge_zeroing_spread = 0.3
   
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
      # Parse an explicit new-grammar code 'R-BEFORE-AFTER' into derived fields.
      # BEFORE/AFTER are digit strings (1=short joint, 2=long joint); count = #digits.
      # Wildcard ('x') codes are not parsed here (ParamGen resolves them first).
      is_finger = (self.type == 'finger')
      # number of add-joint slots (used to size default pad/link lists; matches base_hand)
      self.max_add_joint_slots = 5 if is_finger else 4

      parts = str(self.code).split('-')
      if any('x' in p for p in parts) or len(parts) < 3:
         warning(f"code_data got non-explicit/invalid code '{self.code}'; using default")
         self.code = '0--11' if is_finger else '1--11'
         parts = self.code.split('-')

      R = parts[0]
      before_seg = parts[1]
      after_seg = parts[2]

      self.rotation_mode = R
      
   def generate_pads(self):
      # Default (deterministic, no-bump) pad config. Lists are sized to the number of
      # add-joint slots (max_add_joint_slots), matching base_hand and the original; the
      # Blender side indexes them per pad position with a safe fallback. ParamGen samples
      # over the same length.
      n = self.max_add_joint_slots
      self.bump_type_list = ['gaussian' for _ in range(n)]
      self.bump_number_list = [0 for _ in range(n)]
      self.bump_max_height_intensity_mm_list = [0.0 for _ in range(n)]
      self.bump_height_intensity_list = [[1.0 for _ in range(self.bump_number_list[i])] for i in range(n)]
      self.bumps_spread_list = [[0.25 for _ in range(self.bump_number_list[i])] for i in range(n)]
      self.bumps_aspect_ratio_list = [[1.0 for _ in range(self.bump_number_list[i])] for i in range(n)]
      self.bump_rotation_deg_list = [[0.0 for _ in range(self.bump_number_list[i])] for i in range(n)]
      self.bump_center_angle_deg_list = [[0.0 for _ in range(self.bump_number_list[i])] for i in range(n)]
      self.bump_center_offset_list = [[0.0 for _ in range(self.bump_number_list[i])] for i in range(n)]

   def generate_links(self):
      self.link_added_length_mm_list = [0 for _ in range(self.max_add_joint_slots)]


class HandConfig:
   def __init__(self, data: str = "fixed", origin_offset_mm: tuple[float, float, float] = (0.0, 0.0, 0.0)):
      self.collision_mesh: bool = True
      self.palm_geometry()
      self.thumb_mount_geometry()
      self.origin_offset_mm = origin_offset_mm
      # self.origin_offset_mm = (6.56, -38, 11.4) ## offset for g2a code
      if data == "random":
         raise ValueError("HandConfig(data='random') is no longer supported. Use ParamGen plus HandConfig(data='fixed') to build random hands.")
      if data != "fixed":
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
         # tolerant: keep current default if the saved file lacks this attribute (schema drift)
         setattr(self, k, getattr(module, k, getattr(self, k)))

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
      self.palm_outline_plot_2d: bool = default
