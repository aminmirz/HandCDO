import numpy as np
import matplotlib.pyplot as plt
from Config import *

def warning(message, default=None):
    print(f"Warning: {message}")
    return default

class PalmOutline:
    def __init__(self, cfg: PalmConfig):
        self.cfg = cfg
        self.type = 'palm'        


        self.construct_finger_base_vectors_and_normals()
        self.construct_finger_segments()
        self.construct_thumb_base_frames_segments()
        self.outline_with_segments()
        self.smoothen_outline_chaikin()
        self.smoothen_outline()

        if Plots().palm_outline_plot_2d:
            self.plot_outline()

        # self.heightmap = HeightMap(self)

    
    def construct_finger_base_vectors_and_normals(self):
        # make finger_bases based on the finger points, and the finger_base_offset_mm in the normal direction of the finger points. 
        outline = self.cfg.outline_points.copy()

        def unit_vector(v):
            return v / 1.0 / np.linalg.norm(v)

        def finger_point_normal_vector(p, window_size=1):
            idx = int(np.argmin(np.linalg.norm(outline - p, axis=1)))
            prev_idx = (idx - window_size) % outline.shape[0]
            next_idx = (idx + window_size) % outline.shape[0]
            tangent = unit_vector(outline[next_idx] - outline[prev_idx])
            # 90-deg rotation to get normal vector
            n_cw = np.array([tangent[1], -tangent[0]])
            return n_cw

        bases = []
        base_normal_vectors = []
        for i, p in enumerate(self.cfg.finger_points):
            n_out = finger_point_normal_vector(p)
            t_hat = np.array([-n_out[1], n_out[0]], dtype=float)
            t_hat = -1.0 * t_hat / np.linalg.norm(t_hat)
            bases.append(p + n_out * self.cfg.finger_base_normal_offset_mm_list[i] + self.cfg.finger_base_side_offset_mm_list[i] * t_hat)
            # rotate the normal vector by the finger_angle_deg
            R = np.array([[np.cos(np.deg2rad(self.cfg.finger_angle_deg_list[i])), -np.sin(np.deg2rad(self.cfg.finger_angle_deg_list[i]))],
                          [np.sin(np.deg2rad(self.cfg.finger_angle_deg_list[i])), np.cos(np.deg2rad(self.cfg.finger_angle_deg_list[i]))]])
            n_out = n_out @ R
            base_normal_vectors.append(unit_vector(n_out))
            
        self.cfg.finger_bases = np.array(bases)
        self.cfg.finger_bases_normal_vectors = np.array(base_normal_vectors)

    def construct_finger_segments(self):
        # construct the finger segments using the finger_bases, finger_base_normal_vectors, config.finger_base_seg_mm and config.finger_base_seg_margin_mm.
        # each finger_segment is made of 4 colinear points that form a line perpendicular to the finger_base_normal_vectors and midpoint at the finger_base.
        # The distance between the middle two points is config.finger_base_seg_mm. and side distances are config.finger_base_seg_margin_mm.
        # total length of the segment is config.finger_base_seg_mm + config.finger_base_seg_margin_mm*2.

        segments = []
        for i, base in enumerate(self.cfg.finger_bases):
            n = self.cfg.finger_bases_normal_vectors[i]
    
            t_hat = np.array([-n[1], n[0]], dtype=float)
            t_hat = t_hat / np.linalg.norm(t_hat)

            # Middle segment endpoints around the base (centered at base)
            p1 = base - 0.5 * self.cfg.finger_base_seg_mm * t_hat
            p2 = base + 0.5 * self.cfg.finger_base_seg_mm * t_hat
            # Add margins on both sides
            p0 = p1 - self.cfg.finger_base_seg_margin_mm * t_hat
            p3 = p2 + self.cfg.finger_base_seg_margin_mm * t_hat

            segments.append(np.vstack([p0, p1, p2, p3]))

        self.finger_segments = np.array(segments)

    def construct_thumb_base_frames_segments(self):
        # make thumb_corners based on the thumb points, and the normal direction of the thumb points.
        # the thumb_corner offset is config.thumb_corner_outline_offset_mm.
        # thumb has two segments connected to the thumb_corner. The length of the segments (seg1 and seg2) are config.thumb_base_seg1_mm and config.thumb_base_seg2_mm.
        # two segments are perpendicular to each other.
        # If thumb_point has a positive x value, the seg1 has an angle of arctan(seg2/seg1)+config.thumb_angle_deg with the line connecting the thumb_corner to thumb_point in CCW direction.
        # If thumb_point has a negative x value, the seg1 has an angle of arctan(seg2/seg1)-config.thumb_angle_deg with the line connecting the thumb_corner to thumb_point in CW direction.
        # seg2 has angle of 90 - seg1_angle relatigve to the corner-point line in the other direction (CW or CCW).
        # thumb_bases are the points have the offset of config.thumb_base_corner_offset_mm in the local cordinate frame of the seg1 and seg2 in negatice direction.

        outline = self.cfg.outline_points.copy()

        def unit_vector(v):
            return v / 1.0 / np.linalg.norm(v)

        def outward_normal_at_point(p, window_size=7):
            idx = int(np.argmin(np.linalg.norm(outline - p, axis=1)))
            prev_idx = (idx - window_size) % outline.shape[0]
            next_idx = (idx + window_size) % outline.shape[0]
            tangent = unit_vector(outline[next_idx] - outline[prev_idx])
            n_cw = np.array([tangent[1], -tangent[0]])
            return unit_vector(n_cw)

        def rotate(vec, angle_rad):
            c, s = np.cos(angle_rad), np.sin(angle_rad)
            R = np.array([[c, -s], [s, c]])
            return vec @ R.T

        bases = []
        seg_frames = []
        normal_vectors = []
        for i, p in enumerate(self.cfg.thumb_points):
            n_out = outward_normal_at_point(p)
            corner = p - n_out * self.cfg.thumb_corner_outline_offset_mm #* (1 + 2*np.abs(self.cfg.thumb_points[i, 0]) / self.cfg.palm_size_mm)
            v = p - corner
            v_hat = unit_vector(v)

            theta = np.deg2rad(self.cfg.thumb_angle_deg_list[i]) + np.arctan2(self.cfg.thumb_corner_seg1_mm, self.cfg.thumb_corner_seg2_mm)

            if p[0] >= 0:
                self.cfg.thumb_side_list.append('thumb_right')
                # seg1 CCW from v, seg2 CW by (90 - theta)
                seg1_hat = unit_vector(rotate(v_hat, +theta))
                seg2_hat = unit_vector(rotate(v_hat, -(np.pi / 2.0 - theta)))
            else:
                self.cfg.thumb_side_list.append('thumb_left')
                # seg1 CW from v, seg2 CCW by (90 - theta)
                seg1_hat = unit_vector(rotate(v_hat, -theta))
                seg2_hat = unit_vector(rotate(v_hat, +(np.pi / 2.0 - theta)))

            base = corner - seg1_hat * self.cfg.thumb_base_corner_offset_mm[0] - seg2_hat * self.cfg.thumb_base_corner_offset_mm[1]
            normal_vectors.append(seg1_hat)
            seg1_point = seg1_hat*self.cfg.thumb_corner_seg1_mm +corner
            seg2_point = seg2_hat*self.cfg.thumb_corner_seg2_mm +corner

            # make margin segments for the seg1 and seg2. 
            # seg1_margin_point has offset of (config.thumb_seg1_margin_offset_mm*seg1_hat - config.thumb_seg2_margin_offset_mm*seg2_hat) from the seg1_point.
            # seg2_margin_point has offset of (config.thumb_seg2_margin_offset_mm*seg2_hat - config.thumb_seg1_margin_offset_mm*seg1_hat) from the seg2_point.

            seg1_margin_point = seg1_point + self.cfg.thumb_seg1_margin_offset_mm * (- seg2_hat)#(seg1_hat - 2*seg2_hat)
            seg2_margin_point = seg2_point + self.cfg.thumb_seg2_margin_offset_mm * (seg2_hat - seg1_hat)
            
            # move all the points by the thumb_base_normal_offset_mm in the direction of the normal_vectors
            seg1_margin_point = seg1_margin_point + normal_vectors[i] * self.cfg.thumb_base_normal_offset_mm_list[i]
            seg1_point = seg1_point + normal_vectors[i] * self.cfg.thumb_base_normal_offset_mm_list[i]
            corner = corner + normal_vectors[i] * self.cfg.thumb_base_normal_offset_mm_list[i]
            seg2_point = seg2_point + normal_vectors[i] * self.cfg.thumb_base_normal_offset_mm_list[i]
            seg2_margin_point = seg2_margin_point + normal_vectors[i] * self.cfg.thumb_base_normal_offset_mm_list[i]
            base = base + normal_vectors[i] * self.cfg.thumb_base_normal_offset_mm_list[i]

            # move all the points by the thumb_base_side_offset_mm in the direction normal to normal_vectors
            t_hat = np.array([-normal_vectors[i][1], normal_vectors[i][0]], dtype=float)
            t_hat = t_hat / np.linalg.norm(t_hat)
            seg1_margin_point = seg1_margin_point + t_hat * self.cfg.thumb_base_side_offset_mm_list[i]
            seg1_point = seg1_point + t_hat * self.cfg.thumb_base_side_offset_mm_list[i]
            corner = corner + t_hat * self.cfg.thumb_base_side_offset_mm_list[i]
            seg2_point = seg2_point + t_hat * self.cfg.thumb_base_side_offset_mm_list[i]
            seg2_margin_point = seg2_margin_point + t_hat * self.cfg.thumb_base_side_offset_mm_list[i]
            base = base + t_hat * self.cfg.thumb_base_side_offset_mm_list[i]

            bases.append(base)
            seg_frames.append((seg1_margin_point,seg1_point,corner, seg2_point, seg2_margin_point))

        self.cfg.thumb_bases = np.array(bases) if len(bases) > 0 else np.empty((0, 2))
        self.cfg.thumb_bases_normal_vectors = np.array(normal_vectors) if len(normal_vectors) > 0 else np.empty((0, 2))
        self.thumb_seg_frames = np.array(seg_frames) if len(seg_frames) > 0 else np.empty((0, 2, 2))
    
    def outline_with_segments(self):
        # generate a new outline with the self.outline_points, self.finger_segments, self.thumb_seg_frames.
        # The points on the outline that are closer than the config.finger_min_spacing_mm/2 and config.thumb_min_spacing_mm/2 to the finger and thumb bases are removed -> valid_outline_points
        # the points should be ordered correctly. that is we start from a self.finger_segment and then any point in the valid_outline_points that is before the next finger_base in the self.outline_points is added to the new outline.
        # then, the next finger_segment. and it continues until we get to the thumb. Then for the thumb, if the thumb_point has x>0, then we add the seg1, corner, seg2 in order. otherwise, we add the seg2, corner, seg1 in order.
        # we continue this process until we have added all the points to the new outline.

        orig_outline = self.cfg.outline_points.copy()
        n = orig_outline.shape[0]

        finger_bases = self.cfg.finger_bases
        thumb_bases = self.cfg.thumb_bases
        finger_points = self.cfg.finger_points
        thumb_points = self.cfg.thumb_points

        # Find indices of finger and thumb points in the outline
        finger_indices = []
        for fp in finger_points:
            idx = int(np.argmin(np.linalg.norm(orig_outline - fp, axis=1)))
            finger_indices.append(idx)

        thumb_indices = []
        for tp in thumb_points:
            idx = int(np.argmin(np.linalg.norm(orig_outline - tp, axis=1)))
            thumb_indices.append(idx)

        # Mark points that are too close to finger or thumb bases as invalid
        valid_mask = np.ones(n, dtype=bool)
        for fb in finger_bases:
            dists = np.linalg.norm(orig_outline - fb, axis=1)
            valid_mask &= (dists >= self.cfg.finger_min_spacing_mm / 1.3)
        for tp in thumb_points:
            dists = np.linalg.norm(orig_outline - tp, axis=1)
            valid_mask &= (dists >= self.cfg.thumb_min_spacing_mm / 2)
        # Also filter points too close to any of the 5 thumb segment frame points
        for frame in self.thumb_seg_frames:
            for seg_pt in frame:  # each of the 5 points in the frame
                dists = np.linalg.norm(orig_outline - seg_pt, axis=1)
                valid_mask &= (dists >= self.cfg.thumb_min_spacing_mm / 2)


        # Combine finger and thumb attachments with their outline indices
        # Store: (outline_index, type, segment_index)
        attachments = []
        for i, idx in enumerate(finger_indices):
            attachments.append((idx, 'finger', i))
        for i, idx in enumerate(thumb_indices):
            attachments.append((idx, 'thumb', i))

        # Sort attachments by their index in the outline
        attachments.sort(key=lambda x: x[0])

        # Handle edge case: no fingers or thumbs
        if len(attachments) == 0:
            self.outline_points_with_segments = orig_outline.copy()
            return

        new_pts = []
        is_from_outline = []  # track which points are from original outline
        thumb_segment_starts = []  # track where each thumb segment starts in new_pts
        current_idx = 0

        for att_idx, att_type, seg_i in attachments:
            # Add valid outline points from current_idx up to (but not including) att_idx
            while current_idx < att_idx:
                if valid_mask[current_idx]:
                    new_pts.append(orig_outline[current_idx])
                    is_from_outline.append(True)
                current_idx += 1

            # Insert the appropriate segment
            if att_type == 'finger':
                seg = self.finger_segments[seg_i]  # shape (4, 2): [p0, p1, p2, p3]
                new_pts.extend([seg[0], seg[1], seg[2], seg[3]])
                is_from_outline.extend([False, False, False, False])
            else:  # thumb
                thumb_segment_starts.append(len(new_pts))  # record start index
                frame = self.thumb_seg_frames[seg_i]  # shape (5, 2): [seg1_margin, seg1, corner, seg2, seg2_margin]
                tp = thumb_points[seg_i]
                if tp[0] >= 0:  # thumb on right side (x >= 0)
                    # Add in reverse order: seg2_margin, seg2, corner, seg1, seg1_margin
                    new_pts.extend([frame[4], frame[3], frame[2], frame[1], frame[0]])
                else:  # thumb on left side (x < 0)
                    # Add seg1_margin, seg1, corner, seg2, seg2_margin in order
                    new_pts.extend([frame[0], frame[1], frame[2], frame[3], frame[4]])
                is_from_outline.extend([False, False, False, False, False])

            # Skip the attachment point itself (it's replaced by the segment)
            current_idx = att_idx + 1

        # Add remaining valid points from current_idx to end of outline
        while current_idx < n:
            if valid_mask[current_idx]:
                new_pts.append(orig_outline[current_idx])
                is_from_outline.append(True)
            current_idx += 1

        # if removing the thumb segment margin point (first and last points of the thumb segment) increases the angle 
        # made by the previous (if the last margin point is removed) or next point (if the first margin point is removed) segment with the corner point and the next outline point
        # remove the margin point if it increases the angle.

        def calculate_angle(p1, p2, p3):
            """Calculate angle at p2 formed by vectors p1->p2 and p2->p3. Returns angle in radians."""
            v1 = np.array(p1) - np.array(p2)
            v2 = np.array(p3) - np.array(p2)
            norm1, norm2 = np.linalg.norm(v1), np.linalg.norm(v2)
            if norm1 < 1e-10 or norm2 < 1e-10:
                return np.pi  # treat degenerate case as straight line
            cos_angle = np.dot(v1, v2) / (norm1 * norm2)
            cos_angle = np.clip(cos_angle, -1.0, 1.0)
            return np.arccos(cos_angle)

        # Process thumb segments in reverse order to handle index shifts correctly
        indices_to_remove = []
        for start_idx in thumb_segment_starts:
            # Thumb segment occupies indices: start_idx, start_idx+1, start_idx+2, start_idx+3, start_idx+4
            # These are: [first_margin, second_pt, corner, fourth_pt, last_margin]
            num_pts = len(new_pts)
            first_margin_idx = start_idx
            second_pt_idx = start_idx + 1
            corner_idx = start_idx + 2
            fourth_pt_idx = start_idx + 3
            last_margin_idx = start_idx + 4

            # Check first margin point removal
            prev_pt_idx = (first_margin_idx - 1) % num_pts
            # Angle at second_pt with first_margin vs without (using prev_pt instead)
            angle_with_first_margin = calculate_angle(new_pts[first_margin_idx], new_pts[second_pt_idx], new_pts[corner_idx])
            angle_without_first_margin = calculate_angle(new_pts[prev_pt_idx], new_pts[second_pt_idx], new_pts[corner_idx])
            if angle_without_first_margin > angle_with_first_margin:
                indices_to_remove.append(first_margin_idx)

            # Check last margin point removal
            next_pt_idx = (last_margin_idx + 1) % num_pts
            # Angle at fourth_pt with last_margin vs without (using next_pt instead)
            angle_with_last_margin = calculate_angle(new_pts[corner_idx], new_pts[fourth_pt_idx], new_pts[last_margin_idx])
            angle_without_last_margin = calculate_angle(new_pts[corner_idx], new_pts[fourth_pt_idx], new_pts[next_pt_idx])
            if angle_without_last_margin > angle_with_last_margin:
                indices_to_remove.append(last_margin_idx)

        # Remove marked indices in reverse order to preserve index validity
        for idx in sorted(indices_to_remove, reverse=True):
            new_pts.pop(idx)
            is_from_outline.pop(idx)

        # if in the new sets of point, there are a group of consecutive points that are from the original outline set 
        # and have size of smaller than with size smaller than 3 points they should be removed. they should be consecutively conncted to be removed. if there are ore than 3 connected points from original outline, they stay.
        
        # Find consecutive groups of original outline points
        num_pts = len(new_pts)
        if num_pts > 0:
            indices_to_remove_small_groups = []
            i = 0
            while i < num_pts:
                if is_from_outline[i]:
                    # Start of a group of consecutive outline points
                    group_start = i
                    group_end = i
                    while group_end + 1 < num_pts and is_from_outline[group_end + 1]:
                        group_end += 1
                    group_size = group_end - group_start + 1
                    # If group has fewer than 3 points, mark for removal
                    if group_size < 5:
                        for j in range(group_start, group_end + 1):
                            indices_to_remove_small_groups.append(j)
                    i = group_end + 1
                else:
                    i += 1
            
            # Remove small groups in reverse order
            for idx in sorted(indices_to_remove_small_groups, reverse=True):
                new_pts.pop(idx)

        self.outline_points_with_segments = np.array(new_pts)

    def smoothen_outline_chaikin(self):
        # smoothen the outline using the smoothing_iters and smoothing_t 
        # Chaikin smoothing on a closed polyline, preserving locked edges exactly.
        # locked_edge_keys: finger_segment[1] to finger_segment[2] and thumb_seg_frames[1] to thumb_seg_frames[3]
        if self.outline_points_with_segments.shape[0] < 3:
            return
        iters = int(self.cfg.smoothing_iters)
        t = float(self.cfg.smoothing_t)
        if iters <= 0 or t <= 0.0:
            return

        def key_point(p: np.ndarray):
            return tuple(np.round(p.astype(float), 6))

        def key_edge(a: np.ndarray, b: np.ndarray):
            ka, kb = key_point(a), key_point(b)
            return frozenset((ka, kb))  # undirected edge key

        # Build locked edges set
        locked = set()
        if hasattr(self, 'finger_segments') and self.finger_segments.shape[0] > 0:
            for seg in self.finger_segments:  # shape (4,2)
                p1, p2 = seg[1], seg[2]
                locked.add(key_edge(p1, p2))
        if hasattr(self, 'thumb_seg_frames') and self.thumb_seg_frames.shape[0] > 0:
            for frame in self.thumb_seg_frames:  # expected shape (3,2): [seg1_end, corner, seg2_end]
                # lock both segments connected to the corner
                locked.add(key_edge(frame[1], frame[2]))
                locked.add(key_edge(frame[2], frame[3]))

        poly = self.outline_points_with_segments.astype(float)
        for _ in range(iters):
            if poly.shape[0] < 3:
                break
            new_pts = []
            n = poly.shape[0]
            for i in range(n):
                p = poly[i]
                pn = poly[(i + 1) % n]
                ekey = key_edge(p, pn)
                if ekey in locked:
                    # Preserve the exact edge by appending its endpoints contiguously
                    if len(new_pts) == 0 or not np.allclose(new_pts[-1], p):
                        new_pts.append(p)
                    if not np.allclose(new_pts[-1], pn):
                        new_pts.append(pn)
                    continue
                # Chaikin corner cutting for this edge
                q = (1.0 - t) * p + t * pn
                r = t * p + (1.0 - t) * pn
                if len(new_pts) == 0 or not np.allclose(new_pts[-1], q):
                    new_pts.append(q)
                new_pts.append(r)
            poly = np.array(new_pts, dtype=float)

        self.outline_points_with_segments = poly
    
    def smoothen_outline(self):
        # smoothen the outline using the second method, preserving locked edges exactly, and then down sample the non-locked points down to cfg.resolution_mm
        # locked_edge_keys: finger_segment[1] to finger_segment[2] and thumb_seg_frames[1] to thumb_seg_frames[3]
        if self.outline_points_with_segments.shape[0] < 3:
            return

        iters = int(self.cfg.smoothing_iters) if isinstance(self.cfg.smoothing_iters, int) else warning("Smoothing iterations is not initialized correctly", 5)
        lam = float(self.cfg.smoothing_t) * 0.5 if isinstance(self.cfg.smoothing_t, float) else warning("Smoothing t is not initialized correctly", 0.25) # laplacian weight derived from t
        lam = max(0.0, min(lam, 0.49))       # keep stable
        # lam = 0.2
        res = float(self.cfg.resolution_mm) if isinstance(self.cfg.resolution_mm, float) else warning("Resolution is not initialized correctly", 5.0)
        # res = 5.0
        def key_point(p: np.ndarray):
            return tuple(np.round(p.astype(float), 6))

        def key_edge(a: np.ndarray, b: np.ndarray):
            ka, kb = key_point(a), key_point(b)
            return frozenset((ka, kb))  # undirected edge key

        # Build locked edges set (finger mid edge; thumb consecutive edges in frame)
        locked_edges = set()
        if hasattr(self, 'finger_segments') and self.finger_segments.shape[0] > 0:
            for seg in self.finger_segments:  # shape (4,2)
                locked_edges.add(key_edge(seg[1], seg[2]))
        if hasattr(self, 'thumb_seg_frames') and self.thumb_seg_frames.shape[0] > 0:
            for frame in self.thumb_seg_frames:
                # frame could be (3,2), (5,2), etc. Lock all consecutive edges
                m = frame.shape[0]
                for j in range(m - 1):
                    locked_edges.add(key_edge(frame[j], frame[j + 1]))

        # Determine locked vertices: any vertex that is an endpoint of a locked edge
        poly = self.outline_points_with_segments.astype(float)
        n = poly.shape[0]
        locked_vertices = set()
        for i in range(n):
            a = poly[i]
            b = poly[(i + 1) % n]
            if key_edge(a, b) in locked_edges:
                locked_vertices.add(i)
                locked_vertices.add((i + 1) % n)

        # Laplacian smoothing on non-locked vertices only
        for _ in range(iters):
            if poly.shape[0] < 3:
                break
            new_poly = poly.copy()
            n = poly.shape[0]
            for i in range(n):
                if i in locked_vertices:
                    continue
                p_prev = poly[(i - 1) % n]
                p = poly[i]
                p_next = poly[(i + 1) % n]
                new_poly[i] = (1.0 - 2.0 * lam) * p + lam * (p_prev + p_next)
            poly = new_poly

        # Downsample: keep all locked vertices; for others, enforce spacing ~= resolution_mm
        keep = []
        last_kept = None
        for i in range(poly.shape[0]):
            p = poly[i]
            if i in locked_vertices:
                keep.append(p)
                last_kept = p
                continue
            if last_kept is None or np.linalg.norm(p - last_kept) >= res:
                keep.append(p)
                last_kept = p

        # Ensure closure spacing is reasonable; optional simplification near wrap-around
        if len(keep) >= 2 and np.linalg.norm(keep[0] - keep[-1]) < res * 0.25:
            keep.pop()  # drop last if it's nearly duplicate of first

        self.outline_points_with_segments = np.array(keep, dtype=float)

    def get_outline(self):
        return self.outline_points_with_segments
    
    def get_surface_radius(self):
        return np.max(np.linalg.norm(self.outline_points_with_segments, axis=1))

    def plot_outline(self):
        # Set font
        plt.rcParams['font.family'] = 'Times New Roman'
        fig, ax = plt.subplots()
        # Fill outline polygon
        ax.fill(self.cfg.outline_points[:, 0], self.cfg.outline_points[:, 1], color='gray', alpha=0.2, linewidth=0.0)
        # Outline markers: smaller, darker gray circles
        ax.scatter(self.cfg.outline_points[:, 0], self.cfg.outline_points[:, 1], s=10, color='dimgray', marker='o', alpha=0.25)
        
        # # # Index labels
        # for i, pt in enumerate(self.outline_points):
        #     ax.text(pt[0] + 0.1, pt[1], str(i))

        # Finger and thumb points
        if self.cfg.finger_points.shape[0] > 0:
            ax.scatter(self.cfg.finger_points[:, 0], self.cfg.finger_points[:, 1], color='orangered', marker='o',alpha=0.25, s=60)
        if self.cfg.thumb_points.shape[0] > 0:
            ax.scatter(self.cfg.thumb_points[:, 0], self.cfg.thumb_points[:, 1], color='royalblue', marker='o',alpha=0.25, s=60)

        # Finger and thumb point indices
        if self.cfg.finger_points.shape[0] > 0:
            for i, pt in enumerate(self.cfg.finger_points):
                ax.text(pt[0]+5, pt[1]+5, str(i))
        if self.cfg.thumb_points.shape[0] > 0:
            for i, pt in enumerate(self.cfg.thumb_points):
                ax.text(pt[0]+5, pt[1]+5, str(i))

        # Finger and thumb base
        if self.cfg.finger_bases.shape[0] > 0:
            ax.scatter(self.cfg.finger_bases[:, 0], self.cfg.finger_bases[:, 1], color='orangered', marker='o', s=10, alpha=0.75)
        if self.cfg.thumb_bases.shape[0] > 0:
            ax.scatter(self.cfg.thumb_bases[:, 0], self.cfg.thumb_bases[:, 1], color='royalblue', marker='o', s=10, alpha=0.75)

        # Finger and thumb base normal vectors
        if self.cfg.finger_bases.shape[0] > 0:
            for i, vec in enumerate(self.cfg.finger_bases_normal_vectors):
                ax.quiver(self.cfg.finger_bases[i, 0], self.cfg.finger_bases[i, 1], vec[0], vec[1], color='orangered', scale=15, width=0.005, headwidth=5, headlength=5, alpha=0.75)
        if self.cfg.thumb_bases.shape[0] > 0:
            for i, vec in enumerate(self.cfg.thumb_bases_normal_vectors):
                ax.quiver(self.cfg.thumb_bases[i, 0], self.cfg.thumb_bases[i, 1], vec[0], vec[1], color='royalblue', scale=15, width=0.005, headwidth=5, headlength=5, alpha=0.75)

        # Finger and thumb segments
        if self.finger_segments.shape[0] > 0:
            for i, segment in enumerate(self.finger_segments):
                ax.plot(segment[1:-1, 0], segment[1:-1, 1], color='orangered', linewidth=2, alpha=0.75, )

        if self.thumb_seg_frames.shape[0] > 0:
            for i, frame in enumerate(self.thumb_seg_frames):
                ax.plot(frame[1:-1, 0], frame[1:-1, 1], color='royalblue', linewidth=2, alpha=0.75)
            
        # Outline with segments
        if self.outline_points_with_segments.shape[0] > 0:
            ax.fill(self.outline_points_with_segments[:, 0], self.outline_points_with_segments[:, 1], color='black', linewidth=2, alpha=0.35)
            ax.scatter(self.outline_points_with_segments[:, 0], self.outline_points_with_segments[:, 1], color='black', marker='o', s=10, alpha=0.75)

        ax.set_xlabel('x (mm)')
        ax.set_ylabel('y (mm)')
        ax.grid(False)
        ax.axis('equal')
        plt.show()

        return fig, ax