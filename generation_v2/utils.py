import numpy as np
from math import radians, sin, cos

def signed_polygon_area(points: np.ndarray) -> float:
    pts = np.asarray(points, dtype=float)
    if pts.shape[0] < 3:
        return 0.0
    x = pts[:, 0]
    y = pts[:, 1]
    return 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))

def line_segment_intersection(a0: np.ndarray, a1: np.ndarray, b0: np.ndarray, b1: np.ndarray, tol: float = 1e-9):
    da = np.asarray(a1, dtype=float) - np.asarray(a0, dtype=float)
    db = np.asarray(b1, dtype=float) - np.asarray(b0, dtype=float)
    a0 = np.asarray(a0, dtype=float)
    b0 = np.asarray(b0, dtype=float)
    denom = float(da[0] * db[1] - da[1] * db[0])
    if abs(denom) <= tol:
        return None
    diff = b0 - a0
    t = float(diff[0] * db[1] - diff[1] * db[0]) / denom
    u = float(diff[0] * da[1] - diff[1] * da[0]) / denom
    if t < 0.0 or t > 1.0 or u < 0.0 or u > 1.0:
        return None
    p = a0 + da * t
    return (t, u, p, p)

def resolve_offset_self_intersections(vertices: np.ndarray, offset_mm: float, original_area: float, cyclic: bool = True) -> np.ndarray:
    verts = [np.asarray(v, dtype=float) for v in np.asarray(vertices, dtype=float)]
    if len(verts) < 3:
        return np.asarray(verts, dtype=float)

    sign = 1.0 if offset_mm > 0.0 else -1.0
    i_start = 0 if cyclic else 1
    i = i_start
    while i < len(verts):
        j = i + 2
        while j < len(verts) - (0 if i > 0 else 1):
            inter = line_segment_intersection(verts[i - 1], verts[i], verts[j - 1], verts[j])
            if inter is None:
                j += 1
                continue
            inter_pt = 0.5 * (inter[2] + inter[3])
            tri_inner = np.vstack([inter_pt, verts[i], verts[j - 1]])
            tri_outer = np.vstack([inter_pt, verts[j], verts[i - 1]])
            area_inner = sign * signed_polygon_area(tri_inner)
            area_outer = sign * signed_polygon_area(tri_outer)
            if area_inner > area_outer:
                verts = verts[i:j] + [inter_pt]
                i = i_start
            else:
                verts = verts[:i] + [inter_pt] + verts[j:]
                j = i + 2
        i += 1

    new_area = signed_polygon_area(np.asarray(verts, dtype=float))
    if original_area * new_area < 0.0:
        verts = list(reversed(verts))
    return np.asarray(verts, dtype=float)

def offset_polygon(points: np.ndarray, offset_mm: float) -> np.ndarray:
    """Offset a 2D polygon. Positive distance moves outward; negative moves inward."""
    pts = np.asarray(points, dtype=float)
    n = pts.shape[0]
    if n < 3 or abs(offset_mm) <= 1e-12:
        return pts.copy()

    area = signed_polygon_area(pts)
    if abs(area) < 1e-9:
        return pts.copy()
    ccw = area > 0.0

    def _unit(v):
        nrm = float(np.linalg.norm(v))
        return v / nrm if nrm > 0.0 else v

    def _edge_outward_normal(p0, p1):
        t = _unit(p1 - p0)
        n_out = np.array([t[1], -t[0]], dtype=float) if ccw else np.array([-t[1], t[0]], dtype=float)
        return _unit(n_out)

    def _cross2(a, b):
        return float(a[0] * b[1] - a[1] * b[0])

    new_pts = []
    for i in range(n):
        i_prev = (i - 1) % n
        i_next = (i + 1) % n
        p_prev = pts[i_prev]
        p_curr = pts[i]
        p_next = pts[i_next]
        t_prev = _unit(p_curr - p_prev)
        t_curr = _unit(p_next - p_curr)
        n_prev = _edge_outward_normal(p_prev, p_curr) * offset_mm
        n_curr = _edge_outward_normal(p_curr, p_next) * offset_mm
        a1 = p_prev + n_prev
        b1 = p_curr + n_curr
        denom = _cross2(t_prev, t_curr)
        if abs(denom) < 1e-9:
            q = 0.5 * (p_curr + n_prev + p_curr + n_curr)
        else:
            s = _cross2(b1 - a1, t_curr) / denom
            q = a1 + t_prev * s
        new_pts.append(q)

    return resolve_offset_self_intersections(np.asarray(new_pts, dtype=float), offset_mm=offset_mm, original_area=area, cyclic=True)

def offset_polygon_recursive(points: np.ndarray, total_offset_mm: float, steps: int = 1) -> np.ndarray:
    pts = np.asarray(points, dtype=float)
    if steps <= 1:
        return offset_polygon(pts, total_offset_mm)
    step = float(total_offset_mm) / float(steps)
    for _ in range(steps):
        pts = offset_polygon(pts, step)
    return pts

def clipper_offset_polygon(points: np.ndarray, offset_mm: float, miter_limit: float = 2.0, scale: float = 1000.0) -> np.ndarray:
    """Offset a closed polygon with pyclipper.

    Positive offsets expand the polygon and negative offsets shrink it. Multiple
    returned paths mean the inward offset split into disconnected regions, which
    is not a valid single palm cavity.
    """
    try:
        import pyclipper
    except ImportError as exc:
        raise RuntimeError(
            "pyclipper is required for robust palm wall generation. "
            "Install it in the Python/Blender environment running generation_v2."
        ) from exc

    pts = remove_consecutive_duplicate_points(points)
    if pts.ndim != 2 or pts.shape[1] != 2 or pts.shape[0] < 3:
        raise ValueError("Cannot offset a palm outline with fewer than three 2D points.")
    if abs(signed_polygon_area(pts)) <= 1e-9:
        raise ValueError("Cannot offset a degenerate palm outline with near-zero area.")

    path = [(int(round(float(x) * scale)), int(round(float(y) * scale))) for x, y in pts]
    offsetter = pyclipper.PyclipperOffset(miter_limit=float(miter_limit), arc_tolerance=0.25 * scale)
    offsetter.AddPath(path, pyclipper.JT_MITER, pyclipper.ET_CLOSEDPOLYGON)
    result = offsetter.Execute(float(offset_mm) * scale)
    if not result:
        raise ValueError(
            f"Palm inward offset collapsed at {abs(float(offset_mm)):.3f} mm. "
            "Reduce palm_wall_thickness_mm or use a less concave palm/finger layout."
        )
    if len(result) != 1:
        raise ValueError(
            f"Palm inward offset split into {len(result)} disconnected cavities at "
            f"{abs(float(offset_mm)):.3f} mm. Reduce wall thickness or widen narrow attachment regions."
        )

    out = np.asarray(result[0], dtype=float) / scale
    original_area = signed_polygon_area(pts)
    if signed_polygon_area(out) * original_area < 0.0:
        out = out[::-1]
    return remove_consecutive_duplicate_points(out)

def remove_consecutive_duplicate_points(points: np.ndarray, tol: float = 1e-7) -> np.ndarray:
    pts = np.asarray(points, dtype=float)
    if pts.ndim != 2 or pts.shape[0] == 0:
        return pts.copy()
    out = [pts[0]]
    for p in pts[1:]:
        if np.linalg.norm(p - out[-1]) > tol:
            out.append(p)
    if len(out) > 1 and np.linalg.norm(out[0] - out[-1]) <= tol:
        out.pop()
    return np.asarray(out, dtype=float)

def polygon_has_self_intersections(points: np.ndarray, tol: float = 1e-7) -> bool:
    pts = remove_consecutive_duplicate_points(points, tol=tol)
    n = pts.shape[0]
    if n < 4:
        return False
    for i in range(n):
        a0 = pts[i]
        a1 = pts[(i + 1) % n]
        for j in range(i + 1, n):
            # Adjacent segments share a vertex and are expected to touch.
            if j == i or j == (i + 1) % n or i == (j + 1) % n:
                continue
            b0 = pts[j]
            b1 = pts[(j + 1) % n]
            inter = line_segment_intersection(a0, a1, b0, b1, tol=tol)
            if inter is not None:
                return True
    return False

def point_in_polygon(point: np.ndarray, polygon: np.ndarray, tol: float = 1e-7) -> bool:
    p = np.asarray(point, dtype=float)
    poly = remove_consecutive_duplicate_points(polygon, tol=tol)
    n = poly.shape[0]
    if n < 3:
        return False

    # Treat points on the boundary as inside.
    for i in range(n):
        a = poly[i]
        b = poly[(i + 1) % n]
        ab = b - a
        ap = p - a
        cross = abs(float(ab[0] * ap[1] - ab[1] * ap[0]))
        if cross <= tol:
            dot = float(np.dot(ap, ab))
            if -tol <= dot <= float(np.dot(ab, ab)) + tol:
                return True

    inside = False
    x, y = float(p[0]), float(p[1])
    for i in range(n):
        x0, y0 = poly[i]
        x1, y1 = poly[(i + 1) % n]
        if (y0 > y) != (y1 > y):
            x_cross = (x1 - x0) * (y - y0) / (y1 - y0) + x0
            if x < x_cross + tol:
                inside = not inside
    return inside

def outline_is_valid_inner(candidate: np.ndarray, outer: np.ndarray, min_area_ratio: float = 0.10) -> bool:
    cand = remove_consecutive_duplicate_points(candidate)
    out = remove_consecutive_duplicate_points(outer)
    if cand.ndim != 2 or cand.shape[1] != 2 or cand.shape[0] < 3:
        return False
    if out.ndim != 2 or out.shape[1] != 2 or out.shape[0] < 3:
        return False
    if not np.all(np.isfinite(cand)):
        return False
    if polygon_has_self_intersections(cand):
        return False

    cand_area = abs(signed_polygon_area(cand))
    out_area = abs(signed_polygon_area(out))
    cand_bbox = np.ptp(cand, axis=0)
    out_bbox = np.ptp(out, axis=0)
    if out_area <= 1e-9 or np.any(out_bbox <= 1e-9):
        return False
    if cand_area < min_area_ratio * out_area or cand_area >= 0.98 * out_area:
        return False
    if cand_bbox[0] >= out_bbox[0] or cand_bbox[1] >= out_bbox[1]:
        return False
    if cand_bbox[0] < 0.25 * out_bbox[0] or cand_bbox[1] < 0.25 * out_bbox[1]:
        return False

    # Check vertices and edge midpoints. This catches most cases where a fallback
    # shrink slipped outside a concave modified palm outline.
    for i in range(cand.shape[0]):
        p0 = cand[i]
        p1 = cand[(i + 1) % cand.shape[0]]
        if not point_in_polygon(p0, out):
            return False
        if not point_in_polygon(0.5 * (p0 + p1), out):
            return False
    return True

def normal_shrink_outline(points: np.ndarray, wall_mm: float) -> np.ndarray:
    pts = remove_consecutive_duplicate_points(points)
    n = pts.shape[0]
    if n < 3:
        return pts.copy()
    area = signed_polygon_area(pts)
    if abs(area) < 1e-9:
        return pts.copy()
    ccw = area > 0.0
    center = np.mean(pts, axis=0)

    def _unit(v):
        nrm = float(np.linalg.norm(v))
        return v / nrm if nrm > 1e-12 else v

    def _edge_outward_normal(p0, p1):
        t = _unit(p1 - p0)
        return _unit(np.array([t[1], -t[0]], dtype=float) if ccw else np.array([-t[1], t[0]], dtype=float))

    out = []
    for i in range(n):
        p_prev = pts[(i - 1) % n]
        p = pts[i]
        p_next = pts[(i + 1) % n]
        n_prev = _edge_outward_normal(p_prev, p)
        n_next = _edge_outward_normal(p, p_next)
        inward = -_unit(n_prev + n_next)
        if np.linalg.norm(inward) <= 1e-12:
            inward = _unit(center - p)
        candidate = p + inward * float(wall_mm)
        if not point_in_polygon(candidate, pts):
            # Concave corners can point the averaged normal outward. Pull toward
            # the centroid as a conservative local fallback.
            candidate = p + _unit(center - p) * float(wall_mm)
        out.append(candidate)
    return remove_consecutive_duplicate_points(np.asarray(out, dtype=float))

def radial_inner_outline(points: np.ndarray, wall_mm: float) -> np.ndarray:
    pts = remove_consecutive_duplicate_points(points)
    if pts.shape[0] < 3:
        return pts.copy()
    center = polygon_interior_point(pts)
    bbox = np.ptp(pts, axis=0)
    if np.any(bbox <= 1e-9):
        return pts.copy()
    scale_x = max((bbox[0] - 2.0 * float(wall_mm)) / bbox[0], 0.05)
    scale_y = max((bbox[1] - 2.0 * float(wall_mm)) / bbox[1], 0.05)
    scale = min(scale_x, scale_y)
    return remove_consecutive_duplicate_points(center + (pts - center) * scale)

def polygon_interior_point(points: np.ndarray) -> np.ndarray:
    pts = remove_consecutive_duplicate_points(points)
    if pts.shape[0] < 3:
        return np.mean(pts, axis=0) if pts.size else np.zeros(2, dtype=float)

    # Area centroid first. It is good for many generated palms but can lie
    # outside strongly concave outlines, so it is validated below.
    area2 = float(np.sum(pts[:, 0] * np.roll(pts[:, 1], -1) - pts[:, 1] * np.roll(pts[:, 0], -1)))
    if abs(area2) > 1e-9:
        cx = float(np.sum((pts[:, 0] + np.roll(pts[:, 0], -1)) * (
            pts[:, 0] * np.roll(pts[:, 1], -1) - np.roll(pts[:, 0], -1) * pts[:, 1]
        )) / (3.0 * area2))
        cy = float(np.sum((pts[:, 1] + np.roll(pts[:, 1], -1)) * (
            pts[:, 0] * np.roll(pts[:, 1], -1) - np.roll(pts[:, 0], -1) * pts[:, 1]
        )) / (3.0 * area2))
        c = np.array([cx, cy], dtype=float)
        if point_in_polygon(c, pts):
            return c

    # Robust fallback: choose the inside grid point farthest from the boundary.
    mn = pts.min(axis=0)
    mx = pts.max(axis=0)
    best = np.mean(pts, axis=0)
    best_dist = -1.0
    xs = np.linspace(mn[0], mx[0], 21)
    ys = np.linspace(mn[1], mx[1], 21)
    edges = [(pts[i], pts[(i + 1) % pts.shape[0]]) for i in range(pts.shape[0])]
    for x in xs:
        for y in ys:
            p = np.array([x, y], dtype=float)
            if not point_in_polygon(p, pts):
                continue
            dmin = min(_point_segment_distance(p, a, b) for a, b in edges)
            if dmin > best_dist:
                best = p
                best_dist = dmin
    return best

def _point_segment_distance(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    ab = b - a
    denom = float(np.dot(ab, ab))
    if denom <= 1e-12:
        return float(np.linalg.norm(p - a))
    t = max(0.0, min(1.0, float(np.dot(p - a, ab) / denom)))
    q = a + t * ab
    return float(np.linalg.norm(p - q))

def ray_shrink_outline(points: np.ndarray, wall_mm: float) -> np.ndarray:
    """Shrink each outer point toward an interior point.

    Unlike a uniform radial scale, this uses a per-vertex step capped by the
    available distance to the interior point, keeping concave attachment regions
    inside the modified outline while preserving their shape.
    """
    pts = remove_consecutive_duplicate_points(points)
    if pts.shape[0] < 3:
        return pts.copy()
    center = polygon_interior_point(pts)
    moved = []
    for p in pts:
        v = center - p
        dist = float(np.linalg.norm(v))
        if dist <= 1e-9:
            moved.append(p)
            continue
        step = min(float(wall_mm), 0.45 * dist)
        direction = v / dist
        target = p + direction * step
        if point_in_polygon(target, pts):
            moved.append(target)
            continue
        lo = 0.0
        hi = step
        best = p
        for _ in range(24):
            mid = 0.5 * (lo + hi)
            candidate = p + direction * mid
            if point_in_polygon(candidate, pts):
                best = candidate
                lo = mid
            else:
                hi = mid
        moved.append(best)

    alphas = np.ones(pts.shape[0], dtype=float)
    moved = np.asarray(moved, dtype=float)

    def _build():
        return pts + (moved - pts) * alphas[:, None]

    def _first_intersection(poly):
        n_poly = poly.shape[0]
        for i in range(n_poly):
            for j in range(i + 1, n_poly):
                if j == i or j == (i + 1) % n_poly or i == (j + 1) % n_poly:
                    continue
                if line_segment_intersection(poly[i], poly[(i + 1) % n_poly], poly[j], poly[(j + 1) % n_poly]) is not None:
                    return i, j
        return None

    # Tight concave pockets can make neighboring ray-shrunk edges cross. Relax
    # only the smaller intersecting span back toward the outer boundary.
    poly = _build()
    for _ in range(24):
        hit = _first_intersection(poly)
        if hit is None:
            break
        i, j = hit
        span_a = list(range(i + 1, j + 1))
        span_b = list(range(j + 1, pts.shape[0])) + list(range(0, i + 1))
        span = span_a if len(span_a) <= len(span_b) else span_b
        for idx in span:
            alphas[idx] *= 0.5
        poly = _build()

    # If an extremely tight pocket still crosses after relaxation, keep that
    # smaller span on the modified outer boundary instead of bridging across the
    # concavity. This avoids outside chords and preserves attachment shape.
    for _ in range(12):
        hit = _first_intersection(poly)
        if hit is None:
            break
        i, j = hit
        n_poly = pts.shape[0]
        span_a = list(range(i + 1, j + 1))
        span_b = list(range(j + 1, n_poly)) + list(range(0, i + 1))
        span = span_a if len(span_a) <= len(span_b) else span_b
        expanded = set(span)
        for idx in span:
            for delta in range(-8, 9):
                expanded.add((idx + delta) % n_poly)
        for idx in expanded:
            alphas[idx] = 0.0
        poly = _build()

    for _ in range(12):
        bad_edges = []
        n_poly = poly.shape[0]
        for i in range(n_poly):
            p_mid = 0.5 * (poly[i] + poly[(i + 1) % n_poly])
            if not point_in_polygon(p_mid, pts):
                bad_edges.append(i)
        if not bad_edges:
            break
        for i in bad_edges:
            for delta in range(-8, 10):
                alphas[(i + delta) % pts.shape[0]] = 0.0
        poly = _build()

    return remove_consecutive_duplicate_points(poly)

def make_inner_outline_from_outer(outer_outline: np.ndarray, wall_mm: float, steps: int = 10, allow_fallback: bool = False) -> np.ndarray:
    """Build the palm cavity outline from the final modified body outline.

    The normal path uses pyclipper for a true polygon inward offset. Fallbacks
    are opt-in because the older local shrink methods can create uneven or thin
    walls and should not silently pass through the generation pipeline.
    """
    outer = remove_consecutive_duplicate_points(outer_outline)
    wall = float(wall_mm)
    candidate = clipper_offset_polygon(outer, -wall)
    candidate = remove_consecutive_duplicate_points(candidate)
    if outline_is_valid_inner(candidate, outer, min_area_ratio=0.02):
        return candidate

    if not allow_fallback:
        raise ValueError(
            "pyclipper produced an invalid palm cavity outline. "
            "This usually means the final palm outline is self-intersecting or has a narrow neck smaller than the wall thickness."
        )

    candidates = [offset_polygon_recursive(outer, total_offset_mm=-wall, steps=steps), normal_shrink_outline(outer, wall), ray_shrink_outline(outer, wall), radial_inner_outline(outer, wall)]
    for candidate in candidates:
        candidate = remove_consecutive_duplicate_points(candidate)
        if outline_is_valid_inner(candidate, outer):
            return candidate
    raise ValueError("Unable to build a valid palm cavity outline from the final palm body outline.")

def normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v if n < 1e-12 else v / n

def make_transform(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T

def invert_transform(T: np.ndarray) -> np.ndarray:
    R = T[:3, :3]
    t = T[:3, 3]
    Tinv = np.eye(4)
    Tinv[:3, :3] = R.T
    Tinv[:3, 3] = -R.T @ t
    return Tinv

def euler_xyz_deg_to_matrix(rx_deg: float, ry_deg: float, rz_deg: float) -> np.ndarray:
    rx, ry, rz = np.radians([rx_deg, ry_deg, rz_deg])
    cx, sx = np.cos(rx), np.sin(rx)
    cy, sy = np.cos(ry), np.sin(ry)
    cz, sz = np.cos(rz), np.sin(rz)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx
    
def rotation_about_axis(axis_local: np.ndarray, angle_deg: float) -> np.ndarray:
    axis = normalize(np.asarray(axis_local, dtype=float))
    theta = np.radians(angle_deg)
    c, s = np.cos(theta), np.sin(theta)
    x, y, z = axis
    R = np.array([
        [c + x * x * (1 - c), x * y * (1 - c) - z * s, x * z * (1 - c) + y * s],
        [y * x * (1 - c) + z * s, c + y * y * (1 - c), y * z * (1 - c) - x * s],
        [z * x * (1 - c) - y * s, z * y * (1 - c) + x * s, c + z * z * (1 - c)],
    ])
    return R

def get_attachment_local_normal(attachment: dict):
    # get the normal vector of the attachment after rotation by the attachment["rotation"] (x y z)
    initial_normal = (0.0, 0.0, 1.0)
    rotation = attachment["rotation"]

    rx, ry, rz = (radians(rotation[0]), radians(rotation[1]), radians(rotation[2]))

    x, y, z = initial_normal
    # Rotate around X
    cx, sx = cos(rx), sin(rx)
    y, z = (y * cx - z * sx, y * sx + z * cx)
    # Rotate around Y
    cy, sy = cos(ry), sin(ry)
    x, z = (x * cy + z * sy, -x * sy + z * cy)
    # Rotate around Z
    cz, sz = cos(rz), sin(rz)
    x, y = (x * cz - y * sz, x * sz + y * cz)

    # Clamp tiny values to zero for cleanliness
    def _clamp_zero(v: float, eps: float = 1e-7) -> float:
        return 0.0 if abs(v) < eps else v

    return (_clamp_zero(x), _clamp_zero(y), _clamp_zero(z))

def get_attachment_local_transform(element, attachment_index: int, touple_idx = None) -> np.ndarray:
    att = element.attachments[attachment_index]
    if isinstance(att, tuple):
        touple_idx = touple_idx if touple_idx is not None else element.touple_idx
        att = att[touple_idx]
    R = euler_xyz_deg_to_matrix(*att["rotation"])
    t = np.array(att["position"], dtype=float)
    return make_transform(R, t)

def get_attachment_global_transform(element, attachment_index: int, touple_idx = None) -> np.ndarray:
    return element.transformation @ get_attachment_local_transform(element, attachment_index, touple_idx)
  
def frame_from_normal(normal: np.ndarray, up_hint: np.ndarray = np.array([0.0, 0.0, 1.0])) -> np.ndarray:
    z = normalize(np.asarray(normal, dtype=float))
    up = np.asarray(up_hint, dtype=float)
    x = np.cross(up, z)
    if np.linalg.norm(x) < 1e-8:
        up_alt = np.array([0.0, 1.0, 0.0])
        x = np.cross(up_alt, z)
    x = normalize(x)
    y = np.cross(z, x)
    R = np.column_stack([x, y, z])
    return R

def world_about_frame_rotation(T_frame: np.ndarray, R_local: np.ndarray) -> np.ndarray:
    return T_frame @ make_transform(R_local, np.zeros(3)) @ invert_transform(T_frame)

def transform_points(T: np.ndarray, pts: np.ndarray) -> np.ndarray:
    pts_h = np.c_[pts, np.ones((pts.shape[0], 1))]
    out = (T @ pts_h.T).T[:, :3]
    return out

def face_center_from_normal_and_base_geom(normal: np.ndarray, base_geom: dict) -> np.ndarray:
    sx, sy, sz = base_geom["size_mm"]
    hx, hy, hz = sx / 2.0, sy / 2.0, sz / 2.0
    ox, oy, oz = base_geom["center_location_mm"]
    n = tuple(int(round(c)) for c in normalize(np.array(normal)))
    if n == (1, 0, 0):
        return np.array([ox + hx, oy + 0.0, oz + 0.0])
    if n == (-1, 0, 0):
        return np.array([ox - hx, oy + 0.0, oz + 0.0])
    if n == (0, 0, 1):
        return np.array([ox + 0.0, oy + 0.0, oz + hz])
    if n == (0, 0, -1):
        return np.array([ox + 0.0, oy + 0.0, oz - hz])
    return np.array([ox, oy, oz])

def get_cube_faces(size_mm: tuple, center_location_mm: tuple = (0.0, 0.0, 0.0)) -> list:
    faces = []
    sx, sy, sz = size_mm
    ox, oy, oz = center_location_mm
    hx, hy, hz = sx / 2.0, sy / 2.0, sz / 2.0
    top =np.array([
        [ox - hx, oy - hy, oz + hz],
        [ox + hx, oy - hy, oz + hz],
        [ox + hx, oy + hy, oz + hz],
        [ox - hx, oy + hy, oz + hz],
    ])
    faces.append(top)
    bottom = np.array([
        [ox - hx, oy - hy, oz - hz],
        [ox + hx, oy - hy, oz - hz],
        [ox + hx, oy + hy, oz - hz],
        [ox - hx, oy + hy, oz - hz],
    ])
    faces.append(bottom)
    for k in range(4):
        k2 = (k + 1) % 4
        side = np.array([
            top[k],
            top[k2],
            bottom[k2],
            bottom[k],
        ])
        faces.append(side)
    return faces
