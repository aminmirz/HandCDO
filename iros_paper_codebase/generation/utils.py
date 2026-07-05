import numpy as np
from math import radians, sin, cos

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
