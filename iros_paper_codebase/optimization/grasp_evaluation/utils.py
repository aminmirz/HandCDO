import numpy as np
import torch
from pathlib import Path
from isaaclab.utils.math import quat_from_matrix, matrix_from_quat, quat_from_euler_xyz

def smooth_transforms(transforms, window_size=5):
    """
    Applies a more robust moving average over a list of 4x4 transforms.
    Averages translation directly, and averages quaternions using 
    reference-based approach for proper handling of quaternion structure.
    
    All operations are done with torch tensors.
    
    Parameters:
        transforms (list[torch.Tensor]): List of 4x4 transform tensors.
        window_size (int): Window size for the moving average.
        
    Returns:
        list[torch.Tensor]: Smoothed list of 4x4 transform tensors.
    """
    half_w = window_size // 2
    smoothed = []
    
    n = len(transforms)
    for i in range(n):
        # Define the neighborhood indices
        start = max(0, i - half_w)
        end = min(n, i + half_w + 1)
        
        neighbors = transforms[start:end]
        
        # 1) Average translations
        translations = torch.stack([nbr[:3, 3] for nbr in neighbors], dim=0)  # (k, 3)
        avg_translation = translations.mean(dim=0)
        
        # 2) Average orientations via quaternions with reference-based approach
        quaternions = []
        for nbr in neighbors:
            R = nbr[:3, :3]  # (3,3)
            q = quat_from_matrix(R)  # Expected to return a tensor of shape (4,) in (w, x, y, z) order.
            quaternions.append(q)
        
        # Use first quaternion as reference for averaging
        ref_q = quaternions[0]
        weights = 1.0 / len(quaternions)
        
        # Initialize result quaternion
        result_q = torch.zeros(4, dtype=torch.float32)
        
        for q in quaternions:
            # Ensure the quaternion is on the same hemisphere as the reference
            if torch.dot(ref_q, q) < 0:
                q = -q
            # Add weighted quaternion
            result_q += weights * q
        
        # Re-normalize result quaternion
        result_q = result_q / torch.norm(result_q)
        
        # Convert the averaged quaternion back to a rotation matrix.
        R_avg = matrix_from_quat(result_q)  # (3,3) tensor
        
        # Assemble the new 4x4 transformation.
        T_new = torch.eye(4, dtype=torch.float32)
        T_new[:3, :3] = R_avg
        T_new[:3, 3] = avg_translation
        smoothed.append(T_new)
    
    return smoothed

def load_transforms(folder_path, pos, rot_euler, window_size, sample):
    """
    Loads a list of 4x4 transforms from .txt files in a folder, then:
      1) Re-bases the trajectory so that the first transform becomes the identity.
      2) Applies an initial transformation composed of a rotation (specified as Euler angles)
         and a translation.
      3) Optionally smooths the trajectory using a moving average.

    Parameters:
      folder_path (str or Path): Folder containing transform .txt files.
      pos (list of 3 floats): The translation to apply.
      rot_euler (list of 3 floats): Euler angles (roll, pitch, yaw in radians)
                                          in XYZ convention.
      window_size (int): Window size for smoothing.
      sample (int): Number of samples to resample the trajectory to.
      
    Returns:
      List of 4x4 torch.Tensor transforms.
    """
    folder = Path(folder_path)
    txt_files = sorted(folder.glob("*.txt"))
    transforms = []
    
    # 1) Read all transform files and convert them to torch tensors.
    for f in txt_files:
        T_np = np.loadtxt(f, dtype=float)
        T = torch.tensor(T_np, dtype=torch.float32)
        transforms.append(T)
    
    if len(transforms) == 0:
        print("No transform files found in:", folder_path)
        return transforms
    
    # Smooth the trajectory.
    transforms = smooth_transforms(transforms, window_size=window_size)
    transforms = resample_trajectory(transforms, orig_rate=30, new_rate=sample)
    
    # 2) Invert the first transform to re-base the trajectory.
    base_inv = torch.inverse(transforms[0])
    
    # 3) Create the initial transformation using Euler angles.
    #    Convert Euler angles to a quaternion.
    init_rot_tensor = torch.tensor(rot_euler, dtype=torch.float32).unsqueeze(0)  # Shape: (1,3)
    roll  = init_rot_tensor[:, 0]  # (1,)
    pitch = init_rot_tensor[:, 1]  # (1,)
    yaw   = init_rot_tensor[:, 2]  # (1,)
    
    # Convert Euler angles to quaternion using the provided function.
    quat = quat_from_euler_xyz(roll, pitch, yaw)  # Returns shape (1,4)
    quat = quat[0]  # Extract the 1D tensor (w, x, y, z)
    
    # Convert the quaternion to a rotation matrix.
    rot = matrix_from_quat(quat)  # (3,3) tensor
    # If matrix_from_quat returns a torch tensor, we keep it as such.
    
    # Build the 4x4 homogeneous transformation.
    pos_tensor = torch.tensor(pos, dtype=torch.float32)
    T_init = torch.eye(4, dtype=torch.float32)
    T_init[:3, :3] = rot
    T_init[:3, 3] = pos_tensor
    
    # 4) For each transform, re-base and then apply the initial transformation.
    for i in range(len(transforms)):
        transforms[i] = base_inv @ transforms[i]
        transforms[i] = T_init @ transforms[i]
        transforms[i] = torch.cat((transforms[i][:3, 3], quat_from_matrix(transforms[i][:3, :3])), dim=-1)
    
    return torch.stack(transforms)

def slerp(q0, q1, t):
    """
    Spherical linear interpolation (SLERP) between two normalized quaternions q0 and q1.
    
    Parameters:
      q0 (torch.Tensor): A tensor of shape (4,) representing the start quaternion (w, x, y, z).
      q1 (torch.Tensor): A tensor of shape (4,) representing the end quaternion (w, x, y, z).
      t (float or torch scalar): Interpolation fraction between 0 and 1.
    
    Returns:
      torch.Tensor: The interpolated quaternion of shape (4,).
    """
    # Normalize the quaternions to ensure they are unit quaternions.
    q0 = q0 / torch.norm(q0)
    q1 = q1 / torch.norm(q1)
    
    dot = torch.dot(q0, q1)
    dot = torch.clamp(dot, -1.0, 1.0)
    
    if dot < 0.0:
      q1 = -q1
      dot = -dot
    
    theta_0 = torch.acos(dot)
    theta = theta_0 * t
    q2 = q1 - q0 * dot
    q2 = q2 / torch.norm(q2)
    
    return q0 * torch.cos(theta) + q2 * torch.sin(theta)
  
def resample_trajectory(transforms, orig_rate=30, new_rate=20):
    """
    Resample a trajectory (list of 4x4 transforms) from an original sampling rate
    (default 30Hz) to a new sampling rate (default 20Hz) by time interpolation.
    
    The interpolation is done over the total duration of the trajectory,
    so that the first transform (at t = 0) and the last transform (at t = T) are preserved.
    For each new sample time, we:
      - Find the two original transforms between which this time falls.
      - Linearly interpolate their translations.
      - Convert their 3x3 rotation matrices to quaternions, slerp between them,
        and convert the result back to a 3x3 rotation matrix.
      
    Parameters:
      transforms (list[torch.Tensor]): List of 4x4 transformation matrices.
      orig_rate (float): The original sampling rate (Hz). (Default: 30)
      new_rate (float): The desired new sampling rate (Hz). (Default: 20)
      
    Returns:
      list[torch.Tensor]: The new list of 4x4 transforms resampled at new_rate.
      
    Note:
      In the case where the original trajectory represents exactly one second of motion
      (i.e. 30 frames if using 1-indexed numbering), then the 30th original frame will
      correspond to the 20th (last) frame in the new trajectory.
    """
    if len(transforms) == 0:
        return []
    
    # Compute total time using uniform spacing (in seconds)
    N = len(transforms)
    total_time = (N - 1) / orig_rate

    # Determine how many frames the new trajectory will have.
    # We want the last new frame to exactly correspond to the last original frame.
    new_N = int(round(total_time * new_rate)) + 1

    # Create new sample times between 0 and total_time (inclusive)
    new_times = torch.linspace(0, total_time, steps=new_N)
    
    new_transforms = []
    
    for t in new_times:
        # The corresponding (floating point) index in the original list:
        orig_index = t * orig_rate
        lower = int(torch.floor(orig_index).item())
        upper = int(torch.ceil(orig_index).item())
        
        # Make sure we do not index past the end.
        if upper >= N:
            upper = N - 1
        
        alpha = orig_index - lower  # interpolation factor (0 <= alpha <= 1)
        
        # If the new sample exactly matches an original frame, just copy it.
        if alpha < 1e-6:
            new_transforms.append(transforms[lower])
        else:
            # Interpolate translation linearly.
            T_lower = transforms[lower]
            T_upper = transforms[upper]
            trans_lower = T_lower[:3, 3]
            trans_upper = T_upper[:3, 3]
            interp_translation = (1 - alpha) * trans_lower + alpha * trans_upper
            
            # For rotation, extract the 3x3 rotation matrices and convert to quaternions.
            R_lower = T_lower[:3, :3]
            R_upper = T_upper[:3, :3]
            q_lower = quat_from_matrix(R_lower)  # returns (4,) in (w, x, y, z) order
            q_upper = quat_from_matrix(R_upper)
            
            # Interpolate the quaternions using SLERP.
            interp_q = slerp(q_lower, q_upper, alpha)
            
            # Convert the interpolated quaternion back to a rotation matrix.
            interp_R = matrix_from_quat(interp_q)
            
            # Assemble the new 4x4 transform.
            T_interp = torch.eye(4, dtype=torch.float32)
            T_interp[:3, :3] = interp_R
            T_interp[:3, 3] = interp_translation
            
            new_transforms.append(T_interp)
    
    return new_transforms