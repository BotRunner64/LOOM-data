"""Rigid transforms used to map source poses onto the contract EEF frame."""
import numpy as np

from loom_data.robot.kinematics import rpy_matrix


def eef_offset(value, label='EEF offset'):
    """Explicit source-frame coordinates of the gripper mounting origin, metres."""
    if value is None:
        raise ValueError(f'{label} is required: provide the gripper mounting interface in source axes')
    offset = np.asarray(value, dtype=np.float64)
    if offset.shape != (3,) or not np.isfinite(offset).all():
        raise ValueError(f'{label} requires three finite metres')
    return offset


def pose_matrix(position, quaternion):
    """Source position in metres and quaternion in xyzw order."""
    position, quat = np.asarray(position, dtype=float), np.asarray(quaternion, dtype=float)
    if position.shape != (3,) or quat.shape != (4,) or not np.isfinite(position).all() or not np.isfinite(quat).all():
        raise ValueError('Invalid or nonfinite source pose')
    norm = np.linalg.norm(quat)
    if norm < 1e-6 or abs(norm - 1) > 0.1:
        raise ValueError('Invalid source quaternion')
    x, y, z, w = quat / norm
    transform = np.eye(4)
    transform[:3, :3] = np.array([
        [1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
        [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
        [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)],
    ])
    transform[:3, 3] = position
    return transform


def tool_transform(offset, rpy):
    """T_source_eef: mounting origin in source axes, with contract tool rotation."""
    offset, rpy = eef_offset(offset), np.asarray(rpy, dtype=float)
    if offset.shape != (3,) or rpy.shape != (3,) or not np.isfinite([offset, rpy]).all():
        raise ValueError('Tool offset and RPY require three finite values each')
    matrix = np.eye(4)
    matrix[:3, :3] = rpy_matrix(rpy)
    matrix[:3, 3] = offset
    return matrix
