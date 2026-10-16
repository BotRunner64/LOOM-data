"""URDF forward kinematics; poses map link-local coordinates into base_link.

Only numpy and the self-contained model package are needed. Mimic joints are
resolved recursively. Input positions use radians/metres and canonical names.
"""
from __future__ import annotations
import math
import numpy as np

from .urdf_model import Robot

def _rot_x(a): return np.array([[1, 0, 0], [0, math.cos(a), -math.sin(a)], [0, math.sin(a), math.cos(a)]])
def _rot_y(a): return np.array([[math.cos(a), 0, math.sin(a)], [0, 1, 0], [-math.sin(a), 0, math.cos(a)]])
def _rot_z(a): return np.array([[math.cos(a), -math.sin(a), 0], [math.sin(a), math.cos(a), 0], [0, 0, 1]])


def rpy_matrix(rpy) -> np.ndarray:
    roll, pitch, yaw = rpy
    return _rot_z(yaw) @ _rot_y(pitch) @ _rot_x(roll)


def axis_angle_matrix(axis, angle: float) -> np.ndarray:
    x, y, z = axis
    norm = math.sqrt(x * x + y * y + z * z) or 1.0
    x, y, z = x / norm, y / norm, z / norm
    c, s, C = math.cos(angle), math.sin(angle), 1 - math.cos(angle)
    return np.array([
        [x * x * C + c, x * y * C - z * s, x * z * C + y * s],
        [y * x * C + z * s, y * y * C + c, y * z * C - x * s],
        [z * x * C - y * s, z * y * C + x * s, z * z * C + c],
    ])


def _homogeneous(rotation: np.ndarray, translation) -> np.ndarray:
    matrix = np.eye(4)
    matrix[:3, :3] = rotation
    matrix[:3, 3] = translation
    return matrix


def joint_matrix(joint, value: float) -> np.ndarray:
    matrix = _homogeneous(rpy_matrix(joint.rpy), joint.xyz)
    if joint.type in ("revolute", "continuous"):
        matrix = matrix @ _homogeneous(axis_angle_matrix(joint.axis or (1, 0, 0), value), (0, 0, 0))
    elif joint.type == "prismatic":
        axis = np.array(joint.axis or (1, 0, 0), dtype=float)
        matrix = matrix @ _homogeneous(np.eye(3), value * axis)
    return matrix


def fk_poses(robot: Robot, values: dict[str, float]) -> dict[str, np.ndarray]:
    child_joint = {joint.child: joint for joint in robot.joints}
    child_links = set(child_joint)
    roots = [name for name in robot.links if name not in child_links]
    poses: dict[str, np.ndarray] = {}

    resolved = {}
    resolving = set()
    by_name = {j.name: j for j in robot.joints}

    def joint_value(name):
        if name in resolved:
            return resolved[name]
        if name in resolving:
            raise ValueError(f"Mimic cycle at {name}")
        resolving.add(name)
        joint = by_name[name]
        if joint.mimic:
            value = float(joint.mimic.get("multiplier", 1)) * joint_value(joint.mimic["joint"]) + float(joint.mimic.get("offset", 0))
        else:
            value = float(values.get(name, 0.0))
        if not math.isfinite(value):
            raise ValueError(f"Nonfinite joint value {name}")
        resolving.remove(name)
        resolved[name] = value
        return value

    def compute(link: str) -> np.ndarray:
        if link in poses:
            return poses[link]
        joint = child_joint.get(link)
        if joint is None:
            poses[link] = np.eye(4)
        else:
            value = joint_value(joint.name)
            poses[link] = compute(joint.parent) @ joint_matrix(joint, value)
        return poses[link]

    for root in roots:
        compute(root)
    for link in robot.links:
        compute(link)
    return poses
