"""Reproduce the nominal RealOmin V3 rear mount from the vendor's two URDFs."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

if not __package__:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.robot.kinematics import rpy_matrix
from scripts.robot.urdf_model import parse_urdf


def mounting_geometry(gripper_urdf, controller_urdf):
    """Transfer the controller's rear flange to the capture gripper, in metres.

    Register the rigid palm using the two drive-pivot midpoint and the first
    pivot's fixed axes. This is a nominal geometry correspondence, not a fit to
    the moving fingers or a device-specific camera calibration.
    """
    gripper = {joint.name: joint for joint in parse_urdf(gripper_urdf).joints}
    controller = {joint.name: joint for joint in parse_urdf(controller_urdf).joints}

    def joint(joints, name, kind):
        value = joints[name]
        if (value.parent != 'base_link' or value.type != kind
                or not np.isfinite([value.xyz, value.rpy]).all()):
            raise ValueError(f'{name}: expected a finite {kind} joint on base_link')
        return value

    capture_pivots = [joint(gripper, name, 'revolute') for name in ('joint_1', 'joint_2')]
    control_pivots = [joint(controller, name, 'revolute') for name in ('joint1', 'joint2')]
    camera = joint(gripper, 'joint_ca2', 'fixed')  # Middle of the three cameras.
    flange = joint(controller, 'joint_Flange', 'fixed')
    for pivots in (capture_pivots, control_pivots):
        if not np.allclose(rpy_matrix(pivots[0].rpy), rpy_matrix(pivots[1].rpy), atol=1e-6, rtol=0):
            raise ValueError('Drive-pivot fixed axes differ; unsupported model')
    capture_center = np.mean([pivot.xyz for pivot in capture_pivots], axis=0)
    control_center = np.mean([pivot.xyz for pivot in control_pivots], axis=0)
    capture_from_control = rpy_matrix(capture_pivots[0].rpy) @ rpy_matrix(control_pivots[0].rpy).T
    residuals = [capture_center + capture_from_control @ (np.asarray(c.xyz) - control_center) - g.xyz
                 for g, c in zip(capture_pivots, control_pivots)]
    residual = float(np.max(np.linalg.norm(residuals, axis=1)))
    if residual > .001:
        raise ValueError('Drive-pivot registration differs by more than 1 mm; unsupported models')
    capture_mount = capture_center + capture_from_control @ (np.asarray(flange.xyz) - control_center)
    # Raw EE: camera0 optical center, with X forward/Y left/Z up.
    # URDF joint_ca2: optical X right/Y down/Z forward.
    raw_from_optical = np.array([[0., 0., 1.], [-1., 0., 0.], [0., -1., 0.]])
    capture_from_raw = rpy_matrix(camera.rpy) @ raw_from_optical.T
    raw_mount = capture_from_raw.T @ (capture_mount - camera.xyz)
    return {'eef_offset': raw_mount.tolist(),
            'mount_in_gripper_base_m': capture_mount.tolist(),
            'max_pivot_residual_m': residual}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gripper-urdf', type=Path, required=True)
    parser.add_argument('--controller-urdf', type=Path, required=True)
    args = parser.parse_args()
    result = mounting_geometry(args.gripper_urdf, args.controller_urdf)
    result['gripper_urdf_sha256'] = hashlib.sha256(args.gripper_urdf.read_bytes()).hexdigest()
    result['controller_urdf_sha256'] = hashlib.sha256(args.controller_urdf.read_bytes()).hexdigest()
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
