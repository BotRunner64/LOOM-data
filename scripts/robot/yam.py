"""Shared YAM measured-joint FK and export-contract TCP convention."""
import json

import numpy as np

from scripts.robot.urdf_model import Robot, parse_urdf
from scripts.robot.kinematics import fk_poses


class YamFK:
    def __init__(self, directory, *, arm_local=False):
        manifest = json.loads((directory / 'robot.json').read_text())
        if manifest['robot'] != 'yam':
            raise ValueError('Expected the YAM model')
        robot = parse_urdf(directory / manifest['urdf'])
        parents = {joint.child: joint for joint in robot.joints}
        self.arms = {}
        for side in ('left', 'right'):
            arm = manifest['arms'][side]
            tip = arm['eef_candidates']['grasp']
            base = arm['base_link'] if arm_local else manifest['base_link']
            chain, link = [], tip
            while link != base and link in parents:
                joint = parents[link]
                chain.append(joint)
                link = joint.parent
            if len(arm['joints']) != 6 or {j.name for j in chain if j.type != 'fixed'} != set(arm['joints']):
                raise ValueError('Expected six arm joints and a fixed base-to-grasp chain')
            if link != base:
                raise ValueError('Grasp chain does not reach model base')
            self.arms[side] = (Robot(robot.name, (link, *(j.child for j in reversed(chain))),
                                    tuple(reversed(chain))), arm['joints'], tip)
        # grasp axes: X=-Y_link6, Y=+X_link6, Z=+Z_link6.
        # Contract EEF: +X=+Z_link6 (approach), +Z=+Y_link6 (back of hand),
        # +Y=+Z x +X=+X_link6.
        self.rotation = np.array([[0., 0., -1.], [0., 1., 0.], [1., 0., 0.]])

    def pose(self, side, positions):
        q = np.asarray(positions, dtype=float)
        if q.shape != (6,) or not np.isfinite(q).all():
            raise ValueError(f'{side}: expected six finite measured joint angles')
        robot, names, tip = self.arms[side]
        transform = fk_poses(robot, dict(zip(names, q)))[tip]
        rotation = transform[:3, :3] @ self.rotation
        return np.concatenate((transform[:3, 3], rotation.T.reshape(-1))).tolist()
