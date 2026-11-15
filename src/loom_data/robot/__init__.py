"""Robot models: URDF parsing, forward kinematics, and tools to inspect the model packages."""
import sys

COMMANDS = {
    'fk': ('loom_data.robot.fk', 'pose of one link for given joint values'),
    'check': ('loom_data.robot.packages', 'validate the URDF packages under assets/robot_models'),
    'show': ('loom_data.robot.show_models', 'open the models side by side in the MuJoCo viewer'),
    'realomin-mount': ('loom_data.robot.realomin_mount', 'derive the RealOmin gripper mount from the vendor URDFs'),
}


def main(argv=None):
    from loom_data.cli import dispatch
    return dispatch('loom-data robot', COMMANDS, sys.argv[1:] if argv is None else argv)
