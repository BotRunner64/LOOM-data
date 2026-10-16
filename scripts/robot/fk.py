#!/usr/bin/env python3
"""Compute a canonical link pose from a self-contained URDF package.

python scripts/robot/fk.py --dataset ABC-130K --link arm_left_grasp
python scripts/robot/fk.py --model-dir /path/to/yam --link arm_left_tcp --joints '{"arm_left_joint2": 1.047}'
Omitted joints use zero; --display-pose instead initializes the inspection pose.
Output is T_base_link: p_base = T_base_link @ p_link (homogeneous coordinates).
"""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.robot.urdf_model import parse_urdf
from scripts.robot.kinematics import fk_poses


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    selector = parser.add_mutually_exclusive_group(required=True)
    selector.add_argument("--dataset")
    selector.add_argument("--model-dir", type=Path)
    parser.add_argument("--robot-type", help="source robot_type for datasets with multiple embodiments")
    parser.add_argument("--link", required=True)
    parser.add_argument("--joints", default="{}", help="JSON object, radians/metres, canonical joint names")
    parser.add_argument("--display-pose", action="store_true")
    args = parser.parse_args()
    directory = args.model_dir
    if args.dataset:
        models = Path(__file__).resolve().parents[2] / "assets/robot_models"
        catalog = json.loads((models / "catalog.json").read_text())["datasets"]
        if args.dataset not in catalog:
            parser.error(f"Unknown dataset; choose from {', '.join(catalog)}")
        entry = catalog[args.dataset]
        if "variants" in entry:
            if args.robot_type not in entry["variants"]:
                parser.error(f"{args.dataset} requires --robot-type: {', '.join(entry['variants'])}")
            entry = entry["variants"][args.robot_type]
        elif args.robot_type:
            parser.error("--robot-type is only supported for datasets with model variants")
        if not entry["manifest"]:
            parser.error(entry["reason"])
        directory = (models / entry["manifest"]).parent
    elif args.robot_type:
        parser.error("--robot-type requires --dataset")
    manifest = json.loads((directory / "robot.json").read_text())
    robot = parse_urdf(directory / manifest["urdf"])
    values = dict(manifest["display_configuration"]) if args.display_pose else {}
    supplied = json.loads(args.joints)
    unknown = set(supplied) - set(manifest["joint_order"])
    if unknown:
        parser.error(f"Not independent joint names: {sorted(unknown)}; mimic joints are automatic")
    values.update(supplied)
    if args.link not in robot.links:
        parser.error(f"Unknown link: {args.link}")
    print(json.dumps({"robot": manifest["robot"], "reference": manifest["base_link"],
                      "link": args.link, "T_base_link": fk_poses(robot, values)[args.link].tolist()}, indent=2))


if __name__ == "__main__":
    main()
