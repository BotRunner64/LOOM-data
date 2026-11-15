#!/usr/bin/env python3
"""Open a MuJoCo viewer with the canonical robot models side by side.

    python scripts/robot/show_models.py            # interactive viewer
    python scripts/robot/show_models.py --seconds 3  # auto-close (smoke test)

Why this is not just ``mujoco.viewer.launch(urdf)``
---------------------------------------------------
MuJoCo's URDF importer drops ``<visual>`` geometry by default, so a plain URDF
shows only collision meshes.  This script prepares a MuJoCo-ready scene per
robot by adding ``<mujoco><compiler discardvisual="false"/></mujoco>`` so the
visual meshes are imported. OBJ materials/textures remain in the package for
URDF viewers; MuJoCo's URDF importer does not render those materials.

Requires ``pip install mujoco``.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import mujoco.viewer

MODELS = Path(__file__).resolve().parents[2] / "assets" / "robot_models"
ROBOTS = [("yam", 0.0), ("agibot_g2", 2.4), ("galaxea_r1lite", 4.8), ("galaxea_r1pro", 7.2)]


def load_visual_spec(robot_dir: Path) -> mujoco.MjSpec:
    """Load a robot with visual meshes (materials are ignored)."""
    tree = ET.parse(robot_dir / "robot.urdf")
    root = tree.getroot()
    mujoco_tag = ET.Element("mujoco")
    ET.SubElement(mujoco_tag, "compiler", {"discardvisual": "false"})
    root.insert(0, mujoco_tag)
    # Mesh paths are relative, so the temp URDF must stay in the robot dir.
    with tempfile.NamedTemporaryFile(
        "w", suffix=".urdf", dir=robot_dir, delete=False, encoding="utf-8"
    ) as handle:
        tree.write(handle, encoding="unicode", xml_declaration=True)
        temp_urdf = Path(handle.name)
    try:
        spec = mujoco.MjSpec.from_file(str(temp_urdf))
    finally:
        temp_urdf.unlink(missing_ok=True)
    return spec


def build_model() -> mujoco.MjModel:
    spec = mujoco.MjSpec()
    spec.worldbody.add_geom(
        name="floor",
        type=mujoco.mjtGeom.mjGEOM_PLANE,
        size=[20, 20, 0.05],
        rgba=[0.72, 0.72, 0.75, 1.0],
    )
    light_type = mujoco.mjtLightType.mjLIGHT_DIRECTIONAL
    for position in ([3, -3, 6], [-3, 3, 6], [3, 3, 6]):
        spec.worldbody.add_light(
            type=light_type, pos=position, dir=[0, 0, -1],
            diffuse=[0.9, 0.9, 0.9], ambient=[0.35, 0.35, 0.35],
        )
    for name, offset in ROBOTS:
        child = visual_spec(name)
        mount = spec.worldbody.add_body(name=f"{name}_root", pos=[offset, 0, 0])
        spec.attach(child, prefix=f"{name}_", frame=mount.add_frame())
    return spec.compile()


def visual_spec(name: str) -> mujoco.MjSpec:
    """Visual-only static inspection; collision geometry must not obscure it."""
    spec = load_visual_spec(MODELS / name)
    for geom in list(spec.geoms):
        if geom.contype or geom.conaffinity:
            spec.delete(geom)
    return spec


def set_display_pose(model, data, name: str, prefix: str = "") -> None:
    manifest = json.loads((MODELS / name / "robot.json").read_text())
    values = dict(manifest.get("display_configuration", {}))
    for joint, coupling in manifest.get("mimic_joints", {}).items():
        values[joint] = float(coupling.get("offset", 0)) + float(coupling.get("multiplier", 1)) * values[coupling["joint"]]
    for joint, value in values.items():
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, prefix + joint)
        if jid >= 0:
            data.qpos[model.jnt_qposadr[jid]] = value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seconds", type=float, default=0, help="auto-close after N seconds")
    args = parser.parse_args()

    model = build_model()
    data = mujoco.MjData(model)
    for name, _ in ROBOTS:
        set_display_pose(model, data, name, prefix=name + "_")
    mujoco.mj_forward(model, data)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.lookat[:] = [3.6, 0.0, 0.7]
        viewer.cam.distance = 6.5
        viewer.cam.azimuth = 110.0
        viewer.cam.elevation = -14.0
        viewer.sync()

        started = time.time()
        while viewer.is_running():
            # A URDF viewer holds the supplied pose; it has no controllers.
            mujoco.mj_forward(model, data)
            viewer.sync()
            time.sleep(1.0 / 120.0)
            if args.seconds and time.time() - started > args.seconds:
                break
    return 0


if __name__ == "__main__":
    sys.exit(main())
