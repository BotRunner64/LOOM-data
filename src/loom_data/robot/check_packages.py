#!/usr/bin/env python3
"""Validate distributable URDF packages without upstream sources.

python scripts/robot/check_packages.py --load --report /tmp/model-validation.json
--load requires mujoco; checks visual AND collision geometry after relocation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import shlex
import shutil
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

MODELS = Path(__file__).resolve().parents[2] / "assets/robot_models"


def check_package(directory: Path, load: bool = False) -> dict:
    manifest = json.loads((directory / "robot.json").read_text())
    root = ET.parse(directory / manifest["urdf"]).getroot()

    def require(condition, message):
        if not condition:
            raise ValueError(f"{directory.name}: {message}")

    def asset(relative: str, parent: Path = directory) -> Path:
        require(not Path(relative).is_absolute() and "://" not in relative, f"nonlocal asset {relative}")
        path = (parent / relative).resolve()
        require(path.is_relative_to(directory.resolve()), f"asset escapes package: {relative}")
        require(path.is_file(), f"missing asset {relative}")
        require(not path.read_bytes().startswith(b"version https://git-lfs.github.com/spec"), f"LFS pointer {relative}")
        return path

    links = {x.get("name"): x for x in root.findall("link")}
    joints = {x.get("name"): x for x in root.findall("joint")}
    require(len(links) == len(root.findall("link")), "duplicate link names")
    require(len(joints) == len(root.findall("joint")), "duplicate joint names")
    require(any(directory.glob("LICENSE*")), "missing license")
    children = {}
    for name, joint in joints.items():
        parent, child = joint.find("parent").get("link"), joint.find("child").get("link")
        require(parent in links and child in links, f"dangling joint {name}")
        require(child not in children, f"multiple parents for {child}")
        children[child] = parent
        if joint.get("type") in ("revolute", "prismatic"):
            limit = joint.find("limit")
            require(limit is not None, f"missing limits {name}")
            lo, hi = float(limit.get("lower")), float(limit.get("upper"))
            require(math.isfinite(lo) and math.isfinite(hi) and lo <= hi, f"invalid limits {name}")
        axis = joint.find("axis")
        if axis is not None:
            vector = np.fromstring(axis.get("xyz"), sep=" ")
            require(len(vector) == 3 and np.isfinite(vector).all() and abs(np.linalg.norm(vector)-1) < 1e-5, f"invalid axis {name}")
        chain = set()
        current = name
        while joints[current].find("mimic") is not None:
            require(current not in chain, f"mimic cycle {name}")
            chain.add(current)
            current = joints[current].find("mimic").get("joint")
            require(current in joints, f"missing mimic target {current}")
    require(set(links) - set(children) == {manifest["base_link"]}, "invalid root")
    for name in links:
        seen = set()
        current = name
        while current in children:
            require(current not in seen, f"cycle at {name}")
            seen.add(current)
            current = children[current]
        require(current == manifest["base_link"], f"disconnected link {name}")
    for side, arm in manifest["arms"].items():
        require(arm["dof"] == len(arm["joints"]), f"wrong DOF {side}")
        for name in [arm["base_link"], arm["eef_link"], *arm["links"], *arm["eef_candidates"].values(),
                     arm["camera_link"], arm["gripper"]["mount_link"], *arm["gripper"]["finger_links"]]:
            require(name in links, f"manifest missing link {name}")
        for name in [*arm["joints"], *arm["gripper"]["finger_joints"]]:
            require(name in joints, f"manifest missing joint {name}")
        for joint_name, link_name in zip(arm["joints"], arm["links"]):
            require(joints[joint_name].find("child").get("link") == link_name, f"wrong arm order {side}")
    independent = [name for name,j in joints.items() if j.get("type") != "fixed" and j.find("mimic") is None]
    require(manifest["joint_order"] == independent, "wrong independent joint order")
    for link in links.values():
        inertial = link.find("inertial")
        if inertial is None:
            continue
        mass = float(inertial.find("mass").get("value"))
        a = {k: float(v) for k,v in inertial.find("inertia").attrib.items()}
        tensor = np.array([[a['ixx'],a['ixy'],a['ixz']], [a['ixy'],a['iyy'],a['iyz']], [a['ixz'],a['iyz'],a['izz']]])
        require(np.isfinite(tensor).all() and math.isfinite(mass) and mass > 0, f"invalid inertia {link.get('name')}")
        moments = np.linalg.eigvalsh(tensor)
        require(moments[0] > 0 and moments[2] <= moments[0]+moments[1]+1e-10, f"nonphysical inertia {link.get('name')}")
    pending = [asset(m.get("filename")) for m in root.iter("mesh")]
    pending += [asset(t.get("filename")) for t in root.iter("texture")]
    visited = set()
    while pending:
        path = pending.pop()
        if path in visited:
            continue
        visited.add(path)
        if path.suffix.lower() in (".obj", ".mtl"):
            for line in path.read_text().splitlines():
                parts = shlex.split(line, comments=True)
                if not parts:
                    continue
                if parts[0].lower() == "mtllib":
                    pending += [asset(p, path.parent) for p in parts[1:]]
                elif parts[0].lower().startswith("map_") or parts[0].lower() in ("bump", "disp", "decal", "refl"):
                    pending.append(asset(parts[-1], path.parent))
    require(bool(manifest.get("files")), "missing file hashes")
    for relative, digest in manifest["files"].items():
        require(hashlib.sha256(asset(relative).read_bytes()).hexdigest() == digest, f"checksum mismatch {relative}")
    result = {"robot": manifest["robot"], "links": len(links), "joints": len(joints),
              "independent_joints": len(independent), "asset_dependencies": len(visited), "status": "passed"}
    if load:
        import mujoco
        # A copied package with no sibling sources must load with all visuals.
        with tempfile.TemporaryDirectory(prefix="robot-package-") as temp:
            relocated = Path(temp) / directory.name
            shutil.copytree(directory, relocated)
            tree = ET.parse(relocated / "robot.urdf")
            extension = ET.SubElement(tree.getroot(), "mujoco")
            ET.SubElement(extension, "compiler", {"discardvisual": "false", "fusestatic": "false"})
            tree.write(relocated / "check.urdf")
            model = mujoco.MjModel.from_xml_path(str(relocated / "check.urdf"))
            data = mujoco.MjData(model)
            mujoco.mj_forward(model, data)
            require(np.isfinite(data.xpos).all(), "nonfinite FK after load")
            result["relocated_load"] = {"engine": f"mujoco {mujoco.__version__}", "meshes": model.nmesh, "geoms": model.ngeom}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", type=Path, default=MODELS)
    parser.add_argument("--load", action="store_true")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    results = [check_package(p.parent, args.load) for p in sorted(args.models.glob("*/robot.json"))]
    if not results:
        raise SystemExit("No model packages found")
    report = json.dumps(results, indent=2, ensure_ascii=False)
    print(report)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(report + "\n")


if __name__ == "__main__":
    main()
