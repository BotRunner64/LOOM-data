"""Minimal URDF reader for forward kinematics of the packaged robot models."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import xml.etree.ElementTree as ET


@dataclass(frozen=True)
class Joint:
    name: str
    type: str
    parent: str
    child: str
    xyz: tuple[float, ...]
    rpy: tuple[float, ...]
    axis: tuple[float, ...]
    mimic: dict[str, str] | None


@dataclass(frozen=True)
class Robot:
    name: str
    links: tuple[str, ...]
    joints: tuple[Joint, ...]


def _vector(element: ET.Element | None, attribute: str, default: str) -> tuple[float, ...]:
    text = element.get(attribute, default) if element is not None else default
    return tuple(float(value) for value in text.split())


def parse_urdf(path: Path) -> Robot:
    root = ET.parse(path).getroot()
    joints = []
    for element in root.findall("joint"):
        origin = element.find("origin")
        mimic = element.find("mimic")
        joints.append(Joint(
            name=element.attrib["name"],
            type=element.attrib["type"],
            parent=element.find("parent").attrib["link"],
            child=element.find("child").attrib["link"],
            xyz=_vector(origin, "xyz", "0 0 0"),
            rpy=_vector(origin, "rpy", "0 0 0"),
            axis=_vector(element.find("axis"), "xyz", "1 0 0"),
            mimic=dict(mimic.attrib) if mimic is not None else None,
        ))
    return Robot(
        name=root.get("name", "robot"),
        links=tuple(link.attrib["name"] for link in root.findall("link")),
        joints=tuple(joints),
    )
