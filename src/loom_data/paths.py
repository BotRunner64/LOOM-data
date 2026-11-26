"""Default locations. Data and robot models live beside the checkout, outside Git."""
import os
from pathlib import Path

ROOT = Path(os.environ.get('LOOM_DATA_ROOT') or Path(__file__).resolve().parents[2]).resolve()
RAW = ROOT / 'dataset/raw'
EXPORT = ROOT / 'dataset/export'
PACKED = ROOT / 'dataset/packed'
WORK = ROOT / 'dataset/work'
MANIFESTS = ROOT / 'dataset/manifests'
MODELS = ROOT / 'assets/robot_models'
