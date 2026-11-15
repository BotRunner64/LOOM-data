"""Source datasets. Adding one means adding a module here and its entry in TITLES."""
from importlib import import_module

# Dataset key -> upstream name, which is also its folder under dataset/raw and dataset/export.
TITLES = {
    'abc130k': 'ABC-130K',
    'molmoact2': 'MolmoAct2-BimanualYAM',
    'hifi_umi': 'HiFi-UMI-2K',
    'galaxea': 'Galaxea-Open-World-Dataset',
    'agibotworld2026': 'AgiBotWorld2026',
    'realomin': '10Kh-RealOmin-OpenData',
    'agibotworld_beta': 'AgiBotWorld-Beta',
}
NAMES = tuple(TITLES)


def get(name):
    """The Source class for a dataset key; its dependencies load only when asked for."""
    if name not in TITLES:
        raise ValueError(f'Unknown dataset: {name}')
    return import_module(f'loom_data.sources.{name}').SOURCE
