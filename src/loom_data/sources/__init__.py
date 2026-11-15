"""Source datasets. Adding one means adding a module here and its name to NAMES."""
from importlib import import_module

NAMES = ('abc130k', 'molmoact2', 'hifi_umi', 'galaxea', 'agibotworld2026', 'realomin', 'agibotworld_beta')


def get(name):
    """The Source class for a dataset key; its dependencies load only when asked for."""
    if name not in NAMES:
        raise ValueError(f'Unknown dataset: {name}')
    return import_module(f'loom_data.sources.{name}').SOURCE
