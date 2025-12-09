import importlib

def load_configs(name: str):
    return importlib.import_module(f".{name}", package=__name__)