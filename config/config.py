# config/config.py
import yaml
from pathlib import Path
from functools import lru_cache

@lru_cache()
def load_config():
    config_path = Path(__file__).parent / "config.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)

def get_storage_config():
    return load_config()['storage_manager']

def get_core_config():
    return load_config()['core']

def get_evaluation_config():
    return load_config()['evaluation']