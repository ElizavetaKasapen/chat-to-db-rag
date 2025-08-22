import json
import yaml

config_path = "config.json"
prompt_path = "prompts.yaml"

def load_config(section: str) -> dict:
    try:
        with open(config_path, "r") as file:
            config = json.load(file)
            return config[section]
    except Exception as e:
        print(f"*** You have problems with the configuration file (section '{section}'): {e} ***")
        raise e
    
def load_qdrant_config():
    return load_config("vectorstore")

def load_models_config():
    return load_config("models")

def load_search_config():
    return load_config("search")

def load_prompts() -> dict:
    try:
        with open(prompt_path, "r") as file:
            return yaml.safe_load(file)
    except Exception as e:
        print(f"*** Error loading YAML '{prompt_path}': {e} ***")
        raise