import yaml
import os

CONFIG_PATH = "../datasets.yaml"

def _load_config():
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)

def get_dataset(name):
    config = _load_config()
    if name not in config["datasets"]:
        raise KeyError(f"Dataset '{name}' not found in {CONFIG_PATH}")
    return config["datasets"][name]

def get_root(name):
    return get_dataset(name)["root"]

def get_transcripts_path(name):
    return get_dataset(name)["transcripts"]

def get_pairs_path(name):
    return get_dataset(name)["pairs"]

def get_split_path(name, split):
    splits = get_dataset(name)["splits"]
    if split not in splits:
        raise KeyError(f"Split '{split}' not found for dataset '{name}'")
    return splits[split]
