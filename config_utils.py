"""Configuration loading with optional recursive YAML inheritance."""

import copy
import os

import yaml


def _deep_merge(base, override):
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def load_config(path):
    path = os.path.abspath(path)
    with open(path) as handle:
        config = yaml.safe_load(handle) or {}
    base_path = config.pop("base", None)
    if base_path is None:
        return config
    if not os.path.isabs(base_path):
        base_path = os.path.join(os.path.dirname(path), base_path)
    return _deep_merge(load_config(base_path), config)
