"""Helpers for applying model runtime options from saved training configs."""

import json
import os


def load_model_runtime_options(model_path: str | None) -> tuple[dict, dict]:
    """Load game overrides and env options from a model's train config."""
    if not model_path:
        return {}, {}

    cfg_path = os.path.join(os.path.dirname(model_path), "train_config_used.json")
    if not os.path.isfile(cfg_path):
        return {}, {}

    with open(cfg_path, encoding="utf-8") as f:
        cfg = json.load(f)

    game_overrides = {
        key: value
        for key, value in cfg.get("game_overrides", {}).items()
        if not key.startswith("_") and value is not None
    }
    env_options = {
        key: value
        for key, value in cfg.get("env_options", {}).items()
        if not key.startswith("_")
    }
    return game_overrides, env_options
