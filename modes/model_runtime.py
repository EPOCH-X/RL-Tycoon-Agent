"""모델 런타임 옵션 로더 – train_config_used.json에서 게임 설정을 복원합니다."""

import json
import os


def load_model_runtime_options(model_path: str) -> tuple[dict, dict]:
    """모델 폴더의 train_config_used.json에서 game_overrides와 env_options를 읽습니다.

    Returns:
        (game_overrides, env_options) — 각각 _comment 키 제외.
        파일이 없으면 ({}, {}).
    """
    model_dir = os.path.dirname(model_path) or "."
    cfg_path = os.path.join(model_dir, "train_config_used.json")
    if not os.path.isfile(cfg_path):
        return {}, {}

    with open(cfg_path, encoding="utf-8") as f:
        cfg = json.load(f)

    game_overrides = {
        k: v for k, v in cfg.get("game_overrides", {}).items()
        if not k.startswith("_")
    }
    env_options = {
        k: v for k, v in cfg.get("env_options", {}).items()
        if not k.startswith("_")
    }
    return game_overrides, env_options
