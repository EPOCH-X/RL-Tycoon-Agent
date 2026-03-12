"""Common utilities shared by all algorithm trainers."""

import os
import json
from typing import Any

import torch
import torch.nn as nn
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from config.settings import load_json_config
from ai.gym_env import TycoonEnv


def get_device(force_cpu: bool = False) -> torch.device:
    """GPU 사용 가능 여부를 확인하고 적절한 디바이스를 반환합니다.

    CUDA가 사용 가능하면 GPU를, 아니면 CPU를 반환합니다.
    SB3 알고리즘에는 "auto"를 사용하세요 (SB3가 자체 판단).
    """
    if force_cpu:
        return torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda")
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_mem / (1024**3)
        print(f"  [GPU] Using CUDA: {gpu_name} ({gpu_mem:.1f} GB)")
        return device
    print("  [GPU] CUDA not available, using CPU")
    return torch.device("cpu")


def get_sb3_device() -> str:
    """SB3 알고리즘용 디바이스 문자열을 반환합니다.

    SB3는 'auto', 'cuda', 'cpu' 문자열을 받습니다.
    """
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_mem / (1024**3)
        print(f"  [GPU] SB3 will use CUDA: {gpu_name} ({gpu_mem:.1f} GB)")
        return "auto"
    print("  [GPU] CUDA not available, SB3 will use CPU")
    return "auto"


def load_algo_config(algo_name: str, config_path: str | None = None) -> dict:
    """알고리즘별 설정 JSON을 로드합니다.

    우선순위: config_path > algorithms/<algo>/config.json > config/train_config.json
    """
    if config_path and os.path.isfile(config_path):
        with open(config_path, encoding="utf-8") as f:
            return json.load(f)
    # 알고리즘 폴더 내 기본 config
    algo_dir = os.path.join(
        os.path.dirname(__file__), algo_name.lower(), "config.json"
    )
    if os.path.isfile(algo_dir):
        with open(algo_dir, encoding="utf-8") as f:
            return json.load(f)
    # fallback: 기존 train_config.json
    return load_json_config("train_config.json")


def make_env(rank: int = 0, seed: int = 0,
             game_overrides: dict | None = None,
             reward_config: dict | None = None):
    """환경 생성 팩토리 함수."""
    def _init():
        kwargs = {}
        if game_overrides:
            if game_overrides.get("target_money") is not None:
                kwargs["target_money"] = game_overrides["target_money"]
            if game_overrides.get("day_limit") is not None:
                kwargs["day_limit"] = game_overrides["day_limit"]
        env = TycoonEnv(reward_config=reward_config, **kwargs)
        env.reset(seed=seed + rank)
        return env
    return _init


def make_vec_env(n_envs: int, seed: int = 0,
                 game_overrides: dict | None = None,
                 reward_config: dict | None = None,
                 force_dummy: bool = False):
    """벡터화된 환경을 생성합니다."""
    factories = [make_env(i, seed, game_overrides, reward_config)
                 for i in range(n_envs)]
    if n_envs > 1 and not force_dummy:
        return SubprocVecEnv(factories)
    return DummyVecEnv(factories)


def resolve_activation(name: str):
    """문자열을 PyTorch activation 클래스로 변환합니다."""
    mapping = {"tanh": nn.Tanh, "relu": nn.ReLU, "elu": nn.ELU, "gelu": nn.GELU}
    return mapping.get(name.lower(), nn.Tanh)


def build_policy_kwargs(net_cfg: dict) -> dict:
    """네트워크 설정으로부터 SB3 policy_kwargs를 생성합니다."""
    policy_kwargs: dict[str, Any] = {}
    if net_cfg.get("net_arch"):
        policy_kwargs["net_arch"] = net_cfg["net_arch"]
    if net_cfg.get("activation_fn"):
        policy_kwargs["activation_fn"] = resolve_activation(net_cfg["activation_fn"])
    return policy_kwargs or {}


def save_run_config(save_path: str, cfg: dict) -> None:
    """사용된 설정을 저장 디렉토리에 기록합니다."""
    os.makedirs(save_path, exist_ok=True)
    with open(os.path.join(save_path, "train_config_used.json"), "w",
              encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)
