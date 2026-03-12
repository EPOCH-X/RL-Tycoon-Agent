"""Common utilities shared by all algorithm trainers."""

import os
import json
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.callbacks import BaseCallback

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
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"  [GPU] Using CUDA: {gpu_name} ({gpu_mem:.1f} GB)")
        return device
    print("  [GPU] CUDA not available, using CPU")
    return torch.device("cpu")


def get_sb3_device(policy: str = "MlpPolicy") -> str:
    """SB3 알고리즘용 디바이스 문자열을 반환합니다.

    SB3 MLP 정책은 네트워크가 작아 CPU가 더 빠릅니다 (GPU↔CPU 전송 오버헤드).
    CNN 정책일 때만 GPU를 사용합니다.
    """
    is_cnn = "Cnn" in policy
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        if is_cnn:
            print(f"  [Device] SB3 CNN → CUDA: {gpu_name}")
            return "auto"
        else:
            print(f"  [Device] SB3 MLP → CPU (small network, GPU overhead 불리)")
            return "cpu"
    print("  [Device] CUDA not available → CPU")
    return "cpu"


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


# ────────────────────────────────────────────────
# Early Stopping (SB3 Callback)
# ────────────────────────────────────────────────
class EarlyStopCallback(BaseCallback):
    """SB3 EvalCallback의 callback_after_eval로 사용하는 Early Stopping.

    eval_freq마다 호출되며, patience 횟수 연속 개선이 없으면 학습을 중단합니다.

    Parameters
    ----------
    patience : int
        개선 없이 허용할 최대 평가 횟수.
    min_delta : float
        개선으로 인정할 최소 보상 변화량.
    verbose : int
        0 = 무음, 1 = 중단 시 출력, 2 = 매 평가 출력.
    """

    def __init__(self, patience: int = 50, min_delta: float = 1.0,
                 verbose: int = 1):
        super().__init__(verbose)
        self.patience = patience
        self.min_delta = min_delta
        self.best_reward: float = -np.inf
        self.wait: int = 0

    def _on_step(self) -> bool:
        # parent(EvalCallback)가 self.parent에 mean_reward를 기록
        parent = self.parent
        if parent is None:
            return True

        mean_reward = parent.last_mean_reward
        if mean_reward is None:
            return True

        if mean_reward > self.best_reward + self.min_delta:
            self.best_reward = mean_reward
            self.wait = 0
            if self.verbose >= 2:
                print(f"  [EarlyStop] New best: {mean_reward:.1f}")
        else:
            self.wait += 1
            if self.verbose >= 2:
                print(f"  [EarlyStop] No improve: {self.wait}/{self.patience} "
                      f"(best={self.best_reward:.1f}, current={mean_reward:.1f})")

        if self.wait >= self.patience:
            if self.verbose >= 1:
                print(f"\n  [EarlyStop] ★ 학습 조기 종료! "
                      f"{self.patience}회 연속 개선 없음 "
                      f"(best={self.best_reward:.1f}, "
                      f"steps={self.num_timesteps})")
            return False  # 학습 중단
        return True


# ────────────────────────────────────────────────
# Early Stopping (Custom PyTorch trainers용)
# ────────────────────────────────────────────────
class EarlyStopTracker:
    """Custom PyTorch 트레이너(SAC, A3C, ModelBased)용 Early Stopping 트래커.

    Parameters
    ----------
    patience : int
        개선 없이 허용할 최대 평가 횟수.
    min_delta : float
        개선으로 인정할 최소 보상 변화량.
    """

    def __init__(self, patience: int = 50, min_delta: float = 1.0,
                 verbose: int = 1):
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.best_reward: float = -np.inf
        self.wait: int = 0

    def check(self, eval_reward: float) -> bool:
        """개선 여부를 확인하고, 중단할지 반환합니다.

        Returns
        -------
        bool
            True면 학습 계속, False면 학습 중단.
        """
        if eval_reward > self.best_reward + self.min_delta:
            self.best_reward = eval_reward
            self.wait = 0
            return True

        self.wait += 1
        if self.verbose >= 2:
            print(f"  [EarlyStop] No improve: {self.wait}/{self.patience} "
                  f"(best={self.best_reward:.1f}, current={eval_reward:.1f})")

        if self.wait >= self.patience:
            if self.verbose >= 1:
                print(f"\n  [EarlyStop] ★ 학습 조기 종료! "
                      f"{self.patience}회 연속 개선 없음 "
                      f"(best={self.best_reward:.1f})")
            return False
        return True
