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
        print(f"  [GPU] CUDA 사용: {gpu_name} ({gpu_mem:.1f} GB)")
        return device
    print("  [GPU] CUDA 사용 불가, CPU 사용")
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
            print(f"  [디바이스] SB3 CNN → CUDA: {gpu_name}")
            return "auto"
        else:
            print(f"  [디바이스] SB3 MLP → CPU (소규모 네트워크, GPU 오버헤드 불리)")
            return "cpu"
    print("  [디바이스] CUDA 사용 불가 → CPU")
    return "cpu"


def linear_schedule(initial_value: float):
    """SB3용 선형 학습률 스케줄. progress_remaining 1→0에 따라 lr이 선형 감소."""
    def schedule(progress_remaining: float) -> float:
        return progress_remaining * initial_value
    return schedule


def load_algo_config(algo_name: str, config_path: str | None = None,
                     days: int | None = None) -> dict:
    """알고리즘별 설정 JSON을 로드합니다.

    우선순위: config_path > algorithms/<algo>/config_{days}.json
             > algorithms/<algo>/config.json > config/train_config.json
    """
    if config_path and os.path.isfile(config_path):
        print(f"  [설정 로드] {algo_name} ← 사용자 지정: {config_path}")
        with open(config_path, encoding="utf-8") as f:
            return json.load(f)
    algo_base = os.path.join(os.path.dirname(__file__), algo_name.lower())
    # 60일 전용 config 우선 탐색
    if days and days != 30:
        day_cfg = os.path.join(algo_base, f"config_{days}.json")
        if os.path.isfile(day_cfg):
            print(f"  [설정 로드] {algo_name} ← {days}일 전용: {day_cfg}")
            with open(day_cfg, encoding="utf-8") as f:
                return json.load(f)
    # 알고리즘 폴더 내 기본 config (30일)
    algo_dir = os.path.join(algo_base, "config.json")
    if os.path.isfile(algo_dir):
        print(f"  [설정 로드] {algo_name} ← 전용 설정: {algo_dir}")
        with open(algo_dir, encoding="utf-8") as f:
            return json.load(f)
    # fallback: 기존 train_config.json
    print(f"  [설정 로드] {algo_name} ← 기본 설정: config/train_config.json")
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
# SB3 학습 메트릭 한글 매핑
# ────────────────────────────────────────────────
METRIC_KR: dict[str, str] = {
    "approx_kl":            "근사 KL 발산",
    "clip_fraction":        "클리핑 비율",
    "clip_range":           "클리핑 범위",
    "entropy_loss":         "엔트로피 손실",
    "explained_variance":   "설명 분산",
    "learning_rate":        "학습률",
    "loss":                 "총 손실",
    "n_updates":            "업데이트 횟수",
    "policy_gradient_loss": "정책 경사 손실",
    "value_loss":           "가치 손실",
    "fps":                  "초당 프레임",
    "iterations":           "반복 횟수",
    "time_elapsed":         "경과 시간(초)",
    "total_timesteps":      "총 스텝 수",
    "mean_reward":          "평균 보상",
    "mean_ep_length":       "평균 에피소드 길이",
}


def print_metric_reference():
    """학습 시작 시 메트릭 한글 참조표를 출력합니다."""
    print("\n  ┌─ 메트릭 한글 참조 ─────────────────────────────┐")
    for eng, kr in METRIC_KR.items():
        print(f"  │  {eng:<24s} → {kr}")
    print("  └───────────────────────────────────────────────┘\n")


# ────────────────────────────────────────────────
# Korean Eval + Early Stopping (SB3 Callback)
# ────────────────────────────────────────────────
class KoreanEvalStopCallback(BaseCallback):
    """SB3 EvalCallback의 callback_after_eval로 사용.

    매 평가마다 한글로 결과를 출력하고, Early Stopping을 수행합니다.
    EvalCallback(verbose=0) 과 함께 사용하세요.
    """

    def __init__(self, patience: int = 50, min_delta: float = 1.0,
                 verbose: int = 1):
        super().__init__(verbose)
        self.patience = patience
        self.min_delta = min_delta
        self.best_reward: float = -np.inf
        self.wait: int = 0

    def _on_step(self) -> bool:
        parent = self.parent  # EvalCallback
        if parent is None:
            return True

        mean_reward = parent.last_mean_reward
        if mean_reward is None:
            return True

        # ── 한글 평가 결과 출력 ──
        if self.verbose >= 1:
            std_r = 0.0
            mean_len, std_len = 0.0, 0.0
            if hasattr(parent, "evaluations_results") and parent.evaluations_results:
                last_ep = parent.evaluations_results[-1]
                std_r = float(np.std(last_ep))
            if hasattr(parent, "evaluations_length") and parent.evaluations_length:
                last_len = parent.evaluations_length[-1]
                mean_len = float(np.mean(last_len))
                std_len = float(np.std(last_len))

            print(f"\n  ── 평가 결과 (스텝 {self.num_timesteps:,}) ──────────")
            print(f"  평균 보상 (mean_reward):     {mean_reward:>10.2f} ± {std_r:.2f}")
            print(f"  에피소드 길이 (ep_length):    {mean_len:>10.0f} ± {std_len:.0f}")
            # 개선 표시
            if mean_reward > self.best_reward + self.min_delta:
                print(f"  ★ 신기록! (이전 최고: {self.best_reward:.1f})")
            else:
                print(f"  미개선 ({self.wait + 1}/{self.patience})"
                      f"  최고: {self.best_reward:.1f}")
            print(f"  ─────────────────────────────────────────")

        # ── Early Stopping 로직 ──
        if mean_reward > self.best_reward + self.min_delta:
            self.best_reward = mean_reward
            self.wait = 0
        else:
            self.wait += 1

        if self.wait >= self.patience:
            if self.verbose >= 1:
                print(f"\n  [조기종료] ★ 학습 조기 종료! "
                      f"{self.patience}회 연속 개선 없음 "
                      f"(최고={self.best_reward:.1f}, "
                      f"스텝={self.num_timesteps})")
            return False
        return True


class TrainingDiagnosticsCallback(BaseCallback):
    """Print rolling episode diagnostics from env-provided summaries."""

    def __init__(
        self,
        print_every_episodes: int = 64,
        min_timestep_gap: int = 40000,
        verbose: int = 1,
    ):
        super().__init__(verbose)
        self.print_every_episodes = print_every_episodes
        self.min_timestep_gap = min_timestep_gap
        self._summaries: list[dict] = []
        self._last_print_timestep: int = -1
        self._last_signature: tuple[float, ...] | None = None
        self._suppressed_count: int = 0

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        dones = self.locals.get("dones", [])
        for done, info in zip(dones, infos):
            if not done:
                continue
            summary = info.get("episode_summary")
            if summary:
                self._summaries.append(summary)

        if self.verbose < 1 or not self._summaries:
            return True
        if len(self._summaries) % self.print_every_episodes != 0:
            return True
        if (self._last_print_timestep >= 0
                and self.num_timesteps - self._last_print_timestep < self.min_timestep_gap):
            return True

        window = self._summaries[-self.print_every_episodes:]
        served = np.mean([s.get("customers_served", 0.0) for s in window])
        lost = np.mean([s.get("customers_lost", 0.0) for s in window])
        profit = np.mean([s.get("net_profit", 0.0) for s in window])
        rating = np.mean([s.get("shop_rating", 0.0) for s in window])
        score = np.mean([s.get("final_score", 0.0) for s in window])

        event_keys = [
            "take_order", "submit_kitchen", "pickup_food", "serve_food",
            "pickup_drink", "serve_drink", "lost_customer",
        ]
        reward_keys = [
            "dense_shaping", "time_penalty", "idle_penalty",
            "lost_customer", "take_order", "serve_food",
            "rating_delta", "net_profit_delta",
        ]

        def avg_nested(key: str, nested_key: str) -> float:
            vals = []
            for summary in window:
                nested = summary.get(key, {})
                vals.append(float(nested.get(nested_key, 0.0)))
            return float(np.mean(vals))

        signature = (
            round(served, 2),
            round(lost, 2),
            round(profit, 2),
            round(rating, 3),
            round(score, 2),
            round(avg_nested("event_totals", "take_order"), 2),
            round(avg_nested("event_totals", "serve_food"), 2),
            round(avg_nested("event_totals", "lost_customer"), 2),
            round(avg_nested("reward_totals", "dense_shaping"), 2),
            round(avg_nested("reward_totals", "idle_penalty"), 2),
        )
        if signature == self._last_signature:
            self._suppressed_count += 1
            return True

        if self._suppressed_count > 0:
            print(f"\n  [학습 진단] 동일한 요약 {self._suppressed_count}회 생략")
            self._suppressed_count = 0
        print(f"\n  ── 학습 진단 (최근 {self.print_every_episodes} 에피소드) ─────────")
        print(f"  평균 서빙/이탈: served={served:.1f}, lost={lost:.1f}")
        print(f"  평균 순이익/평점/점수: profit={profit:.1f}, rating={rating:.3f}, score={score:.1f}")
        print("  주요 이벤트:")
        for key in event_keys:
            print(f"    {key:<16} {avg_nested('event_totals', key):>8.2f}")
        print("  주요 보상 성분:")
        for key in reward_keys:
            print(f"    {key:<16} {avg_nested('reward_totals', key):>8.2f}")
        print("  ─────────────────────────────────────────")
        self._last_print_timestep = self.num_timesteps
        self._last_signature = signature
        return True


# ────────────────────────────────────────────────
# Early Stopping (SB3 Callback) – 기존 영문 버전 (호환용)
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
                print(f"  [조기종료] 신기록 (New best): {mean_reward:.1f}")
        else:
            self.wait += 1
            if self.verbose >= 2:
                print(f"  [조기종료] 미개선 (No improve): {self.wait}/{self.patience} "
                      f"(최고={self.best_reward:.1f}, 현재={mean_reward:.1f})")

        if self.wait >= self.patience:
            if self.verbose >= 1:
                print(f"\n  [조기종료] ★ 학습 조기 종료! "
                      f"{self.patience}회 연속 개선 없음 "
                      f"(최고={self.best_reward:.1f}, "
                      f"스텝={self.num_timesteps})")
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
            print(f"  [조기종료] 미개선 (No improve): {self.wait}/{self.patience} "
                  f"(최고={self.best_reward:.1f}, 현재={eval_reward:.1f})")

        if self.wait >= self.patience:
            if self.verbose >= 1:
                print(f"\n  [조기종료] ★ 학습 조기 종료! "
                      f"{self.patience}회 연속 개선 없음 "
                      f"(최고={self.best_reward:.1f})")
            return False
        return True
