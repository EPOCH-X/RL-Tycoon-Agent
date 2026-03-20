"""QR-DQN Trainer – SB3-Contrib QRDQN 기반 학습기.

QR-DQN은 DQN의 분포적(distributional) 변형으로,
각 행동의 기대값 대신 return 분포의 quantile들을 학습합니다.
이 프로젝트처럼 행동 공간이 작고 이산적이며,
보상 shaping이 다소 거친 환경에서 PPO 다음 비교군으로 쓰기 좋습니다.
"""

import copy
import os
from typing import Any

from sb3_contrib import QRDQN
from stable_baselines3.common.callbacks import EvalCallback

from algorithms.base import BaseTrainer
from algorithms.common import (
    load_algo_config, make_vec_env, build_policy_kwargs,
    save_run_config, get_sb3_device, KoreanEvalStopCallback,
    TrainingDiagnosticsCallback,
    print_metric_reference,
)


class QRDQNTrainer(BaseTrainer):
    name = "QRDQN"

    def __init__(self):
        self.model: QRDQN | None = None
        self.train_env = None
        self.eval_env = None
        self.cfg: dict = {}
        self.save_path: str = "models/qrdqn"
        self._timesteps: int = 200_000

    def build(self, cfg: dict | None = None, config_path: str | None = None,
              save_path: str | None = None, **overrides) -> None:
        days = overrides.pop("days", None)
        self.cfg = cfg or load_algo_config("qrdqn", config_path, days=days)
        t = self.cfg.get("training", {})
        hp = self.cfg.get("hyperparameters", {})
        net = self.cfg.get("network", {})
        game_ov = self.cfg.get("game_overrides", {})
        reward_cfg = self.cfg.get("reward_shaping", {})

        self._timesteps = overrides.get("timesteps", t.get("total_timesteps", 500_000))
        seed = overrides.get("seed", t.get("seed", 42))
        eval_freq = t.get("eval_freq", 5000)
        self.save_path = save_path or self.save_path

        os.makedirs(self.save_path, exist_ok=True)
        effective_cfg = copy.deepcopy(self.cfg)
        effective_training = effective_cfg.setdefault("training", {})
        effective_training["total_timesteps"] = self._timesteps
        effective_training["n_envs"] = 1
        effective_training["seed"] = seed
        save_run_config(self.save_path, effective_cfg)

        self.train_env = make_vec_env(1, seed, game_ov, reward_cfg, force_dummy=True)
        self.eval_env = make_vec_env(1, seed + 1000, game_ov, reward_cfg,
                                     force_dummy=True)

        policy_kwargs = build_policy_kwargs(net)
        policy_type = self.cfg.get("policy", "MlpPolicy")
        device = get_sb3_device(policy_type)

        self.model = QRDQN(
            policy_type,
            self.train_env,
            verbose=1,
            learning_rate=hp.get("learning_rate", 1e-4),
            buffer_size=hp.get("buffer_size", 200_000),
            learning_starts=hp.get("learning_starts", 5_000),
            batch_size=hp.get("batch_size", 128),
            gamma=hp.get("gamma", 0.99),
            tau=hp.get("tau", 1.0),
            target_update_interval=hp.get("target_update_interval", 1_000),
            train_freq=hp.get("train_freq", 4),
            gradient_steps=hp.get("gradient_steps", 1),
            exploration_fraction=hp.get("exploration_fraction", 0.3),
            exploration_initial_eps=hp.get("exploration_initial_eps", 1.0),
            exploration_final_eps=hp.get("exploration_final_eps", 0.02),
            max_grad_norm=hp.get("max_grad_norm", 10.0),
            policy_kwargs=policy_kwargs if policy_kwargs else None,
            tensorboard_log=os.path.join(self.save_path, "tb_logs"),
            seed=seed,
            device=device,
        )

        patience = t.get("patience", 100)
        self._eval_cb = EvalCallback(
            self.eval_env,
            best_model_save_path=self.save_path,
            log_path=os.path.join(self.save_path, "eval_logs"),
            eval_freq=eval_freq,
            deterministic=True,
            verbose=0,
            callback_after_eval=KoreanEvalStopCallback(
                patience=patience, min_delta=1.0, verbose=1),
        )

    def train(self, resume_path: str | None = None) -> dict[str, Any]:
        assert self.model is not None, "call build() first"

        if resume_path:
            device = get_sb3_device(self.cfg.get("policy", "MlpPolicy"))
            self.model = QRDQN.load(
                resume_path, env=self.train_env, device=device,
                tensorboard_log=os.path.join(self.save_path, "tb_logs"),
            )
            print(f"  [QRDQN] 체크포인트 복원: {resume_path}")

        print_metric_reference()
        diag_cb = TrainingDiagnosticsCallback(
            print_every_episodes=64,
            min_timestep_gap=max(20000, self.cfg.get("training", {}).get("eval_freq", 5000) * 4),
            verbose=1,
        )
        self.model.learn(total_timesteps=self._timesteps, callback=[self._eval_cb, diag_cb])
        self.save(os.path.join(self.save_path, "final_model"))
        self._cleanup()
        return {
            "algorithm": "QRDQN",
            "timesteps": self._timesteps,
            "save_path": self.save_path,
        }

    def save(self, path: str) -> None:
        if self.model:
            self.model.save(path)

    def load(self, path: str) -> None:
        self.model = QRDQN.load(path, device="cpu")

    def predict(self, obs, deterministic: bool = True) -> int:
        assert self.model is not None
        action, _ = self.model.predict(obs, deterministic=deterministic)
        return int(action)

    def _cleanup(self):
        if self.train_env:
            self.train_env.close()
        if self.eval_env:
            self.eval_env.close()
        print(f"[✓] QRDQN (분위수 회귀 DQN) 학습 완료. 모델 → '{self.save_path}/'")
