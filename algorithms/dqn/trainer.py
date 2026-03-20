"""DQN Trainer – Stable-Baselines3 DQN 기반 학습기.

Deep Q-Network은 이산(discrete) 행동 공간에 적합하며,
Experience Replay와 Target Network를 사용합니다.
레스토랑 게임의 7개 이산 행동에 바로 적용 가능합니다.
"""

import os
from typing import Any

from stable_baselines3 import DQN
from algorithms.base import BaseTrainer
from algorithms.common import (
    load_algo_config, make_vec_env, build_policy_kwargs,
    save_run_config, get_sb3_device, FinalScoreEvalCallback,
    print_metric_reference,
)


class DQNTrainer(BaseTrainer):
    name = "DQN"

    def __init__(self):
        self.model: DQN | None = None
        self.train_env = None
        self.eval_env = None
        self.cfg: dict = {}
        self.save_path: str = "models/dqn"
        self._timesteps: int = 200_000

    def build(self, cfg: dict | None = None, config_path: str | None = None,
              save_path: str | None = None, **overrides) -> None:
        days = overrides.pop("days", None)
        self.cfg = cfg or load_algo_config("dqn", config_path, days=days)
        t = self.cfg.get("training", {})
        hp = self.cfg.get("hyperparameters", {})
        net = self.cfg.get("network", {})
        game_ov = self.cfg.get("game_overrides", {})
        reward_cfg = self.cfg.get("reward_shaping", {})

        self._timesteps = overrides.get("timesteps", t.get("total_timesteps", 200_000))
        # DQN in SB3 does not support vectorised envs > 1
        n_envs = 1
        seed = overrides.get("seed", t.get("seed", 42))
        eval_freq = t.get("eval_freq", 5000)
        self.save_path = save_path or self.save_path

        os.makedirs(self.save_path, exist_ok=True)
        save_run_config(self.save_path, self.cfg)

        self.train_env = make_vec_env(n_envs, seed, game_ov, reward_cfg,
                                      force_dummy=True)
        self.eval_env = make_vec_env(1, seed + 1000, game_ov, reward_cfg,
                                     force_dummy=True)

        policy_kwargs = build_policy_kwargs(net)
        policy_type = self.cfg.get("policy", "MlpPolicy")
        device = get_sb3_device(policy_type)

        self.model = DQN(
            policy_type,
            self.train_env,
            verbose=1,
            learning_rate=hp.get("learning_rate", 1e-4),
            buffer_size=hp.get("buffer_size", 100_000),
            learning_starts=hp.get("learning_starts", 1000),
            batch_size=hp.get("batch_size", 64),
            gamma=hp.get("gamma", 0.99),
            tau=hp.get("tau", 1.0),
            target_update_interval=hp.get("target_update_interval", 1000),
            train_freq=hp.get("train_freq", 4),
            gradient_steps=hp.get("gradient_steps", 1),
            exploration_fraction=hp.get("exploration_fraction", 0.2),
            exploration_initial_eps=hp.get("exploration_initial_eps", 1.0),
            exploration_final_eps=hp.get("exploration_final_eps", 0.05),
            max_grad_norm=hp.get("max_grad_norm", 10.0),
            policy_kwargs=policy_kwargs if policy_kwargs else None,
            tensorboard_log=os.path.join(self.save_path, "tb_logs"),
            seed=seed,
            device=device,
        )

        patience = t.get("patience", 50)
        self._eval_cb = FinalScoreEvalCallback(
            self.eval_env,
            best_model_save_path=self.save_path,
            log_path=os.path.join(self.save_path, "eval_logs"),
            eval_freq=eval_freq,
            deterministic=False,
            n_eval_episodes=t.get("n_eval_episodes", 5),
            patience=patience,
            min_delta=1.0,
            verbose=1,
        )

    def train(self, resume_path: str | None = None) -> dict[str, Any]:
        assert self.model is not None, "call build() first"
        print_metric_reference()
        self.model.learn(total_timesteps=self._timesteps, callback=self._eval_cb)
        self.save(os.path.join(self.save_path, "final_model"))
        self._cleanup()
        return {"algorithm": "DQN", "timesteps": self._timesteps,
                "save_path": self.save_path}

    def save(self, path: str) -> None:
        if self.model:
            self.model.save(path)

    def load(self, path: str) -> None:
        self.model = DQN.load(path, device="cpu")

    def predict(self, obs, deterministic: bool = True) -> int:
        assert self.model is not None
        action, _ = self.model.predict(obs, deterministic=deterministic)
        return int(action)

    def _cleanup(self):
        if self.train_env:
            self.train_env.close()
        if self.eval_env:
            self.eval_env.close()
        print(f"[✓] DQN (심층 Q-네트워크) 학습 완료. 모델 → '{self.save_path}/'")
