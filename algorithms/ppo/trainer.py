"""PPO Trainer – Stable-Baselines3 PPO 기반 학습기.

기존 ai/train.py 의 PPO 로직을 알고리즘 인터페이스로 래핑합니다.
"""

import os
from typing import Any

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback

from algorithms.base import BaseTrainer
from algorithms.common import (
    load_algo_config, make_vec_env, build_policy_kwargs,
    save_run_config,
)


class PPOTrainer(BaseTrainer):
    name = "PPO"

    def __init__(self):
        self.model: PPO | None = None
        self.train_env = None
        self.eval_env = None
        self.cfg: dict = {}
        self.save_path: str = "models/ppo"
        self._timesteps: int = 200_000

    # ── build ────────────────────────────────────
    def build(self, cfg: dict | None = None, config_path: str | None = None,
              save_path: str | None = None, **overrides) -> None:
        self.cfg = cfg or load_algo_config("ppo", config_path)
        t = self.cfg.get("training", {})
        hp = self.cfg.get("hyperparameters", {})
        net = self.cfg.get("network", {})
        game_ov = self.cfg.get("game_overrides", {})
        reward_cfg = self.cfg.get("reward_shaping", {})

        self._timesteps = overrides.get("timesteps", t.get("total_timesteps", 200_000))
        n_envs = overrides.get("n_envs", t.get("n_envs", 4))
        seed = overrides.get("seed", t.get("seed", 42))
        eval_freq = t.get("eval_freq", 5000)
        self.save_path = save_path or self.save_path

        os.makedirs(self.save_path, exist_ok=True)
        save_run_config(self.save_path, self.cfg)

        # 환경
        self.train_env = make_vec_env(n_envs, seed, game_ov, reward_cfg)
        self.eval_env = make_vec_env(1, seed + 1000, game_ov, reward_cfg,
                                     force_dummy=True)

        policy_kwargs = build_policy_kwargs(net)

        self.model = PPO(
            self.cfg.get("policy", "MlpPolicy"),
            self.train_env,
            verbose=1,
            learning_rate=hp.get("learning_rate", 3e-4),
            n_steps=hp.get("n_steps", 2048),
            batch_size=hp.get("batch_size", 64),
            n_epochs=hp.get("n_epochs", 10),
            gamma=hp.get("gamma", 0.99),
            gae_lambda=hp.get("gae_lambda", 0.95),
            clip_range=hp.get("clip_range", 0.2),
            ent_coef=hp.get("ent_coef", 0.01),
            vf_coef=hp.get("vf_coef", 0.5),
            max_grad_norm=hp.get("max_grad_norm", 0.5),
            policy_kwargs=policy_kwargs if policy_kwargs else None,
            tensorboard_log=os.path.join(self.save_path, "tb_logs"),
            seed=seed,
        )

        self._eval_cb = EvalCallback(
            self.eval_env,
            best_model_save_path=self.save_path,
            log_path=os.path.join(self.save_path, "eval_logs"),
            eval_freq=eval_freq,
            deterministic=True,
        )

    # ── train ────────────────────────────────────
    def train(self) -> dict[str, Any]:
        assert self.model is not None, "call build() first"
        self.model.learn(total_timesteps=self._timesteps, callback=self._eval_cb)
        self.save(os.path.join(self.save_path, "final_model"))
        self._cleanup()
        return {"algorithm": "PPO", "timesteps": self._timesteps,
                "save_path": self.save_path}

    # ── save / load ──────────────────────────────
    def save(self, path: str) -> None:
        if self.model:
            self.model.save(path)

    def load(self, path: str) -> None:
        self.model = PPO.load(path)

    def predict(self, obs, deterministic: bool = True) -> int:
        assert self.model is not None
        action, _ = self.model.predict(obs, deterministic=deterministic)
        return int(action)

    def _cleanup(self):
        if self.train_env:
            self.train_env.close()
        if self.eval_env:
            self.eval_env.close()
        print(f"[✓] PPO training complete. Models → '{self.save_path}/'")
