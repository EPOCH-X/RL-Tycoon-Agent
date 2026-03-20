"""MARL Trainer – Multi-Agent RL with Self-play.

경쟁형 경영 게임을 위한 자기 대결(Self-play) 학습:
1. 에이전트를 자기 자신(과거 버전)과 대결시켜 전략을 고도화
2. Opponent Pool: 과거 N개 체크포인트를 유지, 랜덤으로 상대 선택
3. ELO 레이팅으로 에이전트 성장 추적

기반 알고리즘: PPO (SB3)
"""

import os
import copy
import random
from typing import Any

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback

from algorithms.base import BaseTrainer
from algorithms.common import (
    load_algo_config, build_policy_kwargs, save_run_config,
    get_sb3_device, FinalScoreEvalCallback, print_metric_reference,
)
from algorithms.marl.self_play_env import SelfPlayEnv


class OpponentPool:
    """과거 모델들의 풀을 관리합니다."""

    def __init__(self, max_size: int = 5):
        self.max_size = max_size
        self.models: list[dict] = []  # list of state_dicts
        self.elos: list[float] = []

    def add(self, state_dict: dict, elo: float = 1000.0):
        self.models.append(copy.deepcopy(state_dict))
        self.elos.append(elo)
        if len(self.models) > self.max_size:
            self.models.pop(0)
            self.elos.pop(0)

    def sample(self) -> dict | None:
        if not self.models:
            return None
        idx = random.randint(0, len(self.models) - 1)
        return self.models[idx]

    def __len__(self):
        return len(self.models)


class _OpponentAgent:
    """PPO state_dict를 사용해 행동을 예측하는 간단한 래퍼."""

    def __init__(self, model: PPO):
        self._model = model

    def predict(self, obs):
        action, _ = self._model.predict(obs, deterministic=True)
        return int(action)


class SelfPlayCallback(BaseCallback):
    """주기적으로 현재 모델을 Opponent Pool에 추가하고 상대를 교체합니다."""

    def __init__(self, pool: OpponentPool, envs: list[SelfPlayEnv],
                 update_freq: int = 10000, verbose: int = 0):
        super().__init__(verbose)
        self.pool = pool
        self.envs = envs
        self.update_freq = update_freq

    def _on_step(self) -> bool:
        if self.num_timesteps % self.update_freq == 0 and self.num_timesteps > 0:
            # 현재 모델을 풀에 추가
            state_dict = copy.deepcopy(self.model.policy.state_dict())
            self.pool.add(state_dict)

            # 풀에서 상대를 랜덤 선택하여 환경에 설정
            opp_sd = self.pool.sample()
            if opp_sd is not None:
                opp_model = PPO("MlpPolicy", self.envs[0], verbose=0)
                opp_model.policy.load_state_dict(opp_sd)
                opp_agent = _OpponentAgent(opp_model)
                for env in self.envs:
                    if hasattr(env, "set_opponent"):
                        env.set_opponent(opp_agent)
                if self.verbose:
                    print(f"  [MARL (자기대결)] 상대 풀에서 업데이트 완료 "
                          f"(풀 크기={len(self.pool)})")
        return True


class MARLTrainer(BaseTrainer):
    name = "MARL"

    def __init__(self):
        self.model: PPO | None = None
        self.cfg: dict = {}
        self.save_path: str = "models/marl"
        self._timesteps: int = 200_000
        self._pool: OpponentPool | None = None

    def build(self, cfg: dict | None = None, config_path: str | None = None,
              save_path: str | None = None, **overrides) -> None:
        days = overrides.pop("days", None)
        self.cfg = cfg or load_algo_config("marl", config_path, days=days)
        t = self.cfg.get("training", {})
        hp = self.cfg.get("hyperparameters", {})
        net = self.cfg.get("network", {})
        game_ov = self.cfg.get("game_overrides", {})
        reward_cfg = self.cfg.get("reward_shaping", {})

        self._timesteps = overrides.get("timesteps",
                                        t.get("total_timesteps", 200_000))
        n_envs = overrides.get("n_envs", t.get("n_envs", 2))
        seed = overrides.get("seed", t.get("seed", 42))
        pool_size = t.get("opponent_pool_size", 5)
        self._update_freq = t.get("self_play_update_freq", 10000)
        self.save_path = save_path or self.save_path

        os.makedirs(self.save_path, exist_ok=True)
        save_run_config(self.save_path, self.cfg)

        # Self-play 환경 생성
        self._raw_envs: list[SelfPlayEnv] = []

        def _make_sp_env(rank):
            def _init():
                kwargs = {}
                if game_ov.get("target_money") is not None:
                    kwargs["target_money"] = game_ov["target_money"]
                if game_ov.get("day_limit") is not None:
                    kwargs["day_limit"] = game_ov["day_limit"]
                env = SelfPlayEnv(reward_config=reward_cfg, **kwargs)
                env.reset(seed=seed + rank)
                self._raw_envs.append(env)
                return env
            return _init

        self.train_env = DummyVecEnv(
            [_make_sp_env(i) for i in range(n_envs)]
        )

        self._pool = OpponentPool(max_size=pool_size)
        policy_kwargs = build_policy_kwargs(net)
        policy_type = self.cfg.get("policy", "MlpPolicy")
        device = get_sb3_device(policy_type)

        self.model = PPO(
            policy_type,
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
            device=device,
        )

    def train(self, resume_path: str | None = None) -> dict[str, Any]:
        assert self.model is not None, "call build() first"

        reward_cfg = self.cfg.get("reward_shaping", {})
        game_ov = self.cfg.get("game_overrides", {})

        def _make_eval_env():
            kwargs = {}
            if game_ov.get("target_money") is not None:
                kwargs["target_money"] = game_ov["target_money"]
            if game_ov.get("day_limit") is not None:
                kwargs["day_limit"] = game_ov["day_limit"]
            return SelfPlayEnv(reward_config=reward_cfg, **kwargs)

        eval_env = DummyVecEnv([_make_eval_env])
        eval_cb = FinalScoreEvalCallback(
            eval_env,
            best_model_save_path=self.save_path,
            log_path=os.path.join(self.save_path, "eval_logs"),
            eval_freq=self.cfg.get("training", {}).get("eval_freq", 5000),
            deterministic=False,
            n_eval_episodes=self.cfg.get("training", {}).get("n_eval_episodes", 5),
            patience=self.cfg.get("training", {}).get("patience", 50),
            min_delta=1.0,
            verbose=1,
        )

        sp_cb = SelfPlayCallback(
            self._pool, self._raw_envs,
            update_freq=self._update_freq, verbose=1,
        )

        print_metric_reference()
        self.model.learn(total_timesteps=self._timesteps, callback=[eval_cb, sp_cb])
        self.save(os.path.join(self.save_path, "final_model"))
        self.train_env.close()
        print(f"[✓] MARL (멀티에이전트 자기대결) 학습 완료. "
              f"상대 풀 크기={len(self._pool)}. 모델 → '{self.save_path}/'")
        return {"algorithm": "MARL", "timesteps": self._timesteps,
                "pool_size": len(self._pool), "save_path": self.save_path}

    def save(self, path: str) -> None:
        if self.model:
            self.model.save(path)
        # 풀도 저장
        if self._pool:
            import torch
            torch.save({
                "models": self._pool.models,
                "elos": self._pool.elos,
            }, path + "_pool.pt")

    def load(self, path: str) -> None:
        self.model = PPO.load(path, device="cpu")
        pool_path = path + "_pool.pt"
        if os.path.isfile(pool_path):
            import torch
            data = torch.load(pool_path, map_location="cpu")
            if self._pool is None:
                self._pool = OpponentPool()
            self._pool.models = data["models"]
            self._pool.elos = data["elos"]

    def predict(self, obs, deterministic: bool = True) -> int:
        assert self.model is not None
        action, _ = self.model.predict(obs, deterministic=deterministic)
        return int(action)
