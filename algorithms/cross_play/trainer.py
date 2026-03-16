"""CrossPlayTrainer – 교차 알고리즘 자기 대결 학습.

서로 다른 알고리즘으로 학습된 에이전트들을 상대로 PPO 기반 에이전트를 학습합니다.
상대 풀(opponent pool)에서 라운드 로빈으로 상대를 교체하며 경쟁합니다.

Usage:
    python -m algorithms.train_launcher --algo CrossPlay --timesteps 200000
"""

import os
import json
from typing import Any

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback

from algorithms.base import BaseTrainer
from algorithms.common import (
    load_algo_config, save_run_config, build_policy_kwargs,
    get_sb3_device, linear_schedule, KoreanEvalStopCallback,
)
from algorithms.marl.self_play_env import SelfPlayEnv
from ai.agent import load_agent


def _find_trained_models() -> list[dict]:
    """models/ 디렉토리에서 학습된 모델들을 자동 탐색합니다.

    Returns:
        [{"algo": "PPO", "path": "models/ppo/best_model.zip"}, ...]
    """
    models_dir = "models"
    if not os.path.isdir(models_dir):
        return []

    found = []
    for root, _dirs, files in os.walk(models_dir):
        for f in files:
            full = os.path.join(root, f)
            entry = None
            if f == "best_model.zip":
                entry = {"path": full}
            elif f == "best_model.pt":
                entry = {"path": full[:-3]}  # remove .pt extension
            elif f == "final_model.pt":
                entry = {"path": full[:-3]}
            if entry is None:
                continue

            # Detect algorithm from path or config
            cfg_path = os.path.join(root, "train_config_used.json")
            if os.path.isfile(cfg_path):
                with open(cfg_path, encoding="utf-8") as fp:
                    cfg = json.load(fp)
                entry["algo"] = cfg.get("algorithm", "").strip()
            if not entry.get("algo"):
                # Infer from directory name
                from ai.agent import _detect_algo_from_path
                entry["algo"] = _detect_algo_from_path(full) or "PPO"
            found.append(entry)

    return found


class CrossPlayTrainer(BaseTrainer):
    name = "CrossPlay"

    def __init__(self):
        self.model = None
        self.cfg: dict = {}
        self.save_path: str = "models/cross_play"

    def build(self, cfg: dict | None = None, config_path: str | None = None,
              save_path: str | None = None, **overrides) -> None:
        days = overrides.pop("days", None)
        self.cfg = cfg or load_algo_config("cross_play", config_path, days=days)
        t = self.cfg.get("training", {})
        hp = self.cfg.get("hyperparameters", {})
        game_ov = self.cfg.get("game_overrides", {})
        reward_cfg = self.cfg.get("reward_shaping", {})

        self._timesteps = overrides.get("timesteps",
                                        t.get("total_timesteps", 200_000))
        seed = overrides.get("seed", t.get("seed", 42))
        n_envs = overrides.get("n_envs", hp.get("n_envs", 4))
        eval_freq = t.get("eval_freq", 5000)
        self._swap_freq = t.get("opponent_swap_freq", 10_000)
        n_eval = t.get("n_eval_episodes", 5)

        self.save_path = save_path or self.save_path
        os.makedirs(self.save_path, exist_ok=True)
        save_run_config(self.save_path, self.cfg)

        # ── Load opponent pool ──
        self._opponent_pool = self._build_opponent_pool()
        print(f"  [CrossPlay] 상대 풀 크기: {len(self._opponent_pool)}")
        for i, opp in enumerate(self._opponent_pool):
            print(f"    {i+1}. {opp['algo']} → {opp['path']}")

        if not self._opponent_pool:
            print("  [CrossPlay] 학습된 모델 없음! 랜덤 상대로 시작합니다.")

        # ── 첫 번째 상대 선택 ──
        current_opp = self._pick_opponent(0)

        # ── SelfPlayEnv를 벡터화 ──
        shop_kwargs = {}
        if game_ov.get("target_money"):
            shop_kwargs["target_money"] = game_ov["target_money"]
        if game_ov.get("day_limit"):
            shop_kwargs["day_limit"] = game_ov["day_limit"]

        def make_crossplay_env(rank):
            def _init():
                env = SelfPlayEnv(
                    reward_config=reward_cfg,
                    opponent_agent=current_opp,
                    **shop_kwargs,
                )
                env.reset(seed=seed + rank)
                return env
            return _init

        self._shop_kwargs = shop_kwargs
        self._reward_cfg = reward_cfg
        self._seed = seed
        self._n_envs = n_envs

        env_fns = [make_crossplay_env(i) for i in range(n_envs)]
        self._vec_env = DummyVecEnv(env_fns)

        # ── Eval env ──
        eval_env = DummyVecEnv([make_crossplay_env(1000)])

        # ── PPO model ──
        net_cfg = {"net_arch": hp.get("net_arch", [256, 256])}
        policy_kwargs = build_policy_kwargs(net_cfg)
        device = get_sb3_device()

        self.model = PPO(
            "MlpPolicy",
            self._vec_env,
            learning_rate=linear_schedule(hp.get("learning_rate", 3e-4)),
            n_steps=hp.get("n_steps", 2048),
            batch_size=hp.get("batch_size", 64),
            n_epochs=hp.get("n_epochs", 10),
            gamma=hp.get("gamma", 0.99),
            clip_range=hp.get("clip_range", 0.2),
            ent_coef=hp.get("ent_coef", 0.01),
            vf_coef=hp.get("vf_coef", 0.5),
            max_grad_norm=hp.get("max_grad_norm", 0.5),
            policy_kwargs=policy_kwargs,
            seed=seed,
            device=device,
            verbose=0,
            tensorboard_log=os.path.join(self.save_path, "tb_logs"),
        )

        # ── Callbacks ──
        kr_stop = KoreanEvalStopCallback(patience=50)
        eval_cb = EvalCallback(
            eval_env,
            best_model_save_path=self.save_path,
            log_path=os.path.join(self.save_path, "eval_logs"),
            eval_freq=max(eval_freq // n_envs, 1),
            n_eval_episodes=n_eval,
            verbose=0,
            callback_after_eval=kr_stop,
        )
        self._callbacks = [eval_cb]
        self._eval_env = eval_env

    def _build_opponent_pool(self) -> list[dict]:
        """학습된 모델들을 로드하여 상대 풀을 구축합니다."""
        entries = _find_trained_models()
        pool = []
        for entry in entries:
            try:
                agent = load_agent(entry["path"], algo_name=entry["algo"])
                pool.append({
                    "algo": entry["algo"],
                    "path": entry["path"],
                    "agent": agent,
                })
            except Exception as e:
                print(f"  [CrossPlay] 상대 로드 실패: {entry} → {e}")
        return pool

    def _pick_opponent(self, step: int):
        """라운드 로빈으로 상대를 선택합니다."""
        if not self._opponent_pool:
            return None
        idx = (step // max(self._swap_freq, 1)) % len(self._opponent_pool)
        opp = self._opponent_pool[idx]
        print(f"  [CrossPlay] 상대 교체 → {opp['algo']} ({opp['path']})")
        return opp["agent"]

    def _swap_opponents(self, step: int):
        """환경의 상대를 교체합니다."""
        new_opp = self._pick_opponent(step)
        for env in self._vec_env.envs:
            env.set_opponent(new_opp)
        for env in self._eval_env.envs:
            env.set_opponent(new_opp)

    def train(self) -> dict[str, Any]:
        assert self.model is not None, "call build() first"

        steps_done = 0
        swap_interval = self._swap_freq

        print(f"\n  [CrossPlay] 학습 시작: {self._timesteps} 스텝")
        print(f"  [CrossPlay] 상대 교체 주기: {swap_interval} 스텝")

        while steps_done < self._timesteps:
            chunk = min(swap_interval, self._timesteps - steps_done)
            self.model.learn(
                total_timesteps=chunk,
                callback=self._callbacks,
                reset_num_timesteps=(steps_done == 0),
                progress_bar=False,
            )
            steps_done += chunk
            if steps_done < self._timesteps:
                self._swap_opponents(steps_done)

        self.model.save(os.path.join(self.save_path, "final_model"))
        self._vec_env.close()
        self._eval_env.close()

        print(f"[✓] CrossPlay 학습 완료. 모델 → '{self.save_path}/'")
        return {"algorithm": "CrossPlay", "timesteps": self._timesteps,
                "opponents": len(self._opponent_pool),
                "save_path": self.save_path}

    def save(self, path: str) -> None:
        if self.model:
            self.model.save(path)

    def load(self, path: str) -> None:
        self.model = PPO.load(path)

    def predict(self, obs, deterministic: bool = True) -> int:
        assert self.model is not None
        action, _ = self.model.predict(obs, deterministic=deterministic)
        return int(action)
