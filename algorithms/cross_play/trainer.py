"""
CrossPlayTrainer – 교차 알고리즘 자기 대결 학습.

기존 학습된 모델(어떤 알고리즘이든)을 선택하여 다른 모델들과 대결하며 추가 학습합니다.
SB3 모델(PPO, DQN)은 직접 로드하고, 커스텀 모델(SAC, DiscreteSAC 등)은
환경 오버라이드를 통해 SelfPlayEnv에서 원본 트레이너로 학습합니다.

Usage:
    # 대화형 모델 선택
    python -m algorithms.train_launcher --algo CrossPlay --timesteps 200000

    # 특정 모델 지정
    python -m algorithms.train_launcher --algo CrossPlay --model models/discretesac/best_model --timesteps 200000
"""

import os
import json
from typing import Any

import numpy as np
from stable_baselines3 import PPO, DQN
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback

from algorithms.base import BaseTrainer
from algorithms.common import (
    load_algo_config, save_run_config, build_policy_kwargs,
    get_sb3_device, linear_schedule, KoreanEvalStopCallback,
    set_env_override, clear_env_override,
)
from algorithms.marl.self_play_env import SelfPlayEnv
from ai.agent import load_agent, _detect_algo_from_path

# SB3 알고리즘 클래스 매핑
_SB3_ALGO_MAP: dict[str, type] = {
    "PPO": PPO,
    "DQN": DQN,
}

# SelfPlayEnv로 학습된 알고리즘 (compat_mode 불필요)
_SELFPLAY_ALGOS = {"MARL", "CrossPlay"}


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
        self._custom_trainer = None      # 커스텀 알고리즘 학습 시 사용
        self._is_custom: bool = False
        self.cfg: dict = {}
        self.save_path: str = "models/cross_play"

    # ──────────────────────────────────────────────────────
    # build
    # ──────────────────────────────────────────────────────
    def build(self, cfg: dict | None = None, config_path: str | None = None,
              save_path: str | None = None, **overrides) -> None:
        base_model = overrides.pop("base_model", None)
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

        # ── Load opponent pool ──
        self._opponent_pool = self._build_opponent_pool()
        print(f"  [CrossPlay] 상대 풀 크기: {len(self._opponent_pool)}")
        for i, opp in enumerate(self._opponent_pool):
            print(f"    {i+1}. {opp['algo']} → {opp['path']}")

        if not self._opponent_pool:
            print("  [CrossPlay] 학습된 모델 없음! 랜덤 상대로 시작합니다.")

        # ── 학습할 모델 선택 ──
        selected = self._select_learner(base_model)

        # ── Common env setup ──
        shop_kwargs = {}
        if game_ov.get("target_money"):
            shop_kwargs["target_money"] = game_ov["target_money"]
        if game_ov.get("day_limit"):
            shop_kwargs["day_limit"] = game_ov["day_limit"]

        self._shop_kwargs = shop_kwargs
        self._reward_cfg = reward_cfg
        self._seed = seed
        self._n_envs = n_envs

        opponent_agents = [opp["agent"] for opp in self._opponent_pool]

        if selected is None:
            # ── 새 모델 생성 (SB3) ──
            algo_name = self._ask_learner_algo()
            compat = False
            self._build_sb3_fresh(
                algo_name, hp, seed, n_envs, eval_freq, n_eval,
                shop_kwargs, reward_cfg, opponent_agents, compat,
            )
        elif selected["type"] == "sb3":
            # ── SB3 모델 로드 후 추가 학습 ──
            compat = selected["algo"] not in _SELFPLAY_ALGOS
            self._build_sb3_resume(
                selected, seed, n_envs, eval_freq, n_eval,
                shop_kwargs, reward_cfg, opponent_agents, compat,
            )
        else:
            # ── 커스텀 모델 → 환경 오버라이드 후 원본 트레이너로 학습 ──
            self._build_custom_resume(
                selected, shop_kwargs, reward_cfg, opponent_agents, overrides,
            )

        save_run_config(self.save_path, {
            **self.cfg,
            "_crossplay_learner": {
                "algo": selected["algo"] if selected else "PPO",
                "path": selected["path"] if selected else None,
                "type": selected["type"] if selected else "sb3",
            },
        })

    # ──────────────────────────────────────────────────────
    # 모델 선택 로직
    # ──────────────────────────────────────────────────────
    def _select_learner(self, base_model: str | None):
        """학습할 모델을 선택합니다.

        base_model 이 주어지면 해당 모델을 직접 사용하고,
        없으면 대화형으로 선택합니다.
        """
        if base_model:
            return self._resolve_model_path(base_model)
        return self._interactive_select()

    def _resolve_model_path(self, model_path: str) -> dict | None:
        """경로에서 알고리즘 유형을 탐지하여 선택 정보를 반환합니다."""
        algo = _detect_algo_from_path(model_path)
        if not algo:
            cfg_path = os.path.join(os.path.dirname(model_path),
                                    "train_config_used.json")
            if os.path.isfile(cfg_path):
                with open(cfg_path, encoding="utf-8") as f:
                    algo = json.load(f).get("algorithm", "PPO")
            else:
                algo = "PPO"
        mtype = "sb3" if algo in _SB3_ALGO_MAP else "custom"
        return {"algo": algo, "path": model_path, "type": mtype}

    def _interactive_select(self) -> dict | None:
        """대화형으로 학습할 모델을 선택합니다."""
        all_models = _find_trained_models()

        print(f"\n  {'─'*58}")
        print(f"  [CrossPlay] 학습할 모델을 선택하세요")
        print(f"  {'─'*58}")
        print(f"   0. 새로 시작 (새 모델 생성)")
        print(f"  {'─'*58}")
        for i, m in enumerate(all_models, 1):
            algo = m["algo"]
            mtype = "SB3" if algo in _SB3_ALGO_MAP else "커스텀"
            print(f"   {i}. [{algo:<12s}] {m['path']:<36s} ({mtype})")
        print(f"  {'─'*58}")

        if not all_models:
            print("  학습된 모델이 없습니다. 새 모델로 시작합니다.")
            return None

        while True:
            try:
                raw = input("  번호를 입력하세요: ").strip()
                choice = int(raw)
                if choice == 0:
                    return None
                if 1 <= choice <= len(all_models):
                    break
                print(f"  0 ~ {len(all_models)} 범위에서 입력하세요.")
            except (ValueError, EOFError):
                print("  숫자를 입력하세요.")

        sel = all_models[choice - 1]
        algo = sel["algo"]
        mtype = "sb3" if algo in _SB3_ALGO_MAP else "custom"
        print(f"  → 선택: [{algo}] {sel['path']}  ({mtype})")
        return {"algo": algo, "path": sel["path"], "type": mtype}

    @staticmethod
    def _ask_learner_algo() -> str:
        """새 모델 생성 시 SB3 알고리즘을 선택합니다."""
        options = list(_SB3_ALGO_MAP.keys())
        print(f"\n  학습 알고리즘 선택: {', '.join(options)}")
        while True:
            raw = input(f"  알고리즘 [{options[0]}]: ").strip().upper()
            if not raw:
                return options[0]
            if raw in _SB3_ALGO_MAP:
                return raw
            print(f"  지원 알고리즘: {', '.join(options)}")

    # ──────────────────────────────────────────────────────
    # SB3 빌드 경로
    # ──────────────────────────────────────────────────────
    def _make_crossplay_envs(self, n_envs, seed, shop_kwargs, reward_cfg,
                             opponent_agents, compat):
        """SelfPlayEnv 벡터 환경을 생성합니다."""
        first_opp = self._pick_opponent(0)

        def _make(rank):
            def _init():
                env = SelfPlayEnv(
                    reward_config=reward_cfg,
                    opponent_agent=first_opp,
                    opponent_pool=opponent_agents,
                    compat_mode=compat,
                    **shop_kwargs,
                )
                env.reset(seed=seed + rank)
                return env
            return _init

        vec = DummyVecEnv([_make(i) for i in range(n_envs)])
        eval_vec = DummyVecEnv([_make(1000)])
        return vec, eval_vec

    def _build_sb3_fresh(self, algo_name, hp, seed, n_envs, eval_freq,
                         n_eval, shop_kwargs, reward_cfg, opp_agents, compat):
        """새 SB3 모델을 생성합니다."""
        self._is_custom = False
        self._vec_env, self._eval_env = self._make_crossplay_envs(
            n_envs, seed, shop_kwargs, reward_cfg, opp_agents, compat,
        )
        SB3Class = _SB3_ALGO_MAP[algo_name]
        device = get_sb3_device()
        net_cfg = {"net_arch": hp.get("net_arch", [256, 256])}
        policy_kwargs = build_policy_kwargs(net_cfg)

        if algo_name == "PPO":
            self.model = PPO(
                "MlpPolicy", self._vec_env,
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
                seed=seed, device=device, verbose=0,
                tensorboard_log=os.path.join(self.save_path, "tb_logs"),
            )
        else:  # DQN
            self.model = DQN(
                "MlpPolicy", self._vec_env,
                learning_rate=hp.get("learning_rate", 3e-4),
                buffer_size=hp.get("buffer_size", 500_000),
                learning_starts=hp.get("learning_starts", 10_000),
                batch_size=hp.get("batch_size", 64),
                gamma=hp.get("gamma", 0.99),
                tau=hp.get("tau", 0.005),
                exploration_fraction=hp.get("exploration_fraction", 0.1),
                train_freq=hp.get("train_freq", 4),
                policy_kwargs=policy_kwargs,
                seed=seed, device=device, verbose=0,
                tensorboard_log=os.path.join(self.save_path, "tb_logs"),
            )

        self._setup_sb3_callbacks(eval_freq, n_envs, n_eval)
        self._learner_algo = algo_name
        print(f"  [CrossPlay] 새 {algo_name} 모델 생성 완료")

    def _build_sb3_resume(self, selected, seed, n_envs, eval_freq,
                          n_eval, shop_kwargs, reward_cfg, opp_agents, compat):
        """기존 SB3 모델을 로드한 후 CrossPlay 환경으로 추가 학습합니다."""
        self._is_custom = False
        self._vec_env, self._eval_env = self._make_crossplay_envs(
            n_envs, seed, shop_kwargs, reward_cfg, opp_agents, compat,
        )
        algo_name = selected["algo"]
        SB3Class = _SB3_ALGO_MAP[algo_name]
        model_path = selected["path"]

        device = get_sb3_device()
        self.model = SB3Class.load(
            model_path, env=self._vec_env, device=device,
            tensorboard_log=os.path.join(self.save_path, "tb_logs"),
        )
        self._setup_sb3_callbacks(eval_freq, n_envs, n_eval)
        self._learner_algo = algo_name
        print(f"  [CrossPlay] {algo_name} 모델 로드 완료 ← {model_path}")

    def _setup_sb3_callbacks(self, eval_freq, n_envs, n_eval):
        """SB3 학습용 콜백을 설정합니다."""
        kr_stop = KoreanEvalStopCallback(patience=50)
        eval_cb = EvalCallback(
            self._eval_env,
            best_model_save_path=self.save_path,
            log_path=os.path.join(self.save_path, "eval_logs"),
            eval_freq=max(eval_freq // n_envs, 1),
            n_eval_episodes=n_eval,
            verbose=0,
            callback_after_eval=kr_stop,
        )
        self._callbacks = [eval_cb]

    # ──────────────────────────────────────────────────────
    # 커스텀 빌드 경로
    # ──────────────────────────────────────────────────────
    def _build_custom_resume(self, selected, shop_kwargs, reward_cfg,
                             opp_agents, overrides):
        """커스텀 PyTorch 모델을 환경 오버라이드로 CrossPlay 학습합니다."""
        from algorithms.registry import get_algorithm

        algo_name = selected["algo"]
        model_path = selected["path"]
        self._is_custom = True
        self._learner_algo = algo_name

        # SelfPlayEnv 팩토리를 make_env 오버라이드로 등록
        def _crossplay_env_factory(rank, seed, game_ov, reward_cfg_inner):
            def _init():
                kw = {}
                if game_ov:
                    if game_ov.get("target_money") is not None:
                        kw["target_money"] = game_ov["target_money"]
                    if game_ov.get("day_limit") is not None:
                        kw["day_limit"] = game_ov["day_limit"]
                env = SelfPlayEnv(
                    reward_config=reward_cfg_inner,
                    opponent_pool=opp_agents,
                    compat_mode=True,
                    **kw,
                )
                env.reset(seed=seed + rank)
                return env
            return _init

        set_env_override(_crossplay_env_factory)
        try:
            TrainerClass = get_algorithm(algo_name)
            trainer = TrainerClass()
            trainer.build(save_path=self.save_path, **overrides)
            trainer.load(model_path)
        except Exception:
            clear_env_override()
            raise

        self._custom_trainer = trainer
        print(f"  [CrossPlay] 커스텀 {algo_name} 모델 로드 완료 ← {model_path}")
        print(f"  [CrossPlay] 환경 오버라이드: SelfPlayEnv (compat_mode=True)")

    # ──────────────────────────────────────────────────────
    # 상대 풀 관리
    # ──────────────────────────────────────────────────────

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
        if self._is_custom:
            return self._train_custom()
        return self._train_sb3()

    def _train_sb3(self) -> dict[str, Any]:
        assert self.model is not None, "call build() first"

        steps_done = 0
        swap_interval = self._swap_freq

        print(f"\n  [CrossPlay] 학습 시작: {self._timesteps} 스텝 ({self._learner_algo})")
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
        return {"algorithm": "CrossPlay", "learner": self._learner_algo,
                "timesteps": self._timesteps,
                "opponents": len(self._opponent_pool),
                "save_path": self.save_path}

    def _train_custom(self) -> dict[str, Any]:
        assert self._custom_trainer is not None, "call build() first"

        print(f"\n  [CrossPlay] 커스텀 학습 시작: {self._learner_algo}")
        print(f"  [CrossPlay] 상대 풀이 환경에 자동 주입됩니다 (매 에피소드 랜덤 교체)")
        try:
            result = self._custom_trainer.train()
        finally:
            clear_env_override()

        result["crossplay"] = True
        result["opponents"] = len(self._opponent_pool)
        print(f"[✓] CrossPlay 커스텀 학습 완료. 모델 → '{self.save_path}/'")
        return result

    def save(self, path: str) -> None:
        if self._is_custom and self._custom_trainer:
            self._custom_trainer.save(path)
        elif self.model:
            self.model.save(path)

    def load(self, path: str) -> None:
        self.model = PPO.load(path)

    def predict(self, obs, deterministic: bool = True) -> int:
        if self._is_custom and self._custom_trainer:
            return self._custom_trainer.predict(obs, deterministic=deterministic)
        assert self.model is not None
        action, _ = self.model.predict(obs, deterministic=deterministic)
        return int(action)
