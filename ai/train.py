"""Training script – trains a PPO agent on the Tycoon environment using SB3.

설정은 config/train_config.json 에서 관리합니다.
CLI 인자로 오버라이드할 수도 있습니다.

Usage:
    python -m ai.train                              # config 파일 기본값으로 학습
    python -m ai.train --timesteps 500000           # timesteps만 오버라이드
    python -m ai.train --config my_config.json      # 다른 설정 파일 사용
"""

import argparse
import json
import os

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.callbacks import EvalCallback

from config.settings import load_json_config
from ai.gym_env import TycoonEnv


def load_train_config(config_path: str | None = None) -> dict:
    """Load training config JSON. Falls back to config/train_config.json."""
    if config_path and os.path.isfile(config_path):
        with open(config_path, encoding="utf-8") as f:
            return json.load(f)
    return load_json_config("train_config.json")


def _make_env(rank: int = 0, seed: int = 0, game_overrides: dict | None = None,
              reward_config: dict | None = None):
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


def train(args):
    cfg = load_train_config(args.config)
    t_cfg = cfg.get("training", {})
    hp = cfg.get("hyperparameters", {})
    net_cfg = cfg.get("network", {})
    game_ov = cfg.get("game_overrides", {})
    reward_cfg = cfg.get("reward_shaping", {})

    # CLI overrides take priority
    timesteps = args.timesteps or t_cfg.get("total_timesteps", 200_000)
    n_envs = args.n_envs or t_cfg.get("n_envs", 4)
    seed = args.seed if args.seed is not None else t_cfg.get("seed", 42)
    eval_freq = t_cfg.get("eval_freq", 5000)
    save_path = args.save_path

    os.makedirs(save_path, exist_ok=True)

    # Save the config used for this run (reproducibility)
    with open(os.path.join(save_path, "train_config_used.json"), "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)

    # Vectorised training environments
    if n_envs > 1:
        train_env = SubprocVecEnv(
            [_make_env(i, seed, game_ov, reward_cfg) for i in range(n_envs)])
    else:
        train_env = DummyVecEnv([_make_env(0, seed, game_ov, reward_cfg)])

    eval_env = DummyVecEnv([_make_env(0, seed + 1000, game_ov, reward_cfg)])

    # Build policy kwargs from network config
    policy_kwargs = {}
    if net_cfg.get("net_arch"):
        policy_kwargs["net_arch"] = net_cfg["net_arch"]
    if net_cfg.get("activation_fn"):
        import torch.nn as nn
        act_map = {"tanh": nn.Tanh, "relu": nn.ReLU, "elu": nn.ELU}
        act_cls = act_map.get(net_cfg["activation_fn"].lower())
        if act_cls:
            policy_kwargs["activation_fn"] = act_cls

    model = PPO(
        cfg.get("policy", "MlpPolicy"),
        train_env,
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
        tensorboard_log=os.path.join(save_path, "tb_logs"),
        device="cpu",  # MLP 정책은 CPU가 더 빠름 (GPU 전송 오버헤드)
    )

    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=save_path,
        log_path=os.path.join(save_path, "eval_logs"),
        eval_freq=eval_freq,
        deterministic=True,
    )

    model.learn(total_timesteps=timesteps, callback=eval_cb)
    model.save(os.path.join(save_path, "final_model"))

    train_env.close()
    eval_env.close()
    print(f"[✓] Training complete.  Models saved to '{save_path}/'")


# ── CLI ──────────────────────────────────────────
def main():
    p = argparse.ArgumentParser(description="Train RL Tycoon agent")
    p.add_argument("--config",     type=str,   default=None,
                   help="Path to training config JSON (default: config/train_config.json)")
    p.add_argument("--timesteps",  type=int,   default=None,
                   help="Override total_timesteps")
    p.add_argument("--save-path",  type=str,   default="models",
                   help="Directory to save models")
    p.add_argument("--n-envs",     type=int,   default=None,
                   help="Override number of parallel envs")
    p.add_argument("--seed",       type=int,   default=None,
                   help="Override random seed")
    train(p.parse_args())


if __name__ == "__main__":
    main()
