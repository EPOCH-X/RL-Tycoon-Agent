"""Training script – trains a MaskablePPO agent on the Tycoon environment.

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
import importlib.util

from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from config.settings import load_json_config
from ai.gym_env import TycoonEnv


def _tensorboard_log_dir(save_path: str) -> str | None:
    """Return a TensorBoard log dir only when tensorboard is installed."""
    if importlib.util.find_spec("tensorboard") is None:
        return None
    return os.path.join(save_path, "tb_logs")


def _save_training_plots(save_path: str):
    """Persist evaluation and TensorBoard scalar plots when dependencies exist."""
    if importlib.util.find_spec("matplotlib") is None:
        print("[!] matplotlib not installed; skipping plot export.")
        return

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    plots_dir = os.path.join(save_path, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    eval_npz = os.path.join(save_path, "eval_logs", "evaluations.npz")
    if os.path.isfile(eval_npz):
        data = np.load(eval_npz)
        timesteps = data["timesteps"]
        results = data["results"]
        ep_lengths = data["ep_lengths"]

        mean_rewards = results.mean(axis=1)
        std_rewards = results.std(axis=1)
        plt.figure(figsize=(8, 4.5))
        plt.plot(timesteps, mean_rewards, label="mean_reward")
        plt.fill_between(
            timesteps,
            mean_rewards - std_rewards,
            mean_rewards + std_rewards,
            alpha=0.2,
            label="std")
        plt.xlabel("Timesteps")
        plt.ylabel("Eval Reward")
        plt.title("Evaluation Reward")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, "eval_reward.png"), dpi=150)
        plt.close()

        mean_lengths = ep_lengths.mean(axis=1)
        plt.figure(figsize=(8, 4.5))
        plt.plot(timesteps, mean_lengths, label="mean_ep_length")
        plt.xlabel("Timesteps")
        plt.ylabel("Episode Length")
        plt.title("Evaluation Episode Length")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, "eval_episode_length.png"), dpi=150)
        plt.close()

    if importlib.util.find_spec("tensorboard") is None:
        return

    tb_root = os.path.join(save_path, "tb_logs")
    run_dirs = []
    if os.path.isdir(tb_root):
        run_dirs = [
            os.path.join(tb_root, name)
            for name in os.listdir(tb_root)
            if os.path.isdir(os.path.join(tb_root, name))
        ]
    if not run_dirs:
        return

    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    scalar_specs = {
        "train_loss.png": ["train/loss", "train/value_loss"],
        "train_policy.png": ["train/approx_kl", "train/clip_fraction", "train/entropy_loss"],
        "eval_summary.png": ["eval/mean_reward", "eval/mean_ep_length"],
    }

    event_acc = EventAccumulator(run_dirs[0])
    event_acc.Reload()
    available = set(event_acc.Tags().get("scalars", []))

    for filename, tags in scalar_specs.items():
        present_tags = [tag for tag in tags if tag in available]
        if not present_tags:
            continue
        plt.figure(figsize=(8, 4.5))
        for tag in present_tags:
            vals = event_acc.Scalars(tag)
            if not vals:
                continue
            steps = [v.step for v in vals]
            scalars = [v.value for v in vals]
            plt.plot(steps, scalars, label=tag)
        plt.xlabel("Timesteps")
        plt.title(filename.replace(".png", "").replace("_", " ").title())
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, filename), dpi=150)
        plt.close()


def load_train_config(config_path: str | None = None) -> dict:
    """Load training config JSON. Falls back to config/train_config.json."""
    if config_path and os.path.isfile(config_path):
        with open(config_path, encoding="utf-8") as f:
            return json.load(f)
    return load_json_config("train_config.json")


def _make_env(rank: int = 0, seed: int = 0, game_overrides: dict | None = None,
              reward_config: dict | None = None,
              env_options: dict | None = None):
    def _init():
        kwargs = {}
        if game_overrides:
            if game_overrides.get("target_money") is not None:
                kwargs["target_money"] = game_overrides["target_money"]
            if game_overrides.get("day_limit") is not None:
                kwargs["day_limit"] = game_overrides["day_limit"]
        if env_options:
            kwargs.update({
                key: value for key, value in env_options.items()
                if not key.startswith("_")
            })
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
    env_options = cfg.get("env_options", {})

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
            [_make_env(i, seed, game_ov, reward_cfg, env_options) for i in range(n_envs)])
    else:
        train_env = DummyVecEnv([_make_env(0, seed, game_ov, reward_cfg, env_options)])

    eval_env = DummyVecEnv([_make_env(0, seed + 1000, game_ov, reward_cfg, env_options)])

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

    model = MaskablePPO(
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
        tensorboard_log=_tensorboard_log_dir(save_path),
    )

    eval_cb = MaskableEvalCallback(
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
    _save_training_plots(save_path)
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
