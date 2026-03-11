"""Training script – trains a PPO agent on the Tycoon environment using SB3.

Usage:
    python -m ai.train --timesteps 200000 --save-path models
"""

import argparse
import os

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.callbacks import EvalCallback

from ai.gym_env import TycoonEnv


def _make_env(rank: int = 0, seed: int = 0):
    def _init():
        env = TycoonEnv()
        env.reset(seed=seed + rank)
        return env
    return _init


def train(args):
    os.makedirs(args.save_path, exist_ok=True)

    # Vectorised training environments
    if args.n_envs > 1:
        train_env = SubprocVecEnv(
            [_make_env(i, args.seed) for i in range(args.n_envs)])
    else:
        train_env = DummyVecEnv([_make_env(0, args.seed)])

    eval_env = DummyVecEnv([_make_env(0, args.seed + 1000)])

    model = PPO(
        "MlpPolicy",
        train_env,
        verbose=1,
        learning_rate=args.lr,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        gamma=args.gamma,
        tensorboard_log=os.path.join(args.save_path, "tb_logs"),
    )

    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=args.save_path,
        log_path=os.path.join(args.save_path, "eval_logs"),
        eval_freq=args.eval_freq,
        deterministic=True,
    )

    model.learn(total_timesteps=args.timesteps, callback=eval_cb)
    model.save(os.path.join(args.save_path, "final_model"))

    train_env.close()
    eval_env.close()
    print(f"[✓] Training complete.  Models saved to '{args.save_path}/'")


# ── CLI ──────────────────────────────────────────
def main():
    p = argparse.ArgumentParser(description="Train RL Tycoon agent (PPO)")
    p.add_argument("--timesteps",  type=int,   default=200_000)
    p.add_argument("--save-path",  type=str,   default="models")
    p.add_argument("--n-envs",     type=int,   default=4)
    p.add_argument("--seed",       type=int,   default=42)
    p.add_argument("--lr",         type=float, default=3e-4)
    p.add_argument("--n-steps",    type=int,   default=2048)
    p.add_argument("--batch-size", type=int,   default=64)
    p.add_argument("--n-epochs",   type=int,   default=10)
    p.add_argument("--gamma",      type=float, default=0.99)
    p.add_argument("--eval-freq",  type=int,   default=5000)
    train(p.parse_args())


if __name__ == "__main__":
    main()
