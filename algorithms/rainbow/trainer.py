"""Rainbow DQN Trainer.

Implements a practical Rainbow variant for the project's discrete action space:
- Double DQN
- Dueling network
- Noisy linear layers
- Prioritized replay
- C51 distributional value learning
- N-step returns
"""

from __future__ import annotations

import math
import os
from collections import deque
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from algorithms.base import BaseTrainer
from algorithms.common import EarlyStopTracker, load_algo_config, make_env, resolve_activation, save_run_config


class NoisyLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, std_init: float = 0.5):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init

        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.register_buffer("weight_epsilon", torch.empty(out_features, in_features))

        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))
        self.register_buffer("bias_epsilon", torch.empty(out_features))

        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self) -> None:
        bound = 1.0 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-bound, bound)
        self.weight_sigma.data.fill_(self.std_init / math.sqrt(self.in_features))
        self.bias_mu.data.uniform_(-bound, bound)
        self.bias_sigma.data.fill_(self.std_init / math.sqrt(self.out_features))

    def reset_noise(self) -> None:
        eps_in = self._scale_noise(self.in_features)
        eps_out = self._scale_noise(self.out_features)
        self.weight_epsilon.copy_(eps_out.ger(eps_in))
        self.bias_epsilon.copy_(eps_out)

    @staticmethod
    def _scale_noise(size: int) -> torch.Tensor:
        x = torch.randn(size)
        return x.sign() * x.abs().sqrt()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            weight = self.weight_mu
            bias = self.bias_mu
        return F.linear(x, weight, bias)


class DuelingCategoricalNet(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        hidden: list[int],
        atom_size: int,
        support: torch.Tensor,
        activation: type[nn.Module],
        noisy_std: float,
    ):
        super().__init__()
        layers: list[nn.Module] = []
        prev = obs_dim
        for width in hidden:
            layers.extend([nn.Linear(prev, width), activation()])
            prev = width
        self.feature = nn.Sequential(*layers)

        self.adv_1 = NoisyLinear(prev, prev, noisy_std)
        self.adv_2 = NoisyLinear(prev, act_dim * atom_size, noisy_std)
        self.val_1 = NoisyLinear(prev, prev, noisy_std)
        self.val_2 = NoisyLinear(prev, atom_size, noisy_std)

        self.act_dim = act_dim
        self.atom_size = atom_size
        self.register_buffer("support", support)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        feat = self.feature(obs)
        adv = F.relu(self.adv_1(feat))
        adv = self.adv_2(adv).view(-1, self.act_dim, self.atom_size)

        val = F.relu(self.val_1(feat))
        val = self.val_2(val).view(-1, 1, self.atom_size)

        logits = val + adv - adv.mean(dim=1, keepdim=True)
        return F.softmax(logits, dim=-1).clamp(min=1e-6)

    def q_values(self, obs: torch.Tensor) -> torch.Tensor:
        dist = self.forward(obs)
        return torch.sum(dist * self.support.view(1, 1, -1), dim=-1)

    def reset_noise(self) -> None:
        self.adv_1.reset_noise()
        self.adv_2.reset_noise()
        self.val_1.reset_noise()
        self.val_2.reset_noise()


class PrioritizedNStepReplayBuffer:
    def __init__(
        self,
        obs_dim: int,
        capacity: int,
        n_step: int,
        gamma: float,
        alpha: float,
    ):
        self.capacity = capacity
        self.n_step = n_step
        self.gamma = gamma
        self.alpha = alpha

        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)
        self.priorities = np.zeros(capacity, dtype=np.float32)

        self.pos = 0
        self.size = 0
        self.max_priority = 1.0
        self.n_step_queue: deque[tuple[np.ndarray, int, float, np.ndarray, bool]] = deque(maxlen=n_step)

    def __len__(self) -> int:
        return self.size

    def push(self, obs, action, reward, next_obs, done) -> None:
        transition = (np.array(obs, copy=True), int(action), float(reward), np.array(next_obs, copy=True), bool(done))
        self.n_step_queue.append(transition)

        if len(self.n_step_queue) < self.n_step and not done:
            return

        self._append_from_queue()
        if done:
            while self.n_step_queue:
                self._append_from_queue()

    def _append_from_queue(self) -> None:
        reward, next_obs, done = self._compute_n_step_target()
        obs, action, _, _, _ = self.n_step_queue[0]

        self.obs[self.pos] = obs
        self.actions[self.pos] = action
        self.rewards[self.pos] = reward
        self.next_obs[self.pos] = next_obs
        self.dones[self.pos] = float(done)
        self.priorities[self.pos] = self.max_priority

        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
        self.n_step_queue.popleft()

    def _compute_n_step_target(self) -> tuple[float, np.ndarray, bool]:
        reward = 0.0
        next_obs = self.n_step_queue[-1][3]
        done = self.n_step_queue[-1][4]

        for idx, (_, _, r, nobs, d) in enumerate(self.n_step_queue):
            reward += (self.gamma ** idx) * r
            next_obs = nobs
            if d:
                done = True
                break
        return reward, next_obs, done

    def sample(self, batch_size: int, beta: float):
        priorities = self.priorities[:self.size] ** self.alpha
        probs = priorities / priorities.sum()
        indices = np.random.choice(self.size, size=batch_size, p=probs)

        weights = (self.size * probs[indices]) ** (-beta)
        weights /= weights.max()

        batch = {
            "obs": torch.FloatTensor(self.obs[indices]),
            "actions": torch.LongTensor(self.actions[indices]),
            "rewards": torch.FloatTensor(self.rewards[indices]),
            "next_obs": torch.FloatTensor(self.next_obs[indices]),
            "dones": torch.FloatTensor(self.dones[indices]),
            "weights": torch.FloatTensor(weights),
            "indices": indices,
        }
        return batch

    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray) -> None:
        priorities = np.asarray(priorities, dtype=np.float32)
        self.priorities[indices] = priorities
        self.max_priority = max(self.max_priority, float(priorities.max()))


class RainbowTrainer(BaseTrainer):
    name = "Rainbow"

    def __init__(self):
        self.online_net: DuelingCategoricalNet | None = None
        self.target_net: DuelingCategoricalNet | None = None
        self.optimizer: torch.optim.Optimizer | None = None
        self.cfg: dict[str, Any] = {}
        self.save_path = "models/rainbow"
        self._timesteps = 200_000
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._obs_dim = 0
        self._act_dim = 0

    def build(self, cfg: dict | None = None, config_path: str | None = None,
              save_path: str | None = None, **overrides) -> None:
        days = overrides.pop("days", None)
        self.cfg = cfg or load_algo_config("rainbow", config_path, days=days)
        t = self.cfg.get("training", {})
        hp = self.cfg.get("hyperparameters", {})
        net_cfg = self.cfg.get("network", {})
        game_ov = self.cfg.get("game_overrides", {})
        reward_cfg = self.cfg.get("reward_shaping", {})

        self._timesteps = overrides.get("timesteps", t.get("total_timesteps", 600_000))
        seed = overrides.get("seed", t.get("seed", 42))
        self.save_path = save_path or self.save_path
        os.makedirs(self.save_path, exist_ok=True)
        save_run_config(self.save_path, self.cfg)

        env = make_env(0, seed, game_ov, reward_cfg)()
        self._obs_dim = env.observation_space.shape[0]
        self._act_dim = env.action_space.n
        env.close()

        atom_size = hp.get("atoms", 51)
        v_min = hp.get("v_min", -50.0)
        v_max = hp.get("v_max", 100.0)
        support = torch.linspace(v_min, v_max, atom_size)
        hidden = net_cfg.get("net_arch", [256, 256])
        activation = resolve_activation(net_cfg.get("activation_fn", "relu"))
        noisy_std = hp.get("noisy_std", 0.5)

        self.online_net = DuelingCategoricalNet(
            self._obs_dim, self._act_dim, hidden, atom_size, support, activation, noisy_std,
        ).to(self.device)
        self.target_net = DuelingCategoricalNet(
            self._obs_dim, self._act_dim, hidden, atom_size, support, activation, noisy_std,
        ).to(self.device)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.optimizer = torch.optim.Adam(
            self.online_net.parameters(), lr=hp.get("learning_rate", 1e-4)
        )

        self._hp = hp
        self._game_ov = game_ov
        self._reward_cfg = reward_cfg
        self._seed = seed
        self._support = support.to(self.device)
        self._delta_z = (v_max - v_min) / (atom_size - 1)
        self._v_min = v_min
        self._v_max = v_max

    def train(self, resume_path: str | None = None) -> dict[str, Any]:
        assert self.online_net is not None and self.target_net is not None
        assert self.optimizer is not None

        hp = self._hp
        gamma = hp.get("gamma", 0.99)
        n_step = hp.get("n_step", 3)
        batch_size = hp.get("batch_size", 128)
        learning_starts = hp.get("learning_starts", 5_000)
        train_freq = hp.get("train_freq", 1)
        gradient_steps = hp.get("gradient_steps", 1)
        target_update_interval = hp.get("target_update_interval", 2_000)
        max_grad_norm = hp.get("max_grad_norm", 10.0)
        per_beta_start = hp.get("per_beta_start", 0.4)
        per_beta_frames = max(1, hp.get("per_beta_frames", self._timesteps))
        eval_freq = self.cfg.get("training", {}).get("eval_freq", 5_000)

        replay = PrioritizedNStepReplayBuffer(
            self._obs_dim,
            hp.get("buffer_size", 200_000),
            n_step,
            gamma,
            hp.get("per_alpha", 0.6),
        )

        if resume_path:
            self.load(resume_path)
            print(f"  [Rainbow] 체크포인트 복원: {resume_path}")

        env = make_env(0, self._seed, self._game_ov, self._reward_cfg)()
        eval_env = make_env(0, self._seed + 1000, self._game_ov, self._reward_cfg)()
        early_stop = EarlyStopTracker(
            patience=self.cfg.get("training", {}).get("patience", 100),
            min_delta=1.0,
            verbose=1,
            metric_name="mean_final_score",
        )
        from torch.utils.tensorboard import SummaryWriter

        writer = SummaryWriter(os.path.join(self.save_path, "tb_logs"))

        obs, _ = env.reset()
        ep_reward = 0.0
        ep_rewards: list[float] = []
        best_score = float("-inf")
        eval_timesteps: list[int] = []
        eval_rewards: list[float] = []
        eval_scores: list[float] = []
        loss_window: list[float] = []

        gamma_n = gamma ** n_step
        for step in range(1, self._timesteps + 1):
            if step < learning_starts:
                action = env.action_space.sample()
            else:
                action = self.predict(obs, deterministic=True)

            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            replay.push(obs, action, reward, next_obs, done)
            obs = next_obs
            ep_reward += reward

            if done:
                ep_rewards.append(ep_reward)
                writer.add_scalar("rollout/ep_reward", ep_reward, step)
                summary = info.get("episode_summary")
                if summary:
                    writer.add_scalar("rollout/final_score", summary.get("final_score", 0), step)
                    writer.add_scalar("rollout/net_profit", summary.get("net_profit", 0), step)
                    writer.add_scalar("rollout/shop_rating", summary.get("shop_rating", 0), step)
                if len(ep_rewards) % 10 == 0:
                    print(
                        f"  [Rainbow] 스텝 {step}, 에피소드 {len(ep_rewards)}, "
                        f"평균보상(최근10): {np.mean(ep_rewards[-10:]):.1f}"
                    )
                obs, _ = env.reset()
                ep_reward = 0.0

            if step >= learning_starts and step % train_freq == 0 and len(replay) >= batch_size:
                beta = min(1.0, per_beta_start + (1.0 - per_beta_start) * (step / per_beta_frames))
                for _ in range(gradient_steps):
                    metrics = self._update(replay, batch_size, beta, gamma_n, max_grad_norm)
                    loss_window.append(metrics["loss"])
                if step % 1000 == 0 and loss_window:
                    writer.add_scalar("train/loss", float(np.mean(loss_window)), step)
                    loss_window.clear()

            if step % target_update_interval == 0:
                self.target_net.load_state_dict(self.online_net.state_dict())

            if step % eval_freq == 0:
                eval_reward, eval_score = self._evaluate(eval_env)
                print(f"  [Rainbow] 평가 스텝 {step}: mean_reward={eval_reward:.1f}, mean_final_score={eval_score:.1f}")
                writer.add_scalar("eval/mean_reward", eval_reward, step)
                writer.add_scalar("eval/mean_final_score", eval_score, step)
                eval_timesteps.append(step)
                eval_rewards.append(eval_reward)
                eval_scores.append(eval_score)
                if eval_score > best_score:
                    best_score = eval_score
                    self.save(os.path.join(self.save_path, "best_model"))
                if not early_stop.check(eval_score):
                    print(f"  [Rainbow] 조기 종료, 스텝: {step}")
                    break

        self.save(os.path.join(self.save_path, "final_model"))
        eval_dir = os.path.join(self.save_path, "eval_logs")
        os.makedirs(eval_dir, exist_ok=True)
        np.savez(
            os.path.join(eval_dir, "evaluations.npz"),
            timesteps=np.array(eval_timesteps),
            results=np.array(eval_rewards),
            final_scores=np.array(eval_scores),
        )
        writer.close()
        env.close()
        eval_env.close()
        print(f"[✓] Rainbow 학습 완료. 모델 → '{self.save_path}/'")
        return {"algorithm": "Rainbow", "timesteps": self._timesteps, "save_path": self.save_path}

    def _update(self, replay: PrioritizedNStepReplayBuffer, batch_size: int,
                beta: float, gamma_n: float, max_grad_norm: float) -> dict[str, float]:
        assert self.online_net is not None and self.target_net is not None and self.optimizer is not None

        batch = replay.sample(batch_size, beta)
        obs = batch["obs"].to(self.device)
        actions = batch["actions"].to(self.device)
        rewards = batch["rewards"].to(self.device)
        next_obs = batch["next_obs"].to(self.device)
        dones = batch["dones"].to(self.device)
        weights = batch["weights"].to(self.device)

        dist = self.online_net(obs)
        chosen_dist = dist[torch.arange(batch_size, device=self.device), actions]

        with torch.no_grad():
            next_q = self.online_net.q_values(next_obs)
            next_actions = next_q.argmax(dim=1)
            next_dist = self.target_net(next_obs)[torch.arange(batch_size, device=self.device), next_actions]

            tz = rewards.unsqueeze(1) + (1.0 - dones.unsqueeze(1)) * gamma_n * self._support.view(1, -1)
            tz = tz.clamp(self._v_min, self._v_max)
            b = (tz - self._v_min) / self._delta_z
            lower = b.floor().long()
            upper = b.ceil().long()

            proj_dist = torch.zeros_like(next_dist)
            offset = (torch.arange(batch_size, device=self.device) * next_dist.size(1)).unsqueeze(1)

            proj_dist.view(-1).index_add_(0, (lower + offset).view(-1), (next_dist * (upper.float() - b)).view(-1))
            proj_dist.view(-1).index_add_(0, (upper + offset).view(-1), (next_dist * (b - lower.float())).view(-1))

            same_bin = upper == lower
            if same_bin.any():
                proj_dist[same_bin] += next_dist[same_bin]

        loss_per_sample = -(proj_dist * chosen_dist.log()).sum(dim=1)
        loss = (loss_per_sample * weights).mean()

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online_net.parameters(), max_grad_norm)
        self.optimizer.step()

        priorities = loss_per_sample.detach().cpu().numpy() + 1e-6
        replay.update_priorities(batch["indices"], priorities)
        self.online_net.reset_noise()
        self.target_net.reset_noise()
        return {"loss": float(loss.item())}

    def _evaluate(self, env, n_episodes: int | None = None) -> tuple[float, float]:
        episodes = n_episodes or self.cfg.get("training", {}).get("n_eval_episodes", 5)
        rewards = []
        final_scores = []
        for _ in range(episodes):
            obs, _ = env.reset()
            done = False
            total_reward = 0.0
            while not done:
                action = self.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                total_reward += reward
                done = terminated or truncated
            rewards.append(total_reward)
            summary = info.get("episode_summary", {})
            final_scores.append(float(summary.get("final_score", 0.0)))
        return float(np.mean(rewards)), float(np.mean(final_scores))

    def save(self, path: str) -> None:
        assert self.online_net is not None and self.target_net is not None and self.optimizer is not None
        torch.save(
            {
                "online": self.online_net.state_dict(),
                "target": self.target_net.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "cfg": self.cfg,
            },
            path if path.endswith(".pt") else f"{path}.pt",
        )

    def load(self, path: str) -> None:
        assert self.online_net is not None and self.target_net is not None and self.optimizer is not None
        actual_path = path if path.endswith(".pt") else f"{path}.pt"
        payload = torch.load(actual_path, map_location=self.device)
        self.online_net.load_state_dict(payload["online"])
        self.target_net.load_state_dict(payload["target"])
        self.optimizer.load_state_dict(payload["optimizer"])
        self.online_net.to(self.device)
        self.target_net.to(self.device)

    def predict(self, obs, deterministic: bool = True) -> int:
        assert self.online_net is not None
        obs_t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        self.online_net.eval()
        with torch.no_grad():
            q_values = self.online_net.q_values(obs_t)
            action = int(q_values.argmax(dim=1).item())
        self.online_net.train()
        return action
