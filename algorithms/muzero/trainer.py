"""MuZero-style trainer with latent dynamics and MCTS planning.

This is a compact project-oriented implementation rather than a paper-complete
replica. It keeps the core MuZero loop intact:
- representation network for observations
- dynamics model over latent states
- prediction heads for policy and value
- Monte Carlo tree search for action selection
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from algorithms.base import BaseTrainer
from algorithms.common import EarlyStopTracker, load_algo_config, make_env, resolve_activation, save_run_config


class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden: list[int], out_dim: int, activation: type[nn.Module]):
        super().__init__()
        layers: list[nn.Module] = []
        prev = in_dim
        for width in hidden:
            layers.extend([nn.Linear(prev, width), activation()])
            prev = width
        layers.append(nn.Linear(prev, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MuZeroNet(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, latent_dim: int,
                 hidden_dim: int, activation: type[nn.Module]):
        super().__init__()
        trunk = [hidden_dim, hidden_dim]
        self.representation = MLP(obs_dim, trunk, latent_dim, activation)
        self.dynamics_body = MLP(latent_dim + act_dim, trunk, latent_dim, activation)
        self.reward_head = MLP(latent_dim, [hidden_dim], 1, activation)
        self.policy_head = MLP(latent_dim, [hidden_dim], act_dim, activation)
        self.value_head = MLP(latent_dim, [hidden_dim], 1, activation)
        self.act_dim = act_dim

    def initial_inference(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        latent = torch.tanh(self.representation(obs))
        logits = self.policy_head(latent)
        value = self.value_head(latent).squeeze(-1)
        return latent, logits, value

    def recurrent_inference(self, latent: torch.Tensor, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        action_onehot = F.one_hot(action.long(), self.act_dim).float()
        next_latent = torch.tanh(self.dynamics_body(torch.cat([latent, action_onehot], dim=-1)))
        reward = self.reward_head(next_latent).squeeze(-1)
        logits = self.policy_head(next_latent)
        value = self.value_head(next_latent).squeeze(-1)
        return next_latent, reward, logits, value


@dataclass
class SearchNode:
    prior: float
    latent: torch.Tensor | None = None
    reward: float = 0.0
    visit_count: int = 0
    value_sum: float = 0.0
    children: dict[int, "SearchNode"] = field(default_factory=dict)

    @property
    def value(self) -> float:
        return self.value_sum / self.visit_count if self.visit_count > 0 else 0.0


class EpisodeReplay:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.episodes: list[list[dict[str, Any]]] = []

    def add(self, episode: list[dict[str, Any]]) -> None:
        if not episode:
            return
        self.episodes.append(episode)
        if len(self.episodes) > self.capacity:
            self.episodes.pop(0)

    def __len__(self) -> int:
        return len(self.episodes)

    def sample(self, batch_size: int, unroll_steps: int):
        obs_batch = []
        action_batch = []
        reward_batch = []
        value_batch = []
        policy_batch = []

        for _ in range(batch_size):
            episode = self.episodes[np.random.randint(0, len(self.episodes))]
            start = np.random.randint(0, len(episode))
            root = episode[start]
            obs_batch.append(root["obs"])

            action_seq = []
            reward_seq = []
            value_seq = []
            policy_seq = []
            for offset in range(unroll_steps + 1):
                idx = start + offset
                if idx < len(episode):
                    step = episode[idx]
                    value_seq.append(step["return"])
                    policy_seq.append(step["policy"])
                    if offset < unroll_steps:
                        action_seq.append(step["action"])
                        reward_seq.append(step["reward"])
                else:
                    value_seq.append(0.0)
                    policy_seq.append(np.full_like(root["policy"], 1.0 / len(root["policy"])))
                    if offset < unroll_steps:
                        action_seq.append(0)
                        reward_seq.append(0.0)

            action_batch.append(action_seq)
            reward_batch.append(reward_seq)
            value_batch.append(value_seq)
            policy_batch.append(policy_seq)

        return {
            "obs": torch.FloatTensor(np.array(obs_batch, dtype=np.float32)),
            "actions": torch.LongTensor(np.array(action_batch, dtype=np.int64)),
            "rewards": torch.FloatTensor(np.array(reward_batch, dtype=np.float32)),
            "values": torch.FloatTensor(np.array(value_batch, dtype=np.float32)),
            "policies": torch.FloatTensor(np.array(policy_batch, dtype=np.float32)),
        }


class MuZeroTrainer(BaseTrainer):
    name = "MuZero"

    def __init__(self):
        self.model: MuZeroNet | None = None
        self.optimizer: torch.optim.Optimizer | None = None
        self.cfg: dict[str, Any] = {}
        self.save_path = "models/muzero"
        self._timesteps = 200_000
        self._obs_dim = 0
        self._act_dim = 0
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def build(self, cfg: dict | None = None, config_path: str | None = None,
              save_path: str | None = None, **overrides) -> None:
        days = overrides.pop("days", None)
        self.cfg = cfg or load_algo_config("muzero", config_path, days=days)
        t = self.cfg.get("training", {})
        hp = self.cfg.get("hyperparameters", {})
        net_cfg = self.cfg.get("network", {})
        game_ov = self.cfg.get("game_overrides", {})
        reward_cfg = self.cfg.get("reward_shaping", {})

        self._timesteps = overrides.get("timesteps", t.get("total_timesteps", 400_000))
        seed = overrides.get("seed", t.get("seed", 42))
        self.save_path = save_path or self.save_path
        os.makedirs(self.save_path, exist_ok=True)
        save_run_config(self.save_path, self.cfg)

        env = make_env(0, seed, game_ov, reward_cfg)()
        self._obs_dim = env.observation_space.shape[0]
        self._act_dim = env.action_space.n
        env.close()

        activation = resolve_activation(net_cfg.get("activation_fn", "relu"))
        self.model = MuZeroNet(
            self._obs_dim,
            self._act_dim,
            hp.get("latent_dim", 128),
            hp.get("hidden_dim", 256),
            activation,
        ).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=hp.get("learning_rate", 3e-4))

        self._hp = hp
        self._game_ov = game_ov
        self._reward_cfg = reward_cfg
        self._seed = seed

    def train(self, resume_path: str | None = None) -> dict[str, Any]:
        assert self.model is not None and self.optimizer is not None
        hp = self._hp
        gamma = hp.get("gamma", 0.99)
        learning_starts = hp.get("learning_starts", 2_000)
        train_freq = hp.get("train_freq", 100)
        gradient_steps = hp.get("gradient_steps", 8)
        unroll_steps = hp.get("unroll_steps", 5)
        batch_size = hp.get("batch_size", 32)
        max_grad_norm = hp.get("max_grad_norm", 10.0)
        eval_freq = self.cfg.get("training", {}).get("eval_freq", 5_000)

        if resume_path:
            self.load(resume_path)
            print(f"  [MuZero] 체크포인트 복원: {resume_path}")

        replay = EpisodeReplay(hp.get("buffer_size", 200))
        env = make_env(0, self._seed, self._game_ov, self._reward_cfg)()
        eval_env = make_env(0, self._seed + 1000, self._game_ov, self._reward_cfg)()
        early_stop = EarlyStopTracker(
            patience=self.cfg.get("training", {}).get("patience", 50),
            min_delta=1.0,
            verbose=1,
            metric_name="mean_final_score",
        )
        from torch.utils.tensorboard import SummaryWriter

        writer = SummaryWriter(os.path.join(self.save_path, "tb_logs"))

        obs, _ = env.reset()
        current_episode: list[dict[str, Any]] = []
        episode_rewards: list[float] = []
        ep_reward = 0.0
        best_score = float("-inf")
        eval_timesteps: list[int] = []
        eval_rewards: list[float] = []
        eval_scores: list[float] = []
        losses: list[float] = []

        for step in range(1, self._timesteps + 1):
            if step < learning_starts or len(replay) == 0:
                action = env.action_space.sample()
                policy = np.full(self._act_dim, 1.0 / self._act_dim, dtype=np.float32)
            else:
                action, policy, _ = self._run_mcts(obs, add_exploration=True)

            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            current_episode.append({
                "obs": np.array(obs, copy=True),
                "action": int(action),
                "reward": float(reward),
                "policy": np.array(policy, copy=True),
            })
            obs = next_obs
            ep_reward += reward

            if done:
                self._annotate_returns(current_episode, gamma)
                replay.add(current_episode)
                current_episode = []
                episode_rewards.append(ep_reward)
                writer.add_scalar("rollout/ep_reward", ep_reward, step)
                summary = info.get("episode_summary")
                if summary:
                    writer.add_scalar("rollout/final_score", summary.get("final_score", 0), step)
                    writer.add_scalar("rollout/net_profit", summary.get("net_profit", 0), step)
                if len(episode_rewards) % 5 == 0:
                    print(
                        f"  [MuZero] 스텝 {step}, 에피소드 {len(episode_rewards)}, "
                        f"평균보상(최근5): {np.mean(episode_rewards[-5:]):.1f}"
                    )
                obs, _ = env.reset()
                ep_reward = 0.0

            if step >= learning_starts and len(replay) > 0 and step % train_freq == 0:
                for _ in range(gradient_steps):
                    loss = self._update(replay, batch_size, unroll_steps, max_grad_norm)
                    losses.append(loss)
                if losses:
                    writer.add_scalar("train/loss", float(np.mean(losses)), step)
                    losses.clear()

            if step % eval_freq == 0:
                eval_reward, eval_score = self._evaluate(eval_env)
                print(f"  [MuZero] 평가 스텝 {step}: mean_reward={eval_reward:.1f}, mean_final_score={eval_score:.1f}")
                writer.add_scalar("eval/mean_reward", eval_reward, step)
                writer.add_scalar("eval/mean_final_score", eval_score, step)
                eval_timesteps.append(step)
                eval_rewards.append(eval_reward)
                eval_scores.append(eval_score)
                if eval_score > best_score:
                    best_score = eval_score
                    self.save(os.path.join(self.save_path, "best_model"))
                if not early_stop.check(eval_score):
                    print(f"  [MuZero] 조기 종료, 스텝: {step}")
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
        print(f"[✓] MuZero 학습 완료. 모델 → '{self.save_path}/'")
        return {"algorithm": "MuZero", "timesteps": self._timesteps, "save_path": self.save_path}

    def _annotate_returns(self, episode: list[dict[str, Any]], gamma: float) -> None:
        running = 0.0
        for step in reversed(episode):
            running = step["reward"] + gamma * running
            step["return"] = float(running)

    def _run_mcts(self, obs, add_exploration: bool) -> tuple[int, np.ndarray, float]:
        assert self.model is not None
        self.model.eval()
        obs_t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)

        with torch.no_grad():
            root_latent, root_logits, root_value = self.model.initial_inference(obs_t)
            priors = torch.softmax(root_logits, dim=-1).squeeze(0).cpu().numpy()

        if add_exploration:
            noise = np.random.dirichlet([self._hp.get("dirichlet_alpha", 0.3)] * self._act_dim)
            eps = self._hp.get("dirichlet_epsilon", 0.25)
            priors = (1.0 - eps) * priors + eps * noise

        root = SearchNode(prior=1.0, latent=root_latent)
        for action, prior in enumerate(priors):
            root.children[action] = SearchNode(prior=float(prior))

        for _ in range(self._hp.get("mcts_simulations", 32)):
            node = root
            search_path = [root]
            actions = []

            while node.children:
                action, node = self._select_child(search_path[-1])
                actions.append(action)
                search_path.append(node)

            parent = search_path[-2] if len(search_path) > 1 else root
            parent_latent = parent.latent
            action_tensor = torch.tensor([actions[-1]], device=self.device)
            with torch.no_grad():
                latent, reward, logits, value = self.model.recurrent_inference(parent_latent, action_tensor)
                node.latent = latent
                node.reward = float(reward.item())
                priors = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()
                for action, prior in enumerate(priors):
                    node.children[action] = SearchNode(prior=float(prior))
                leaf_value = float(value.item())

            self._backpropagate(search_path, leaf_value, self._hp.get("gamma", 0.99))

        visits = np.array([root.children[a].visit_count for a in range(self._act_dim)], dtype=np.float32)
        if visits.sum() == 0:
            policy = np.full(self._act_dim, 1.0 / self._act_dim, dtype=np.float32)
        else:
            policy = visits / visits.sum()
        action = int(np.argmax(policy))
        self.model.train()
        return action, policy, root_value.item()

    def _select_child(self, node: SearchNode) -> tuple[int, SearchNode]:
        total_visits = math.sqrt(max(1, node.visit_count))
        c_puct = self._hp.get("c_puct", 1.25)
        best_score = float("-inf")
        best_action = 0
        best_child = next(iter(node.children.values()))
        for action, child in node.children.items():
            prior_score = c_puct * child.prior * total_visits / (1 + child.visit_count)
            score = child.value + prior_score
            if score > best_score:
                best_score = score
                best_action = action
                best_child = child
        return best_action, best_child

    def _backpropagate(self, search_path: list[SearchNode], value: float, gamma: float) -> None:
        for node in reversed(search_path):
            node.value_sum += value
            node.visit_count += 1
            value = node.reward + gamma * value

    def _update(self, replay: EpisodeReplay, batch_size: int, unroll_steps: int,
                max_grad_norm: float) -> float:
        assert self.model is not None and self.optimizer is not None
        batch = replay.sample(batch_size, unroll_steps)
        obs = batch["obs"].to(self.device)
        actions = batch["actions"].to(self.device)
        rewards = batch["rewards"].to(self.device)
        values = batch["values"].to(self.device)
        policies = batch["policies"].to(self.device)

        latent, logits, value = self.model.initial_inference(obs)
        policy_loss = -(policies[:, 0] * F.log_softmax(logits, dim=-1)).sum(dim=-1).mean()
        value_loss = F.mse_loss(value, values[:, 0])
        reward_loss = torch.zeros(1, device=self.device)

        for step_idx in range(unroll_steps):
            latent, pred_reward, pred_logits, pred_value = self.model.recurrent_inference(
                latent, actions[:, step_idx]
            )
            reward_loss = reward_loss + F.mse_loss(pred_reward, rewards[:, step_idx])
            policy_loss = policy_loss + (-(policies[:, step_idx + 1] * F.log_softmax(pred_logits, dim=-1)).sum(dim=-1).mean())
            value_loss = value_loss + F.mse_loss(pred_value, values[:, step_idx + 1])
            latent = latent.detach() + (latent - latent.detach()) * 0.5

        loss = policy_loss + value_loss + reward_loss
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
        self.optimizer.step()
        return float(loss.item())

    def _evaluate(self, env, n_episodes: int | None = None) -> tuple[float, float]:
        episodes = n_episodes or self.cfg.get("training", {}).get("n_eval_episodes", 5)
        rewards = []
        final_scores = []
        for _ in range(episodes):
            obs, _ = env.reset()
            done = False
            total_reward = 0.0
            while not done:
                action, _, _ = self._run_mcts(obs, add_exploration=False)
                obs, reward, terminated, truncated, info = env.step(action)
                total_reward += reward
                done = terminated or truncated
            rewards.append(total_reward)
            summary = info.get("episode_summary", {})
            final_scores.append(float(summary.get("final_score", 0.0)))
        return float(np.mean(rewards)), float(np.mean(final_scores))

    def save(self, path: str) -> None:
        assert self.model is not None and self.optimizer is not None
        torch.save(
            {
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "cfg": self.cfg,
            },
            path if path.endswith(".pt") else f"{path}.pt",
        )

    def load(self, path: str) -> None:
        assert self.model is not None and self.optimizer is not None
        actual_path = path if path.endswith(".pt") else f"{path}.pt"
        payload = torch.load(actual_path, map_location=self.device)
        self.model.load_state_dict(payload["model"])
        self.optimizer.load_state_dict(payload["optimizer"])
        self.model.to(self.device)

    def predict(self, obs, deterministic: bool = True) -> int:
        action, _, _ = self._run_mcts(obs, add_exploration=not deterministic)
        return action