"""SAC Trainer – Soft Actor-Critic for Discrete actions.

SB3의 기본 SAC는 연속 행동 공간만 지원하므로,
이 구현에서는 PyTorch로 Discrete SAC를 직접 구현합니다.

핵심 특징:
- Maximum entropy RL: 탐색과 활용의 자동 균형
- Automatic temperature (alpha) tuning
- Twin Q-networks (Double Q-learning)
- Replay Buffer 사용
"""

import os
from typing import Any
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

from algorithms.base import BaseTrainer
from algorithms.common import load_algo_config, make_env, save_run_config, EarlyStopTracker


# ────────────────────────────────────────────────
# Networks
# ────────────────────────────────────────────────
class SoftQNetwork(nn.Module):
    """Twin Q-Network for discrete actions."""

    def __init__(self, obs_dim: int, act_dim: int,
                 hidden: list[int] | None = None):
        super().__init__()
        hidden = hidden or [256, 256]
        layers1, layers2 = [], []
        prev = obs_dim
        for h in hidden:
            layers1 += [nn.Linear(prev, h), nn.ReLU()]
            layers2 += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        layers1.append(nn.Linear(prev, act_dim))
        layers2.append(nn.Linear(prev, act_dim))
        self.q1 = nn.Sequential(*layers1)
        self.q2 = nn.Sequential(*layers2)

    def forward(self, obs):
        return self.q1(obs), self.q2(obs)


class PolicyNetwork(nn.Module):
    """Categorical policy for discrete actions."""

    def __init__(self, obs_dim: int, act_dim: int,
                 hidden: list[int] | None = None):
        super().__init__()
        hidden = hidden or [256, 256]
        layers = []
        prev = obs_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        layers.append(nn.Linear(prev, act_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, obs):
        logits = self.net(obs)
        probs = F.softmax(logits, dim=-1)
        return probs

    def get_action(self, obs, deterministic=False):
        probs = self.forward(obs)
        if deterministic:
            return probs.argmax(dim=-1)
        dist = torch.distributions.Categorical(probs)
        return dist.sample()


# ────────────────────────────────────────────────
# Replay Buffer
# ────────────────────────────────────────────────
class ReplayBuffer:
    def __init__(self, capacity: int = 100_000):
        self.buffer = deque(maxlen=capacity)

    def push(self, obs, action, reward, next_obs, done):
        self.buffer.append((obs, action, reward, next_obs, done))

    def sample(self, batch_size: int):
        idxs = np.random.choice(len(self.buffer), batch_size, replace=False)
        batch = [self.buffer[i] for i in idxs]
        obs, act, rew, nobs, done = zip(*batch)
        return (np.array(obs, dtype=np.float32),
                np.array(act, dtype=np.int64),
                np.array(rew, dtype=np.float32),
                np.array(nobs, dtype=np.float32),
                np.array(done, dtype=np.float32))

    def __len__(self):
        return len(self.buffer)


# ────────────────────────────────────────────────
# SAC Discrete Trainer
# ────────────────────────────────────────────────
class SACTrainer(BaseTrainer):
    name = "SAC"

    def __init__(self):
        self.policy: PolicyNetwork | None = None
        self.q_net: SoftQNetwork | None = None
        self.q_target: SoftQNetwork | None = None
        self.cfg: dict = {}
        self.save_path: str = "models/sac"
        self._timesteps: int = 200_000
        self._obs_dim: int = 0
        self._act_dim: int = 0
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def build(self, cfg: dict | None = None, config_path: str | None = None,
              save_path: str | None = None, **overrides) -> None:
        days = overrides.pop("days", None)
        self.cfg = cfg or load_algo_config("sac", config_path, days=days)
        t = self.cfg.get("training", {})
        hp = self.cfg.get("hyperparameters", {})
        net = self.cfg.get("network", {})
        game_ov = self.cfg.get("game_overrides", {})
        reward_cfg = self.cfg.get("reward_shaping", {})

        self._timesteps = overrides.get("timesteps",
                                        t.get("total_timesteps", 200_000))
        seed = overrides.get("seed", t.get("seed", 42))
        self.save_path = save_path or self.save_path
        os.makedirs(self.save_path, exist_ok=True)
        save_run_config(self.save_path, self.cfg)

        # 환경 차원
        tmp = make_env(0, seed, game_ov, reward_cfg)()
        self._obs_dim = tmp.observation_space.shape[0]
        self._act_dim = tmp.action_space.n
        tmp.close()

        hidden = net.get("net_arch", [256, 256])

        self.policy = PolicyNetwork(self._obs_dim, self._act_dim, hidden).to(self.device)
        self.q_net = SoftQNetwork(self._obs_dim, self._act_dim, hidden).to(self.device)
        self.q_target = SoftQNetwork(self._obs_dim, self._act_dim, hidden).to(self.device)
        self.q_target.load_state_dict(self.q_net.state_dict())

        lr = hp.get("learning_rate", 3e-4)
        self._policy_opt = Adam(self.policy.parameters(), lr=lr)
        self._q_opt = Adam(self.q_net.parameters(), lr=lr)

        # Auto temperature
        target_ent = hp.get("target_entropy", "auto")
        if target_ent == "auto":
            self._target_entropy = -np.log(1.0 / self._act_dim) * 0.98
        else:
            self._target_entropy = float(target_ent)

        self._log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self._alpha_opt = Adam([self._log_alpha], lr=lr)

        self._hp = hp
        self._game_ov = game_ov
        self._reward_cfg = reward_cfg
        self._seed = seed

    def train(self) -> dict[str, Any]:
        assert self.policy is not None, "call build() first"
        hp = self._hp
        gamma = hp.get("gamma", 0.99)
        tau = hp.get("tau", 0.005)
        batch_size = hp.get("batch_size", 256)
        buffer_size = hp.get("buffer_size", 100_000)
        learning_starts = hp.get("learning_starts", 1000)
        train_freq = hp.get("train_freq", 1)
        gradient_steps = hp.get("gradient_steps", 1)
        eval_freq = self.cfg.get("training", {}).get("eval_freq", 5000)

        replay = ReplayBuffer(buffer_size)
        env = make_env(0, self._seed, self._game_ov, self._reward_cfg)()
        eval_env = make_env(0, self._seed + 1000, self._game_ov, self._reward_cfg)()
        early_stop = EarlyStopTracker(patience=50, min_delta=1.0, verbose=1)

        obs, _ = env.reset()
        episode_rewards = []
        ep_reward = 0.0
        best_eval = float("-inf")

        for step in range(1, self._timesteps + 1):
            if step < learning_starts:
                action = env.action_space.sample()
            else:
                obs_t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
                action = self.policy.get_action(obs_t).item()

            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            replay.push(obs, action, reward, next_obs, float(done))
            obs = next_obs
            ep_reward += reward

            if done:
                episode_rewards.append(ep_reward)
                if len(episode_rewards) % 10 == 0:
                    print(f"  [SAC (소프트 액터-크리틱)] 스텝(Step) {step}, 에피소드(Episodes) {len(episode_rewards)}, "
                          f"평균보상(AvgReward, 최근10): {np.mean(episode_rewards[-10:]):.1f}")
                obs, _ = env.reset()
                ep_reward = 0.0

            # Train
            if step >= learning_starts and step % train_freq == 0:
                for _ in range(gradient_steps):
                    self._update(replay, batch_size, gamma, tau)

            # Eval
            if step % eval_freq == 0:
                eval_r = self._evaluate(eval_env, n_episodes=5)
                print(f"  [SAC] 평가(Eval) 스텝 {step}: 평균보상(mean_reward)={eval_r:.1f}")
                if eval_r > best_eval:
                    best_eval = eval_r
                    self.save(os.path.join(self.save_path, "best_model"))
                if not early_stop.check(eval_r):
                    print(f"  [SAC] 조기 종료 (Early stopped), 스텝: {step}")
                    break

        self.save(os.path.join(self.save_path, "final_model"))
        env.close()
        eval_env.close()
        print(f"[✓] SAC (소프트 액터-크리틱) 학습 완료. 모델 → '{self.save_path}/'")
        return {"algorithm": "SAC", "timesteps": self._timesteps,
                "episodes": len(episode_rewards), "save_path": self.save_path}

    def _update(self, replay: ReplayBuffer, batch_size: int,
                gamma: float, tau: float):
        obs_b, act_b, rew_b, nobs_b, done_b = replay.sample(batch_size)
        obs_t = torch.FloatTensor(obs_b).to(self.device)
        act_t = torch.LongTensor(act_b).to(self.device)
        rew_t = torch.FloatTensor(rew_b).to(self.device)
        nobs_t = torch.FloatTensor(nobs_b).to(self.device)
        done_t = torch.FloatTensor(done_b).to(self.device)

        alpha = self._log_alpha.exp().detach()

        # Q target
        with torch.no_grad():
            next_probs = self.policy(nobs_t)
            next_log_probs = torch.log(next_probs + 1e-8)
            q1_next, q2_next = self.q_target(nobs_t)
            q_next = torch.min(q1_next, q2_next)
            v_next = (next_probs * (q_next - alpha * next_log_probs)).sum(dim=-1)
            q_target = rew_t + gamma * (1.0 - done_t) * v_next

        # Q loss
        q1, q2 = self.q_net(obs_t)
        q1_a = q1.gather(1, act_t.unsqueeze(-1)).squeeze(-1)
        q2_a = q2.gather(1, act_t.unsqueeze(-1)).squeeze(-1)
        q_loss = F.mse_loss(q1_a, q_target) + F.mse_loss(q2_a, q_target)

        self._q_opt.zero_grad()
        q_loss.backward()
        self._q_opt.step()

        # Policy loss
        probs = self.policy(obs_t)
        log_probs = torch.log(probs + 1e-8)
        q1_pi, q2_pi = self.q_net(obs_t)
        q_pi = torch.min(q1_pi.detach(), q2_pi.detach())
        policy_loss = (probs * (alpha * log_probs - q_pi)).sum(dim=-1).mean()

        self._policy_opt.zero_grad()
        policy_loss.backward()
        self._policy_opt.step()

        # Alpha loss
        alpha_loss = -(self._log_alpha *
                       (probs.detach() *
                        (log_probs.detach() + self._target_entropy)
                        ).sum(dim=-1)).mean()

        self._alpha_opt.zero_grad()
        alpha_loss.backward()
        self._alpha_opt.step()

        # Soft update target
        for tp, sp in zip(self.q_target.parameters(), self.q_net.parameters()):
            tp.data.copy_(tau * sp.data + (1.0 - tau) * tp.data)

    def _evaluate(self, env, n_episodes: int = 5) -> float:
        total = 0.0
        for _ in range(n_episodes):
            obs, _ = env.reset()
            done = False
            while not done:
                obs_t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
                action = self.policy.get_action(obs_t, deterministic=True).item()
                obs, r, terminated, truncated, _ = env.step(action)
                total += r
                done = terminated or truncated
        return total / n_episodes

    def save(self, path: str) -> None:
        torch.save({
            "policy": self.policy.state_dict(),
            "q_net": self.q_net.state_dict(),
            "q_target": self.q_target.state_dict(),
            "log_alpha": self._log_alpha.data,
        }, path + ".pt")

    def load(self, path: str) -> None:
        ckpt = torch.load(path + ".pt", map_location=self.device)
        if self.policy:
            self.policy.load_state_dict(ckpt["policy"])
        if self.q_net:
            self.q_net.load_state_dict(ckpt["q_net"])
        if self.q_target:
            self.q_target.load_state_dict(ckpt["q_target"])
        self._log_alpha.data = ckpt["log_alpha"]

    def predict(self, obs, deterministic: bool = True) -> int:
        assert self.policy is not None
        obs_t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        action = self.policy.get_action(obs_t, deterministic=deterministic)
        return int(action.item())
