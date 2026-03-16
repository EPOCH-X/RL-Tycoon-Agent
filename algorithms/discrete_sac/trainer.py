"""Discrete SAC v2 – Quantile-based Distributional Soft Actor-Critic.

기존 SAC와의 차이점:
- 3개 Q-네트워크 (TQC 스타일, 상위 quantile 드롭으로 과대추정 억제)
- Quantile Regression: Q-값 분포를 학습하여 불확실성까지 모델링
- 적응형 α (temperature) 자동 튜닝
- 개선된 리플레이 버퍼 (numpy 배열 기반)
"""

import os
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter

from algorithms.base import BaseTrainer
from algorithms.common import load_algo_config, make_env, save_run_config, EarlyStopTracker


# ────────────────────────────────────────────────
# Quantile Q-Network
# ────────────────────────────────────────────────
class QuantileQNetwork(nn.Module):
    """각 행동에 대해 N개의 quantile 값을 출력하는 Q-네트워크."""

    def __init__(self, obs_dim: int, act_dim: int, n_quantiles: int = 25,
                 hidden: list[int] | None = None):
        super().__init__()
        hidden = hidden or [256, 256]
        layers = []
        prev = obs_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        layers.append(nn.Linear(prev, act_dim * n_quantiles))
        self.net = nn.Sequential(*layers)
        self.act_dim = act_dim
        self.n_quantiles = n_quantiles

    def forward(self, obs):
        # → [batch, act_dim, n_quantiles]
        return self.net(obs).view(-1, self.act_dim, self.n_quantiles)


class PolicyNetwork(nn.Module):
    """이산 행동 공간용 Categorical 정책."""

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
        return F.softmax(self.net(obs), dim=-1)

    def get_action(self, obs, deterministic=False):
        probs = self.forward(obs)
        if deterministic:
            return probs.argmax(dim=-1)
        return torch.distributions.Categorical(probs).sample()


# ────────────────────────────────────────────────
# Efficient Replay Buffer (numpy-based)
# ────────────────────────────────────────────────
class NumpyReplayBuffer:
    def __init__(self, obs_dim: int, capacity: int = 500_000):
        self.capacity = capacity
        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)
        self.pos = 0
        self.size = 0

    def push(self, obs, action, reward, next_obs, done):
        self.obs[self.pos] = obs
        self.actions[self.pos] = action
        self.rewards[self.pos] = reward
        self.next_obs[self.pos] = next_obs
        self.dones[self.pos] = done
        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int):
        idxs = np.random.randint(0, self.size, size=batch_size)
        return (self.obs[idxs], self.actions[idxs], self.rewards[idxs],
                self.next_obs[idxs], self.dones[idxs])

    def __len__(self):
        return self.size


# ────────────────────────────────────────────────
# Trainer
# ────────────────────────────────────────────────
class DiscreteSACTrainer(BaseTrainer):
    name = "DiscreteSAC"

    def __init__(self):
        self.policy: PolicyNetwork | None = None
        self.q_nets: list[QuantileQNetwork] = []
        self.q_targets: list[QuantileQNetwork] = []
        self.cfg: dict = {}
        self.save_path: str = "models/discrete_sac"
        self._timesteps: int = 200_000
        self._obs_dim: int = 0
        self._act_dim: int = 0
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def build(self, cfg: dict | None = None, config_path: str | None = None,
              save_path: str | None = None, **overrides) -> None:
        days = overrides.pop("days", None)
        self.cfg = cfg or load_algo_config("discrete_sac", config_path, days=days)
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

        tmp = make_env(0, seed, game_ov, reward_cfg)()
        self._obs_dim = tmp.observation_space.shape[0]
        self._act_dim = tmp.action_space.n
        tmp.close()

        hidden = net.get("net_arch", [256, 256])
        n_quantiles = hp.get("n_quantiles", 25)
        n_critics = hp.get("n_critics", 3)

        self.policy = PolicyNetwork(self._obs_dim, self._act_dim, hidden).to(self.device)
        self.q_nets = [QuantileQNetwork(self._obs_dim, self._act_dim, n_quantiles, hidden).to(self.device)
                       for _ in range(n_critics)]
        self.q_targets = [QuantileQNetwork(self._obs_dim, self._act_dim, n_quantiles, hidden).to(self.device)
                          for _ in range(n_critics)]
        for qn, qt in zip(self.q_nets, self.q_targets):
            qt.load_state_dict(qn.state_dict())

        lr = hp.get("learning_rate", 3e-4)
        self._policy_opt = Adam(self.policy.parameters(), lr=lr)
        q_params = [p for qn in self.q_nets for p in qn.parameters()]
        self._q_opt = Adam(q_params, lr=lr)

        ent_ratio = hp.get("target_entropy_ratio", 0.45)
        self._target_entropy = -np.log(1.0 / self._act_dim) * ent_ratio
        self._log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self._alpha_opt = Adam([self._log_alpha], lr=lr)

        self._n_quantiles = n_quantiles
        self._n_critics = n_critics
        self._top_drop = hp.get("top_quantiles_to_drop", 2)
        self._max_grad_norm = hp.get("max_grad_norm", 1.0)
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
        buffer_size = hp.get("buffer_size", 500_000)
        learning_starts = hp.get("learning_starts", 20000)
        train_freq = hp.get("train_freq", 1)
        gradient_steps = hp.get("gradient_steps", 2)
        eval_freq = self.cfg.get("training", {}).get("eval_freq", 5000)
        n_envs = self.cfg.get("training", {}).get("n_envs", 1)

        replay = NumpyReplayBuffer(self._obs_dim, buffer_size)
        envs = [make_env(i, self._seed + i, self._game_ov, self._reward_cfg)()
                for i in range(n_envs)]
        eval_env = make_env(0, self._seed + 1000, self._game_ov, self._reward_cfg)()
        early_stop = EarlyStopTracker(patience=50, min_delta=1.0, verbose=1)

        obs_all = []
        ep_reward_all = [0.0] * n_envs
        for env in envs:
            o, _ = env.reset()
            obs_all.append(o)
        episode_rewards = []
        best_eval = float("-inf")

        writer = SummaryWriter(os.path.join(self.save_path, "tb_logs"))
        eval_timesteps, eval_results = [], []
        loss_acc = {"q_loss": [], "policy_loss": [], "alpha": [], "entropy": []}

        for step in range(1, self._timesteps + 1):
            env_idx = (step - 1) % n_envs
            env = envs[env_idx]

            if step < learning_starts:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    obs_t = torch.FloatTensor(obs_all[env_idx]).unsqueeze(0).to(self.device)
                    action = self.policy.get_action(obs_t).item()

            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            replay.push(obs_all[env_idx], action, reward, next_obs, float(done))
            obs_all[env_idx] = next_obs
            ep_reward_all[env_idx] += reward

            if done:
                episode_rewards.append(ep_reward_all[env_idx])
                writer.add_scalar("rollout/ep_reward", ep_reward_all[env_idx], step)
                summary = info.get("episode_summary")
                if summary:
                    writer.add_scalar("rollout/served", summary.get("customers_served", 0), step)
                    writer.add_scalar("rollout/lost", summary.get("customers_lost", 0), step)
                if len(episode_rewards) % 10 == 0:
                    print(f"  [DiscreteSAC] 스텝 {step}, 에피소드 {len(episode_rewards)}, "
                          f"평균보상(최근10): {np.mean(episode_rewards[-10:]):.1f}")
                obs_all[env_idx], _ = env.reset()
                ep_reward_all[env_idx] = 0.0

            # Train
            if step >= learning_starts and step % train_freq == 0:
                for _ in range(gradient_steps):
                    if len(replay) >= batch_size:
                        metrics = self._update(replay, batch_size, gamma, tau)
                        for k, v in metrics.items():
                            loss_acc[k].append(v)
                if step % 1000 == 0 and loss_acc["q_loss"]:
                    for k, vals in loss_acc.items():
                        writer.add_scalar(f"train/{k}", np.mean(vals), step)
                    loss_acc = {k: [] for k in loss_acc}

            # Eval
            if step % eval_freq == 0:
                eval_r = self._evaluate(eval_env, n_episodes=5)
                print(f"  [DiscreteSAC] 평가 스텝 {step}: mean_reward={eval_r:.1f}")
                writer.add_scalar("eval/mean_reward", eval_r, step)
                eval_timesteps.append(step)
                eval_results.append(eval_r)
                if eval_r > best_eval:
                    best_eval = eval_r
                    self.save(os.path.join(self.save_path, "best_model"))
                if not early_stop.check(eval_r):
                    print(f"  [DiscreteSAC] 조기 종료, 스텝: {step}")
                    break

        self.save(os.path.join(self.save_path, "final_model"))
        eval_log_dir = os.path.join(self.save_path, "eval_logs")
        os.makedirs(eval_log_dir, exist_ok=True)
        np.savez(os.path.join(eval_log_dir, "evaluations.npz"),
                 timesteps=np.array(eval_timesteps),
                 results=np.array(eval_results))
        writer.close()
        for env in envs:
            env.close()
        eval_env.close()
        print(f"[✓] DiscreteSAC 학습 완료. 모델 → '{self.save_path}/'")
        return {"algorithm": "DiscreteSAC", "timesteps": self._timesteps,
                "episodes": len(episode_rewards), "save_path": self.save_path}

    def _update(self, replay, batch_size, gamma, tau):
        obs_b, act_b, rew_b, nobs_b, done_b = replay.sample(batch_size)
        obs_t = torch.FloatTensor(obs_b).to(self.device)
        act_t = torch.LongTensor(act_b).to(self.device)
        rew_t = torch.FloatTensor(rew_b).to(self.device)
        nobs_t = torch.FloatTensor(nobs_b).to(self.device)
        done_t = torch.FloatTensor(done_b).to(self.device)

        alpha = self._log_alpha.exp().detach()

        # ── Q target (quantile, drop top-K) ──
        with torch.no_grad():
            next_probs = self.policy(nobs_t)  # [B, A]
            next_log = torch.log(next_probs + 1e-8)

            # 각 target network의 quantile 수집
            all_q_next = []  # list of [B, A, N]
            for qt in self.q_targets:
                all_q_next.append(qt(nobs_t))

            # 모든 critic의 quantile을 합침 → [B, A, n_critics*N]
            cat_q = torch.cat(all_q_next, dim=2)
            # 상위 quantile 드롭 (과대추정 억제)
            sorted_q, _ = torch.sort(cat_q, dim=2)
            n_keep = cat_q.shape[2] - self._top_drop
            truncated_q = sorted_q[:, :, :n_keep]  # [B, A, n_keep]
            q_mean = truncated_q.mean(dim=2)  # [B, A]

            v_next = (next_probs * (q_mean - alpha * next_log)).sum(dim=-1)
            q_target_scalar = rew_t + gamma * (1.0 - done_t) * v_next  # [B]

        # ── Quantile Huber loss ──
        total_q_loss = torch.tensor(0.0, device=self.device)
        taus = (torch.arange(self._n_quantiles, device=self.device, dtype=torch.float32) + 0.5
                ) / self._n_quantiles  # [N]
        target_expanded = q_target_scalar.unsqueeze(-1).expand(-1, self._n_quantiles)  # [B, N]

        for qn in self.q_nets:
            q_all = qn(obs_t)  # [B, A, N]
            q_a = q_all.gather(1, act_t.view(-1, 1, 1).expand(-1, 1, self._n_quantiles)).squeeze(1)  # [B, N]

            td = target_expanded - q_a  # [B, N]
            huber = torch.where(td.abs() < 1.0, 0.5 * td.pow(2), td.abs() - 0.5)
            quantile_loss = (taus.unsqueeze(0) - (td < 0).float()).abs() * huber
            total_q_loss = total_q_loss + quantile_loss.mean()

        self._q_opt.zero_grad()
        total_q_loss.backward()
        for qn in self.q_nets:
            nn.utils.clip_grad_norm_(qn.parameters(), self._max_grad_norm)
        self._q_opt.step()

        # ── Policy loss ──
        probs = self.policy(obs_t)
        log_probs = torch.log(probs + 1e-8)
        with torch.no_grad():
            q_vals = []
            for qn in self.q_nets:
                q_vals.append(qn(obs_t).mean(dim=2))  # [B, A]
            q_pi = torch.stack(q_vals).min(dim=0).values

        policy_loss = (probs * (alpha * log_probs - q_pi)).sum(dim=-1).mean()

        self._policy_opt.zero_grad()
        policy_loss.backward()
        nn.utils.clip_grad_norm_(self.policy.parameters(), self._max_grad_norm)
        self._policy_opt.step()

        # ── Alpha loss ──
        entropy = -(probs.detach() * log_probs.detach()).sum(dim=-1)
        alpha_loss = -(self._log_alpha * (self._target_entropy - entropy)).mean()
        self._alpha_opt.zero_grad()
        alpha_loss.backward()
        self._alpha_opt.step()

        # ── Soft update targets ──
        for qt, qn in zip(self.q_targets, self.q_nets):
            for tp, sp in zip(qt.parameters(), qn.parameters()):
                tp.data.copy_(tau * sp.data + (1.0 - tau) * tp.data)

        return {
            "q_loss": total_q_loss.item(),
            "policy_loss": policy_loss.item(),
            "alpha": alpha.item(),
            "entropy": entropy.mean().item(),
        }

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
        data = {
            "policy": self.policy.state_dict(),
            "log_alpha": self._log_alpha.data,
        }
        for i, (qn, qt) in enumerate(zip(self.q_nets, self.q_targets)):
            data[f"q_net_{i}"] = qn.state_dict()
            data[f"q_target_{i}"] = qt.state_dict()
        torch.save(data, path + ".pt")

    def load(self, path: str) -> None:
        ckpt = torch.load(path + ".pt", map_location=self.device)
        if self.policy:
            self.policy.load_state_dict(ckpt["policy"])
        for i, (qn, qt) in enumerate(zip(self.q_nets, self.q_targets)):
            qn.load_state_dict(ckpt[f"q_net_{i}"])
            qt.load_state_dict(ckpt[f"q_target_{i}"])
        self._log_alpha.data = ckpt["log_alpha"]

    def predict(self, obs, deterministic: bool = True) -> int:
        assert self.policy is not None
        obs_t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        action = self.policy.get_action(obs_t, deterministic=deterministic)
        return int(action.item())
