"""Model-Based RL Trainer – World Model + MPC(Model Predictive Control).

핵심 아이디어:
1. 실제 환경 데이터로 World Model(전이/보상 예측)을 학습
2. World Model 안에서 시뮬레이션(Imagination)하여 추가 학습 데이터 생성
3. Dyna 스타일: 실제 데이터 + 상상 데이터를 섞어 정책 학습
4. 선택적으로 MPC(Model Predictive Control)로 행동 선택 가능

장점: 데이터 효율성이 높음 (적은 환경 상호작용으로 학습)
"""

import os
from typing import Any
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.tensorboard import SummaryWriter

from algorithms.base import BaseTrainer
from algorithms.common import load_algo_config, make_env, save_run_config, EarlyStopTracker
from algorithms.model_based.world_model import WorldModel, WorldModelTrainer


# ────────────────────────────────────────────────
# Policy Network (Actor-Critic for Model-Based)
# ────────────────────────────────────────────────
class PolicyNet(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int,
                 hidden: list[int] | None = None):
        super().__init__()
        hidden = hidden or [128, 128]
        layers = []
        prev = obs_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        self.shared = nn.Sequential(*layers)
        self.actor = nn.Linear(prev, act_dim)
        self.critic = nn.Linear(prev, 1)

    def forward(self, x):
        feat = self.shared(x)
        logits = self.actor(feat)
        value = self.critic(feat)
        return logits, value

    def get_action(self, obs, deterministic=False):
        logits, value = self(obs)
        probs = F.softmax(logits, dim=-1)
        if deterministic:
            return probs.argmax(dim=-1), value
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        return action, value, dist.log_prob(action), dist.entropy()


# ────────────────────────────────────────────────
# Replay Buffer
# ────────────────────────────────────────────────
class TransitionBuffer:
    def __init__(self, capacity: int = 100_000):
        self.buffer = deque(maxlen=capacity)

    def push(self, obs, action, reward, next_obs, done):
        self.buffer.append((obs, action, reward, next_obs, done))

    def sample(self, n: int):
        idxs = np.random.choice(len(self.buffer), min(n, len(self.buffer)),
                                replace=False)
        batch = [self.buffer[i] for i in idxs]
        obs, act, rew, nobs, done = zip(*batch)
        return (np.array(obs, dtype=np.float32),
                np.array(act, dtype=np.float32),
                np.array(rew, dtype=np.float32),
                np.array(nobs, dtype=np.float32),
                np.array(done, dtype=np.float32))

    def __len__(self):
        return len(self.buffer)


# ────────────────────────────────────────────────
# MPC Planner
# ────────────────────────────────────────────────
class MPCPlanner:
    """Model Predictive Control – World Model을 사용하여 최적 행동을 탐색합니다."""

    def __init__(self, world_model: WorldModel, act_dim: int,
                 horizon: int = 10, n_trajectories: int = 100,
                 gamma: float = 0.99, device: str = "cpu"):
        self.world_model = world_model
        self.act_dim = act_dim
        self.horizon = horizon
        self.n_traj = n_trajectories
        self.gamma = gamma
        self.device = device

    def plan(self, obs: np.ndarray) -> int:
        """현재 관측에서 최적 첫 번째 행동을 반환합니다."""
        obs_t = torch.FloatTensor(obs).unsqueeze(0).repeat(
            self.n_traj, 1).to(self.device)

        total_rewards = torch.zeros(self.n_traj, device=self.device)
        first_actions = torch.randint(0, self.act_dim, (self.n_traj,),
                                      device=self.device)

        current_obs = obs_t
        discount = 1.0

        for t in range(self.horizon):
            if t == 0:
                actions = first_actions.float()
            else:
                actions = torch.randint(
                    0, self.act_dim, (self.n_traj,),
                    device=self.device).float()

            with torch.no_grad():
                next_obs, reward, done = self.world_model.predict_mean(
                    current_obs, actions)

            total_rewards += discount * reward
            discount *= self.gamma
            current_obs = next_obs

            # 종료된 궤적은 더 이상 보상 누적하지 않음
            alive_mask = (done < 0.5).float()
            discount *= alive_mask.mean().item()

        # 각 첫 번째 행동별 평균 보상 계산
        best_returns = torch.zeros(self.act_dim, device=self.device)
        counts = torch.zeros(self.act_dim, device=self.device)
        for a in range(self.act_dim):
            mask = (first_actions == a)
            if mask.sum() > 0:
                best_returns[a] = total_rewards[mask].mean()
                counts[a] = mask.sum()

        return int(best_returns.argmax().item())


# ────────────────────────────────────────────────
# Model-Based RL Trainer
# ────────────────────────────────────────────────
class ModelBasedTrainer(BaseTrainer):
    name = "ModelBased"

    def __init__(self):
        self.world_model: WorldModel | None = None
        self.policy: PolicyNet | None = None
        self.cfg: dict = {}
        self.save_path: str = "models/model_based"
        self._timesteps: int = 200_000
        self._obs_dim: int = 0
        self._act_dim: int = 0
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def build(self, cfg: dict | None = None, config_path: str | None = None,
              save_path: str | None = None, **overrides) -> None:
        days = overrides.pop("days", None)
        self.cfg = cfg or load_algo_config("model_based", config_path, days=days)
        t = self.cfg.get("training", {})
        hp = self.cfg.get("hyperparameters", {})
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

        # World Model
        wm_hidden = hp.get("world_model_hidden", [256, 256])
        self.world_model = WorldModel(
            self._obs_dim, self._act_dim, wm_hidden
        ).to(self.device)
        self._wm_trainer = WorldModelTrainer(
            self.world_model,
            lr=hp.get("world_model_lr", 1e-3),
            device=str(self.device),
        )

        # Policy
        policy_hidden = hp.get("policy_hidden", [128, 128])
        self.policy = PolicyNet(
            self._obs_dim, self._act_dim, policy_hidden
        ).to(self.device)
        self._policy_opt = torch.optim.Adam(
            self.policy.parameters(), lr=hp.get("learning_rate", 3e-4))

        # MPC Planner
        self._planner = MPCPlanner(
            self.world_model, self._act_dim,
            horizon=t.get("planning_horizon", 10),
            n_trajectories=t.get("num_simulated_trajectories", 100),
            gamma=hp.get("gamma", 0.99),
            device=str(self.device),
        )

        self._hp = hp
        self._t_cfg = t
        self._game_ov = game_ov
        self._reward_cfg = reward_cfg
        self._seed = seed

    def train(self, resume_path: str | None = None) -> dict[str, Any]:
        assert self.world_model is not None, "call build() first"
        t = self._t_cfg
        hp = self._hp
        gamma = hp.get("gamma", 0.99)

        wm_train_freq = t.get("world_model_train_freq", 1000)
        wm_epochs = t.get("world_model_epochs", 10)
        wm_batch = t.get("world_model_batch_size", 128)
        real_ratio = t.get("real_data_ratio", 0.5)
        eval_freq = t.get("eval_freq", 5000)
        buffer_size = t.get("buffer_size", 100_000)

        replay = TransitionBuffer(buffer_size)
        env = make_env(0, self._seed, self._game_ov, self._reward_cfg)()
        eval_env = make_env(0, self._seed + 1000, self._game_ov,
                            self._reward_cfg)()

        obs, _ = env.reset()
        ep_rewards = []
        ep_reward = 0.0
        best_score = float("-inf")
        use_mpc = True  # 초반엔 MPC, 후반엔 정책 네트워크 사용
        early_stop = EarlyStopTracker(patience=50, min_delta=1.0, verbose=1,
                          metric_name="mean_final_score")

        # TensorBoard + 평가 기록
        writer = SummaryWriter(os.path.join(self.save_path, "tb_logs"))
        eval_timesteps, eval_results, eval_final_scores = [], [], []

        for step in range(1, self._timesteps + 1):
            # 행동 선택: MPC 또는 Policy
            if use_mpc and len(replay) >= wm_batch:
                action = self._planner.plan(obs)
            elif len(replay) < 1000:
                action = env.action_space.sample()
            else:
                obs_t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    action, _ = self.policy.get_action(obs_t, deterministic=False)
                    action = action.item()

            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            replay.push(obs, float(action), reward, next_obs, float(done))
            obs = next_obs
            ep_reward += reward

            if done:
                ep_rewards.append(ep_reward)
                if len(ep_rewards) % 10 == 0:
                    print(f"  [ModelBased (모델기반)] 스텝(Step) {step}, 에피소드(Episodes) {len(ep_rewards)}, "
                          f"평균보상(AvgReward, 최근10): {np.mean(ep_rewards[-10:]):.1f}")
                writer.add_scalar("rollout/ep_reward", ep_reward, step)
                summary = info.get("episode_summary")
                if summary:
                    writer.add_scalar("rollout/served", summary.get("customers_served", 0), step)
                    writer.add_scalar("rollout/lost", summary.get("customers_lost", 0), step)
                    writer.add_scalar("rollout/profit", summary.get("net_profit", 0), step)
                    writer.add_scalar("rollout/rating", summary.get("shop_rating", 0), step)
                    writer.add_scalar("rollout/final_score", summary.get("final_score", 0), step)
                obs, _ = env.reset()
                ep_reward = 0.0

            # World Model 학습
            if step % wm_train_freq == 0 and len(replay) >= wm_batch:
                wm_losses = []
                for _ in range(wm_epochs):
                    o, a, r, no, d = replay.sample(wm_batch)
                    loss = self._wm_trainer.train_step(o, a, no, r, d)
                    wm_losses.append(loss["total_loss"])
                avg_wm_loss = np.mean(wm_losses)
                print(f"  [WorldModel (월드모델)] 평균 손실(Avg loss): {avg_wm_loss:.4f}")
                writer.add_scalar("train/world_model_loss", avg_wm_loss, step)

                # 상상 데이터 생성 → 정책 학습 (Dyna-style)
                self._train_policy_from_imagination(replay, wm_batch, gamma)

                # 충분히 학습되면 MPC → Policy 전환
                if step > self._timesteps * 0.3:
                    use_mpc = False

            # 정책 학습 (실제 데이터)
            if step % 256 == 0 and len(replay) >= wm_batch:
                self._train_policy_on_real(replay, wm_batch, gamma)

            # 평가
            if step % eval_freq == 0:
                eval_r, eval_score = self._evaluate(eval_env)
                print(f"  [ModelBased] 평가(Eval) 스텝 {step}: "
                      f"평균보상(mean_reward)={eval_r:.1f}, 평균최종점수(mean_final_score)={eval_score:.1f}")
                writer.add_scalar("eval/mean_reward", eval_r, step)
                writer.add_scalar("eval/mean_final_score", eval_score, step)
                eval_timesteps.append(step)
                eval_results.append(eval_r)
                eval_final_scores.append(eval_score)
                if eval_score > best_score:
                    best_score = eval_score
                    self.save(os.path.join(self.save_path, "best_model"))
                if not early_stop.check(eval_score):
                    print(f"  [ModelBased] 조기 종료 (Early stopped), 스텝: {step}")
                    break

        self.save(os.path.join(self.save_path, "final_model"))
        eval_log_dir = os.path.join(self.save_path, "eval_logs")
        os.makedirs(eval_log_dir, exist_ok=True)
        np.savez(os.path.join(eval_log_dir, "evaluations.npz"),
                 timesteps=np.array(eval_timesteps),
                 results=np.array(eval_results),
                 final_scores=np.array(eval_final_scores))
        writer.close()
        env.close()
        eval_env.close()
        print(f"[✓] Model-Based RL (모델기반 강화학습) 학습 완료. "
              f"모델 → '{self.save_path}/'")
        return {"algorithm": "ModelBased", "timesteps": self._timesteps,
                "episodes": len(ep_rewards), "save_path": self.save_path}

    def _train_policy_from_imagination(self, replay: TransitionBuffer,
                                       batch_size: int, gamma: float):
        """World Model 내부에서 상상 궤적을 생성하여 정책을 학습합니다."""
        obs_b, _, _, _, _ = replay.sample(batch_size)
        obs_t = torch.FloatTensor(obs_b).to(self.device)

        total_policy_loss = 0.0
        horizon = min(5, self._t_cfg.get("planning_horizon", 10))

        current = obs_t
        discount = 1.0
        for _ in range(horizon):
            logits, values = self.policy(current)
            probs = F.softmax(logits, dim=-1)
            dist = torch.distributions.Categorical(probs)
            actions = dist.sample()

            with torch.no_grad():
                next_obs, rewards, dones = self.world_model.predict_mean(
                    current, actions.float())

            # PPO-style advantage (simplified)
            _, next_values = self.policy(next_obs)
            advantages = rewards + gamma * next_values.squeeze(-1) * (
                1.0 - dones) - values.squeeze(-1)

            policy_loss = -(dist.log_prob(actions) * advantages.detach()).mean()
            value_loss = F.mse_loss(values.squeeze(-1),
                                    (rewards + gamma * next_values.squeeze(-1).detach()))
            entropy = dist.entropy().mean()

            loss = policy_loss + 0.5 * value_loss - 0.01 * entropy
            total_policy_loss += discount * loss.item()

            self._policy_opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
            self._policy_opt.step()

            current = next_obs.detach()
            discount *= gamma

    def _train_policy_on_real(self, replay: TransitionBuffer,
                              batch_size: int, gamma: float):
        """실제 경험 데이터로 정책을 학습합니다."""
        obs_b, act_b, rew_b, nobs_b, done_b = replay.sample(batch_size)
        obs_t = torch.FloatTensor(obs_b).to(self.device)
        rew_t = torch.FloatTensor(rew_b).to(self.device)
        nobs_t = torch.FloatTensor(nobs_b).to(self.device)
        done_t = torch.FloatTensor(done_b).to(self.device)
        act_t = torch.LongTensor(act_b.astype(int)).to(self.device)

        logits, values = self.policy(obs_t)
        _, next_values = self.policy(nobs_t)

        targets = rew_t + gamma * next_values.squeeze(-1).detach() * (1 - done_t)
        advantages = targets - values.squeeze(-1).detach()

        probs = F.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        log_probs = dist.log_prob(act_t)

        policy_loss = -(log_probs * advantages).mean()
        value_loss = F.mse_loss(values.squeeze(-1), targets)
        entropy = dist.entropy().mean()

        loss = policy_loss + 0.5 * value_loss - 0.01 * entropy

        self._policy_opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
        self._policy_opt.step()

    def _evaluate(self, env, n_episodes: int = 5) -> tuple[float, float]:
        total = 0.0
        total_score = 0.0
        for _ in range(n_episodes):
            obs, _ = env.reset()
            done = False
            ep_info = {}
            while not done:
                obs_t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    action, _ = self.policy.get_action(obs_t, deterministic=False)
                obs, r, terminated, truncated, info = env.step(action.item())
                total += r
                done = terminated or truncated
                if done:
                    ep_info = info
            summary = ep_info.get("episode_summary", {})
            total_score += summary.get("final_score", 0.0)
        return total / n_episodes, total_score / n_episodes

    def save(self, path: str) -> None:
        torch.save({
            "world_model": self.world_model.state_dict() if self.world_model else None,
            "policy": self.policy.state_dict() if self.policy else None,
        }, path + ".pt")

    def load(self, path: str) -> None:
        ckpt = torch.load(path + ".pt", map_location=self.device)
        if self.world_model and ckpt.get("world_model"):
            self.world_model.load_state_dict(ckpt["world_model"])
        if self.policy and ckpt.get("policy"):
            self.policy.load_state_dict(ckpt["policy"])

    def predict(self, obs, deterministic: bool = True) -> int:
        assert self.policy is not None
        obs_t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        with torch.no_grad():
            action, _ = self.policy.get_action(obs_t, deterministic=deterministic)
        return int(action.item())
