"""DreamerV3 Trainer – RSSM 세계 모델 + 상상 기반 Actor-Critic.

학습 흐름:
1. 환경에서 데이터 수집 → 시퀀스 버퍼에 저장
2. 시퀀스 샘플링 → RSSM 세계 모델 학습 (관측/보상/종료 예측)
3. 세계 모델 안에서 상상 궤적 생성 → Actor-Critic 학습
4. 학습된 Actor로 환경에서 행동 선택
"""

import os
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from algorithms.base import BaseTrainer
from algorithms.common import load_algo_config, make_env, save_run_config, EarlyStopTracker
from algorithms.dreamer.networks import (
    RSSM, RewardPredictor, ContinuePredictor, ObsDecoder, Actor, Critic,
)


# ────────────────────────────────────────────────
# Sequence Replay Buffer
# ────────────────────────────────────────────────
class SequenceBuffer:
    """에피소드를 시퀀스 단위로 저장하는 버퍼."""

    def __init__(self, obs_dim: int, capacity: int = 500_000):
        self.capacity = capacity
        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)
        self.pos = 0
        self.size = 0

    def push(self, obs, action, reward, done):
        self.obs[self.pos] = obs
        self.actions[self.pos] = action
        self.rewards[self.pos] = reward
        self.dones[self.pos] = done
        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample_sequences(self, batch_size: int, seq_len: int):
        """랜덤 시작점에서 seq_len 길이의 시퀀스를 batch_size개 샘플링."""
        max_start = self.size - seq_len
        if max_start <= 0:
            return None
        starts = np.random.randint(0, max_start, size=batch_size)
        idxs = starts[:, None] + np.arange(seq_len)[None, :]  # [B, T]

        return {
            "obs": self.obs[idxs],          # [B, T, obs_dim]
            "actions": self.actions[idxs],   # [B, T]
            "rewards": self.rewards[idxs],   # [B, T]
            "dones": self.dones[idxs],       # [B, T]
        }

    def __len__(self):
        return self.size


# ────────────────────────────────────────────────
# DreamerV3 Trainer
# ────────────────────────────────────────────────
class DreamerTrainer(BaseTrainer):
    name = "Dreamer"

    def __init__(self):
        self.rssm: RSSM | None = None
        self.actor: Actor | None = None
        self.critic: Critic | None = None
        self.cfg: dict = {}
        self.save_path: str = "models/dreamer"
        self._timesteps: int = 200_000
        self._obs_dim: int = 0
        self._act_dim: int = 0
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def build(self, cfg: dict | None = None, config_path: str | None = None,
              save_path: str | None = None, **overrides) -> None:
        days = overrides.pop("days", None)
        self.cfg = cfg or load_algo_config("dreamer", config_path, days=days)
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

        tmp = make_env(0, seed, game_ov, reward_cfg)()
        self._obs_dim = tmp.observation_space.shape[0]
        self._act_dim = tmp.action_space.n
        tmp.close()

        rssm_h = hp.get("rssm_hidden", 256)
        rssm_s = hp.get("rssm_stochastic", 32)
        rssm_c = hp.get("rssm_discrete_classes", 32)

        self.rssm = RSSM(self._obs_dim, self._act_dim,
                         rssm_h, rssm_s, rssm_c).to(self.device)
        feat_dim = self.rssm.feature_dim

        self.reward_pred = RewardPredictor(feat_dim).to(self.device)
        self.continue_pred = ContinuePredictor(feat_dim).to(self.device)
        self.obs_decoder = ObsDecoder(feat_dim, self._obs_dim).to(self.device)
        self.actor = Actor(feat_dim, self._act_dim).to(self.device)
        self.critic = Critic(feat_dim).to(self.device)

        lr_w = hp.get("learning_rate_world", 3e-4)
        lr_a = hp.get("learning_rate_actor", 1e-4)
        lr_c = hp.get("learning_rate_critic", 3e-4)

        world_params = (list(self.rssm.parameters()) +
                        list(self.reward_pred.parameters()) +
                        list(self.continue_pred.parameters()) +
                        list(self.obs_decoder.parameters()))
        self._world_opt = torch.optim.Adam(world_params, lr=lr_w)
        self._actor_opt = torch.optim.Adam(self.actor.parameters(), lr=lr_a)
        self._critic_opt = torch.optim.Adam(self.critic.parameters(), lr=lr_c)

        self._hp = hp
        self._game_ov = game_ov
        self._reward_cfg = reward_cfg
        self._seed = seed

    def train(self) -> dict[str, Any]:
        assert self.rssm is not None, "call build() first"
        hp = self._hp
        gamma = hp.get("gamma", 0.997)
        lam = hp.get("lambda_", 0.95)
        batch_size = hp.get("batch_size", 64)
        seq_len = hp.get("seq_len", 32)
        buffer_size = hp.get("buffer_size", 500_000)
        wm_train_freq = hp.get("world_model_train_freq", 100)
        imagine_horizon = hp.get("imagination_horizon", 15)
        entropy_coef = hp.get("entropy_coef", 0.003)
        max_grad_norm = hp.get("max_grad_norm", 100.0)
        free_nats = hp.get("free_nats", 1.0)
        eval_freq = self.cfg.get("training", {}).get("eval_freq", 5000)

        replay = SequenceBuffer(self._obs_dim, buffer_size)
        env = make_env(0, self._seed, self._game_ov, self._reward_cfg)()
        eval_env = make_env(0, self._seed + 1000, self._game_ov, self._reward_cfg)()
        early_stop = EarlyStopTracker(patience=50, min_delta=1.0, verbose=1)

        writer = SummaryWriter(os.path.join(self.save_path, "tb_logs"))
        eval_timesteps, eval_results = [], []

        obs, _ = env.reset()
        episode_rewards, ep_reward = [], 0.0
        best_eval = float("-inf")

        # RSSM running state for action selection
        h, z = self.rssm.initial_state(1, self.device)
        prev_action = torch.zeros(1, self._act_dim, device=self.device)

        for step in range(1, self._timesteps + 1):
            # Select action
            with torch.no_grad():
                obs_t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
                h, z, _, _ = self.rssm.observe_step(h, z, prev_action, obs_t)
                feat = self.rssm.get_feature(h, z)
                action_dist = self.actor.get_dist(feat)
                action = action_dist.sample().item()

            prev_action = F.one_hot(torch.tensor([action], device=self.device),
                                    self._act_dim).float()

            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            replay.push(obs, action, reward, float(done))
            obs = next_obs
            ep_reward += reward

            if done:
                episode_rewards.append(ep_reward)
                writer.add_scalar("rollout/ep_reward", ep_reward, step)
                summary = info.get("episode_summary")
                if summary:
                    writer.add_scalar("rollout/served", summary.get("customers_served", 0), step)
                    writer.add_scalar("rollout/lost", summary.get("customers_lost", 0), step)
                if len(episode_rewards) % 10 == 0:
                    print(f"  [Dreamer] 스텝 {step}, 에피소드 {len(episode_rewards)}, "
                          f"평균보상(최근10): {np.mean(episode_rewards[-10:]):.1f}")
                obs, _ = env.reset()
                ep_reward = 0.0
                h, z = self.rssm.initial_state(1, self.device)
                prev_action = torch.zeros(1, self._act_dim, device=self.device)

            # Train world model + actor-critic
            if step % wm_train_freq == 0 and len(replay) > batch_size * seq_len:
                metrics = self._train_step(
                    replay, batch_size, seq_len, gamma, lam,
                    imagine_horizon, entropy_coef, max_grad_norm, free_nats)
                for k, v in metrics.items():
                    writer.add_scalar(f"train/{k}", v, step)

            # Eval
            if step % eval_freq == 0:
                eval_r = self._evaluate(eval_env)
                print(f"  [Dreamer] 평가 스텝 {step}: mean_reward={eval_r:.1f}")
                writer.add_scalar("eval/mean_reward", eval_r, step)
                eval_timesteps.append(step)
                eval_results.append(eval_r)
                if eval_r > best_eval:
                    best_eval = eval_r
                    self.save(os.path.join(self.save_path, "best_model"))
                if not early_stop.check(eval_r):
                    print(f"  [Dreamer] 조기 종료, 스텝: {step}")
                    break

        self.save(os.path.join(self.save_path, "final_model"))
        eval_log_dir = os.path.join(self.save_path, "eval_logs")
        os.makedirs(eval_log_dir, exist_ok=True)
        np.savez(os.path.join(eval_log_dir, "evaluations.npz"),
                 timesteps=np.array(eval_timesteps),
                 results=np.array(eval_results))
        writer.close()
        env.close()
        eval_env.close()
        print(f"[✓] Dreamer 학습 완료. 모델 → '{self.save_path}/'")
        return {"algorithm": "Dreamer", "timesteps": self._timesteps,
                "episodes": len(episode_rewards), "save_path": self.save_path}

    def _train_step(self, replay, batch_size, seq_len, gamma, lam,
                    imagine_horizon, entropy_coef, max_grad_norm, free_nats):
        """세계 모델 + Actor-Critic 한 스텝 학습."""
        batch = replay.sample_sequences(batch_size, seq_len)
        if batch is None:
            return {}

        obs_seq = torch.FloatTensor(batch["obs"]).to(self.device)       # [B, T, O]
        act_seq = torch.LongTensor(batch["actions"]).to(self.device)    # [B, T]
        rew_seq = torch.FloatTensor(batch["rewards"]).to(self.device)   # [B, T]
        done_seq = torch.FloatTensor(batch["dones"]).to(self.device)    # [B, T]

        act_onehot = F.one_hot(act_seq, self._act_dim).float()  # [B, T, A]

        # ── 1) World Model: RSSM forward through sequence ──
        h, z = self.rssm.initial_state(batch_size, self.device)
        priors, posteriors, features_list = [], [], []

        for t in range(seq_len):
            a_prev = act_onehot[:, t - 1] if t > 0 else torch.zeros(batch_size, self._act_dim, device=self.device)
            h, z, prior_logits, post_logits = self.rssm.observe_step(
                h, z, a_prev, obs_seq[:, t])
            priors.append(prior_logits)
            posteriors.append(post_logits)
            features_list.append(self.rssm.get_feature(h, z))

        features = torch.stack(features_list, dim=1)  # [B, T, feat]
        priors = torch.stack(priors, dim=1)            # [B, T, stoch, classes]
        posteriors = torch.stack(posteriors, dim=1)

        # Losses
        # Observation reconstruction
        obs_pred = self.obs_decoder(features)
        obs_loss = F.mse_loss(obs_pred, obs_seq)

        # Reward prediction
        rew_pred = self.reward_pred(features)
        rew_loss = F.mse_loss(rew_pred, rew_seq)

        # Continue prediction
        cont_pred = self.continue_pred(features)
        cont_target = 1.0 - done_seq
        cont_loss = F.binary_cross_entropy_with_logits(cont_pred, cont_target)

        # KL divergence (free nats)
        prior_probs = F.softmax(priors, dim=-1)
        post_probs = F.softmax(posteriors, dim=-1)
        kl = (post_probs * (torch.log(post_probs + 1e-8) - torch.log(prior_probs + 1e-8))).sum(dim=-1)
        kl = kl.sum(dim=-1)  # sum over stoch categories
        kl = torch.clamp(kl.mean(), min=free_nats)

        world_loss = obs_loss + rew_loss + cont_loss + kl

        self._world_opt.zero_grad()
        world_loss.backward()
        nn.utils.clip_grad_norm_(
            list(self.rssm.parameters()) + list(self.reward_pred.parameters()) +
            list(self.continue_pred.parameters()) + list(self.obs_decoder.parameters()),
            max_grad_norm)
        self._world_opt.step()

        # ── 2) Actor-Critic: Imagine trajectories ──
        with torch.no_grad():
            # Start imagination from last posterior state
            init_h = h.detach()
            init_z = z.detach()

        imagined_features = []
        imagined_rewards = []
        imagined_continues = []
        im_h, im_z = init_h, init_z

        for _ in range(imagine_horizon):
            feat = self.rssm.get_feature(im_h, im_z)
            imagined_features.append(feat)

            action_dist = self.actor.get_dist(feat)
            action = action_dist.sample()
            action_oh = F.one_hot(action, self._act_dim).float()

            im_h, im_z = self.rssm.imagine_step(im_h, im_z, action_oh)

            with torch.no_grad():
                r = self.reward_pred(self.rssm.get_feature(im_h, im_z))
                c = torch.sigmoid(self.continue_pred(self.rssm.get_feature(im_h, im_z)))
            imagined_rewards.append(r)
            imagined_continues.append(c)

        im_feats = torch.stack(imagined_features, dim=0)  # [H, B, feat]
        im_rews = torch.stack(imagined_rewards, dim=0)     # [H, B]
        im_conts = torch.stack(imagined_continues, dim=0)  # [H, B]

        # Compute lambda-returns
        with torch.no_grad():
            last_feat = self.rssm.get_feature(im_h, im_z)
            last_value = self.critic(last_feat)

        values = self.critic(im_feats.reshape(-1, im_feats.shape[-1])
                             ).reshape(imagine_horizon, batch_size)

        returns = torch.zeros_like(im_rews)
        last_return = last_value
        for t in reversed(range(imagine_horizon)):
            last_return = im_rews[t] + gamma * im_conts[t] * (
                (1 - lam) * values[t] + lam * last_return)
            returns[t] = last_return

        # ── Critic loss ──
        critic_loss = F.mse_loss(values, returns.detach())
        self._critic_opt.zero_grad()
        critic_loss.backward(retain_graph=True)
        nn.utils.clip_grad_norm_(self.critic.parameters(), max_grad_norm)
        self._critic_opt.step()

        # ── Actor loss (REINFORCE with baseline + entropy) ──
        advantages = (returns - values).detach()
        actor_loss = torch.tensor(0.0, device=self.device)
        total_entropy = torch.tensor(0.0, device=self.device)

        im_h2, im_z2 = init_h.detach(), init_z.detach()
        for t in range(imagine_horizon):
            feat = self.rssm.get_feature(im_h2, im_z2)
            dist = self.actor.get_dist(feat)
            action = dist.sample()

            log_prob = dist.log_prob(action)
            actor_loss = actor_loss - (log_prob * advantages[t]).mean()
            total_entropy = total_entropy + dist.entropy().mean()

            action_oh = F.one_hot(action, self._act_dim).float()
            im_h2, im_z2 = self.rssm.imagine_step(im_h2.detach(), im_z2.detach(), action_oh)

        actor_loss = actor_loss / imagine_horizon - entropy_coef * total_entropy / imagine_horizon

        self._actor_opt.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), max_grad_norm)
        self._actor_opt.step()

        return {
            "world_loss": world_loss.item(),
            "obs_loss": obs_loss.item(),
            "reward_loss": rew_loss.item(),
            "kl": kl.item(),
            "actor_loss": actor_loss.item(),
            "critic_loss": critic_loss.item(),
            "entropy": (total_entropy / imagine_horizon).item(),
        }

    def _evaluate(self, env, n_episodes: int = 5) -> float:
        total = 0.0
        for _ in range(n_episodes):
            obs, _ = env.reset()
            done = False
            h, z = self.rssm.initial_state(1, self.device)
            prev_action = torch.zeros(1, self._act_dim, device=self.device)
            while not done:
                with torch.no_grad():
                    obs_t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
                    h, z, _, _ = self.rssm.observe_step(h, z, prev_action, obs_t)
                    feat = self.rssm.get_feature(h, z)
                    probs = self.actor(feat)
                    action = probs.argmax(dim=-1).item()
                prev_action = F.one_hot(torch.tensor([action], device=self.device),
                                        self._act_dim).float()
                obs, r, terminated, truncated, _ = env.step(action)
                total += r
                done = terminated or truncated
        return total / n_episodes

    def save(self, path: str) -> None:
        torch.save({
            "rssm": self.rssm.state_dict(),
            "reward_pred": self.reward_pred.state_dict(),
            "continue_pred": self.continue_pred.state_dict(),
            "obs_decoder": self.obs_decoder.state_dict(),
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
        }, path + ".pt")

    def load(self, path: str) -> None:
        ckpt = torch.load(path + ".pt", map_location=self.device)
        if self.rssm:
            self.rssm.load_state_dict(ckpt["rssm"])
        if hasattr(self, "reward_pred"):
            self.reward_pred.load_state_dict(ckpt["reward_pred"])
        if hasattr(self, "continue_pred"):
            self.continue_pred.load_state_dict(ckpt["continue_pred"])
        if hasattr(self, "obs_decoder"):
            self.obs_decoder.load_state_dict(ckpt["obs_decoder"])
        if self.actor:
            self.actor.load_state_dict(ckpt["actor"])
        if self.critic:
            self.critic.load_state_dict(ckpt["critic"])

    def predict(self, obs, deterministic: bool = True) -> int:
        assert self.actor is not None and self.rssm is not None
        if not hasattr(self, "_running_h"):
            self._running_h, self._running_z = self.rssm.initial_state(1, self.device)
            self._running_action = torch.zeros(1, self._act_dim, device=self.device)

        obs_t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        with torch.no_grad():
            self._running_h, self._running_z, _, _ = self.rssm.observe_step(
                self._running_h, self._running_z, self._running_action, obs_t)
            feat = self.rssm.get_feature(self._running_h, self._running_z)
            probs = self.actor(feat)
            if deterministic:
                action = probs.argmax(dim=-1).item()
            else:
                action = torch.distributions.Categorical(probs).sample().item()
        self._running_action = F.one_hot(torch.tensor([action], device=self.device),
                                          self._act_dim).float()
        return action
