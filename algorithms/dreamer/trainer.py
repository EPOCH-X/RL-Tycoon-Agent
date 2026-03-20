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
def symlog(x: torch.Tensor) -> torch.Tensor:
    """DreamerV3 symlog 변환: sign(x) * ln(|x| + 1)"""
    return torch.sign(x) * torch.log(torch.abs(x) + 1.0)


def symexp(x: torch.Tensor) -> torch.Tensor:
    """symlog의 역변환."""
    return torch.sign(x) * (torch.exp(torch.abs(x)) - 1.0)


class SequenceBuffer:
    """에피소드를 시퀀스 단위로 저장하는 버퍼 (에피소드 경계 인식)."""

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
        """에피소드 경계를 넘지 않는 시퀀스를 batch_size개 샘플링."""
        max_start = self.size - seq_len
        if max_start <= 0:
            return None

        # 에피소드 경계를 포함하지 않는 유효 시작점 수집
        valid_starts = []
        for s in range(max_start):
            # 시퀀스 내부(마지막 스텝 제외)에 done이 없으면 유효
            if not np.any(self.dones[s:s + seq_len - 1] > 0.5):
                valid_starts.append(s)

        if len(valid_starts) < batch_size:
            # 유효 시작점 부족 시 일반 샘플링 (fallback)
            starts = np.random.randint(0, max_start, size=batch_size)
        else:
            starts = np.array(valid_starts)
            starts = starts[np.random.randint(0, len(starts), size=batch_size)]

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

    def train(self, resume_path: str | None = None) -> dict[str, Any]:
        assert self.rssm is not None, "call build() first"
        hp = self._hp
        gamma = hp.get("gamma", 0.997)
        lam = hp.get("lambda_", 0.95)
        batch_size = hp.get("batch_size", 64)
        seq_len = hp.get("seq_len", 32)
        buffer_size = hp.get("buffer_size", 500_000)
        wm_train_freq = hp.get("world_model_train_freq", 100)
        imagine_horizon = hp.get("imagination_horizon", 5)
        entropy_coef = hp.get("entropy_coef", 0.003)
        max_grad_norm = hp.get("max_grad_norm", 1.0)
        free_nats = hp.get("free_nats", 1.0)
        learning_starts = hp.get("learning_starts", 20000)
        n_envs = self.cfg.get("training", {}).get("n_envs", 1)
        eval_freq = self.cfg.get("training", {}).get("eval_freq", 5000)

        replay = SequenceBuffer(self._obs_dim, buffer_size)
        # 멀티-환경 지원
        envs = [make_env(i, self._seed + i, self._game_ov, self._reward_cfg)()
                for i in range(n_envs)]
        eval_env = make_env(0, self._seed + 1000, self._game_ov, self._reward_cfg)()
        early_stop = EarlyStopTracker(patience=50, min_delta=1.0, verbose=1,
                          metric_name="mean_final_score")

        writer = SummaryWriter(os.path.join(self.save_path, "tb_logs"))
        eval_timesteps, eval_results, eval_final_scores = [], [], []

        # 각 환경의 상태 초기화
        obs_list = []
        h_list, z_list, prev_action_list = [], [], []
        ep_reward_list = [0.0] * n_envs
        for i in range(n_envs):
            o, _ = envs[i].reset()
            obs_list.append(o)
            _h, _z = self.rssm.initial_state(1, self.device)
            h_list.append(_h)
            z_list.append(_z)
            prev_action_list.append(torch.zeros(1, self._act_dim, device=self.device))

        episode_rewards = []
        best_score = float("-inf")
        env_idx = 0  # round-robin 인덱스

        for step in range(1, self._timesteps + 1):
            # round-robin 환경 선택
            ei = env_idx % n_envs
            env_idx += 1
            obs = obs_list[ei]
            h, z = h_list[ei], z_list[ei]
            prev_action = prev_action_list[ei]

            # Select action (learning_starts 이전에는 랜덤)
            if step < learning_starts:
                action = np.random.randint(self._act_dim)
            else:
                with torch.no_grad():
                    obs_t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
                    h, z, _, _ = self.rssm.observe_step(h, z, prev_action, obs_t)
                    feat = self.rssm.get_feature(h, z)
                    action_dist = self.actor.get_dist(feat)
                    action = action_dist.sample().item()

            prev_action = F.one_hot(torch.tensor([action], device=self.device),
                                    self._act_dim).float()

            next_obs, reward, terminated, truncated, info = envs[ei].step(action)
            done = terminated or truncated
            replay.push(obs, action, reward, float(done))
            obs_list[ei] = next_obs
            h_list[ei], z_list[ei] = h, z
            prev_action_list[ei] = prev_action
            ep_reward_list[ei] += reward

            if done:
                episode_rewards.append(ep_reward_list[ei])
                writer.add_scalar("rollout/ep_reward", ep_reward_list[ei], step)
                summary = info.get("episode_summary")
                if summary:
                    writer.add_scalar("rollout/served", summary.get("customers_served", 0), step)
                    writer.add_scalar("rollout/lost", summary.get("customers_lost", 0), step)
                    writer.add_scalar("rollout/final_score", summary.get("final_score", 0), step)
                if len(episode_rewards) % 10 == 0:
                    print(f"  [Dreamer] 스텝 {step}, 에피소드 {len(episode_rewards)}, "
                          f"평균보상(최근10): {np.mean(episode_rewards[-10:]):.1f}")
                o, _ = envs[ei].reset()
                obs_list[ei] = o
                ep_reward_list[ei] = 0.0
                _h, _z = self.rssm.initial_state(1, self.device)
                h_list[ei], z_list[ei] = _h, _z
                prev_action_list[ei] = torch.zeros(1, self._act_dim, device=self.device)

            # Train world model + actor-critic (learning_starts 이후만)
            if (step >= learning_starts and step % wm_train_freq == 0
                    and len(replay) > batch_size * seq_len):
                metrics = self._train_step(
                    replay, batch_size, seq_len, gamma, lam,
                    imagine_horizon, entropy_coef, max_grad_norm, free_nats)
                for k, v in metrics.items():
                    writer.add_scalar(f"train/{k}", v, step)

            # Eval
            if step % eval_freq == 0:
                eval_r, eval_score = self._evaluate(eval_env)
                print(f"  [Dreamer] 평가 스텝 {step}: mean_reward={eval_r:.1f}, mean_final_score={eval_score:.1f}")
                writer.add_scalar("eval/mean_reward", eval_r, step)
                writer.add_scalar("eval/mean_final_score", eval_score, step)
                eval_timesteps.append(step)
                eval_results.append(eval_r)
                eval_final_scores.append(eval_score)
                if eval_score > best_score:
                    best_score = eval_score
                    self.save(os.path.join(self.save_path, "best_model"))
                if not early_stop.check(eval_score):
                    print(f"  [Dreamer] 조기 종료, 스텝: {step}")
                    break

        self.save(os.path.join(self.save_path, "final_model"))
        eval_log_dir = os.path.join(self.save_path, "eval_logs")
        os.makedirs(eval_log_dir, exist_ok=True)
        np.savez(os.path.join(eval_log_dir, "evaluations.npz"),
                 timesteps=np.array(eval_timesteps),
                 results=np.array(eval_results),
                 final_scores=np.array(eval_final_scores))
        writer.close()
        for e in envs:
            e.close()
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

        # Reward prediction (symlog 변환)
        rew_pred = self.reward_pred(features)
        rew_target = symlog(rew_seq)
        rew_loss = F.mse_loss(rew_pred, rew_target)

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

        # ── 2) Actor-Critic: Imagine trajectories (단일 루프 — advantage-action 일치) ──
        with torch.no_grad():
            init_h = h.detach()
            init_z = z.detach()

        imagined_features = []
        imagined_rewards = []
        imagined_continues = []
        imagined_log_probs = []
        imagined_entropies = []
        im_h, im_z = init_h, init_z

        for _ in range(imagine_horizon):
            feat = self.rssm.get_feature(im_h, im_z)
            imagined_features.append(feat)

            action_dist = self.actor.get_dist(feat)
            action = action_dist.sample()
            imagined_log_probs.append(action_dist.log_prob(action))
            imagined_entropies.append(action_dist.entropy())
            action_oh = F.one_hot(action, self._act_dim).float()

            im_h, im_z = self.rssm.imagine_step(im_h, im_z, action_oh)

            with torch.no_grad():
                next_feat = self.rssm.get_feature(im_h, im_z)
                r_sym = self.reward_pred(next_feat)
                r = symexp(r_sym)  # symlog → 원래 스케일 복원
                c = torch.sigmoid(self.continue_pred(next_feat))
            imagined_rewards.append(r)
            imagined_continues.append(c)

        im_feats = torch.stack(imagined_features, dim=0)  # [H, B, feat]
        im_rews = torch.stack(imagined_rewards, dim=0)     # [H, B]
        im_conts = torch.stack(imagined_continues, dim=0)  # [H, B]
        im_log_probs = torch.stack(imagined_log_probs, dim=0)  # [H, B]
        im_entropies = torch.stack(imagined_entropies, dim=0)  # [H, B]

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

        # ── Actor loss (같은 루프의 log_prob × advantage — 불일치 해결) ──
        advantages = (returns - values).detach()
        actor_loss = -(im_log_probs * advantages).mean()
        actor_loss = actor_loss - entropy_coef * im_entropies.mean()

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
            "entropy": im_entropies.mean().item(),
        }

    def _evaluate(self, env, n_episodes: int = 5) -> tuple[float, float]:
        total = 0.0
        total_score = 0.0
        for _ in range(n_episodes):
            obs, _ = env.reset()
            done = False
            h, z = self.rssm.initial_state(1, self.device)
            prev_action = torch.zeros(1, self._act_dim, device=self.device)
            ep_info = {}
            while not done:
                with torch.no_grad():
                    obs_t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
                    h, z, _, _ = self.rssm.observe_step(h, z, prev_action, obs_t)
                    feat = self.rssm.get_feature(h, z)
                    probs = self.actor(feat)
                    action = torch.distributions.Categorical(probs).sample().item()
                prev_action = F.one_hot(torch.tensor([action], device=self.device),
                                        self._act_dim).float()
                obs, r, terminated, truncated, info = env.step(action)
                total += r
                done = terminated or truncated
                if done:
                    ep_info = info
                summary = ep_info.get("episode_summary", {})
                total_score += summary.get("final_score", 0.0)
            return total / n_episodes, total_score / n_episodes

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
