"""A3C Trainer – Asynchronous Advantage Actor-Critic (PyTorch 직접 구현).

SB3에는 A3C가 포함되어 있지 않으므로 PyTorch로 직접 구현합니다.
멀티프로세스 워커가 각자 환경과 상호작용하며 글로벌 모델을 비동기 업데이트합니다.

핵심 특징:
- 비동기 멀티 워커 학습 (SharedAdam + mp.Process)
- n-step return 기반 Advantage 계산
- Entropy 보너스로 탐색 유도
"""

import os
import copy
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import torch.multiprocessing as mp

from algorithms.base import BaseTrainer
from algorithms.common import load_algo_config, make_env, save_run_config
from algorithms.a3c.network import ActorCritic


class SharedAdam(torch.optim.Adam):
    """프로세스 간 공유되는 Adam optimizer."""

    def __init__(self, params, **kwargs):
        super().__init__(params, **kwargs)
        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                state["step"] = torch.zeros(1)
                state["exp_avg"] = torch.zeros_like(p.data)
                state["exp_avg_sq"] = torch.zeros_like(p.data)
                # Share in memory
                state["step"].share_memory_()
                state["exp_avg"].share_memory_()
                state["exp_avg_sq"].share_memory_()


def _worker(rank: int, global_model: ActorCritic, optimizer: SharedAdam,
            cfg: dict, global_counter: mp.Value, max_steps: int,
            save_path: str, results_queue: mp.Queue):
    """A3C 워커 프로세스."""
    hp = cfg.get("hyperparameters", {})
    net_cfg = cfg.get("network", {})
    game_ov = cfg.get("game_overrides", {})
    reward_cfg = cfg.get("reward_shaping", {})
    seed = cfg.get("training", {}).get("seed", 42)

    gamma = hp.get("gamma", 0.99)
    gae_lambda = hp.get("gae_lambda", 0.95)
    entropy_coef = hp.get("entropy_coef", 0.01)
    value_loss_coef = hp.get("value_loss_coef", 0.5)
    max_grad_norm = hp.get("max_grad_norm", 40.0)
    n_steps = hp.get("n_steps", 20)

    env_fn = make_env(rank, seed, game_ov, reward_cfg)
    env = env_fn()

    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n

    local_model = ActorCritic(
        obs_dim, act_dim,
        hidden_sizes=net_cfg.get("net_arch", [128, 128]),
        activation=net_cfg.get("activation_fn", "relu"),
    )

    obs, _ = env.reset()
    episode_reward = 0.0
    episode_count = 0

    while True:
        with global_counter.get_lock():
            if global_counter.value >= max_steps:
                break
        # Sync local ← global
        local_model.load_state_dict(global_model.state_dict())

        log_probs = []
        values = []
        rewards = []
        entropies = []

        for _ in range(n_steps):
            obs_t = torch.FloatTensor(obs).unsqueeze(0)
            action, log_prob, entropy, value = local_model.act(obs_t)
            obs_next, reward, terminated, truncated, info = env.step(action.item())

            log_probs.append(log_prob)
            values.append(value.squeeze())
            rewards.append(reward)
            entropies.append(entropy)

            obs = obs_next
            episode_reward += reward

            with global_counter.get_lock():
                global_counter.value += 1

            if terminated or truncated:
                episode_count += 1
                if episode_count % 10 == 0:
                    results_queue.put(("episode", rank, episode_count,
                                       episode_reward))
                obs, _ = env.reset()
                episode_reward = 0.0
                break

        # Compute returns
        R = torch.tensor(0.0)
        if not (terminated or truncated):
            obs_t = torch.FloatTensor(obs).unsqueeze(0)
            _, R = local_model.evaluate(obs_t)
            R = R.squeeze().detach()

        returns = []
        for r in reversed(rewards):
            R = r + gamma * R
            returns.insert(0, R)

        returns = torch.stack(returns)
        log_probs = torch.stack(log_probs)
        values = torch.stack(values)
        entropies = torch.stack(entropies)

        advantages = returns - values.detach()

        policy_loss = -(log_probs * advantages).mean()
        value_loss = F.mse_loss(values, returns.detach())
        entropy_loss = -entropies.mean()

        loss = (policy_loss
                + value_loss_coef * value_loss
                + entropy_coef * entropy_loss)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(local_model.parameters(), max_grad_norm)

        # Push local gradients → global model
        for lp, gp in zip(local_model.parameters(), global_model.parameters()):
            if gp.grad is None:
                gp._grad = lp.grad
            else:
                gp.grad = lp.grad
        optimizer.step()

    env.close()
    results_queue.put(("done", rank, episode_count, 0.0))


class A3CTrainer(BaseTrainer):
    name = "A3C"

    def __init__(self):
        self.global_model: ActorCritic | None = None
        self.cfg: dict = {}
        self.save_path: str = "models/a3c"
        self._timesteps: int = 200_000
        self._obs_dim: int = 0
        self._act_dim: int = 0

    def build(self, cfg: dict | None = None, config_path: str | None = None,
              save_path: str | None = None, **overrides) -> None:
        self.cfg = cfg or load_algo_config("a3c", config_path)
        t = self.cfg.get("training", {})
        net = self.cfg.get("network", {})
        game_ov = self.cfg.get("game_overrides", {})
        reward_cfg = self.cfg.get("reward_shaping", {})

        self._timesteps = overrides.get("timesteps",
                                        t.get("total_timesteps", 200_000))
        self.save_path = save_path or self.save_path
        os.makedirs(self.save_path, exist_ok=True)
        save_run_config(self.save_path, self.cfg)

        # 관측/행동 차원을 알아내기 위해 임시 환경 생성
        seed = t.get("seed", 42)
        tmp_env = make_env(0, seed, game_ov, reward_cfg)()
        self._obs_dim = tmp_env.observation_space.shape[0]
        self._act_dim = tmp_env.action_space.n
        tmp_env.close()

        self.global_model = ActorCritic(
            self._obs_dim, self._act_dim,
            hidden_sizes=net.get("net_arch", [128, 128]),
            activation=net.get("activation_fn", "relu"),
        )
        self.global_model.share_memory()

    def train(self) -> dict[str, Any]:
        assert self.global_model is not None, "call build() first"
        mp.set_start_method("spawn", force=True)

        n_workers = self.cfg.get("training", {}).get("n_workers", 4)
        lr = self.cfg.get("hyperparameters", {}).get("learning_rate", 1e-4)

        optimizer = SharedAdam(self.global_model.parameters(), lr=lr)
        global_counter = mp.Value("i", 0)
        results_queue = mp.Queue()

        workers = []
        for rank in range(n_workers):
            p = mp.Process(
                target=_worker,
                args=(rank, self.global_model, optimizer, self.cfg,
                      global_counter, self._timesteps, self.save_path,
                      results_queue),
            )
            p.start()
            workers.append(p)

        # Monitor
        total_episodes = 0
        while any(p.is_alive() for p in workers):
            while not results_queue.empty():
                msg = results_queue.get_nowait()
                if msg[0] == "episode":
                    _, rank, ep_cnt, ep_rew = msg
                    print(f"  [Worker {rank}] Episode {ep_cnt}, "
                          f"Reward: {ep_rew:.1f}")
                elif msg[0] == "done":
                    _, rank, ep_cnt, _ = msg
                    total_episodes += ep_cnt

        for p in workers:
            p.join()

        self.save(os.path.join(self.save_path, "final_model"))
        print(f"[✓] A3C training complete ({total_episodes} episodes). "
              f"Models → '{self.save_path}/'")
        return {"algorithm": "A3C", "timesteps": self._timesteps,
                "total_episodes": total_episodes,
                "save_path": self.save_path}

    def save(self, path: str) -> None:
        if self.global_model:
            torch.save(self.global_model.state_dict(), path + ".pt")

    def load(self, path: str) -> None:
        if self.global_model is None:
            raise RuntimeError("build() must be called before load()")
        state = torch.load(path + ".pt", map_location="cpu")
        self.global_model.load_state_dict(state)

    def predict(self, obs, deterministic: bool = True) -> int:
        assert self.global_model is not None
        obs_t = torch.FloatTensor(obs).unsqueeze(0)
        if deterministic:
            action, _ = self.global_model.evaluate(obs_t)
        else:
            action, _, _, _ = self.global_model.act(obs_t)
        return int(action.item())
