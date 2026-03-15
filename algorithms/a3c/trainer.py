"""A3C Trainer – Asynchronous Advantage Actor-Critic (PyTorch 직접 구현).

SB3에는 A3C가 포함되어 있지 않으므로 PyTorch로 직접 구현합니다.

GPU 사용 가능 시:
  → 단일 프로세스 GPU A2C 모드 (GPU 텐서 연산으로 속도 향상)
  → GPU에서는 멀티프로세스 공유 메모리가 불가능하므로
    벡터화 환경 + 배치 업데이트로 대체합니다.

GPU 없을 시:
  → 기존 멀티프로세스 CPU A3C 모드

핵심 특징:
- GPU: 벡터화 환경 + 배치 Actor-Critic 업데이트 (A2C)
- CPU: 비동기 멀티 워커 학습 (SharedAdam + mp.Process)
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
from algorithms.common import load_algo_config, make_env, make_vec_env, save_run_config, get_device, EarlyStopTracker
from algorithms.a3c.network import ActorCritic


class SharedAdam(torch.optim.Adam):
    """프로세스 간 공유되는 Adam optimizer (CPU A3C용)."""

    def __init__(self, params, **kwargs):
        super().__init__(params, **kwargs)
        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                state["step"] = torch.zeros(1)
                state["exp_avg"] = torch.zeros_like(p.data)
                state["exp_avg_sq"] = torch.zeros_like(p.data)
                state["step"].share_memory_()
                state["exp_avg"].share_memory_()
                state["exp_avg_sq"].share_memory_()


def _worker(rank: int, global_model: ActorCritic, optimizer: SharedAdam,
            cfg: dict, global_counter: mp.Value, max_steps: int,
            save_path: str, results_queue: mp.Queue):
    """A3C 워커 프로세스 (CPU only)."""
    hp = cfg.get("hyperparameters", {})
    net_cfg = cfg.get("network", {})
    game_ov = cfg.get("game_overrides", {})
    reward_cfg = cfg.get("reward_shaping", {})
    seed = cfg.get("training", {}).get("seed", 42)

    gamma = hp.get("gamma", 0.99)
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
        local_model.load_state_dict(global_model.state_dict())

        log_probs, values, rewards, entropies = [], [], [], []

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

        loss = policy_loss + value_loss_coef * value_loss + entropy_coef * entropy_loss

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(local_model.parameters(), max_grad_norm)

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
        self.device: torch.device = torch.device("cpu")

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

        seed = t.get("seed", 42)
        tmp_env = make_env(0, seed, game_ov, reward_cfg)()
        self._obs_dim = tmp_env.observation_space.shape[0]
        self._act_dim = tmp_env.action_space.n
        tmp_env.close()

        self.device = get_device()

        self.global_model = ActorCritic(
            self._obs_dim, self._act_dim,
            hidden_sizes=net.get("net_arch", [128, 128]),
            activation=net.get("activation_fn", "relu"),
        )

        if self.device.type == "cuda":
            # GPU 모드: 모델을 GPU로 이동
            self.global_model = self.global_model.to(self.device)
        else:
            # CPU 모드: 공유 메모리로 설정 (멀티프로세스용)
            self.global_model.share_memory()

    def train(self) -> dict[str, Any]:
        assert self.global_model is not None, "call build() first"

        if self.device.type == "cuda":
            return self._train_gpu()
        else:
            return self._train_cpu()

    # ── GPU A2C 모드 (벡터화 환경 + 배치 업데이트) ──
    def _train_gpu(self) -> dict[str, Any]:
        """GPU 가속 A2C 학습 (단일 프로세스, 벡터화 환경)."""
        hp = self.cfg.get("hyperparameters", {})
        t = self.cfg.get("training", {})
        game_ov = self.cfg.get("game_overrides", {})
        reward_cfg = self.cfg.get("reward_shaping", {})

        gamma = hp.get("gamma", 0.99)
        entropy_coef = hp.get("entropy_coef", 0.01)
        value_loss_coef = hp.get("value_loss_coef", 0.5)
        max_grad_norm = hp.get("max_grad_norm", 40.0)
        n_steps = hp.get("n_steps", 20)
        lr = hp.get("learning_rate", 1e-4)
        seed = t.get("seed", 42)
        n_envs = t.get("n_workers", 4)
        eval_freq = t.get("eval_freq", 5000)

        optimizer = torch.optim.Adam(self.global_model.parameters(), lr=lr)

        # 벡터화 환경 (CPU에서 실행, 관측만 GPU로 전송)
        vec_env = make_vec_env(n_envs, seed, game_ov, reward_cfg, force_dummy=True)
        eval_env = make_env(0, seed + 1000, game_ov, reward_cfg)()

        obs_arr = vec_env.reset()
        total_steps = 0
        episode_count = 0
        best_eval = float("-inf")
        early_stop = EarlyStopTracker(patience=50, min_delta=1.0, verbose=1)

        print(f"  [A3C-GPU (비동기 액터-크리틱)] {n_envs}개 병렬 환경으로 학습 시작, 디바이스: {self.device}")

        while total_steps < self._timesteps:
            # ── 1) Rollout 수집 (no_grad – 행동 선택만) ──
            mb_obs, mb_actions, mb_rewards, mb_dones = [], [], [], []

            for _ in range(n_steps):
                obs_t = torch.FloatTensor(obs_arr).to(self.device)
                with torch.no_grad():
                    logits, _ = self.global_model(obs_t)
                    probs = F.softmax(logits, dim=-1)
                    dist = torch.distributions.Categorical(probs)
                    actions = dist.sample()

                mb_obs.append(obs_t)
                mb_actions.append(actions)

                actions_np = actions.cpu().numpy()
                obs_arr, rewards, dones, infos = vec_env.step(actions_np)

                mb_rewards.append(torch.FloatTensor(rewards).to(self.device))
                mb_dones.append(torch.FloatTensor(dones).to(self.device))

                total_steps += n_envs
                episode_count += sum(dones)

            # ── 2) 그래디언트 계산을 위해 forward pass 재실행 ──
            all_obs = torch.stack(mb_obs)              # [n_steps, n_envs, obs_dim]
            all_actions = torch.stack(mb_actions)      # [n_steps, n_envs]

            flat_obs = all_obs.reshape(-1, all_obs.shape[-1])
            logits_flat, values_flat = self.global_model(flat_obs)

            logits_2d = logits_flat.reshape(n_steps, n_envs, -1)
            values = values_flat.squeeze(-1).reshape(n_steps, n_envs)

            probs_2d = F.softmax(logits_2d, dim=-1)
            dist_2d = torch.distributions.Categorical(probs_2d)
            log_probs = dist_2d.log_prob(all_actions)   # [n_steps, n_envs]
            entropies = dist_2d.entropy()               # [n_steps, n_envs]

            # ── 3) Returns 계산 (no_grad) ──
            with torch.no_grad():
                last_obs_t = torch.FloatTensor(obs_arr).to(self.device)
                _, last_values = self.global_model(last_obs_t)
                last_values = last_values.squeeze(-1)

            returns = []
            R = last_values
            for t_idx in reversed(range(n_steps)):
                R = mb_rewards[t_idx] + gamma * R * (1.0 - mb_dones[t_idx])
                returns.insert(0, R)

            returns = torch.stack(returns).detach()    # [n_steps, n_envs]

            # ── 4) Loss 계산 & 역전파 ──
            advantages = returns - values.detach()

            policy_loss = -(log_probs * advantages).mean()
            value_loss = F.mse_loss(values, returns)
            entropy_loss = -entropies.mean()

            loss = policy_loss + value_loss_coef * value_loss + entropy_coef * entropy_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.global_model.parameters(), max_grad_norm)
            optimizer.step()

            # 로그
            if total_steps % (eval_freq // 2) < n_envs * n_steps:
                print(f"  [A3C-GPU] 스텝(Steps): {total_steps}, 에피소드(Episodes): {episode_count}, "
                      f"손실(Loss): {loss.item():.4f}")

            # 평가
            if total_steps % eval_freq < n_envs * n_steps:
                eval_r = self._evaluate_gpu(eval_env)
                print(f"  [A3C-GPU] 평가(Eval) 스텝 {total_steps}: "
                      f"평균보상(mean_reward)={eval_r:.1f}")
                if eval_r > best_eval:
                    best_eval = eval_r
                    self.save(os.path.join(self.save_path, "best_model"))
                if not early_stop.check(eval_r):
                    print(f"  [A3C-GPU] 조기 종료 (Early stopped), 스텝: {total_steps}")
                    break

        vec_env.close()
        eval_env.close()
        self.save(os.path.join(self.save_path, "final_model"))
        print(f"[✓] A3C-GPU (비동기 액터-크리틱) 학습 완료 ({episode_count}개 에피소드). "
              f"모델 → '{self.save_path}/'")
        return {"algorithm": "A3C", "mode": "GPU-A2C",
                "timesteps": self._timesteps,
                "total_episodes": episode_count,
                "save_path": self.save_path}

    def _evaluate_gpu(self, env, n_episodes: int = 5) -> float:
        total = 0.0
        for _ in range(n_episodes):
            obs, _ = env.reset()
            done = False
            while not done:
                obs_t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    action, _ = self.global_model.evaluate(obs_t)
                obs, r, terminated, truncated, _ = env.step(action.item())
                total += r
                done = terminated or truncated
        return total / n_episodes

    # ── CPU A3C 모드 (멀티프로세스) ──
    def _train_cpu(self) -> dict[str, Any]:
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

        total_episodes = 0
        while any(p.is_alive() for p in workers):
            while not results_queue.empty():
                msg = results_queue.get_nowait()
                if msg[0] == "episode":
                    _, rank, ep_cnt, ep_rew = msg
                    print(f"  [워커 {rank}] 에피소드(Episode) {ep_cnt}, "
                          f"보상(Reward): {ep_rew:.1f}")
                elif msg[0] == "done":
                    _, rank, ep_cnt, _ = msg
                    total_episodes += ep_cnt

        for p in workers:
            p.join()

        self.save(os.path.join(self.save_path, "final_model"))
        print(f"[✓] A3C-CPU (비동기 액터-크리틱) 학습 완료 ({total_episodes}개 에피소드). "
              f"모델 → '{self.save_path}/'")
        return {"algorithm": "A3C", "mode": "CPU-A3C",
                "timesteps": self._timesteps,
                "total_episodes": total_episodes,
                "save_path": self.save_path}

    def save(self, path: str) -> None:
        if self.global_model:
            # CPU로 옮겨서 저장 (호환성)
            state_dict = {k: v.cpu() for k, v in self.global_model.state_dict().items()}
            torch.save(state_dict, path + ".pt")

    def load(self, path: str) -> None:
        if self.global_model is None:
            raise RuntimeError("build() must be called before load()")
        state = torch.load(path + ".pt", map_location=self.device)
        self.global_model.load_state_dict(state)
        self.global_model = self.global_model.to(self.device)

    def predict(self, obs, deterministic: bool = True) -> int:
        assert self.global_model is not None
        obs_t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        with torch.no_grad():
            if deterministic:
                action, _ = self.global_model.evaluate(obs_t)
            else:
                action, _, _, _ = self.global_model.act(obs_t)
        return int(action.item())
