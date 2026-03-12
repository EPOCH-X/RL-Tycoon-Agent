"""World Model – 환경의 전이 함수(transition)와 보상 함수를 학습하는 신경망.

입력: (state, action) → 출력: (next_state, reward, done_prob)
이를 통해 실제 환경 없이 시뮬레이션(Imagination)이 가능합니다.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class WorldModel(nn.Module):
    """환경 전이/보상을 예측하는 앙상블 모델.

    앙상블을 사용하여 예측 불확실성도 추정합니다.
    """

    def __init__(self, obs_dim: int, act_dim: int,
                 hidden: list[int] | None = None,
                 n_ensemble: int = 3):
        super().__init__()
        hidden = hidden or [256, 256]
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.n_ensemble = n_ensemble

        # 행동을 one-hot으로 인코딩
        input_dim = obs_dim + act_dim

        self.models = nn.ModuleList()
        for _ in range(n_ensemble):
            layers = []
            prev = input_dim
            for h in hidden:
                layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(0.05)]
                prev = h
            model = nn.Sequential(*layers)
            self.models.append(model)

        # 공유 출력 헤드 (각 앙상블 멤버별)
        self.state_heads = nn.ModuleList(
            [nn.Linear(hidden[-1], obs_dim) for _ in range(n_ensemble)])
        self.reward_heads = nn.ModuleList(
            [nn.Linear(hidden[-1], 1) for _ in range(n_ensemble)])
        self.done_heads = nn.ModuleList(
            [nn.Linear(hidden[-1], 1) for _ in range(n_ensemble)])

    def _encode_action(self, action: torch.Tensor, batch_size: int) -> torch.Tensor:
        """행동 인덱스를 one-hot 벡터로 변환."""
        one_hot = torch.zeros(batch_size, self.act_dim,
                              device=action.device)
        one_hot.scatter_(1, action.long().unsqueeze(-1), 1.0)
        return one_hot

    def forward(self, obs: torch.Tensor, action: torch.Tensor):
        """앙상블 예측. 반환: (next_states, rewards, dones) — 각각 리스트."""
        batch_size = obs.shape[0]
        act_onehot = self._encode_action(action, batch_size)
        x = torch.cat([obs, act_onehot], dim=-1)

        next_states, rewards, dones = [], [], []
        for i in range(self.n_ensemble):
            feat = self.models[i](x)
            ns = self.state_heads[i](feat) + obs  # residual prediction
            r = self.reward_heads[i](feat)
            d = torch.sigmoid(self.done_heads[i](feat))
            next_states.append(ns)
            rewards.append(r.squeeze(-1))
            dones.append(d.squeeze(-1))

        return next_states, rewards, dones

    def predict_mean(self, obs: torch.Tensor, action: torch.Tensor):
        """앙상블 평균 예측."""
        ns_list, r_list, d_list = self(obs, action)
        next_state = torch.stack(ns_list).mean(0)
        reward = torch.stack(r_list).mean(0)
        done = torch.stack(d_list).mean(0)
        return next_state, reward, done

    def predict_with_uncertainty(self, obs: torch.Tensor, action: torch.Tensor):
        """평균 + 표준편차(불확실성) 반환."""
        ns_list, r_list, d_list = self(obs, action)
        ns_stack = torch.stack(ns_list)
        r_stack = torch.stack(r_list)
        return (ns_stack.mean(0), ns_stack.std(0),
                r_stack.mean(0), r_stack.std(0))


class WorldModelTrainer:
    """World Model의 학습을 관리합니다."""

    def __init__(self, world_model: WorldModel,
                 lr: float = 1e-3, device: str = "cpu"):
        self.model = world_model
        self.optimizer = torch.optim.Adam(world_model.parameters(), lr=lr)
        self.device = device

    def train_step(self, obs_batch, act_batch, next_obs_batch,
                   reward_batch, done_batch) -> dict[str, float]:
        """한 배치에 대해 world model을 학습합니다."""
        obs = torch.FloatTensor(obs_batch).to(self.device)
        act = torch.FloatTensor(act_batch).to(self.device)
        next_obs = torch.FloatTensor(next_obs_batch).to(self.device)
        rew = torch.FloatTensor(reward_batch).to(self.device)
        done = torch.FloatTensor(done_batch).to(self.device)

        ns_list, r_list, d_list = self.model(obs, act)

        state_loss = sum(F.mse_loss(ns, next_obs) for ns in ns_list)
        reward_loss = sum(F.mse_loss(r, rew) for r in r_list)
        done_loss = sum(F.binary_cross_entropy(d, done) for d in d_list)

        loss = state_loss + reward_loss + 0.1 * done_loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {
            "state_loss": state_loss.item() / self.model.n_ensemble,
            "reward_loss": reward_loss.item() / self.model.n_ensemble,
            "done_loss": done_loss.item() / self.model.n_ensemble,
            "total_loss": loss.item(),
        }
