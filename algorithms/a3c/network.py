"""A3C 네트워크 – Actor-Critic shared backbone."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ActorCritic(nn.Module):
    """A3C 용 Actor-Critic 네트워크.

    Shared feature extractor + policy head + value head.
    """

    def __init__(self, obs_dim: int, act_dim: int,
                 hidden_sizes: list[int] | None = None,
                 activation: str = "relu"):
        super().__init__()
        hidden_sizes = hidden_sizes or [128, 128]
        act_fn = {"relu": nn.ReLU, "tanh": nn.Tanh, "elu": nn.ELU}.get(
            activation, nn.ReLU)
        self._act_name = activation   # _init_weights에서 gain 결정용

        # Shared feature extractor
        layers: list[nn.Module] = []
        prev = obs_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(prev, h))
            layers.append(act_fn())
            prev = h
        self.shared = nn.Sequential(*layers)

        # Policy (actor) head
        self.policy_head = nn.Linear(prev, act_dim)
        # Value (critic) head
        self.value_head = nn.Linear(prev, 1)

        self._init_weights()

    def _init_weights(self):
        """레이어별 적절한 gain 으로 초기화.

        히든 레이어: gain=√2 (ReLU 표준)
        정책 헤드:   gain=0.01 (초기 탐색 균등)
        가치 헤드:   gain=1.0
        """
        act_gains = {"relu": 2**0.5, "tanh": 1.0, "elu": 1.0}
        hidden_gain = act_gains.get(
            getattr(self, '_act_name', 'relu'), 2**0.5)
        for m in self.shared.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=hidden_gain)
                nn.init.zeros_(m.bias)
        nn.init.orthogonal_(self.policy_head.weight, gain=0.01)
        nn.init.zeros_(self.policy_head.bias)
        nn.init.orthogonal_(self.value_head.weight, gain=1.0)
        nn.init.zeros_(self.value_head.bias)

    def forward(self, x: torch.Tensor):
        features = self.shared(x)
        logits = self.policy_head(features)
        value = self.value_head(features)
        return logits, value

    def act(self, obs: torch.Tensor):
        """확률적 행동 샘플링 (학습용)."""
        logits, value = self(obs)
        probs = F.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        return action, dist.log_prob(action), dist.entropy(), value

    def evaluate(self, obs: torch.Tensor):
        """결정적 행동 선택 (평가용)."""
        logits, value = self(obs)
        return logits.argmax(dim=-1), value
