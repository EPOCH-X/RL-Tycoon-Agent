"""DreamerV3 RSSM Networks – Recurrent State-Space Model.

핵심 구성요소:
- RSSM: GRU 기반 상태전이 + 이산 확률 벡터 (stochastic state)
- Reward/Continue 예측기
- Actor (정책), Critic (가치 함수)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D


class RSSM(nn.Module):
    """Recurrent State-Space Model (DreamerV3 스타일).

    State = (deterministic h, stochastic z)
    - h: GRU hidden [hidden_size]
    - z: categorical stochastic [n_categoricals * n_classes] (one-hot flattened)
    """

    def __init__(self, obs_dim: int, act_dim: int,
                 hidden_size: int = 256,
                 stoch_size: int = 32,
                 n_classes: int = 32):
        super().__init__()
        self.hidden_size = hidden_size
        self.stoch_size = stoch_size
        self.n_classes = n_classes
        self.stoch_dim = stoch_size * n_classes  # flattened

        # Prior: h → z_prior (predict stochastic from deterministic alone)
        self.prior_net = nn.Sequential(
            nn.Linear(hidden_size, hidden_size), nn.ELU(),
            nn.Linear(hidden_size, stoch_size * n_classes),
        )

        # Posterior: h + obs → z_post (correct stochastic using observation)
        self.posterior_net = nn.Sequential(
            nn.Linear(hidden_size + obs_dim, hidden_size), nn.ELU(),
            nn.Linear(hidden_size, stoch_size * n_classes),
        )

        # Sequence model: (z_prev, action) → h_next
        self.gru_input = nn.Linear(self.stoch_dim + act_dim, hidden_size)
        self.gru = nn.GRUCell(hidden_size, hidden_size)

    def initial_state(self, batch_size: int, device):
        """초기 (h, z) 상태."""
        h = torch.zeros(batch_size, self.hidden_size, device=device)
        z = torch.zeros(batch_size, self.stoch_dim, device=device)
        return h, z

    def observe_step(self, prev_h, prev_z, action, obs):
        """한 스텝 관측: (h_t-1, z_t-1, a_t-1, o_t) → (h_t, z_post_t, prior_logits, post_logits)."""
        # Sequence model → h_t
        x = self.gru_input(torch.cat([prev_z, action], dim=-1))
        h = self.gru(F.elu(x), prev_h)

        # Prior
        prior_logits = self.prior_net(h).view(-1, self.stoch_size, self.n_classes)

        # Posterior
        post_logits = self.posterior_net(
            torch.cat([h, obs], dim=-1)
        ).view(-1, self.stoch_size, self.n_classes)

        # Sample z from posterior (straight-through)
        z_post = self._sample_stoch(post_logits)
        return h, z_post, prior_logits, post_logits

    def imagine_step(self, prev_h, prev_z, action):
        """상상 스텝 (관측 없이): (h_t-1, z_t-1, a_t-1) → (h_t, z_prior_t)."""
        x = self.gru_input(torch.cat([prev_z, action], dim=-1))
        h = self.gru(F.elu(x), prev_h)
        prior_logits = self.prior_net(h).view(-1, self.stoch_size, self.n_classes)
        z = self._sample_stoch(prior_logits)
        return h, z

    def _sample_stoch(self, logits):
        """Categorical에서 샘플링 (straight-through Gumbel-Softmax)."""
        probs = F.softmax(logits, dim=-1)
        # Straight-through: one-hot in forward, gradient through softmax
        indices = D.Categorical(probs=probs).sample()
        one_hot = F.one_hot(indices, self.n_classes).float()
        z = one_hot + probs - probs.detach()  # straight-through
        return z.view(-1, self.stoch_dim)

    def get_feature(self, h, z):
        """(h, z)를 합쳐서 feature vector로."""
        return torch.cat([h, z], dim=-1)

    @property
    def feature_dim(self):
        return self.hidden_size + self.stoch_dim


class RewardPredictor(nn.Module):
    def __init__(self, feature_dim: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim, hidden), nn.ELU(),
            nn.Linear(hidden, hidden), nn.ELU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, features):
        return self.net(features).squeeze(-1)


class ContinuePredictor(nn.Module):
    """에피소드 계속 확률 예측 (1 - done)."""
    def __init__(self, feature_dim: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim, hidden), nn.ELU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, features):
        return self.net(features).squeeze(-1)


class ObsDecoder(nn.Module):
    """관측 복원 디코더."""
    def __init__(self, feature_dim: int, obs_dim: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim, hidden), nn.ELU(),
            nn.Linear(hidden, hidden), nn.ELU(),
            nn.Linear(hidden, obs_dim),
        )

    def forward(self, features):
        return self.net(features)


class Actor(nn.Module):
    """이산 행동 정책."""
    def __init__(self, feature_dim: int, act_dim: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim, hidden), nn.ELU(),
            nn.Linear(hidden, hidden), nn.ELU(),
            nn.Linear(hidden, act_dim),
        )

    def forward(self, features):
        return F.softmax(self.net(features), dim=-1)

    def get_dist(self, features):
        probs = self.forward(features)
        return D.Categorical(probs=probs)


class Critic(nn.Module):
    """가치 함수."""
    def __init__(self, feature_dim: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim, hidden), nn.ELU(),
            nn.Linear(hidden, hidden), nn.ELU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, features):
        return self.net(features).squeeze(-1)
