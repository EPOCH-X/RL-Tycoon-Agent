"""Self-play 환경 래퍼 – 두 에이전트가 동시에 각자의 레스토랑을 경영하며 경쟁.

각 에이전트는 자신의 Shop 인스턴스를 갖고,
상대방의 성과(돈, 평점)를 관측에 포함하여 상대적 전략을 학습합니다.
"""

import numpy as np
import gymnasium
from gymnasium import spaces

from config.settings import NUM_ACTIONS, TILE_SIZE
from core.shop import Shop
from ai.gym_env import build_observation, _obs_size
from ai.reward import RewardCalculator


class SelfPlayEnv(gymnasium.Env):
    """두 매장 경쟁 환경.

    에이전트(obs)는 자기 매장 관측 + 상대 매장 요약(돈, 평점, 서빙수)을 봅니다.
    보상에 상대 대비 성과 보너스가 추가됩니다.
    """

    metadata = {"render_modes": ["human"], "render_fps": 10}

    def __init__(self, render_mode=None, reward_config=None,
                 opponent_agent=None, **shop_kwargs):
        super().__init__()
        self.shop = Shop(**shop_kwargs)
        self.opponent_shop = Shop(**shop_kwargs)
        self.render_mode = render_mode
        self._reward_calc = RewardCalculator(reward_config)

        # 상대 요약 차원: money_ratio, rating, customers_served_ratio
        self._opponent_summary_dim = 3
        base_obs_len = _obs_size(self.shop)
        total_obs_len = base_obs_len + self._opponent_summary_dim

        self.action_space = spaces.Discrete(NUM_ACTIONS)
        self.observation_space = spaces.Box(
            low=0.0, high=2.0, shape=(total_obs_len,), dtype=np.float32)

        self._opponent_agent = opponent_agent
        self._reward_cfg = reward_config or {}

    def set_opponent(self, agent):
        """상대 에이전트를 설정합니다 (Self-play에서 주기적으로 교체)."""
        self._opponent_agent = agent

    def _get_opponent_action(self, opp_obs: np.ndarray) -> int:
        if self._opponent_agent is None:
            return np.random.randint(0, NUM_ACTIONS)
        if hasattr(self._opponent_agent, "predict"):
            result = self._opponent_agent.predict(opp_obs)
            if isinstance(result, tuple):
                return int(result[0])
            return int(result)
        return np.random.randint(0, NUM_ACTIONS)

    def _build_obs(self) -> np.ndarray:
        base = build_observation(self.shop)
        # 상대 요약
        opp_money = min(2.0, self.opponent_shop.money /
                        max(1, self.opponent_shop.target_money))
        opp_rating = self.opponent_shop.shop_rating
        opp_served = min(2.0, self.opponent_shop.customers_served / 20.0)
        summary = np.array([opp_money, opp_rating, opp_served],
                           dtype=np.float32)
        return np.concatenate([base, summary])

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.shop.reset()
        self.opponent_shop.reset()
        return self._build_obs(), {}

    def step(self, action):
        # 내 매장 스텝
        events = self.shop.step(int(action))
        self.shop.auto_select_trait()

        # 상대 매장 스텝
        opp_base_obs = build_observation(self.opponent_shop)
        opp_action = self._get_opponent_action(opp_base_obs)
        self.opponent_shop.step(int(opp_action))
        self.opponent_shop.auto_select_trait()

        # 보상 계산 (기본 + 상대적 보너스)
        reward = self._reward_calc(events)

        # 상대적 보상: 돈 차이, 평점 차이
        money_diff = (self.shop.money - self.opponent_shop.money) / \
                     max(1, self.shop.target_money)
        rating_diff = self.shop.shop_rating - self.opponent_shop.shop_rating
        reward += self._reward_cfg.get("relative_money_bonus", 0.5) * money_diff
        reward += self._reward_cfg.get("relative_rating_bonus", 1.0) * rating_diff

        obs = self._build_obs()
        terminated = self.shop.done or self.opponent_shop.done
        truncated = False
        info = {
            "my_money": self.shop.money,
            "opp_money": self.opponent_shop.money,
            "my_rating": self.shop.shop_rating,
            "opp_rating": self.opponent_shop.shop_rating,
            "won": self.shop.won,
            "opp_won": self.opponent_shop.won,
        }
        return obs, reward, terminated, truncated, info

    def render(self):
        pass  # Rendering skipped for training

    def close(self):
        pass
