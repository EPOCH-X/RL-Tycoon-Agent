"""Self-play 환경 래퍼 – 두 에이전트가 동시에 각자의 레스토랑을 경영하며 경쟁.

각 에이전트는 자신의 Shop 인스턴스를 갖고,
상대방의 성과(돈, 평점)를 관측에 포함하여 상대적 전략을 학습합니다.
"""

import random as _random
import numpy as np
import gymnasium
from gymnasium import spaces

from config.settings import NUM_ACTIONS, ACTION_INTERACT, TILE_SIZE
from core.shop import Shop
from ai.gym_env import build_observation, _obs_size, _get_primary_target_point
from ai.reward import RewardCalculator


class SelfPlayEnv(gymnasium.Env):
    """두 매장 경쟁 환경.

    에이전트(obs)는 자기 매장 관측 + 상대 매장 요약(돈, 평점, 서빙수)을 봅니다.
    보상에 상대 대비 성과 보너스가 추가됩니다.
    """

    metadata = {"render_modes": ["human"], "render_fps": 10}

    def __init__(self, render_mode=None, reward_config=None,
                 opponent_agent=None, opponent_pool=None,
                 compat_mode=False, **shop_kwargs):
        super().__init__()
        self.shop = Shop(**shop_kwargs)
        self.opponent_shop = Shop(**shop_kwargs)
        self.render_mode = render_mode
        self._reward_calc = RewardCalculator(reward_config)

        # 상대 요약 차원: money_ratio, rating, customers_served_ratio
        self._opponent_summary_dim = 3
        base_obs_len = _obs_size(self.shop)
        # compat_mode=True → 기존 TycoonEnv 관측 차원 유지 (이전 모델 호환)
        self._compat_mode = compat_mode
        total_obs_len = base_obs_len if compat_mode else (base_obs_len + self._opponent_summary_dim)

        self.action_space = spaces.Discrete(NUM_ACTIONS)
        obs_high = 1.0 if compat_mode else 2.0
        self.observation_space = spaces.Box(
            low=-1.0, high=obs_high, shape=(total_obs_len,), dtype=np.float32)

        self._opponent_agent = opponent_agent
        self._opponent_pool = opponent_pool or []
        self._reward_cfg = reward_config or {}
        self._prev_potential: float = 0.0
        self._prev_net_profit: float = 0.0
        self._prev_rating: float = 0.0
        self._prev_final_score: float = 0.0

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
        if self._compat_mode:
            return base
        # 상대 요약
        opp_money = min(2.0, self.opponent_shop.money /
                        max(1, self.opponent_shop.target_money))
        opp_rating = self.opponent_shop.shop_rating
        opp_served = min(2.0, self.opponent_shop.customers_served / 20.0)
        summary = np.array([opp_money, opp_rating, opp_served],
                           dtype=np.float32)
        return np.concatenate([base, summary])

    def _calc_potential(self) -> float:
        px = self.shop.player.center_x
        py = self.shop.player.center_y
        map_diag = np.hypot(self.shop.grid_width * TILE_SIZE,
                            self.shop.grid_height * TILE_SIZE)
        if map_diag <= 0:
            return 0.0
        target = _get_primary_target_point(self.shop)
        if target is None:
            return 0.0
        return -float(np.hypot(px - target[0], py - target[1]) / map_diag)

    def _dense_shaping(self) -> float:
        new_potential = self._calc_potential()
        shaped = 0.99 * new_potential - self._prev_potential
        self._prev_potential = new_potential
        return shaped * 3.0

    def _auto_face_nearest(self, shop: Shop):
        px = shop.player.center_x
        py = shop.player.center_y
        best_dist = 80 * 1.5
        best_dx, best_dy = 0.0, 0.0
        found = False

        def consider(target_x: float, target_y: float):
            nonlocal best_dist, best_dx, best_dy, found
            dist = float(np.hypot(target_x - px, target_y - py))
            if dist < best_dist:
                best_dist = dist
                best_dx = target_x - px
                best_dy = target_y - py
                found = True

        if shop.player.has_food or shop.player.has_drink:
            first = shop.player.first_carried
            if first:
                tid = first.get("table_id", -1)
                for table in shop.tables:
                    if table.table_id == tid:
                        consider(table.center_x, table.center_y)
                        break
        elif shop.player.has_order:
            for gx, gy in shop.kitchen_counter_positions:
                consider(gx * TILE_SIZE + TILE_SIZE / 2,
                         gy * TILE_SIZE + TILE_SIZE / 2)
        else:
            for table in shop.tables:
                if table.customer is not None:
                    consider(table.center_x, table.center_y)
            if shop.kitchen.ready:
                for gx, gy in shop.kitchen_counter_positions:
                    consider(gx * TILE_SIZE + TILE_SIZE / 2,
                             gy * TILE_SIZE + TILE_SIZE / 2)

        if found:
            if abs(best_dx) > abs(best_dy):
                shop.player.facing = 3 if best_dx > 0 else 2
            else:
                shop.player.facing = 1 if best_dy > 0 else 0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        # opponent_pool이 있으면 매 에피소드마다 랜덤 상대 선택
        if self._opponent_pool:
            self._opponent_agent = _random.choice(self._opponent_pool)
        self.shop.reset()
        self.opponent_shop.reset()
        self._prev_potential = self._calc_potential()
        self._prev_net_profit = float(self.shop.net_profit)
        self._prev_rating = float(self.shop.shop_rating)
        self._prev_final_score = float(self.shop.final_score)
        return self._build_obs(), {}

    def step(self, action):
        if int(action) == ACTION_INTERACT:
            self._auto_face_nearest(self.shop)

        # 내 매장 스텝
        events = self.shop.step(int(action))

        # 상대 매장 스텝
        opp_base_obs = build_observation(self.opponent_shop)
        opp_action = self._get_opponent_action(opp_base_obs)
        if int(opp_action) == ACTION_INTERACT:
            self._auto_face_nearest(self.opponent_shop)
        self.opponent_shop.step(int(opp_action))

        net_profit_delta = float(self.shop.net_profit) - self._prev_net_profit
        rating_delta = float(self.shop.shop_rating) - self._prev_rating
        final_score_delta = float(self.shop.final_score) - self._prev_final_score
        if abs(net_profit_delta) > 1e-6:
            events.append(("net_profit_delta", net_profit_delta))
        if abs(rating_delta) > 1e-6:
            events.append(("rating_delta", rating_delta))
        if abs(final_score_delta) > 1e-6:
            events.append(("final_score_delta", final_score_delta))

        # 보상 계산 (기본 + 상대적 보너스)
        reward = self._reward_calc(events)
        reward += self._dense_shaping()

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
        if terminated or truncated:
            info["episode_summary"] = {
                "customers_served": self.shop.customers_served,
                "customers_lost": self.shop.customers_lost,
                "net_profit": float(self.shop.net_profit),
                "shop_rating": float(self.shop.shop_rating),
                "final_score": float(self.shop.final_score),
            }
        self._prev_net_profit = float(self.shop.net_profit)
        self._prev_rating = float(self.shop.shop_rating)
        self._prev_final_score = float(self.shop.final_score)
        return obs, reward, terminated, truncated, info

    def render(self):
        pass  # Rendering skipped for training

    def close(self):
        pass
