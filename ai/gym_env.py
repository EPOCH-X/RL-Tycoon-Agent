"""Gymnasium environment wrapper for the restaurant management sim.

Wraps ``core.shop.Shop`` into a standard Gymnasium ``Env`` so that
Stable-Baselines3 (and any other Gymnasium-compatible library) can
train agents on it.
"""

import numpy as np
import gymnasium
from gymnasium import spaces

from config.settings import (
    TILE_SIZE, UI_HEIGHT, COLORS, ASSETS_DIR, NUM_ACTIONS,
)
from core.shop import Shop
from core.customer import CustomerState

# ────────────────────────────────────────────────
# Observation encoding helpers
# ────────────────────────────────────────────────
# Menu-item id mapping (must stay in sync with config/menu.json)
MENU_IDS = {
    "coffee": 1, "sandwich": 2, "pasta": 3,
    "steak": 4, "sushi_set": 5, "lobster": 6,
    "wagyu": 7, "truffle_course": 8,
}
NUM_MENU = len(MENU_IDS) + 1   # +1 for empty/unknown

# Customer-state encoding for observation
_STATE_ENC = {
    CustomerState.WAITING_TO_ORDER: 0.25,
    CustomerState.ORDER_TAKEN:      0.50,
    CustomerState.EATING:           0.75,
}


def _obs_size(shop: Shop) -> int:
    """Compute observation-vector length for a shop."""
    return (
        4                            # player x, y, facing, carry_type
        + 2                          # carry_table_id, carry_menu_id
        + shop.max_tables * 4        # table: occupied, state, menu_id, patience
        + 3                          # kitchen: cooking count, ready count, capacity ratio
        + 8                          # money_ratio, day_ratio, time_ratio, shop_rating,
                                     # can_upgrade, net_profit_ratio, employee_count, bar+delivery
    )


def build_observation(shop: Shop) -> np.ndarray:
    """Convert a Shop snapshot into a flat float32 observation vector."""
    size = _obs_size(shop)
    obs = np.zeros(size, dtype=np.float32)
    idx = 0

    # ── Player (pixel-based, normalised) ─────────
    map_px_w = max(1, shop.grid_width * TILE_SIZE)
    map_px_h = max(1, shop.grid_height * TILE_SIZE)
    obs[idx]     = shop.player.x / map_px_w
    obs[idx + 1] = shop.player.y / map_px_h
    obs[idx + 2] = shop.player.facing / 3.0
    # carry type: 0=idle, 0.33=order, 0.66=food, 1.0=drink
    if shop.player.has_order:
        obs[idx + 3] = 0.33
    elif shop.player.has_food:
        obs[idx + 3] = 0.66
    elif shop.player.has_drink:
        obs[idx + 3] = 1.0
    idx += 4

    # Carry details (first carried item)
    first = shop.player.first_carried
    if first:
        tid = first.get("table_id", 0)
        obs[idx] = tid / max(1, shop.max_tables)
        mi = first.get("menu_item") or (first.get("items", [None])[0] if first.get("items") else None)
        if mi:
            mid = MENU_IDS.get(mi.get("id", ""), 0)
            obs[idx + 1] = mid / NUM_MENU
    idx += 2

    # ── Tables (fixed-size: max_tables slots) ────
    for i in range(shop.max_tables):
        if i < len(shop.tables):
            cust = shop.tables[i].customer
            if cust is not None:
                obs[idx]     = 1.0
                obs[idx + 1] = _STATE_ENC.get(cust.state, 0.0)
                mi = cust.menu_item
                if mi:
                    obs[idx + 2] = MENU_IDS.get(mi["id"], 0) / NUM_MENU
                obs[idx + 3] = cust.patience_ratio
        idx += 4

    # ── Kitchen ──────────────────────────────────
    obs[idx]     = shop.kitchen.num_cooking / max(1, shop.kitchen.capacity)
    obs[idx + 1] = len(shop.kitchen.ready) / max(1, shop.kitchen.capacity)
    obs[idx + 2] = (shop.kitchen.num_cooking + len(shop.kitchen.ready)
                     ) / max(1, shop.kitchen.capacity)
    idx += 3

    # ── Game state ───────────────────────────────
    obs[idx]     = min(1.0, shop.money / max(1, shop.target_money))
    obs[idx + 1] = shop.current_day / max(1, shop.day_limit)
    obs[idx + 2] = 1.0 - min(1.0, shop.time_elapsed / max(1, shop.total_time_limit))
    obs[idx + 3] = shop.shop_rating
    # Can afford any upgrade?
    can_buy = 0.0
    for upg in shop.upgrades_data:
        uid = upg["id"]
        level = shop.upgrade_levels[uid]
        if level < upg["max_level"]:
            cost = int(upg["base_cost"] * (upg["cost_multiplier"] ** level))
            if shop.money >= cost:
                can_buy = 1.0
                break
    obs[idx + 4] = can_buy
    obs[idx + 5] = min(1.0, shop.net_profit / max(1, shop.target_money))
    obs[idx + 6] = len(shop.employees) / 4.0
    bar_del = 0.0
    if shop.bartender_hired:
        bar_del += 0.5
    if shop.delivery_unlocked:
        bar_del += 0.5
    obs[idx + 7] = bar_del

    return obs


# ────────────────────────────────────────────────
# Gymnasium Environment
# ────────────────────────────────────────────────
class TycoonEnv(gymnasium.Env):
    """Gymnasium environment for the restaurant management game."""

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 10}

    def __init__(self, render_mode=None, **shop_kwargs):
        super().__init__()
        self.shop = Shop(**shop_kwargs)
        self.render_mode = render_mode

        self.action_space = spaces.Discrete(NUM_ACTIONS)
        obs_len = _obs_size(self.shop)
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(obs_len,), dtype=np.float32)

        self._screen = None
        self._renderer = None

    # ── Gymnasium API ────────────────────────────
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.shop.reset()
        return build_observation(self.shop), {}

    def step(self, action):
        reward = self.shop.step(int(action))
        # Auto-select traits for RL agent
        self.shop.auto_select_trait()
        obs = build_observation(self.shop)
        terminated = self.shop.done
        truncated = False
        info = {
            "money": self.shop.money,
            "day": self.shop.current_day,
            "time_elapsed": self.shop.time_elapsed,
            "customers_served": self.shop.customers_served,
            "customers_lost": self.shop.customers_lost,
            "shop_rating": self.shop.shop_rating,
            "won": self.shop.won,
            "tables_active": len(self.shop.tables),
        }
        return obs, reward, terminated, truncated, info

    def render(self):
        import pygame
        if self.render_mode == "human":
            if self._screen is None:
                pygame.init()
                from rendering.asset_manager import AssetManager
                from rendering.renderer import Renderer
                w = self.shop.grid_width * TILE_SIZE
                h = self.shop.grid_height * TILE_SIZE + UI_HEIGHT
                self._screen = pygame.display.set_mode((w, h))
                pygame.display.set_caption("RL 타이쿤 – 학습 중")
                self._renderer = Renderer(AssetManager(ASSETS_DIR))

            self._screen.fill(COLORS["background"])
            self._renderer.draw(self._screen, self.shop)
            if self.shop.done:
                self._renderer.draw_game_over(self._screen, self.shop)
            pygame.display.flip()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.close()

        elif self.render_mode == "rgb_array":
            import pygame
            w = self.shop.grid_width * TILE_SIZE
            h = self.shop.grid_height * TILE_SIZE + UI_HEIGHT
            surf = pygame.Surface((w, h))
            surf.fill(COLORS["background"])
            if self._renderer is None:
                from rendering.asset_manager import AssetManager
                from rendering.renderer import Renderer
                self._renderer = Renderer(AssetManager(ASSETS_DIR))
            self._renderer.draw(surf, self.shop)
            return np.transpose(
                pygame.surfarray.array3d(surf), axes=(1, 0, 2))

    def close(self):
        if self._screen is not None:
            import pygame
            pygame.quit()
            self._screen = None
