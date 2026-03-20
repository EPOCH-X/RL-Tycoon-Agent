"""Gymnasium environment wrapper for the restaurant management sim.

Wraps ``core.shop.Shop`` into a standard Gymnasium ``Env`` so that
Stable-Baselines3 (and any other Gymnasium-compatible library) can
train agents on it.
"""

import math

import numpy as np
import gymnasium
from gymnasium import spaces

from config.settings import (
    TILE_SIZE, UI_HEIGHT, COLORS, NUM_ACTIONS,
    INTERACT_RANGE, ACTION_INTERACT, ACTION_NONE,
    UPGRADE_ACTION_MIN, UPGRADE_ACTION_MAX,
    TRAIT_ACTION_MIN, TRAIT_ACTION_MAX,
    ACTION_TO_UPGRADE,
    load_json_config,
)
from core.shop import Shop
from core.player import Player
from core.customer import CustomerState
from ai.reward import RewardCalculator

# ────────────────────────────────────────────────
# Observation encoding helpers
# ────────────────────────────────────────────────
# Menu-item id mapping (auto-loaded from config/menu.json)
_menu_data = load_json_config("menu.json")
MENU_IDS = {item["id"]: i + 1 for i, item in enumerate(_menu_data)}
NUM_MENU = len(MENU_IDS) + 1   # +1 for empty/unknown

# Customer-state encoding for observation
_STATE_ENC = {
    CustomerState.WAITING_TO_ORDER: 0.25,
    CustomerState.ORDER_TAKEN:      0.50,
    CustomerState.EATING:           0.75,
}

# Trait ID → numeric encoding for observation
_TRAIT_IDS: dict[str, int] = {
    "gourmet": 1,
    "master_chef": 2,
    "charming": 3,
    "efficient": 4,
    "popular": 5,
    "patient_service": 6,
    "tip_jar": 7,
}
_NUM_TRAIT_TYPES = len(_TRAIT_IDS) + 1  # +1 for unknown

# Number of upgrade types (for observation)
_NUM_UPGRADES = 9


def _norm_point(shop: Shop, x: float, y: float) -> tuple[float, float]:
    map_px_w = max(1, shop.grid_width * TILE_SIZE)
    map_px_h = max(1, shop.grid_height * TILE_SIZE)
    return x / map_px_w, y / map_px_h


def _get_primary_target_point(shop: Shop) -> tuple[float, float] | None:
    px = shop.player.center_x
    py = shop.player.center_y

    if shop.player.has_food or shop.player.has_drink:
        first = shop.player.first_carried
        if first:
            tid = first.get("table_id", -1)
            # 고아 아이템 확인: 해당 테이블에 손님이 없으면 → 쓰레기통
            target_table = None
            for table in shop.tables:
                if table.table_id == tid:
                    target_table = table
                    break
            if target_table is None or target_table.customer is None:
                # 고아 음식/음료 → 쓰레기통으로 유도
                return shop.get_station_interaction_point(
                    shop.trash_can_positions, px, py)
            return shop.get_table_interaction_point(target_table, px, py)
        return None

    if shop.player.has_order:
        return shop.get_station_interaction_point(
            shop.kitchen_counter_positions, px, py)

    candidates: list[tuple[float, float]] = []
    for table in shop.tables:
        if (table.customer is not None
                and table.customer.state == CustomerState.WAITING_TO_ORDER):
            candidates.append(shop.get_table_interaction_point(table, px, py))

    if shop.kitchen.ready:
        candidates.append(
            shop.get_station_interaction_point(
                shop.kitchen_counter_positions, px, py)
        )

    if not candidates:
        return None
    return min(candidates, key=lambda pos: math.hypot(pos[0] - px, pos[1] - py))


def _get_primary_target_signature(shop: Shop) -> tuple[str, int] | None:
    if shop.player.has_food or shop.player.has_drink:
        first = shop.player.first_carried
        if first:
            tid = int(first.get("table_id", -1))
            # 고아 아이템 → 쓰레기통이 타겟
            target_table = None
            for table in shop.tables:
                if table.table_id == tid:
                    target_table = table
                    break
            if target_table is None or target_table.customer is None:
                return ("trash", 0)
            return ("table", tid)
        return None

    if shop.player.has_order:
        return ("kitchen", 0)

    best_table = None
    best_dist = float("inf")
    px = shop.player.center_x
    py = shop.player.center_y
    for table in shop.tables:
        if (table.customer is not None
                and table.customer.state == CustomerState.WAITING_TO_ORDER):
            tx, ty = shop.get_table_interaction_point(table, px, py)
            dist = math.hypot(tx - px, ty - py)
            if dist < best_dist:
                best_dist = dist
                best_table = table.table_id

    if best_table is not None:
        return ("table", int(best_table))
    if shop.kitchen.ready:
        return ("kitchen", 0)
    return None


def _target_in_range(shop: Shop) -> bool:
    target = _get_primary_target_point(shop)
    if target is None:
        return False
    px = shop.player.center_x
    py = shop.player.center_y
    return math.hypot(target[0] - px, target[1] - py) <= INTERACT_RANGE * 0.9


def _obs_size(shop: Shop) -> int:
    """Compute observation-vector length for a shop."""
    return (
        4                            # player x, y, facing, carry_type
        + 2                          # carry_table_id, carry_menu_id
        + 4                          # can_move: up, down, left, right
        + shop.max_tables * 6        # table: x, y, occupied, state, menu_id, patience
        + 3                          # kitchen: cooking count, ready count, capacity ratio
        + 6                          # kitchen/bar/trash landmark positions (3×2)
        + 2                          # nearest-target direction vector (dx, dy)
        + 3                          # waiting queue: count ratio, first patience ratio, queue_full
        + 8                          # money_ratio, day_ratio, time_ratio, shop_rating,
                                     # can_upgrade, net_profit_ratio, employee_count, bartender
        + _NUM_UPGRADES * 3          # per-upgrade: level_ratio, affordable, unlocked
        + 1                          # trait_selection_active
        + 3 * 2                      # 3 trait choices × (trait_id_encoded, stacks_ratio)
    )


def build_observation(shop: Shop) -> np.ndarray:
    """Convert a Shop snapshot into a flat float32 observation vector."""
    size = _obs_size(shop)
    obs = np.zeros(size, dtype=np.float32)
    idx = 0

    # ── Player (pixel-based, normalised) ─────────
    map_px_w = max(1, shop.grid_width * TILE_SIZE)
    map_px_h = max(1, shop.grid_height * TILE_SIZE)
    obs[idx]     = shop.player.center_x / map_px_w
    obs[idx + 1] = shop.player.center_y / map_px_h
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

    # ── Local move feasibility (collision-aware) ─────────
    move_step = shop.player.speed * 0.2
    candidate_moves = [
        (0.0, -move_step),
        (0.0, move_step),
        (-move_step, 0.0),
        (move_step, 0.0),
    ]
    for move_idx, (dx, dy) in enumerate(candidate_moves):
        if shop._can_move_to(shop.player.x + dx, shop.player.y + dy):
            obs[idx + move_idx] = 1.0
    idx += 4

    # ── Tables (fixed-size: max_tables slots, now with XY) ────
    for i in range(shop.max_tables):
        if i < len(shop.tables):
            t = shop.tables[i]
            tx, ty = shop.get_table_interaction_point(t)
            obs[idx], obs[idx + 1] = _norm_point(shop, tx, ty)
            cust = t.customer
            if cust is not None:
                obs[idx + 2] = 1.0
                obs[idx + 3] = _STATE_ENC.get(cust.state, 0.0)
                mi = cust.menu_item
                if mi:
                    obs[idx + 4] = MENU_IDS.get(mi["id"], 0) / NUM_MENU
                obs[idx + 5] = cust.patience_ratio
        idx += 6

    # ── Kitchen ──────────────────────────────────
    obs[idx]     = shop.kitchen.num_cooking / max(1, shop.kitchen.cooking_capacity)
    obs[idx + 1] = len(shop.kitchen.ready) / max(1, shop.kitchen.storage_capacity)
    obs[idx + 2] = (shop.kitchen.num_cooking + len(shop.kitchen.ready)
                     ) / max(1, shop.kitchen.cooking_capacity + shop.kitchen.storage_capacity)
    idx += 3

    # ── Landmark positions (kitchen, bar, trash — averaged) ──
    kx, ky = shop.get_station_interaction_point(shop.kitchen_counter_positions)
    obs[idx], obs[idx + 1] = _norm_point(shop, kx, ky)
    bx, by = shop.get_station_interaction_point(shop.bar_counter_positions)
    obs[idx + 2], obs[idx + 3] = _norm_point(shop, bx, by)
    tx, ty = shop.get_station_interaction_point(shop.trash_can_positions)
    obs[idx + 4], obs[idx + 5] = _norm_point(shop, tx, ty)
    idx += 6

    # ── Nearest target direction vector (dx, dy) ─────────
    px_norm = shop.player.center_x / map_px_w
    py_norm = shop.player.center_y / map_px_h
    target_dx, target_dy = 0.0, 0.0
    target_point = _get_primary_target_point(shop)
    if target_point is None and shop.tables:
        # Fallback: direction toward average table position
        xs = [shop.get_table_interaction_point(t)[0] for t in shop.tables]
        ys = [shop.get_table_interaction_point(t)[1] for t in shop.tables]
        target_point = (sum(xs) / len(xs), sum(ys) / len(ys))
    if target_point is not None:
        tx, ty = _norm_point(shop, *target_point)
        target_dx = tx - px_norm
        target_dy = ty - py_norm
    obs[idx]     = np.clip(target_dx, -1.0, 1.0)
    obs[idx + 1] = np.clip(target_dy, -1.0, 1.0)
    idx += 2

    # ── Waiting queue (밖에서 대기 중인 잠재고객) ──
    max_q = max(1, shop.max_waiting_queue)
    obs[idx]     = len(shop.waiting_queue) / max_q  # queue utilization
    if shop.waiting_queue:
        obs[idx + 1] = shop.waiting_queue[0].waiting_patience_ratio  # first customer patience
    obs[idx + 2] = 1.0 if len(shop.waiting_queue) >= shop.max_waiting_queue else 0.0  # queue full
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
    obs[idx + 7] = 1.0 if shop.bartender_hired else 0.0
    idx += 8

    # ── Per-upgrade details (9 upgrades × 3) ────────────
    for upg in shop.upgrades_data:
        uid = upg["id"]
        level = shop.upgrade_levels.get(uid, 0)
        max_lv = max(1, upg["max_level"])
        obs[idx]     = level / max_lv                           # level ratio
        cost = int(upg["base_cost"] * (upg["cost_multiplier"] ** level))
        obs[idx + 1] = 1.0 if (level < max_lv and shop.money >= cost) else 0.0  # affordable
        req = upg.get("unlock_net_profit", 0)
        obs[idx + 2] = 1.0 if shop.net_profit >= req else 0.0  # unlocked
        idx += 3

    # ── Trait selection (1 + 3×2 = 7) ───────────────────
    obs[idx] = 1.0 if shop.trait_selection_active else 0.0
    idx += 1
    choices = getattr(shop, "trait_choices", []) or []
    for i in range(3):
        if i < len(choices):
            tid = choices[i].get("id", "")
            obs[idx]     = _TRAIT_IDS.get(tid, 0) / _NUM_TRAIT_TYPES
            ms = max(1, choices[i].get("max_stacks", 1))
            cur = shop.traits.get(tid, 0)
            obs[idx + 1] = cur / ms
        idx += 2

    return obs


# ────────────────────────────────────────────────
# Gymnasium Environment
# ────────────────────────────────────────────────
class TycoonEnv(gymnasium.Env):
    """Gymnasium environment for the restaurant management game."""

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 10}

    def __init__(self, render_mode=None, reward_config=None, **shop_kwargs):
        super().__init__()
        self.shop = Shop(**shop_kwargs)
        self.render_mode = render_mode
        self._reward_calc = RewardCalculator(reward_config)

        self.action_space = spaces.Discrete(NUM_ACTIONS)
        obs_len = _obs_size(self.shop)
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(obs_len,), dtype=np.float32)

        self._screen = None
        self._renderer = None
        self._prev_served: int = 0
        self._prev_lost: int = 0
        self._prev_potential: float = 0.0
        self._prev_net_profit: float = 0.0
        self._prev_rating: float = 0.0
        self._prev_final_score: float = 0.0
        self._prev_target_signature: tuple[str, int] | None = None
        self._prev_target_in_range: bool = False
        self._episode_event_totals: dict[str, float] = {}
        self._episode_reward_totals: dict[str, float] = {}
        self._episode_steps: int = 0

    # ── Potential-based reward shaping ────────────
    def _calc_potential(self) -> float:
        """State potential for reward shaping.

        Returns a value in [-1, 0] proportional to how close the player
        is to their current task objective.  Higher = closer to target.
        When no task target exists, guides toward the map center.
        """
        shop = self.shop
        px, py = shop.player.center_x, shop.player.center_y
        map_diag = math.hypot(shop.grid_width * TILE_SIZE,
                              shop.grid_height * TILE_SIZE)
        if map_diag == 0:
            return 0.0
        target_point = _get_primary_target_point(shop)
        if target_point is None:
            # Fallback: guide toward average of all table interaction points
            # so the agent naturally stays near the service area.
            if shop.tables:
                xs = [shop.get_table_interaction_point(t)[0] for t in shop.tables]
                ys = [shop.get_table_interaction_point(t)[1] for t in shop.tables]
                target_point = (sum(xs) / len(xs), sum(ys) / len(ys))
            else:
                return 0.0
        d = math.hypot(px - target_point[0], py - target_point[1])
        return -d / map_diag

    def _dense_shaping(self) -> float:
        """Potential-based shaping: F = φ(s') - φ(s).

        Guides the agent toward the nearest task-relevant target.
        γ=1.0 within episode to prevent stationary agents from
        accumulating spurious positive reward (0.99 leak bug fix).
        Capped per-step to prevent accumulation from dominating
        the actual gameplay rewards (service chain).
        """
        new_potential = self._calc_potential()
        F = new_potential - self._prev_potential   # γ=1.0 → 정지 시 F=0
        self._prev_potential = new_potential
        # Scale=3.0, cap ±1.0 → strong guidance, bounded per step
        return max(-1.0, min(1.0, F * 3.0))

    def _auto_face_nearest(self):
        """Face the player toward the nearest interactable within range.

        Called before INTERACT so the agent doesn't need to learn facing —
        a critical difficulty reduction for RL.
        """
        shop = self.shop
        px = shop.player.center_x
        py = shop.player.center_y
        best_dist = INTERACT_RANGE * 1.5
        best_dx, best_dy = 0.0, 0.0
        found = False

        def consider(target_x: float, target_y: float):
            nonlocal best_dist, best_dx, best_dy, found
            d = math.hypot(target_x - px, target_y - py)
            if d < best_dist:
                best_dist = d
                best_dx = target_x - px
                best_dy = target_y - py
                found = True

        if shop.player.has_food or shop.player.has_drink:
            first = shop.player.first_carried
            if first:
                tid = first.get("table_id", -1)
                # 고아 아이템 → 쓰레기통 방향
                target_table = None
                for table in shop.tables:
                    if table.table_id == tid:
                        target_table = table
                        break
                if target_table is None or target_table.customer is None:
                    # 쓰레기통으로 향함
                    for gx, gy in shop.trash_can_positions:
                        consider(gx * TILE_SIZE + TILE_SIZE / 2,
                                 gy * TILE_SIZE + TILE_SIZE / 2)
                else:
                    consider(target_table.center_x, target_table.center_y)
        elif shop.player.has_order:
            for gx, gy in shop.kitchen_counter_positions:
                consider(gx * TILE_SIZE + TILE_SIZE / 2,
                         gy * TILE_SIZE + TILE_SIZE / 2)
        else:
            for table in shop.tables:
                if (table.customer is not None
                        and table.customer.state == CustomerState.WAITING_TO_ORDER):
                    consider(table.center_x, table.center_y)
            if shop.kitchen.ready:
                for gx, gy in shop.kitchen_counter_positions:
                    consider(gx * TILE_SIZE + TILE_SIZE / 2,
                             gy * TILE_SIZE + TILE_SIZE / 2)
            if shop.bartender_hired and shop.bar.has_ready:
                for gx, gy in shop.bar_counter_positions:
                    consider(gx * TILE_SIZE + TILE_SIZE / 2,
                             gy * TILE_SIZE + TILE_SIZE / 2)
            if shop.player.carrying:
                for gx, gy in shop.trash_can_positions:
                    consider(gx * TILE_SIZE + TILE_SIZE / 2,
                             gy * TILE_SIZE + TILE_SIZE / 2)

        if found:
            if abs(best_dx) > abs(best_dy):
                shop.player.facing = (Player.FACING_RIGHT if best_dx > 0
                                      else Player.FACING_LEFT)
            else:
                shop.player.facing = (Player.FACING_DOWN if best_dy > 0
                                      else Player.FACING_UP)

    # ── Gymnasium API ────────────────────────────
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.shop.reset()
        self._prev_served = 0
        self._prev_lost = 0
        self._prev_potential = self._calc_potential()
        self._prev_net_profit = float(self.shop.net_profit)
        self._prev_rating = float(self.shop.shop_rating)
        self._prev_final_score = float(self.shop.final_score)
        self._prev_target_signature = _get_primary_target_signature(self.shop)
        self._prev_target_in_range = _target_in_range(self.shop)
        self._episode_event_totals = {}
        self._episode_reward_totals = {}
        self._episode_steps = 0
        return build_observation(self.shop), {}

    def step(self, action):
        action = int(action)

        # Auto-face nearest interactable on INTERACT
        if action == ACTION_INTERACT:
            self._auto_face_nearest()

        before_x = float(self.shop.player.x)
        before_y = float(self.shop.player.y)

        events = self.shop.step(action)

        # ── time_penalty: 매 스텝 소량 비용 (긴박감 유도) ──
        events.append(("time_penalty", 1.0))
        # ── idle_penalty: WAIT 액션 시 추가 페널티 ──
        if action == ACTION_NONE:
            events.append(("idle_penalty", 1.0))

        # ── upgrade_available: 업그레이드 가능한데 안 살 때 매 스텝 페널티 ──
        if not (UPGRADE_ACTION_MIN <= action <= UPGRADE_ACTION_MAX):
            shop = self.shop
            for upg in shop.upgrades_data:
                uid = upg["id"]
                level = shop.upgrade_levels[uid]
                if level < upg["max_level"]:
                    if upg.get("effect_type") == "hire_chef" and shop.num_chefs >= shop.max_chefs:
                        continue
                    req = upg.get("unlock_profit", 0)
                    if shop.net_profit < req:
                        continue
                    cost = (upg["cost_list"][level]
                            if "cost_list" in upg and level < len(upg["cost_list"])
                            else int(upg["base_cost"] * (upg["cost_multiplier"] ** level)))
                    if shop.money >= cost:
                        events.append(("upgrade_available", 1.0))
                        break

        if action in (0, 1, 2, 3):
            moved_dist = math.hypot(self.shop.player.x - before_x,
                                    self.shop.player.y - before_y)
            if moved_dist < 1e-3:
                events.append(("blocked_move", 1.0))
        current_target_signature = _get_primary_target_signature(self.shop)
        current_target_in_range = _target_in_range(self.shop)
        if (current_target_signature is not None
                and current_target_in_range
                and (current_target_signature != self._prev_target_signature
                     or not self._prev_target_in_range)):
            self._episode_event_totals["target_ready"] = (
                self._episode_event_totals.get("target_ready", 0.0) + 1.0
            )

        net_profit_delta = float(self.shop.net_profit) - self._prev_net_profit
        rating_delta = float(self.shop.shop_rating) - self._prev_rating
        final_score_delta = float(self.shop.final_score) - self._prev_final_score
        if abs(net_profit_delta) > 1e-6:
            events.append(("net_profit_delta", net_profit_delta))
        if abs(rating_delta) > 1e-6:
            events.append(("rating_delta", rating_delta))
        if abs(final_score_delta) > 1e-6:
            events.append(("final_score_delta", final_score_delta))
        reward, reward_breakdown, event_values = self._reward_calc.details(events)

        # Potential-based shaping (replaces old absolute proximity)
        dense_reward = self._dense_shaping()
        reward += dense_reward
        reward_breakdown["dense_shaping"] = reward_breakdown.get("dense_shaping", 0.0) + dense_reward

        self._episode_steps += 1
        for name, value in event_values.items():
            self._episode_event_totals[name] = self._episode_event_totals.get(name, 0.0) + value
        for name, value in reward_breakdown.items():
            self._episode_reward_totals[name] = self._episode_reward_totals.get(name, 0.0) + value

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
            "shop_rating_stars": self.shop.shop_rating_stars,
            "final_score": self.shop.final_score,
            "won": self.shop.won,
            "tables_active": len(self.shop.tables),
            "waiting_queue": len(self.shop.waiting_queue),
            "last_events": dict(event_values),
            "last_reward_breakdown": dict(reward_breakdown),
        }
        if terminated or truncated:
            info["episode_summary"] = {
                "steps": self._episode_steps,
                "customers_served": self.shop.customers_served,
                "customers_lost": self.shop.customers_lost,
                "net_profit": float(self.shop.net_profit),
                "shop_rating": float(self.shop.shop_rating),
                "final_score": float(self.shop.final_score),
                "event_totals": dict(self._episode_event_totals),
                "reward_totals": dict(self._episode_reward_totals),
            }
        self._prev_served = self.shop.customers_served
        self._prev_lost = self.shop.customers_lost
        self._prev_net_profit = float(self.shop.net_profit)
        self._prev_rating = float(self.shop.shop_rating)
        self._prev_final_score = float(self.shop.final_score)
        self._prev_target_signature = current_target_signature
        self._prev_target_in_range = current_target_in_range
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
                am = AssetManager()
                am.ensure_loaded()
                self._renderer = Renderer(am)

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
                am = AssetManager()
                am.ensure_loaded()
                self._renderer = Renderer(am)
            self._renderer.draw(surf, self.shop)
            return np.transpose(
                pygame.surfarray.array3d(surf), axes=(1, 0, 2))

    def close(self):
        if self._screen is not None:
            import pygame
            pygame.quit()
            self._screen = None
