"""Gymnasium environment wrapper for the restaurant management sim."""

import json
import os
import numpy as np
import gymnasium
from gymnasium import spaces

from config.settings import (
    TILE_SIZE,
    UI_HEIGHT,
    COLORS,
    ASSETS_DIR,
    NUM_ACTIONS,
    load_json_config,
)
from core.shop import Shop
from core.customer import CustomerState
from ai.reward import RewardCalculator

_menu_data = load_json_config("menu.json")
MENU_IDS = {item["id"]: i + 1 for i, item in enumerate(_menu_data)}
NUM_MENU = len(MENU_IDS) + 1

_STATE_ENC = {
    CustomerState.WAITING_TO_ORDER: 0.25,
    CustomerState.ORDER_TAKEN: 0.50,
    CustomerState.EATING: 0.75,
}


def _priority_table(shop: Shop):
    if shop.player.carrying:
        target_ids = {
            item.get("table_id") for item in shop.player.carrying
            if item.get("table_id") is not None
        }
        for table in shop.tables:
            if table.table_id in target_ids and table.customer is not None:
                return table

    best_table = None
    best_score = float("-inf")
    px = shop.player.center_x
    py = shop.player.center_y
    for table in shop.tables:
        cust = table.customer
        if cust is None:
            continue
        if cust.state == CustomerState.WAITING_TO_ORDER:
            state_score = 3.0
        elif cust.state == CustomerState.ORDER_TAKEN:
            state_score = 2.0
        elif cust.state == CustomerState.EATING:
            state_score = 1.0
        else:
            continue

        dist = np.hypot(table.center_x - px, table.center_y - py)
        urgency = (1.0 - cust.patience_ratio) * 2.0
        score = state_score + urgency - (dist / max(1.0, shop.grid_width * TILE_SIZE))
        if score > best_score:
            best_score = score
            best_table = table
    return best_table


def _obs_size(shop: Shop) -> int:
    return (
        4
        + 4
        + 6
        + shop.max_tables * 6
        + 8
        + 25
        + 10
    )


def build_observation(shop: Shop) -> np.ndarray:
    size = _obs_size(shop)
    obs = np.zeros(size, dtype=np.float32)
    idx = 0

    map_px_w = max(1, shop.grid_width * TILE_SIZE)
    map_px_h = max(1, shop.grid_height * TILE_SIZE)
    obs[idx] = shop.player.x / map_px_w
    obs[idx + 1] = shop.player.y / map_px_h
    obs[idx + 2] = shop.player.facing / 3.0
    if shop.player.has_order:
        obs[idx + 3] = 0.33
    elif shop.player.has_food:
        obs[idx + 3] = 0.66
    elif shop.player.has_drink:
        obs[idx + 3] = 1.0
    idx += 4

    first = shop.player.first_carried
    if first:
        tid = first.get("table_id", 0)
        obs[idx] = tid / max(1, shop.max_tables)
        mi = first.get("menu_item") or (first.get("items", [None])[0] if first.get("items") else None)
        if mi:
            mid = MENU_IDS.get(mi.get("id", ""), 0)
            obs[idx + 1] = mid / NUM_MENU
        target_table = next((t for t in shop.tables if t.table_id == tid), None)
        if target_table is not None:
            obs[idx + 2] = np.clip(
                (target_table.center_x - shop.player.center_x) / map_px_w, -1.0, 1.0)
            obs[idx + 3] = np.clip(
                (target_table.center_y - shop.player.center_y) / map_px_h, -1.0, 1.0)
    idx += 4

    priority_table = _priority_table(shop)
    if priority_table is not None:
        cust = priority_table.customer
        obs[idx] = 1.0
        if cust is not None:
            obs[idx + 1] = _STATE_ENC.get(cust.state, 0.0)
            mi = cust.menu_item
            if mi:
                obs[idx + 2] = MENU_IDS.get(mi["id"], 0) / NUM_MENU
            obs[idx + 3] = cust.patience_ratio
        obs[idx + 4] = np.clip(
            (priority_table.center_x - shop.player.center_x) / map_px_w, -1.0, 1.0)
        obs[idx + 5] = np.clip(
            (priority_table.center_y - shop.player.center_y) / map_px_h, -1.0, 1.0)
    idx += 6

    for i in range(shop.max_tables):
        if i < len(shop.tables):
            table = shop.tables[i]
            cust = table.customer
            if cust is not None:
                obs[idx] = 1.0
                obs[idx + 1] = _STATE_ENC.get(cust.state, 0.0)
                mi = cust.menu_item
                if mi:
                    obs[idx + 2] = MENU_IDS.get(mi["id"], 0) / NUM_MENU
                obs[idx + 3] = cust.patience_ratio
            obs[idx + 4] = np.clip(
                (table.center_x - shop.player.center_x) / map_px_w, -1.0, 1.0)
            obs[idx + 5] = np.clip(
                (table.center_y - shop.player.center_y) / map_px_h, -1.0, 1.0)
        idx += 6

    kitchen_x, kitchen_y = shop._kitchen_center()
    obs[idx] = np.clip((kitchen_x - shop.player.center_x) / map_px_w, -1.0, 1.0)
    obs[idx + 1] = np.clip((kitchen_y - shop.player.center_y) / map_px_h, -1.0, 1.0)
    if shop.bar_counter_positions:
        bar_x, bar_y = shop._bar_center()
        obs[idx + 2] = np.clip((bar_x - shop.player.center_x) / map_px_w, -1.0, 1.0)
        obs[idx + 3] = np.clip((bar_y - shop.player.center_y) / map_px_h, -1.0, 1.0)
    if shop.trash_can_positions:
        trash_x, trash_y = shop._trash_center()
        obs[idx + 4] = np.clip((trash_x - shop.player.center_x) / map_px_w, -1.0, 1.0)
        obs[idx + 5] = np.clip((trash_y - shop.player.center_y) / map_px_h, -1.0, 1.0)
    obs[idx + 6] = shop.kitchen.num_cooking / max(1, shop.kitchen.cooking_capacity)
    obs[idx + 7] = len(shop.kitchen.ready) / max(1, shop.kitchen.storage_capacity)
    idx += 8

    obs[idx] = min(1.0, shop.money / max(1, shop.target_money))
    obs[idx + 1] = shop.current_day / max(1, shop.day_limit)
    obs[idx + 2] = 1.0 - min(1.0, shop.time_elapsed / max(1, shop.total_time_limit))
    obs[idx + 3] = shop.shop_rating

    tracked_upgrades = [
        "buy_table",
        "hire_waiter",
        "hire_bartender",
        "kitchen_expand",
        "hire_chef",
    ]
    can_buy = 1.0 if any(shop.can_buy_upgrade(uid) for uid in tracked_upgrades) else 0.0
    obs[idx + 4] = can_buy
    obs[idx + 5] = min(1.0, shop.net_profit / max(1, shop.target_money))
    obs[idx + 6] = len(shop.employees) / 4.0

    bar_del = 0.0
    if shop.bartender_hired:
        bar_del += 0.5
    if getattr(shop, "delivery_unlocked", False):
        bar_del += 0.5
    obs[idx + 7] = bar_del
    obs[idx + 8] = 1.0 if shop.has_stale_carry() else 0.0
    obs[idx + 9] = 1.0 if shop.has_urgent_work() else 0.0
    obs[idx + 10] = 1.0 if shop.trait_selection_active else 0.0
    best_buy = shop._get_best_auto_buy_choice()
    if best_buy is not None:
        obs[idx + 11] = np.clip(best_buy["score"] / 150.0, 0.0, 1.0)
    queue_len = len(shop.waiting_queue)
    obs[idx + 12] = queue_len / max(1, shop.max_waiting_queue)
    obs[idx + 13] = 1.0 if queue_len >= shop.max_waiting_queue else 0.0
    if queue_len > 0:
        oldest_wait = shop.waiting_queue[0]
        obs[idx + 14] = oldest_wait.patience_ratio
    else:
        obs[idx + 14] = 0.0
    for offset, upgrade_id in enumerate(tracked_upgrades):
        obs[idx + 15 + offset] = 1.0 if shop.can_buy_upgrade(upgrade_id) else 0.0
    for offset, upgrade_id in enumerate(tracked_upgrades):
        cost = shop.get_upgrade_next_cost(upgrade_id)
        if cost is None:
            obs[idx + 20 + offset] = 0.0
        else:
            obs[idx + 20 + offset] = min(1.0, cost / max(1, shop.target_money))
    idx += 25

    occupied_tables = sum(1 for table in shop.tables if table.customer is not None)
    empty_tables = max(0, len(shop.tables) - occupied_tables)
    obs[idx] = occupied_tables / max(1, len(shop.tables))
    obs[idx + 1] = empty_tables / max(1, len(shop.tables))
    obs[idx + 2] = shop.num_chefs / max(1, shop.max_chefs)
    waiter_level = float(shop.upgrade_levels.get("hire_waiter", 0))
    obs[idx + 3] = waiter_level / max(1.0, shop._get_upgrade_data("hire_waiter")["max_level"])
    obs[idx + 4] = shop.kitchen.num_cooking / max(1, shop.kitchen.cooking_capacity)
    obs[idx + 5] = len(shop.kitchen.ready) / max(1, shop.kitchen.storage_capacity)

    waiter_cost = shop.get_upgrade_next_cost("hire_waiter")
    table_cost = shop.get_upgrade_next_cost("buy_table")
    chef_cost = shop.get_upgrade_next_cost("hire_chef")
    kitchen_cost = shop.get_upgrade_next_cost("kitchen_expand")
    obs[idx + 6] = 0.0 if waiter_cost is None else min(1.0, shop.money / max(1, waiter_cost))
    obs[idx + 7] = 0.0 if table_cost is None else min(1.0, shop.money / max(1, table_cost))
    obs[idx + 8] = 0.0 if chef_cost is None else min(1.0, shop.money / max(1, chef_cost))
    obs[idx + 9] = 0.0 if kitchen_cost is None else min(1.0, shop.money / max(1, kitchen_cost))

    return obs


class TycoonEnv(gymnasium.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 10}

    def __init__(self, render_mode=None, reward_config=None,
                 analysis_log_dir: str | None = None,
                 env_rank: int = 0,
                 **shop_kwargs):
        super().__init__()
        self.shop = Shop(**shop_kwargs)
        self.render_mode = render_mode
        self._reward_calc = RewardCalculator(reward_config)
        self.analysis_log_dir = analysis_log_dir
        self.env_rank = env_rank

        self.action_space = spaces.Discrete(NUM_ACTIONS)
        obs_len = _obs_size(self.shop)
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(obs_len,), dtype=np.float32)

        self._screen = None
        self._renderer = None

    def _write_episode_analysis(self, summary: dict):
        if not self.analysis_log_dir:
            return
        os.makedirs(self.analysis_log_dir, exist_ok=True)
        out_path = os.path.join(
            self.analysis_log_dir, f"episode_analysis_env{self.env_rank}.jsonl")
        with open(out_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(summary, ensure_ascii=False) + "\n")

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.shop.reset()
        return build_observation(self.shop), {}

    def step(self, action):
        events = self.shop.step(int(action))
        self.shop.auto_select_trait()
        reward = self._reward_calc(events)
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
        if terminated:
            summary = self.shop.get_game_result()
            info["episode_summary"] = summary
            self._write_episode_analysis(summary)
        return obs, reward, terminated, truncated, info

    def action_masks(self):
        return self.shop.get_action_mask()

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
                pygame.display.set_caption("RL Tycoon")
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
            return np.transpose(pygame.surfarray.array3d(surf), axes=(1, 0, 2))

    def close(self):
        if self._screen is not None:
            import pygame

            pygame.quit()
            self._screen = None
