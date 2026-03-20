"""HumanMode – single-player mode controlled by keyboard.

Movement is handled every frame (tick) for smooth pixel movement.
Game logic (cooking, customers, interaction) runs at the fixed step rate.
"""

import pygame
from config.settings import (
    TILE_SIZE, UI_HEIGHT, COLORS,
    ACTION_INTERACT, ACTION_NONE,
)
from modes.base_mode import BaseMode
from core.shop import Shop
from core.ranking import RankingManager
from rendering.asset_manager import AssetManager
from rendering.renderer import Renderer


class HumanMode(BaseMode):

    def __init__(self, *, target_money=None, day_limit=None,
                 time_scale: float = 1.0):
        self.shop = Shop(target_money=target_money, day_limit=day_limit)
        w = self.shop.grid_width * TILE_SIZE
        h = self.shop.grid_height * TILE_SIZE + UI_HEIGHT
        super().__init__(w, h, title="RL 타이쿤 – 솔로 모드",
                         time_scale=time_scale)

        self.am = AssetManager()
        self.am.ensure_loaded()
        self.renderer = Renderer(self.am, background_key="sample1")
        self.ranking = RankingManager()
        self._interact_pressed = False
        self._result_recorded = False

    # ── events ───────────────────────────────────
    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
                return
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_RIGHTBRACKET:
                    self.increase_speed()
                    continue
                if event.key == pygame.K_LEFTBRACKET:
                    self.decrease_speed()
                    continue
                if event.key == pygame.K_0:
                    self.reset_speed()
                    continue
                if event.key == pygame.K_ESCAPE:
                    if self.shop.upgrade_mode:
                        self.shop.upgrade_mode = False
                    elif self.shop.trait_selection_active:
                        pass  # can't dismiss trait popup
                    else:
                        self.running = False
                        return

                # ── Trait selection (highest priority) ──
                if self.shop.trait_selection_active:
                    for i, k in enumerate([pygame.K_1, pygame.K_2, pygame.K_3]):
                        if event.key == k:
                            self.shop.select_trait(i)
                    continue

                if event.key == pygame.K_u:
                    self.shop.upgrade_mode = not self.shop.upgrade_mode

                # Tab key to cycle upgrade tabs
                if self.shop.upgrade_mode and event.key == pygame.K_TAB:
                    self.shop.upgrade_tab = (self.shop.upgrade_tab + 1) % 3

                # Upgrade buying (when panel is open)
                if self.shop.upgrade_mode:
                    num_keys = [pygame.K_1, pygame.K_2, pygame.K_3,
                                pygame.K_4, pygame.K_5, pygame.K_6,
                                pygame.K_7, pygame.K_8, pygame.K_9]
                    for i, k in enumerate(num_keys):
                        if event.key == k:
                            self.shop.buy_upgrade_by_index(i)

                if event.key in (pygame.K_SPACE, pygame.K_RETURN):
                    self._interact_pressed = True
                if event.key == pygame.K_r and self.shop.done:
                    self.shop.reset()
                    self._result_recorded = False

    # ── per-frame smooth movement ────────────────
    def tick(self, dt):
        if self.shop.done:
            return
        keys = pygame.key.get_pressed()
        dx, dy = 0.0, 0.0
        if keys[pygame.K_UP] or keys[pygame.K_w]:
            dy -= 1
        if keys[pygame.K_DOWN] or keys[pygame.K_s]:
            dy += 1
        if keys[pygame.K_LEFT] or keys[pygame.K_a]:
            dx -= 1
        if keys[pygame.K_RIGHT] or keys[pygame.K_d]:
            dx += 1
        self.shop.move_player_continuous(dx, dy, dt)

    # ── logic tick (fixed rate) ──────────────────
    def update(self):
        if self.shop.done:
            return
        action = ACTION_NONE
        if self._interact_pressed:
            action = ACTION_INTERACT
            self._interact_pressed = False
        self.shop.step_logic(action)

    # ── draw ─────────────────────────────────────
    def render(self):
        self.screen.fill(COLORS["background"])
        self.renderer.draw(self.screen, self.shop)
        if self.shop.done:
            # Record ranking on first game-over frame
            if not self._result_recorded:
                result = self.shop.get_game_result()
                self.ranking.record_result("Player", result)
                rank = self.ranking.get_rank(
                    self.shop.final_score, self.shop.day_limit)
                self._rank_text = f"랭킹: #{rank}"
                self._result_recorded = True
            self.renderer.draw_game_over(
                self.screen, self.shop,
                extra_text=getattr(self, "_rank_text", ""))
        pygame.display.flip()
