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
from rendering.asset_manager import AssetManager
from rendering.renderer import Renderer


class HumanMode(BaseMode):

    def __init__(self, *, target_money=None, day_limit=None):
        self.shop = Shop(target_money=target_money, day_limit=day_limit)
        w = self.shop.grid_width * TILE_SIZE
        h = self.shop.grid_height * TILE_SIZE + UI_HEIGHT
        super().__init__(w, h, title="RL Tycoon – Human Mode")

        self.am = AssetManager()
        self.renderer = Renderer(self.am)
        self._interact_pressed = False

    # ── events ───────────────────────────────────
    def handle_events(self):
        self._interact_pressed = False
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
                return
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    if self.shop.upgrade_mode:
                        self.shop.upgrade_mode = False
                    else:
                        self.running = False
                        return
                if event.key == pygame.K_u:
                    self.shop.upgrade_mode = not self.shop.upgrade_mode
                # Upgrade buying (when panel is open)
                if self.shop.upgrade_mode:
                    for i, k in enumerate([pygame.K_1, pygame.K_2,
                                           pygame.K_3, pygame.K_4,
                                           pygame.K_5]):
                        if event.key == k:
                            self.shop.buy_upgrade_by_index(i)
                if event.key in (pygame.K_SPACE, pygame.K_RETURN):
                    self._interact_pressed = True
                if event.key == pygame.K_r and self.shop.done:
                    self.shop.reset()

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
            self.renderer.draw_game_over(self.screen, self.shop)
        pygame.display.flip()
