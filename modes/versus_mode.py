"""VersusMode – Human (left) vs AI (right) split-screen competition.

Human side uses smooth per-frame movement (tick) + step_logic.
AI side uses the standard step() (movement + logic combined).
"""

import pygame
from config.settings import (
    TILE_SIZE, UI_HEIGHT, COLORS,
    VERSUS_DIVIDER_WIDTH, VERSUS_DIVIDER_COLOR,
    ACTION_INTERACT, ACTION_NONE,
)
from modes.base_mode import BaseMode
from core.shop import Shop
from rendering.asset_manager import AssetManager
from rendering.renderer import Renderer
from ai.agent import load_agent


class VersusMode(BaseMode):

    def __init__(self, *, model_path=None,
                 target_money=None, day_limit=None):
        self.human_shop = Shop(target_money=target_money,
                               day_limit=day_limit)
        self.ai_shop = Shop(target_money=target_money,
                            day_limit=day_limit)

        self.map_w = self.human_shop.grid_width * TILE_SIZE
        self.map_h = self.human_shop.grid_height * TILE_SIZE + UI_HEIGHT

        screen_w = self.map_w * 2 + VERSUS_DIVIDER_WIDTH
        screen_h = self.map_h
        super().__init__(screen_w, screen_h,
                         title="RL Tycoon – Human vs AI")

        self.am = AssetManager()
        self.renderer_left = Renderer(self.am)
        self.renderer_right = Renderer(self.am)

        self.agent = load_agent(model_path)

        self._interact_pressed = False
        self.winner: str | None = None

    # ── events ───────────────────────────────────
    def handle_events(self):
        self._interact_pressed = False
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
                return
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    if self.human_shop.upgrade_mode:
                        self.human_shop.upgrade_mode = False
                    else:
                        self.running = False
                        return
                if event.key == pygame.K_u:
                    self.human_shop.upgrade_mode = not self.human_shop.upgrade_mode
                if self.human_shop.upgrade_mode:
                    for i, k in enumerate([pygame.K_1, pygame.K_2,
                                           pygame.K_3, pygame.K_4,
                                           pygame.K_5]):
                        if event.key == k:
                            self.human_shop.buy_upgrade_by_index(i)
                if event.key in (pygame.K_SPACE, pygame.K_RETURN):
                    self._interact_pressed = True
                if event.key == pygame.K_r and self._game_over():
                    self.human_shop.reset()
                    self.ai_shop.reset()
                    self.winner = None

    # ── per-frame smooth movement (human) ────────
    def tick(self, dt):
        if self._game_over():
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
        self.human_shop.move_player_continuous(dx, dy, dt)

    # ── logic tick ───────────────────────────────
    def update(self):
        if self._game_over():
            return

        # Human – game logic only (movement in tick)
        action = ACTION_NONE
        if self._interact_pressed:
            action = ACTION_INTERACT
            self._interact_pressed = False
        self.human_shop.step_logic(action)

        # AI – full step (movement + game logic)
        obs = self._build_ai_obs()
        ai_action = self.agent.predict(obs)
        self.ai_shop.step(ai_action)

        # Sync time (both shops share the same clock)
        self.ai_shop.time_elapsed = self.human_shop.time_elapsed

        # Check winner
        if self.human_shop.won and self.ai_shop.won:
            self.winner = "draw"
        elif self.human_shop.won:
            self.winner = "human"
            self.ai_shop.done = True
        elif self.ai_shop.won:
            self.winner = "ai"
            self.human_shop.done = True
        elif self.human_shop.done and self.ai_shop.done:
            if self.human_shop.money > self.ai_shop.money:
                self.winner = "human"
            elif self.ai_shop.money > self.human_shop.money:
                self.winner = "ai"
            else:
                self.winner = "draw"

    # ── draw ─────────────────────────────────────
    def render(self):
        self.screen.fill(COLORS["background"])

        # Left half – human
        self.renderer_left.draw(self.screen, self.human_shop,
                                offset_x=0, offset_y=0)
        # Divider
        dv_x = self.map_w
        pygame.draw.rect(self.screen, VERSUS_DIVIDER_COLOR,
                         (dv_x, 0, VERSUS_DIVIDER_WIDTH, self.map_h))

        # Right half – AI
        right_ox = self.map_w + VERSUS_DIVIDER_WIDTH
        self.renderer_right.draw(self.screen, self.ai_shop,
                                 offset_x=right_ox, offset_y=0)

        # Labels
        font = pygame.font.SysFont(None, 20)
        lbl_h = font.render("HUMAN", True, (100, 200, 255))
        self.screen.blit(lbl_h, (self.map_w // 2
                                 - lbl_h.get_width() // 2, 2))
        lbl_a = font.render("AI", True, (255, 150, 100))
        self.screen.blit(lbl_a, (right_ox + self.map_w // 2
                                 - lbl_a.get_width() // 2, 2))

        # Game-over overlays
        if self._game_over():
            extra = self._winner_text()
            self.renderer_left.draw_game_over(
                self.screen, self.human_shop, extra_text=extra)
            self.renderer_right.draw_game_over(
                self.screen, self.ai_shop,
                offset_x=right_ox, extra_text=extra)

        pygame.display.flip()

    # ── helpers ──────────────────────────────────
    def _game_over(self) -> bool:
        return self.human_shop.done and self.ai_shop.done

    def _winner_text(self) -> str:
        if self.winner == "human":
            return "Winner: HUMAN!"
        if self.winner == "ai":
            return "Winner: AI!"
        return "It's a DRAW!"

    def _build_ai_obs(self):
        """Lightweight observation for the AI agent.

        Uses the same format as ``ai.gym_env.build_observation``.
        Importing it here avoids heavyweight numpy at module level.
        """
        from ai.gym_env import build_observation
        return build_observation(self.ai_shop)
