"""AIMode - single-shop observer mode driven by a trained AI model."""

import pygame

from config.settings import TILE_SIZE, UI_HEIGHT, COLORS
from modes.base_mode import BaseMode
from modes.model_runtime import load_model_runtime_options
from core.shop import Shop
from rendering.asset_manager import AssetManager
from rendering.renderer import Renderer
from ai.agent import load_agent
from ai.controller import decide_override_action


class AIMode(BaseMode):
    """Run one shop controlled entirely by the trained AI."""

    def __init__(self, *, model_path=None,
                 target_money=None, day_limit=None,
                 time_scale: float = 1.0,
                 use_rule_controller: bool = False,
                 stochastic: bool = False):
        game_overrides, env_options = load_model_runtime_options(model_path)
        if target_money is None:
            target_money = game_overrides.get("target_money")
        if day_limit is None:
            day_limit = game_overrides.get("day_limit")
        self.shop = Shop(
            target_money=target_money,
            day_limit=day_limit,
            **env_options,
        )
        self.use_rule_controller = use_rule_controller
        w = self.shop.grid_width * TILE_SIZE
        h = self.shop.grid_height * TILE_SIZE + UI_HEIGHT
        super().__init__(w, h, title="RL 타이쿤 - AI 관찰 모드",
                         time_scale=time_scale)

        self.am = AssetManager()
        self.renderer = Renderer(self.am)
        self.agent = load_agent(model_path)
        if hasattr(self.agent, "deterministic"):
            self.agent.deterministic = not stochastic

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
                    self.running = False
                    return
                if event.key == pygame.K_r and self.shop.done:
                    self.shop.reset()

    def update(self):
        if self.shop.done:
            return

        action = self._decide_ai_action()
        self.shop.step(action)
        self.shop.auto_select_trait()

    def render(self):
        self.screen.fill(COLORS["background"])
        self.renderer.draw(self.screen, self.shop)
        if self.shop.done:
            self.renderer.draw_game_over(self.screen, self.shop)
        pygame.display.flip()

    def _build_ai_obs(self):
        from ai.gym_env import build_observation
        return build_observation(self.shop)

    def _decide_ai_action(self):
        if self.use_rule_controller:
            heuristic_action = decide_override_action(self.shop)
            if heuristic_action is not None:
                return heuristic_action
        obs = self._build_ai_obs()
        return self.agent.predict(
            obs, action_mask=self.shop.get_action_mask())
