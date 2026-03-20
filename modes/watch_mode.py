"""WatchMode – spectate a trained AI agent playing the game with full rendering."""

import json
import os

import numpy as np
import pygame

from config.settings import TILE_SIZE, UI_HEIGHT, COLORS, ACTION_NONE, NUM_ACTIONS
from modes.base_mode import BaseMode
from modes.model_runtime import load_model_runtime_options
from core.ranking import RankingManager
from core.shop import Shop
from rendering.asset_manager import AssetManager
from rendering.renderer import Renderer
from ai.agent import load_agent, _detect_algo_from_path
from ai.gym_env import build_observation

ACTION_NAMES = [
    "↑위", "↓아래", "←좌", "→우", "★상호작용", "·대기",
    "₩자동구매", "₩테이블", "₩종업원", "₩바텐더", "₩주방", "₩요리사",
]


def _auto_find_best_model() -> tuple[str | None, str | None]:
    models_dir = "models"
    if not os.path.isdir(models_dir):
        return None, None
    candidates = []
    for root, _dirs, files in os.walk(models_dir):
        for f in files:
            if f == "best_model.zip":
                candidates.append(os.path.join(root, f))
            elif f == "best_model.pt":
                candidates.append(os.path.join(root, f[:-3]))
    if not candidates:
        return None, None
    path = candidates[0]
    algo = _detect_algo_from_path(path)
    cfg_path = os.path.join(os.path.dirname(path), "train_config_used.json")
    if os.path.isfile(cfg_path):
        with open(cfg_path, encoding="utf-8") as fp:
            cfg = json.load(fp)
        algo = cfg.get("algorithm", algo) or algo
    return path, algo


class WatchMode(BaseMode):
    def __init__(self, *, model_path=None, algo_name=None,
                 target_money=None, day_limit=None,
                 speed_multiplier: float = 1.0):
        if model_path is None:
            model_path, algo_name = _auto_find_best_model()
            if model_path:
                print(f"  [관전 모드] 자동 탐지 모델: {model_path} ({algo_name})")

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
        w = self.shop.grid_width * TILE_SIZE
        h = self.shop.grid_height * TILE_SIZE + UI_HEIGHT
        super().__init__(w, h, title="RL 타이쿤 – 관전 모드")

        self.am = AssetManager()
        self.am.ensure_loaded()
        self.renderer = Renderer(self.am, background_key="sample3")
        self.ranking = RankingManager()

        self.agent = load_agent(model_path, algo_name=algo_name)
        if hasattr(self.agent, "deterministic"):
            self.agent.deterministic = False
        self.speed_multiplier = max(0.5, speed_multiplier)

        self._model_path = model_path
        self._algo_name = algo_name
        self._result_recorded = False
        self._last_action: int = ACTION_NONE
        self._last_probs: np.ndarray | None = None
        self._step_count: int = 0
        self._action_counts = [0] * NUM_ACTIONS

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
                return
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.running = False
                    return
                if event.key == pygame.K_r and self.shop.done:
                    self.shop.reset()
                    self._result_recorded = False
                    self._step_count = 0
                    self._action_counts = [0] * NUM_ACTIONS
                if event.key == pygame.K_d and hasattr(self.agent, "deterministic"):
                    self.agent.deterministic = not self.agent.deterministic
                if event.key == pygame.K_UP:
                    self.speed_multiplier = min(10.0, self.speed_multiplier + 0.5)
                if event.key == pygame.K_DOWN:
                    self.speed_multiplier = max(0.5, self.speed_multiplier - 0.5)

    def tick(self, dt):
        pass

    def update(self):
        if self.shop.done:
            return

        steps = max(1, int(self.speed_multiplier))
        for _ in range(steps):
            if self.shop.done:
                break
            obs = build_observation(self.shop)
            mask = self.shop.get_action_mask()
            action = self.agent.predict(obs, action_mask=mask)
            self._last_action = action
            self._step_count += 1
            if 0 <= action < NUM_ACTIONS:
                self._action_counts[action] += 1

            if hasattr(self.agent, "get_action_probs"):
                try:
                    self._last_probs = self.agent.get_action_probs(obs)
                except Exception:
                    self._last_probs = None

            self.shop.step(action)
            self.shop.auto_select_trait()

    def render(self):
        self.screen.fill(COLORS["background"])
        self.renderer.draw(self.screen, self.shop)

        available = [f.lower() for f in pygame.font.get_fonts()]
        kr_font = None
        for fn in ["malgungothic", "gulim", "dotum", "nanumgothic"]:
            if fn in available:
                kr_font = fn
                break
        font = pygame.font.SysFont(kr_font, 14)
        font_sm = pygame.font.SysFont(kr_font, 12)

        det_mode = "확정적" if getattr(self.agent, "deterministic", True) else "확률적"
        header = f"관전 모드 ×{self.speed_multiplier:.1f}  |  정책: {det_mode} [D 전환]  |  스텝: {self._step_count}"
        lbl = font.render(header, True, (180, 220, 255))
        self.screen.blit(lbl, (4, 2))

        panel_x = self.screen.get_width() - 180
        panel_y = 20
        a_name = ACTION_NAMES[self._last_action] if 0 <= self._last_action < NUM_ACTIONS else "?"
        act_lbl = font.render(f"행동: {a_name}", True, (255, 255, 200))
        self.screen.blit(act_lbl, (panel_x, panel_y))
        panel_y += 18

        if self._last_probs is not None:
            for i, (name, prob) in enumerate(zip(ACTION_NAMES, self._last_probs)):
                bar_w = int(prob * 100)
                color = (100, 200, 100) if i == self._last_action else (80, 80, 120)
                pygame.draw.rect(self.screen, color, (panel_x, panel_y, bar_w, 10))
                txt = font_sm.render(f"{name} {prob:.0%}", True, (200, 200, 200))
                self.screen.blit(txt, (panel_x + 105, panel_y - 1))
                panel_y += 13

        if self._step_count > 0:
            panel_y += 4
            dist_lbl = font_sm.render("── 행동 분포 ──", True, (150, 150, 150))
            self.screen.blit(dist_lbl, (panel_x, panel_y))
            panel_y += 14
            for name, cnt in zip(ACTION_NAMES, self._action_counts):
                pct = cnt / self._step_count * 100
                txt = font_sm.render(f"{name}: {pct:.0f}%", True, (170, 170, 170))
                self.screen.blit(txt, (panel_x, panel_y))
                panel_y += 12

        hint = font_sm.render("↑↓: 속도  D: 정책전환  R: 재시작  ESC: 종료", True, (120, 120, 120))
        h = self.screen.get_height()
        self.screen.blit(hint, (4, h - 14))

        if self.shop.done:
            if not self._result_recorded:
                result = self.shop.get_game_result()
                self.ranking.record_result("AI", result)
                self._result_recorded = True
                rank = self.ranking.get_rank(self.shop.final_score, self.shop.day_limit)
                self._rank_text = f"랭킹: #{rank}"
            self.renderer.draw_game_over(
                self.screen,
                self.shop,
                extra_text=getattr(self, "_rank_text", ""),
            )

        pygame.display.flip()
