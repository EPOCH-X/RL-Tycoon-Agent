"""WatchMode – spectate a trained AI agent playing the game with full rendering.

The agent controls the shop autonomously while you watch.
  ESC   – quit
  R     – restart (after game-over)
  D     – toggle deterministic / stochastic policy
  ↑/↓   – adjust speed (×0.5 ~ ×10)
"""

import json
import os
import pygame
import numpy as np

from config.settings import (
    TILE_SIZE, UI_HEIGHT, COLORS,
    ACTION_NONE,
)
from modes.base_mode import BaseMode
from core.shop import Shop
from core.ranking import RankingManager
from rendering.asset_manager import AssetManager
from rendering.renderer import Renderer
from ai.agent import load_agent
from ai.gym_env import build_observation, _get_primary_target_signature
from core.customer import CustomerState

ACTION_NAMES = ["↑위", "↓아래", "←좌", "→우", "★상호작용", "·대기", "₩업그레이드"]

# 디버그: True면 전반 분석용 로그를 watch_debug.log에 기록
WATCH_DEBUG_IDLE_AT_TABLE = True
WATCH_DEBUG_LOG_FILE = "watch_debug.log"  # 프로젝트 루트
# 주기적 요약(스텝 간격), 0이면 주기 로그 없음
WATCH_DEBUG_SUMMARY_EVERY_N_STEPS = 50


class WatchMode(BaseMode):

    def __init__(self, *, model_path=None, algo_name=None,
                 target_money=None, day_limit=None,
                 speed_multiplier: float = 1.0):
        # Auto-detect day_limit from saved model config if not specified
        if day_limit is None and model_path:
            cfg_path = os.path.join(os.path.dirname(model_path),
                                    "train_config_used.json")
            if os.path.isfile(cfg_path):
                with open(cfg_path, encoding="utf-8") as f:
                    tcfg = json.load(f)
                ov = tcfg.get("game_overrides", {})
                if ov.get("day_limit") is not None:
                    day_limit = ov["day_limit"]

        self.shop = Shop(target_money=target_money, day_limit=day_limit)
        w = self.shop.grid_width * TILE_SIZE
        h = self.shop.grid_height * TILE_SIZE + UI_HEIGHT
        super().__init__(w, h, title="RL 타이쿤 – 관전 모드")

        self.am = AssetManager()
        self.renderer = Renderer(self.am)
        self.ranking = RankingManager()

        self.agent = load_agent(model_path, algo_name=algo_name)
        self.speed_multiplier = max(0.5, speed_multiplier)

        self._model_path = model_path
        self._algo_name = algo_name
        self._result_recorded = False

        # Debug state
        self._last_action: int = ACTION_NONE
        self._last_probs: np.ndarray | None = None
        self._step_count: int = 0
        self._action_counts = [0] * 7

        self._debug_log_path = None
        if WATCH_DEBUG_IDLE_AT_TABLE and WATCH_DEBUG_LOG_FILE:
            self._debug_log_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                WATCH_DEBUG_LOG_FILE,
            )
            try:
                import datetime
                with open(self._debug_log_path, "a", encoding="utf-8") as f:
                    f.write(f"=== watch 디버그 세션 시작 {datetime.datetime.now().isoformat()} ===\n")
            except Exception:
                pass

    # ── events ───────────────────────────────────
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
                    self._action_counts = [0] * 7
                # Toggle deterministic
                if event.key == pygame.K_d:
                    if hasattr(self.agent, 'deterministic'):
                        self.agent.deterministic = not self.agent.deterministic
                # Speed controls
                if event.key == pygame.K_UP:
                    self.speed_multiplier = min(10.0, self.speed_multiplier + 0.5)
                if event.key == pygame.K_DOWN:
                    self.speed_multiplier = max(0.5, self.speed_multiplier - 0.5)

    # ── tick (no manual movement) ────────────────
    def tick(self, dt):
        pass

    # ── logic tick – agent acts ──────────────────
    def update(self):
        if self.shop.done:
            return

        steps = max(1, int(self.speed_multiplier))
        for _ in range(steps):
            if self.shop.done:
                break
            obs = build_observation(self.shop)
            target_sig = _get_primary_target_signature(self.shop)
            action = self.agent.predict(obs)
            self._last_action = action
            self._step_count += 1
            if 0 <= action < 7:
                self._action_counts[action] += 1

            # 전반 분석용 디버그 로그 (파일에만 기록)
            if WATCH_DEBUG_IDLE_AT_TABLE and self._debug_log_path:
                waiting_tids = [
                    t.table_id for t in self.shop.tables
                    if t.customer is not None
                    and t.customer.state == CustomerState.WAITING_TO_ORDER
                ]
                carry = "order" if self.shop.player.has_order else (
                    "food" if self.shop.player.has_food else (
                        "drink" if self.shop.player.has_drink else "idle"
                    )
                )
                target_str = f"{target_sig[0]}{target_sig[1]}" if target_sig else "None"
                act_name = ACTION_NAMES[action] if 0 <= action < 7 else f"act{action}"
                px, py = self.shop.player.center_x, self.shop.player.center_y
                line = (
                    f"step={self._step_count} | target={target_str} | action={action}({act_name}) | "
                    f"carry={carry} | waiting_tables={waiting_tids} | pos=({px:.0f},{py:.0f})\n"
                )
                do_log = False
                if waiting_tids:
                    do_log = True
                if action == ACTION_NONE:
                    do_log = True
                if WATCH_DEBUG_SUMMARY_EVERY_N_STEPS and self._step_count % WATCH_DEBUG_SUMMARY_EVERY_N_STEPS == 0:
                    do_log = True
                if do_log:
                    try:
                        with open(self._debug_log_path, "a", encoding="utf-8") as f:
                            f.write(line)
                    except Exception:
                        pass

            # Get action probabilities for debug display
            if hasattr(self.agent, 'get_action_probs'):
                try:
                    self._last_probs = self.agent.get_action_probs(obs)
                except Exception:
                    self._last_probs = None

            self.shop.step(action)
            self.shop.auto_select_trait()

    # ── draw ─────────────────────────────────────
    def render(self):
        self.screen.fill(COLORS["background"])
        self.renderer.draw(self.screen, self.shop)

        # ── Find Korean font ──
        available = [f.lower() for f in pygame.font.get_fonts()]
        kr_font = None
        for fn in ["malgungothic", "gulim", "dotum", "nanumgothic"]:
            if fn in available:
                kr_font = fn
                break
        font = pygame.font.SysFont(kr_font, 14)
        font_sm = pygame.font.SysFont(kr_font, 12)

        # ── Top-left: mode info ──
        det_mode = "확정적" if getattr(self.agent, 'deterministic', True) else "확률적"
        header = f"관전 모드 ×{self.speed_multiplier:.1f}  |  정책: {det_mode} [D 전환]  |  스텝: {self._step_count}"
        lbl = font.render(header, True, (180, 220, 255))
        self.screen.blit(lbl, (4, 2))

        # ── Right side: action debug panel ──
        panel_x = self.screen.get_width() - 180
        panel_y = 20
        # Current action
        a_name = ACTION_NAMES[self._last_action] if 0 <= self._last_action < 7 else "?"
        act_lbl = font.render(f"행동: {a_name}", True, (255, 255, 200))
        self.screen.blit(act_lbl, (panel_x, panel_y))
        panel_y += 18

        # Action probabilities (bar chart)
        if self._last_probs is not None:
            for i, (name, prob) in enumerate(zip(ACTION_NAMES, self._last_probs)):
                # Bar background
                bar_w = int(prob * 100)
                color = (100, 200, 100) if i == self._last_action else (80, 80, 120)
                pygame.draw.rect(self.screen, color,
                                 (panel_x, panel_y, bar_w, 10))
                txt = font_sm.render(f"{name} {prob:.0%}", True, (200, 200, 200))
                self.screen.blit(txt, (panel_x + 105, panel_y - 1))
                panel_y += 13

        # Action distribution over episode
        if self._step_count > 0:
            panel_y += 4
            dist_lbl = font_sm.render("── 행동 분포 ──", True, (150, 150, 150))
            self.screen.blit(dist_lbl, (panel_x, panel_y))
            panel_y += 14
            for i, (name, cnt) in enumerate(zip(ACTION_NAMES, self._action_counts)):
                pct = cnt / self._step_count * 100
                txt = font_sm.render(f"{name}: {pct:.0f}%", True, (170, 170, 170))
                self.screen.blit(txt, (panel_x, panel_y))
                panel_y += 12

        # ── Bottom hint ──
        hint = font_sm.render("↑↓: 속도  D: 정책전환  R: 재시작  ESC: 종료", True, (120, 120, 120))
        h = self.screen.get_height()
        self.screen.blit(hint, (4, h - 14))

        if self.shop.done:
            if not self._result_recorded:
                result = self.shop.get_game_result()
                self.ranking.record_result("AI", result)
                self._result_recorded = True
                rank = self.ranking.get_rank(
                    self.shop.final_score, self.shop.day_limit)
                self._rank_text = f"랭킹: #{rank}"
            self.renderer.draw_game_over(
                self.screen, self.shop,
                extra_text=getattr(self, "_rank_text", ""))

        pygame.display.flip()
