"""TournamentMode – 학습된 모델들이 순차적으로 경쟁하는 토너먼트 관전 모드.

각 참가자가 동일 조건에서 매장을 운영하고, 최종 스코어로 순위를 매깁니다.
--participants 옵션으로 모델을 지정하거나, models/ 디렉토리에서 자동 탐색합니다.

Usage:
    python main.py --mode tournament
    python main.py --mode tournament --days 60 --speed 3
"""

import json
import os
import pygame
import numpy as np

from config.settings import (
    TILE_SIZE, UI_HEIGHT, COLORS, FPS, STEP_INTERVAL,
    VERSUS_DIVIDER_WIDTH, VERSUS_DIVIDER_COLOR,
)
from modes.base_mode import BaseMode
from modes.model_runtime import load_model_runtime_options
from core.shop import Shop
from rendering.asset_manager import AssetManager
from rendering.renderer import Renderer
from ai.agent import load_agent
from ai.gym_env import build_observation


def _find_all_models() -> list[dict]:
    """models/ 에서 학습된 모델들을 자동 탐색합니다."""
    from algorithms.cross_play.trainer import _find_trained_models
    return _find_trained_models()


class TournamentMode(BaseMode):
    """토너먼트 관전 모드 – 학습된 AI 에이전트들이 동시에 경쟁합니다.

    최대 4명의 참가자가 화면을 분할하여 동시에 매장을 운영합니다.
    게임 종료 후 최종 스코어 순위표를 표시합니다.
    """

    MAX_PARTICIPANTS = 4

    @staticmethod
    def _normalize_participants(participants: list[dict | str]) -> list[dict]:
        normalized: list[dict] = []
        for entry in participants:
            if isinstance(entry, str):
                normalized.append({
                    "algo": os.path.splitext(os.path.basename(entry))[0],
                    "path": entry,
                })
            else:
                normalized.append(entry)
        return normalized

    def __init__(self, *, participants: list[dict] | None = None,
                 target_money=None, day_limit=None,
                 speed_multiplier: float = 1.0):

        # ── 참가자 로드 ──
        if participants is None:
            entries = _find_all_models()
            # Deduplicate: keep best_model per algorithm
            seen_algos: dict[str, dict] = {}
            for entry in entries:
                algo = entry.get("algo", "Unknown")
                path = entry.get("path", "")
                # Prefer best_model over final_model
                if algo not in seen_algos or "best" in path:
                    seen_algos[algo] = entry
            participants = list(seen_algos.values())[:self.MAX_PARTICIPANTS]

        if not participants:
            raise RuntimeError(
                "토너먼트 참가자 없음! 먼저 모델을 학습하세요.\n"
                "  python -m algorithms.train_launcher --algo PPO --timesteps 100000"
            )

        participants = self._normalize_participants(participants)

        # Limit to 4
        participants = participants[:self.MAX_PARTICIPANTS]
        n = len(participants)

        # ── Agents & Shops ──
        self._entries: list[dict] = []
        for entry in participants:
            algo = entry.get("algo", "Unknown")
            path = entry.get("path", "")
            game_overrides, env_options = load_model_runtime_options(path)
            shop_target_money = target_money if target_money is not None else game_overrides.get("target_money")
            shop_day_limit = day_limit if day_limit is not None else game_overrides.get("day_limit")
            try:
                agent = load_agent(path, algo_name=None)
            except Exception as e:
                print(f"  [Tournament] {algo} 로드 실패 ({path}): {e}")
                continue
            self._entries.append({
                "algo": algo,
                "path": path,
                "agent": agent,
                "shop": Shop(
                    target_money=shop_target_money,
                    day_limit=shop_day_limit,
                    **env_options,
                ),
            })

        if not self._entries:
            raise RuntimeError("로드 가능한 참가자가 없습니다!")

        n = len(self._entries)
        print(f"\n  [Tournament] 참가자 {n}명:")
        for i, e in enumerate(self._entries):
            print(f"    {i+1}. {e['algo']} → {e['path']}")

        # ── Layout: 1→1col, 2→2col, 3-4→2×2 grid ──
        sample_shop = self._entries[0]["shop"]
        self._panel_w = sample_shop.grid_width * TILE_SIZE
        self._panel_h = sample_shop.grid_height * TILE_SIZE + UI_HEIGHT
        div = VERSUS_DIVIDER_WIDTH

        if n == 1:
            cols, rows = 1, 1
        elif n == 2:
            cols, rows = 2, 1
        else:
            cols, rows = 2, 2

        self._cols = cols
        self._rows = rows
        screen_w = cols * self._panel_w + (cols - 1) * div
        screen_h = rows * self._panel_h + (rows - 1) * div

        # Add scoreboard area at bottom
        self._scoreboard_h = 160
        screen_h += self._scoreboard_h

        super().__init__(screen_w, screen_h, title="RL 타이쿤 – 토너먼트")

        self.am = AssetManager()
        self._renderers = [Renderer(self.am) for _ in range(n)]

        self.speed_multiplier = max(0.5, speed_multiplier)
        self._all_done = False
        self._rankings: list[dict] = []

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
                if event.key == pygame.K_r and self._all_done:
                    self._restart()
                if event.key == pygame.K_UP:
                    self.speed_multiplier = min(10.0, self.speed_multiplier + 0.5)
                if event.key == pygame.K_DOWN:
                    self.speed_multiplier = max(0.5, self.speed_multiplier - 0.5)

    def tick(self, dt):
        pass

    # ── update ───────────────────────────────────
    def update(self):
        if self._all_done:
            return

        steps = max(1, int(self.speed_multiplier))
        for _ in range(steps):
            any_running = False
            for entry in self._entries:
                shop = entry["shop"]
                if shop.done:
                    continue
                any_running = True
                obs = build_observation(shop)
                action = entry["agent"].predict(
                    obs, action_mask=shop.get_action_mask())
                shop.step(action)
                shop.auto_select_trait()

            if not any_running:
                self._all_done = True
                self._compute_rankings()
                break

    # ── render ───────────────────────────────────
    def render(self):
        self.screen.fill((20, 20, 35))
        n = len(self._entries)
        div = VERSUS_DIVIDER_WIDTH

        for idx, entry in enumerate(self._entries):
            col = idx % self._cols
            row = idx // self._cols
            ox = col * (self._panel_w + div)
            oy = row * (self._panel_h + div)

            self._renderers[idx].draw(self.screen, entry["shop"],
                                      offset_x=ox, offset_y=oy)

            # Label
            font = self._get_font(16)
            color = self._participant_color(idx)
            label = font.render(f"{entry['algo']}", True, color)
            self.screen.blit(label, (ox + 4, oy + 2))

        # ── Dividers ──
        total_w = self._cols * self._panel_w + (self._cols - 1) * div
        total_h = self._rows * self._panel_h + (self._rows - 1) * div

        for c in range(1, self._cols):
            x = c * self._panel_w + (c - 1) * div
            pygame.draw.rect(self.screen, VERSUS_DIVIDER_COLOR,
                             (x, 0, div, total_h))
        for r in range(1, self._rows):
            y = r * self._panel_h + (r - 1) * div
            pygame.draw.rect(self.screen, VERSUS_DIVIDER_COLOR,
                             (0, y, total_w, div))

        # ── Scoreboard ──
        self._draw_scoreboard(total_h)

        # ── Speed info ──
        font_sm = self._get_font(14)
        speed_txt = font_sm.render(
            f"속도: ×{self.speed_multiplier:.1f}  ↑↓: 속도조절  R: 재시작  ESC: 종료",
            True, (140, 140, 160))
        self.screen.blit(speed_txt, (4, self.screen.get_height() - 18))

        # ── Game over overlay ──
        if self._all_done:
            for idx, entry in enumerate(self._entries):
                col = idx % self._cols
                row = idx // self._cols
                ox = col * (self._panel_w + div)
                oy = row * (self._panel_h + div)
                rank = self._get_rank(entry["algo"])
                extra = f"#{rank}" if rank else ""
                self._renderers[idx].draw_game_over(
                    self.screen, entry["shop"],
                    offset_x=ox, offset_y=oy, extra_text=extra)

        pygame.display.flip()

    def _draw_scoreboard(self, y_start: int):
        """화면 하단에 실시간 스코어보드를 그립니다."""
        font = self._get_font(18)
        font_sm = self._get_font(14)

        # Background
        board_rect = pygame.Rect(0, y_start, self.screen.get_width(),
                                 self._scoreboard_h)
        pygame.draw.rect(self.screen, (25, 25, 45), board_rect)
        pygame.draw.line(self.screen, (80, 80, 120),
                         (0, y_start), (self.screen.get_width(), y_start), 2)

        # Title
        title = font.render("📊 토너먼트 스코어보드", True, (255, 215, 0))
        self.screen.blit(title, (10, y_start + 8))

        # Headers
        hx = 10
        hy = y_start + 35
        headers = ["순위", "알고리즘", "스코어", "수익($)", "평점",
                    "서빙", "이탈", "상태"]
        col_widths = [50, 120, 100, 100, 80, 60, 60, 80]
        for header, cw in zip(headers, col_widths):
            h_surf = font_sm.render(header, True, (160, 180, 200))
            self.screen.blit(h_surf, (hx, hy))
            hx += cw

        # Sort entries by score descending
        scored = []
        for idx, entry in enumerate(self._entries):
            shop = entry["shop"]
            scored.append({
                "idx": idx,
                "algo": entry["algo"],
                "score": shop.final_score,
                "money": shop.money,
                "net_profit": shop.net_profit,
                "rating": shop.shop_rating_stars,
                "served": shop.customers_served,
                "lost": shop.customers_lost,
                "done": shop.done,
                "won": shop.won,
            })
        scored.sort(key=lambda x: x["score"], reverse=True)

        # Rows
        for rank, s in enumerate(scored, 1):
            rx = 10
            ry = hy + 20 + (rank - 1) * 20
            color = self._participant_color(s["idx"])

            status = "✓완료" if s["done"] else "진행중..."
            if s["won"]:
                status = "★승리!"

            values = [
                f"#{rank}",
                s["algo"],
                f"{s['score']:,.0f}",
                f"${s['net_profit']:,}",
                f"{s['rating']:.1f}★",
                str(s["served"]),
                str(s["lost"]),
                status,
            ]
            for val, cw in zip(values, col_widths):
                v_surf = font_sm.render(val, True, color)
                self.screen.blit(v_surf, (rx, ry))
                rx += cw

    def _compute_rankings(self):
        """게임 종료 후 최종 순위를 계산합니다."""
        self._rankings = []
        for entry in self._entries:
            self._rankings.append({
                "algo": entry["algo"],
                "score": entry["shop"].final_score,
                "won": entry["shop"].won,
            })
        self._rankings.sort(key=lambda x: x["score"], reverse=True)
        print("\n  ╔══════════════════════════════════════╗")
        print("  ║      🏆 토너먼트 최종 결과 🏆        ║")
        print("  ╠══════════════════════════════════════╣")
        for i, r in enumerate(self._rankings, 1):
            medal = {1: "🥇", 2: "🥈", 3: "🥉"}.get(i, "  ")
            won_str = " ★승리" if r["won"] else ""
            print(f"  ║ {medal} #{i}  {r['algo']:<12s}"
                  f"  스코어: {r['score']:>8,.0f}{won_str}")
        print("  ╚══════════════════════════════════════╝")

    def _get_rank(self, algo: str) -> int | None:
        for i, r in enumerate(self._rankings, 1):
            if r["algo"] == algo:
                return i
        return None

    def _restart(self):
        for entry in self._entries:
            entry["shop"].reset()
        self._all_done = False
        self._rankings = []

    @staticmethod
    def _participant_color(idx: int) -> tuple[int, int, int]:
        palette = [
            (100, 200, 255),  # Blue
            (255, 150, 100),  # Orange
            (100, 255, 150),  # Green
            (255, 200, 100),  # Yellow
        ]
        return palette[idx % len(palette)]

    def _get_font(self, size: int) -> pygame.font.Font:
        if not hasattr(self, "_font_cache"):
            self._font_cache: dict[int, pygame.font.Font] = {}
        if size not in self._font_cache:
            available = [f.lower() for f in pygame.font.get_fonts()]
            kr_font = None
            for fn in ["malgungothic", "gulim", "dotum", "nanumgothic"]:
                if fn in available:
                    kr_font = fn
                    break
            self._font_cache[size] = pygame.font.SysFont(kr_font, size)
        return self._font_cache[size]
