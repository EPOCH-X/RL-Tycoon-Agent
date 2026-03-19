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


def _auto_find_best_model() -> tuple[str | None, str | None]:
    """models/ 디렉토리에서 best_model을 자동 탐색합니다.
    Returns (model_path, algo_name) or (None, None).
    """
    import os, json
    from ai.agent import _detect_algo_from_path
    models_dir = "models"
    if not os.path.isdir(models_dir):
        return None, None
    # Prefer best_model.zip, then best_model.pt
    candidates = []
    for root, _dirs, files in os.walk(models_dir):
        for f in files:
            if f == "best_model.zip":
                candidates.append(os.path.join(root, f))
            elif f == "best_model.pt":
                candidates.append(os.path.join(root, f[:-3]))
    if not candidates:
        return None, None
    # Pick the first one found
    path = candidates[0]
    algo = _detect_algo_from_path(path)
    # Also check config
    cfg_path = os.path.join(os.path.dirname(path), "train_config_used.json")
    if os.path.isfile(cfg_path):
        with open(cfg_path, encoding="utf-8") as fp:
            cfg = json.load(fp)
        algo = cfg.get("algorithm", algo) or algo
    return path, algo


class VersusMode(BaseMode):

    def __init__(self, *, model_path=None,
                 target_money=None, day_limit=None):
        self.human_shop = Shop(target_money=target_money,
                               day_limit=day_limit)
        self.ai_shop = Shop(target_money=target_money,
                            day_limit=day_limit)

        self.map_w = self.human_shop.grid_width * TILE_SIZE
        self.map_h = self.human_shop.grid_height * TILE_SIZE + UI_HEIGHT

        # Single-panel screen: human at full size, AI as PiP overlay
        screen_w = self.map_w
        screen_h = self.map_h
        super().__init__(screen_w, screen_h,
                         title="RL 타이쿤 – 대결 모드")

        self.am = AssetManager()
        self.am.ensure_loaded()
        self.renderer_left = Renderer(self.am, background_key="sample2")
        self.renderer_right = Renderer(self.am, background_key="sample2")

        # Auto-find model if not specified
        algo_name = None
        if model_path is None:
            model_path, algo_name = _auto_find_best_model()
            if model_path:
                print(f"  [대결 모드] 자동 탐지 모델: {model_path} ({algo_name})")
        self.agent = load_agent(model_path, algo_name=algo_name)
        if hasattr(self.agent, 'deterministic'):
            self.agent.deterministic = False  # 확률적 정책 사용
        self._algo_name = algo_name or "Random"

        self._interact_pressed = False
        self.winner: str | None = None

        # PiP settings for AI view
        self._pip_scale = 0.25
        self._pip_w = int(self.map_w * self._pip_scale)
        self._pip_h = int(self.map_h * self._pip_scale)
        self._pip_surface = pygame.Surface((self.map_w, self.map_h))

    # ── events ───────────────────────────────────
    def handle_events(self):
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

                # Trait selection for human side
                if self.human_shop.trait_selection_active:
                    for i, k in enumerate([pygame.K_1, pygame.K_2, pygame.K_3]):
                        if event.key == k:
                            self.human_shop.select_trait(i)
                    continue

                if event.key == pygame.K_u:
                    self.human_shop.upgrade_mode = not self.human_shop.upgrade_mode
                if self.human_shop.upgrade_mode and event.key == pygame.K_TAB:
                    self.human_shop.upgrade_tab = (self.human_shop.upgrade_tab + 1) % 3
                if self.human_shop.upgrade_mode:
                    num_keys = [pygame.K_1, pygame.K_2, pygame.K_3,
                                pygame.K_4, pygame.K_5, pygame.K_6,
                                pygame.K_7, pygame.K_8, pygame.K_9]
                    for i, k in enumerate(num_keys):
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

        # AI auto-selects traits
        self.ai_shop.auto_select_trait()

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
            h_score = self.human_shop.final_score
            a_score = self.ai_shop.final_score
            if h_score > a_score:
                self.winner = "human"
            elif a_score > h_score:
                self.winner = "ai"
            else:
                self.winner = "draw"

    # ── draw ─────────────────────────────────────
    def render(self):
        self.screen.fill(COLORS["background"])

        # Full-size human view
        self.renderer_left.draw(self.screen, self.human_shop,
                                offset_x=0, offset_y=0)

        # PiP: render AI to off-screen surface, then scale down
        self._pip_surface.fill(COLORS["background"])
        self.renderer_right.draw(self._pip_surface, self.ai_shop,
                                 offset_x=0, offset_y=0)
        pip_scaled = pygame.transform.smoothscale(
            self._pip_surface, (self._pip_w, self._pip_h))

        # Position PiP: bottom-right, above UI area
        pip_x = self.map_w - self._pip_w - 6
        pip_y = self.map_h - UI_HEIGHT - self._pip_h - 6
        # Border
        pygame.draw.rect(self.screen, (200, 200, 200),
                         (pip_x - 2, pip_y - 2,
                          self._pip_w + 4, self._pip_h + 4), 2)
        self.screen.blit(pip_scaled, (pip_x, pip_y))

        # PiP label + AI stats overlay
        available = [f.lower() for f in pygame.font.get_fonts()]
        kr_font = None
        for fn in ["malgungothic", "gulim", "dotum", "nanumgothic"]:
            if fn in available:
                kr_font = fn
                break
        font_pip = pygame.font.SysFont(kr_font, 18, bold=True)
        ai_shop = self.ai_shop
        stars = ai_shop.shop_rating * 5.0
        info_text = (f"[{self._algo_name}]  ${ai_shop.money}"
                     f"(${ai_shop.net_profit})  {stars:.1f}")
        info_surf = font_pip.render(info_text, True, (255, 255, 255))
        # Semi-transparent background bar
        bar_w = self._pip_w
        bar_h = info_surf.get_height() + 4
        bar_surf = pygame.Surface((bar_w, bar_h), pygame.SRCALPHA)
        bar_surf.fill((0, 0, 0, 160))
        self.screen.blit(bar_surf, (pip_x, pip_y))
        self.screen.blit(info_surf, (pip_x + 4, pip_y + 2))

        # Game-over overlays
        if self._game_over():
            extra = self._winner_text()
            self.renderer_left.draw_game_over(
                self.screen, self.human_shop, extra_text=extra)

        pygame.display.flip()

    # ── helpers ──────────────────────────────────
    def _game_over(self) -> bool:
        return self.human_shop.done and self.ai_shop.done

    def _winner_text(self) -> str:
        if self.winner == "human":
            return "승자: 플레이어!"
        if self.winner == "ai":
            return "승자: AI!"
        return "무승부!"

    def _build_ai_obs(self):
        """Lightweight observation for the AI agent.

        Uses the same format as ``ai.gym_env.build_observation``.
        Importing it here avoids heavyweight numpy at module level.
        """
        from ai.gym_env import build_observation
        return build_observation(self.ai_shop)
