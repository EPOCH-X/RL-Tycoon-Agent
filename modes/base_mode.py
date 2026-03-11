"""BaseMode – shared Pygame game-loop skeleton for all game modes."""

import pygame
from config.settings import FPS, STEP_INTERVAL


class BaseMode:
    """Abstract base for game modes.

    Subclasses must override ``handle_events``, ``update``, and ``render``.
    The fixed-timestep loop guarantees that ``update()`` is called at exactly
    1 / STEP_INTERVAL Hz regardless of rendering frame rate.
    """

    def __init__(self, screen_width: int, screen_height: int,
                 title: str = "RL Tycoon"):
        pygame.init()
        self.screen = pygame.display.set_mode((screen_width, screen_height))
        pygame.display.set_caption(title)
        self.clock = pygame.time.Clock()
        self.running = True
        self._step_accum = 0.0

    # ── override these ───────────────────────────
    def handle_events(self):
        raise NotImplementedError

    def tick(self, dt: float):
        """Per-frame update (e.g. smooth movement). Override in subclass."""
        pass

    def update(self):
        raise NotImplementedError

    def render(self):
        raise NotImplementedError

    # ── main loop ────────────────────────────────
    def run(self):
        while self.running:
            dt = self.clock.tick(FPS) / 1000.0
            self.handle_events()

            self.tick(dt)   # per-frame (smooth movement)

            self._step_accum += dt
            while self._step_accum >= STEP_INTERVAL:
                self.update()
                self._step_accum -= STEP_INTERVAL

            self.render()

        pygame.quit()
