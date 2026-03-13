"""BaseMode – shared Pygame game-loop skeleton for all game modes."""

import pygame
from config.settings import FPS, STEP_INTERVAL


class BaseMode:
    """Abstract base for game modes.

    Subclasses must override ``handle_events``, ``update``, and ``render``.
    The fixed-timestep loop guarantees that ``update()`` is called at exactly
    1 / STEP_INTERVAL Hz regardless of rendering frame rate.
    """

    SPEED_LEVELS = (0.5, 1.0, 2.0, 4.0, 8.0)

    def __init__(self, screen_width: int, screen_height: int,
                 title: str = "RL Tycoon",
                 time_scale: float = 1.0):
        pygame.init()
        self.title = title
        self.screen = pygame.display.set_mode((screen_width, screen_height))
        pygame.display.set_caption(title)
        self.clock = pygame.time.Clock()
        self.running = True
        self._step_accum = 0.0
        self.time_scale = 1.0
        self.set_time_scale(time_scale)

    def set_time_scale(self, value: float):
        """Set simulation speed multiplier."""
        self.time_scale = max(0.25, float(value))
        pygame.display.set_caption(f"{self.title} ({self.time_scale:.1f}x)")

    def increase_speed(self):
        """Move to the next predefined speed level."""
        for level in self.SPEED_LEVELS:
            if level > self.time_scale:
                self.set_time_scale(level)
                return
        self.set_time_scale(self.SPEED_LEVELS[-1])

    def decrease_speed(self):
        """Move to the previous predefined speed level."""
        for level in reversed(self.SPEED_LEVELS):
            if level < self.time_scale:
                self.set_time_scale(level)
                return
        self.set_time_scale(self.SPEED_LEVELS[0])

    def reset_speed(self):
        """Restore realtime simulation speed."""
        self.set_time_scale(1.0)

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
            sim_dt = dt * self.time_scale
            self.handle_events()

            self.tick(sim_dt)   # per-frame (smooth movement)

            self._step_accum += sim_dt
            while self._step_accum >= STEP_INTERVAL:
                self.update()
                self._step_accum -= STEP_INTERVAL

            self.render()

        pygame.quit()
