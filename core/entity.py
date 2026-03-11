"""Base Entity class – the abstraction layer between dot-rendering (Phase 1)
and sprite-rendering (Phase 3).

Every visible game object (player, customer, …) inherits from Entity.
Positioning is pixel-based (float x, y).
"""

import pygame
from config.settings import TILE_SIZE


class Entity:
    """Pixel-positioned game object with pluggable visual representation."""

    def __init__(self, x: float, y: float, *,
                 color=(255, 255, 255), sprite_key: str | None = None):
        self.x = float(x)
        self.y = float(y)
        self.color = color
        self.sprite_key = sprite_key

        # Animation state (used by AssetManager in Phase 3)
        self.animation_state = "idle"
        self.animation_frame = 0
        self.animation_timer = 0.0
        self.animation_speed = 0.15      # seconds per frame

        self.visible = True

    # ── pixel / grid helpers ─────────────────────
    @property
    def pixel_x(self) -> float:
        return self.x

    @property
    def pixel_y(self) -> float:
        return self.y

    @property
    def center_x(self) -> float:
        return self.x + TILE_SIZE / 2

    @property
    def center_y(self) -> float:
        return self.y + TILE_SIZE / 2

    @property
    def grid_x(self) -> int:
        return int(self.x) // TILE_SIZE

    @property
    def grid_y(self) -> int:
        return int(self.y) // TILE_SIZE

    def get_rect(self, offset_x=0, offset_y=0):
        return pygame.Rect(
            int(self.x) + offset_x,
            int(self.y) + offset_y,
            TILE_SIZE, TILE_SIZE,
        )

    # ── animation ────────────────────────────────
    def update_animation(self, dt: float):
        self.animation_timer += dt
        if self.animation_timer >= self.animation_speed:
            self.animation_timer -= self.animation_speed
            self.animation_frame += 1

    # ── rendering ────────────────────────────────
    def render(self, surface: pygame.Surface, asset_manager,
               offset_x=0, offset_y=0):
        """Draw the entity.  Uses sprite if available, else a simple rect."""
        if not self.visible:
            return

        rect = self.get_rect(offset_x, offset_y)

        if (self.sprite_key
                and asset_manager.has_sprite(self.sprite_key,
                                             self.animation_state)):
            frame = asset_manager.get_frame(
                self.sprite_key, self.animation_state, self.animation_frame)
            surface.blit(frame, rect.topleft)
        else:
            # Phase 1 – coloured rectangle with a small margin
            margin = TILE_SIZE // 8
            inner = rect.inflate(-margin * 2, -margin * 2)
            pygame.draw.rect(surface, self.color, inner)
