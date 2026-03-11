"""Player entity – the server/waiter controlled by human or AI.

The player walks between tables and the kitchen to:
1. Take orders from seated customers
2. Deliver order slips to the kitchen counter
3. Pick up cooked food from the kitchen
4. Serve food to the correct table

Positioning is pixel-based (distance movement, not grid snapping).
"""

import pygame
from core.entity import Entity
from config.settings import TILE_SIZE, COLORS, PLAYER_SPEED


class Player(Entity):
    # Facing constants
    FACING_UP = 0
    FACING_DOWN = 1
    FACING_LEFT = 2
    FACING_RIGHT = 3

    DIRECTION_VEC = {
        FACING_UP:    (0, -1),
        FACING_DOWN:  (0,  1),
        FACING_LEFT:  (-1, 0),
        FACING_RIGHT: (1,  0),
    }

    def __init__(self, x: float, y: float):
        super().__init__(x, y, color=COLORS["player"], sprite_key="player")
        self.facing = self.FACING_DOWN
        self.speed = float(PLAYER_SPEED)   # pixels per second (upgradeable)

        # Carrying state: None | {"type": "order", ...} | {"type": "food", ...}
        self.carrying: dict | None = None

    # ── carrying helpers ─────────────────────────
    @property
    def is_idle(self) -> bool:
        return self.carrying is None

    @property
    def has_order(self) -> bool:
        return self.carrying is not None and self.carrying["type"] == "order"

    @property
    def has_food(self) -> bool:
        return self.carrying is not None and self.carrying["type"] == "food"

    def pick_up_order(self, table_id: int, menu_item: dict):
        self.carrying = {
            "type": "order",
            "table_id": table_id,
            "menu_item": menu_item,
        }

    def pick_up_food(self, table_id: int, menu_item: dict):
        self.carrying = {
            "type": "food",
            "table_id": table_id,
            "menu_item": menu_item,
        }

    def drop(self) -> dict | None:
        item = self.carrying
        self.carrying = None
        return item

    # ── rendering (Phase 1) ──────────────────────
    def render(self, surface: pygame.Surface, asset_manager,
               offset_x=0, offset_y=0):
        if not self.visible:
            return

        rect = self.get_rect(offset_x, offset_y)

        # Try sprite first
        if (self.sprite_key
                and asset_manager.has_sprite(self.sprite_key,
                                             self.animation_state)):
            frame = asset_manager.get_frame(
                self.sprite_key, self.animation_state, self.animation_frame)
            surface.blit(frame, rect.topleft)
            return

        # Phase 1: coloured body + direction arrow
        margin = TILE_SIZE // 6
        body = rect.inflate(-margin * 2, -margin * 2)
        col = COLORS["player_carry"] if self.carrying else COLORS["player"]
        pygame.draw.rect(surface, col, body)

        # Direction indicator (small triangle)
        cx, cy = body.center
        s = TILE_SIZE // 6
        if self.facing == self.FACING_UP:
            pts = [(cx, cy - s), (cx - s, cy), (cx + s, cy)]
        elif self.facing == self.FACING_DOWN:
            pts = [(cx, cy + s), (cx - s, cy), (cx + s, cy)]
        elif self.facing == self.FACING_LEFT:
            pts = [(cx - s, cy), (cx, cy - s), (cx, cy + s)]
        else:
            pts = [(cx + s, cy), (cx, cy - s), (cx, cy + s)]
        pygame.draw.polygon(surface, (255, 255, 255), pts)

        # Carrying indicator (small icon above head)
        if self.carrying:
            tag_rect = pygame.Rect(rect.x + TILE_SIZE // 4,
                                   rect.y - 10, TILE_SIZE // 2, 10)
            if self.has_order:
                pygame.draw.rect(surface, (200, 200, 100), tag_rect)
            else:
                pygame.draw.rect(surface, (100, 220, 100), tag_rect)
            pygame.draw.rect(surface, (0, 0, 0), tag_rect, 1)
