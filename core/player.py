"""Player entity – the server/waiter controlled by human or AI.

The player walks between tables and the kitchen to:
1. Take orders from seated customers
2. Deliver order slips to the kitchen counter
3. Pick up cooked food from the kitchen
4. Serve food to the correct table

Carrying is list-based: the player can hold up to *carry_capacity* items.
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
        self.speed = float(PLAYER_SPEED)

        # Carrying state: list of dicts
        # Each item: {"type": "order"|"food"|"drink", "table_id": int, ...}
        self.carrying: list[dict] = []
        self.carry_capacity: int = 1

    # ── carrying helpers ─────────────────────────
    @property
    def is_idle(self) -> bool:
        return len(self.carrying) == 0

    @property
    def has_order(self) -> bool:
        return any(c["type"] == "order" for c in self.carrying)

    @property
    def has_food(self) -> bool:
        return any(c["type"] == "food" for c in self.carrying)

    @property
    def has_drink(self) -> bool:
        return any(c["type"] == "drink" for c in self.carrying)

    @property
    def can_carry_more(self) -> bool:
        return len(self.carrying) < self.carry_capacity

    def pick_up_order(self, table_id: int, items: list[dict],
                      drink_item: dict | None = None):
        """Pick up an order slip (may contain multiple items for a family)."""
        order = {
            "type": "order",
            "table_id": table_id,
            "items": items,
        }
        if drink_item:
            order["drink_item"] = drink_item
        self.carrying.append(order)

    def pick_up_food(self, table_id: int, menu_item: dict):
        self.carrying.append({
            "type": "food",
            "table_id": table_id,
            "menu_item": menu_item,
        })

    def pick_up_drink(self, table_id: int, drink_item: dict):
        self.carrying.append({
            "type": "drink",
            "table_id": table_id,
            "drink_item": drink_item,
        })

    def drop_orders(self) -> list[dict]:
        """Remove and return all order slips."""
        orders = [c for c in self.carrying if c["type"] == "order"]
        self.carrying = [c for c in self.carrying if c["type"] != "order"]
        return orders

    def drop_food_for_table(self, table_id: int) -> list[dict]:
        """Remove and return all food/drink items for a specific table."""
        matching = [c for c in self.carrying
                    if c["type"] in ("food", "drink")
                    and c["table_id"] == table_id]
        self.carrying = [c for c in self.carrying if c not in matching]
        return matching

    def drop_all(self) -> list[dict]:
        items = self.carrying[:]
        self.carrying.clear()
        return items

    # ── backward compat: first carried item dict ──
    @property
    def first_carried(self) -> dict | None:
        """First carried item (for UI display etc.)."""
        return self.carrying[0] if self.carrying else None

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
            elif self.has_drink:
                pygame.draw.rect(surface, (160, 100, 220), tag_rect)
            else:
                pygame.draw.rect(surface, (100, 220, 100), tag_rect)
            pygame.draw.rect(surface, (0, 0, 0), tag_rect, 1)
            # Show count if carrying multiple
            if len(self.carrying) > 1:
                font = pygame.font.SysFont(None, 14)
                cnt = font.render(str(len(self.carrying)), True, (0, 0, 0))
                surface.blit(cnt, cnt.get_rect(center=tag_rect.center))
