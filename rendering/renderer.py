"""Renderer – draws the full game view using image assets from data/images/.

Layer order (back to front):
  1. Background image (mode-dependent)
  2. Wall tiles + Kitchen back-wall tiles (주방벽면)
  3. Bar counter tiles
  4. Chef images (요리사/1-6.png)
  5. Kitchen counter tiles (주방.png) + cooking overlays
  6. Trash can tiles
  7. Hall table tiles + Purchasable ghost tables
  8. Customer sprites (seated/waiting/eating)
  9. Chair tiles
  10. Player / Employees / Walking customers / Food & Drink items
  11. UI overlays
"""

import pygame
from config.settings import TILE_SIZE, UI_HEIGHT, COLORS
from core.customer import CustomerState
from rendering.asset_manager import AssetManager

# Customer type JSON id → image folder name
_CUST_TYPE_FOLDER = {
    "budget": "budget",
    "normal": "normal",
    "tourist": "tourist",
    "wealthy": "richman",
    "vip": "vip",
    "critic": "critic",
}


class Renderer:
    """Draws a Shop's state onto a Pygame surface using image assets."""

    def __init__(self, asset_manager: AssetManager, *,
                 background_key: str = "sample1"):
        self.am = asset_manager
        self.background_key = background_key
        self._fonts_ok = False
        self.font_sm: pygame.font.Font | None = None
        self.font_md: pygame.font.Font | None = None
        self.font_lg: pygame.font.Font | None = None
        self._scaled_bg: pygame.Surface | None = None
        self._scaled_bg_key: str | None = None
        self._scaled_bg_size: tuple[int, int] | None = None

    # ── lazy font init (needs pygame.init first) ─
    def _ensure_fonts(self):
        if not self._fonts_ok:
            font_name = self._find_korean_font()
            self.font_sm = pygame.font.SysFont(font_name, 16)
            self.font_md = pygame.font.SysFont(font_name, 24)
            self.font_lg = pygame.font.SysFont(font_name, 36)
            self._fonts_ok = True

    @staticmethod
    def _find_korean_font():
        available = [f.lower() for f in pygame.font.get_fonts()]
        korean_fonts = ["malgungothic", "malgunbd", "gulim", "dotum",
                        "batang", "nanumgothic", "nanumbarungothic",
                        "applegothic", "nanumgothicbold"]
        for name in korean_fonts:
            if name in available:
                return name
        return None

    # ═══════════════════════════════════════════════
    #  Main entry point
    # ═══════════════════════════════════════════════
    def draw(self, surface: pygame.Surface, shop, *,
             offset_x: int = 0, offset_y: int = 0):
        """Draw the complete shop view with proper layer ordering."""
        self._ensure_fonts()
        self.am.ensure_loaded()
        ox, oy = offset_x, offset_y

        # Layer 1: Background
        self._draw_background(surface, shop, ox, oy)
        # Layer 2: Wall tiles + Kitchen back-wall
        self._draw_walls(surface, shop, ox, oy)
        # Layer 3: Bar counters
        self._draw_bar(surface, shop, ox, oy)
        # Layer 4: Chef images
        self._draw_chefs(surface, shop, ox, oy)
        # Layer 5: Kitchen counter tiles + cooking overlays
        self._draw_kitchen(surface, shop, ox, oy)
        # Layer 6: Trash cans
        self._draw_trash_cans(surface, shop, ox, oy)
        # Layer 7: Hall tables
        self._draw_tables(surface, shop, ox, oy)
        # Layer 8: Seated customers
        self._draw_customers(surface, shop, ox, oy)
        # Layer 9: Chairs
        self._draw_chairs(surface, shop, ox, oy)
        # Layer 10: Moving entities
        self._draw_leaving_customers(surface, shop, ox, oy)
        self._draw_waiting_queue(surface, shop, ox, oy)
        self._draw_employees(surface, shop, ox, oy)
        shop.player.render(surface, self.am, ox, oy)
        # Overlays
        self._draw_carry_labels(surface, shop, ox, oy)
        self._draw_floating_texts(surface, shop, ox, oy)
        self._draw_ui(surface, shop, ox, oy)
        self._draw_upgrade_panel(surface, shop, ox, oy)
        self._draw_trait_popup(surface, shop, ox, oy)

    # ═══════════════════════════════════════════════
    #  Layer 1: Background
    # ═══════════════════════════════════════════════
    def _draw_background(self, surface, shop, ox, oy):
        map_w = shop.grid_width * TILE_SIZE
        map_h = shop.grid_height * TILE_SIZE
        bg_raw = self.am.get_background(self.background_key)
        if bg_raw:
            target_size = (map_w, map_h)
            if (self._scaled_bg_key != self.background_key
                    or self._scaled_bg_size != target_size):
                self._scaled_bg = pygame.transform.smoothscale(
                    bg_raw, target_size)
                self._scaled_bg_key = self.background_key
                self._scaled_bg_size = target_size
            surface.blit(self._scaled_bg, (ox, oy))
        else:
            pygame.draw.rect(surface, COLORS["floor"],
                             (ox, oy, map_w, map_h))

    # ═══════════════════════════════════════════════
    #  Layer 2: Wall tiles + Kitchen back-wall
    # ═══════════════════════════════════════════════
    def _draw_walls(self, surface, shop, ox, oy):
        kitchen_xs = {pos[0] for pos in shop.kitchen_counter_positions}
        for y, row in enumerate(shop.layout):
            for x, tile in enumerate(row):
                if tile != 1:
                    continue
                # Only draw kitchen back-wall tiles (decorative);
                # plain wall tiles are covered by the background image.
                if (y + 1 < len(shop.layout)
                        and x in kitchen_xs
                        and shop.layout[y + 1][x] == 3):
                    rect = pygame.Rect(x * TILE_SIZE + ox,
                                       y * TILE_SIZE + oy,
                                       TILE_SIZE, TILE_SIZE)
                    img = (self.am.tile_kitchen_wall1
                           if x % 2 == 0
                           else self.am.tile_kitchen_wall2)
                    if img:
                        surface.blit(img, rect.topleft)

    # ═══════════════════════════════════════════════
    #  Layer 7: Tables (active)
    # ═══════════════════════════════════════════════
    def _draw_tables(self, surface, shop, ox, oy):
        for table in shop.tables:
            rect = pygame.Rect(table.grid_x * TILE_SIZE + ox,
                               table.grid_y * TILE_SIZE + oy,
                               TILE_SIZE, TILE_SIZE)
            if self.am.tile_table:
                surface.blit(self.am.tile_table, rect.topleft)
            else:
                col = COLORS["table_occupied"] if table.is_occupied else COLORS["table"]
                pygame.draw.rect(surface, col, rect)

    # ═══════════════════════════════════════════════
    #  Purchasable table ghosts
    # ═══════════════════════════════════════════════
    def _draw_purchasable_tables(self, surface, shop, ox, oy):
        for tdata in shop._purchasable_tables:
            rect = pygame.Rect(tdata["grid_x"] * TILE_SIZE + ox,
                               tdata["grid_y"] * TILE_SIZE + oy,
                               TILE_SIZE, TILE_SIZE)
            if self.am.tile_table:
                ghost = self.am.tile_table.copy()
                ghost.set_alpha(80)
                surface.blit(ghost, rect.topleft)
            else:
                ghost = pygame.Surface((TILE_SIZE, TILE_SIZE), pygame.SRCALPHA)
                ghost.fill((139, 90, 43, 50))
                surface.blit(ghost, rect.topleft)
            lbl = self.font_sm.render("$", True, (200, 180, 100))
            surface.blit(lbl, lbl.get_rect(center=rect.center))

    # ═══════════════════════════════════════════════
    #  Layer 9: Chairs (drawn on top of customers)
    # ═══════════════════════════════════════════════
    def _draw_chairs(self, surface, shop, ox, oy):
        if not self.am.tile_chair:
            return
        for table in shop.tables:
            if not table.is_occupied:
                continue
            rect = pygame.Rect(table.grid_x * TILE_SIZE + ox,
                               table.grid_y * TILE_SIZE + oy,
                               TILE_SIZE, TILE_SIZE)
            surface.blit(self.am.tile_chair, rect.topleft)

    # ═══════════════════════════════════════════════
    #  Layer 4: Chef images (요리사/1-6.png)
    # ═══════════════════════════════════════════════
    def _draw_chefs(self, surface, shop, ox, oy):
        positions = sorted(shop.kitchen_counter_positions)
        for i in range(min(shop.num_chefs, len(positions))):
            pos = positions[i]
            # Place chef above kitchen counter: shift up by half tile
            # so the belly/waist aligns with the counter top
            chef_x = pos[0] * TILE_SIZE + ox
            chef_y = pos[1] * TILE_SIZE + oy - TILE_SIZE // 2
            chef_img = self.am.get_chef_image(i)
            if chef_img:
                surface.blit(chef_img, (chef_x, chef_y))

    # ═══════════════════════════════════════════════
    #  Layer 5: Kitchen counters + cooking overlays
    # ═══════════════════════════════════════════════
    def _draw_kitchen(self, surface, shop, ox, oy):
        kitchen = shop.kitchen
        positions = sorted(shop.kitchen_counter_positions)

        tile_slots: list[tuple[str, dict] | None] = [None] * len(positions)
        idx = 0
        for order in kitchen.cooking:
            if idx < len(tile_slots):
                tile_slots[idx] = ("cooking", order)
                idx += 1
        for item in kitchen.ready:
            if idx < len(tile_slots):
                tile_slots[idx] = ("ready", item)
                idx += 1

        for i, pos in enumerate(positions):
            rect = pygame.Rect(pos[0] * TILE_SIZE + ox,
                               pos[1] * TILE_SIZE + oy,
                               TILE_SIZE, TILE_SIZE)
            # Kitchen counter tile image
            if self.am.tile_kitchen:
                surface.blit(self.am.tile_kitchen, rect.topleft)
            else:
                slot = tile_slots[i] if i < len(tile_slots) else None
                if slot and slot[0] == "ready":
                    col = COLORS["kitchen_ready"]
                elif slot and slot[0] == "cooking":
                    col = COLORS["kitchen_cooking"]
                else:
                    col = COLORS["kitchen"]
                pygame.draw.rect(surface, col, rect)

            slot = tile_slots[i] if i < len(tile_slots) else None
            if slot is None:
                continue

            state, data = slot

            if state == "cooking":
                # Food icon overlay
                food_id = data["menu_item"].get("id", "")
                food_icon = self.am.get_food_icon(food_id)
                if food_icon:
                    surface.blit(food_icon, (rect.x + 16, rect.y + 2))
                else:
                    name = data["menu_item"]["name"]
                    nm = self.font_sm.render(name[:4], True, (255, 255, 255))
                    surface.blit(nm, nm.get_rect(
                        centerx=rect.centerx, top=rect.y + 4))

                # Table ID
                tid = self.font_sm.render(
                    f"T{data['table_id']}", True, (180, 180, 180))
                surface.blit(tid, (rect.right - 22, rect.y + 4))

                # Progress bar
                total = data["menu_item"]["cook_time"]
                ratio = max(0.0, 1.0 - data["timer"] / total) if total else 1.0
                bw = TILE_SIZE - 8
                bx = rect.x + 4
                by = rect.bottom - 12
                pygame.draw.rect(surface, (60, 60, 60), (bx, by, bw, 8))
                pygame.draw.rect(surface, (80, 220, 80),
                                 (bx, by, int(bw * ratio), 8))
                pct = self.font_sm.render(
                    f"{int(ratio * 100)}%", True, (200, 200, 200))
                surface.blit(pct, pct.get_rect(
                    centerx=rect.centerx, bottom=rect.bottom - 14))

            elif state == "ready":
                done = self.font_sm.render("완성", True, (255, 200, 80))
                surface.blit(done, (rect.x + 2, rect.y + 2))
                food_id = data["menu_item"].get("id", "")
                food_icon = self.am.get_food_icon(food_id)
                if food_icon:
                    surface.blit(food_icon, (rect.x + 16, rect.centery - 16))
                else:
                    name = data["menu_item"]["name"]
                    nm = self.font_sm.render(name[:4], True, (255, 255, 100))
                    surface.blit(nm, nm.get_rect(center=rect.center))
                tid = self.font_sm.render(
                    f"T{data['table_id']}", True, (180, 180, 180))
                surface.blit(tid, tid.get_rect(
                    centerx=rect.centerx, bottom=rect.bottom - 4))

    # ═══════════════════════════════════════════════
    #  Layer 6: Trash cans
    # ═══════════════════════════════════════════════
    def _draw_trash_cans(self, surface, shop, ox, oy):
        for pos in shop.trash_can_positions:
            rect = pygame.Rect(pos[0] * TILE_SIZE + ox,
                               pos[1] * TILE_SIZE + oy,
                               TILE_SIZE, TILE_SIZE)
            if self.am.tile_trash:
                surface.blit(self.am.tile_trash, rect.topleft)
            else:
                col = COLORS.get("trash_can", (120, 100, 80))
                pygame.draw.rect(surface, col, rect)

    # ═══════════════════════════════════════════════
    #  Layer 8: Customers (drawn at their table position)
    # ═══════════════════════════════════════════════
    @staticmethod
    def _customer_sprite_key(cust) -> str:
        ctype_id = cust.customer_type.get("id", "normal")
        folder = _CUST_TYPE_FOLDER.get(ctype_id, ctype_id)
        return f"customer_{folder}"

    @staticmethod
    def _customer_anim_state(cust) -> str:
        state = cust.state
        if state == CustomerState.WALKING_TO_TABLE:
            # walk_back = moving UP (away from camera, toward tables)
            return "walk_back"
        elif state == CustomerState.WAITING_TO_ORDER:
            return "sit_wait_order_front"
        elif state == CustomerState.ORDER_TAKEN:
            return "sit_wait_food_front"
        elif state == CustomerState.EATING:
            return "sit_eating_front"
        elif state == CustomerState.LEAVING_HAPPY:
            return "exit_happy_front"
        elif state == CustomerState.LEAVING_ANGRY:
            return "exit_angry_front"
        elif state == CustomerState.WAITING_OUTSIDE:
            return "stand_wait_order_back"
        elif state == CustomerState.WALKING_TO_EXIT:
            # walk_front = moving DOWN (toward camera, toward exit)
            if cust._happy:
                return "walk_front"
            return "walk_front"
        return "walk_front"

    def _draw_customer_sprite(self, surface, cust, cx, cy):
        """Draw a single customer with sprite or fallback."""
        sprite_key = self._customer_sprite_key(cust)
        anim_state = self._customer_anim_state(cust)
        rect = pygame.Rect(cx, cy, TILE_SIZE, TILE_SIZE)

        # Fallback chain: try primary state, then back variant, then walk_front
        if self.am.has_sprite(sprite_key, anim_state):
            frame = self.am.get_frame(
                sprite_key, anim_state, cust.animation_frame)
            surface.blit(frame, rect.topleft)
        elif anim_state.endswith("_front"):
            back_state = anim_state.replace("_front", "_back")
            if self.am.has_sprite(sprite_key, back_state):
                frame = self.am.get_frame(
                    sprite_key, back_state, cust.animation_frame)
                surface.blit(frame, rect.topleft)
            elif self.am.has_sprite(sprite_key, "walk_front"):
                frame = self.am.get_frame(
                    sprite_key, "walk_front", cust.animation_frame)
                surface.blit(frame, rect.topleft)
            else:
                self._draw_customer_fallback(surface, cust, rect)
        elif self.am.has_sprite(sprite_key, "walk_front"):
            frame = self.am.get_frame(
                sprite_key, "walk_front", cust.animation_frame)
            surface.blit(frame, rect.topleft)
        else:
            self._draw_customer_fallback(surface, cust, rect)

    @staticmethod
    def _draw_customer_fallback(surface, cust, rect):
        margin = TILE_SIZE // 5
        body = rect.inflate(-margin * 2, -margin * 2)
        pygame.draw.rect(surface, cust.color, body)
        pygame.draw.rect(surface, (0, 0, 0), body, 1)

    def _draw_customers(self, surface, shop, ox, oy):
        for table in shop.tables:
            cust = table.customer
            if cust is None:
                continue

            cx = cust.pixel_x + ox
            cy = cust.pixel_y + oy
            self._draw_customer_sprite(surface, cust, cx, cy)

            # Drink indicator
            if cust.drink_item and not cust.drink_served:
                drink_id = cust.drink_item.get("id", "")
                drink_icon = self.am.get_drink_icon(drink_id)
                if drink_icon:
                    surface.blit(drink_icon, (cx + TILE_SIZE - 16, cy))
                else:
                    di = self.font_sm.render("음", True, (180, 100, 255))
                    surface.blit(di, (cx + TILE_SIZE - 14, cy + 2))

            # Patience bar
            if cust.state in (CustomerState.WAITING_TO_ORDER,
                              CustomerState.ORDER_TAKEN,
                              CustomerState.WALKING_TO_TABLE):
                bw = TILE_SIZE - 8
                bh = 4
                bx = cx + 4
                by = cy + TILE_SIZE - bh - 2
                pygame.draw.rect(surface, (60, 60, 60), (bx, by, bw, bh))
                ratio = cust.patience_ratio
                fw = int(bw * ratio)
                if fw > 0:
                    bcol = (int(255 * (1 - ratio)), int(255 * ratio), 0)
                    pygame.draw.rect(surface, bcol, (bx, by, fw, bh))

    # ═══════════════════════════════════════════════
    #  Waiting Queue (entrance area)
    # ═══════════════════════════════════════════════
    def _draw_leaving_customers(self, surface, shop, ox, oy):
        """Draw customers walking to the exit after payment/penalty."""
        for cust in shop.leaving_customers:
            cx = cust.pixel_x + ox
            cy = cust.pixel_y + oy
            self._draw_customer_sprite(surface, cust, cx, cy)

    def _draw_waiting_queue(self, surface, shop, ox, oy):
        if not shop.waiting_queue:
            return
        ex = int(shop.entrance_x) + ox
        ey = int(shop.entrance_y) + oy
        for i, cust in enumerate(shop.waiting_queue):
            cx = ex + (i % 2) * (TILE_SIZE + 4)
            cy = ey + (i // 2) * (TILE_SIZE + 4)
            self._draw_customer_sprite(surface, cust, cx, cy)
            # Patience bar
            bw = TILE_SIZE - 16
            bh = 3
            bx = cx + 8
            by = cy + TILE_SIZE - 10
            pygame.draw.rect(surface, (60, 60, 60), (bx, by, bw, bh))
            ratio = cust.waiting_patience_ratio
            fw = int(bw * ratio)
            if fw > 0:
                bcol = (int(255 * (1 - ratio)), int(255 * ratio), 0)
                pygame.draw.rect(surface, bcol, (bx, by, fw, bh))
        qlbl = self.font_sm.render(
            f"대기:{len(shop.waiting_queue)}", True, (255, 200, 80))
        surface.blit(qlbl, (ex, ey - 14))

    # ═══════════════════════════════════════════════
    #  Bottom UI panel
    # ═══════════════════════════════════════════════
    def _draw_ui(self, surface, shop, ox, oy):
        ui_y = shop.grid_height * TILE_SIZE + oy
        map_w = shop.grid_width * TILE_SIZE
        panel = pygame.Rect(ox, ui_y, map_w, UI_HEIGHT)
        pygame.draw.rect(surface, COLORS["ui_bg"], panel)

        y0 = ui_y + 8

        # ── money + net profit (순이익=판매총액) ────
        mtxt = self.font_md.render(
            f"${shop.money}  (순이익: ${shop.net_profit})",
            True, COLORS["money"])
        surface.blit(mtxt, (ox + 10, y0))

        # ── day / timer ──────────────────────────
        rem = shop.time_remaining
        m, s = int(rem) // 60, int(rem) % 60
        dtxt = self.font_md.render(
            f"{shop.current_day}일차/{shop.day_limit}일  {m}:{s:02d}",
            True, COLORS["timer"])
        surface.blit(dtxt, dtxt.get_rect(
            centerx=ox + map_w // 2, top=y0))

        # ── satisfaction ─────────────────────────
        rating_stars = shop.shop_rating_stars
        stxt = self.font_md.render(
            f"평점: {rating_stars:.1f}/5.0★", True, COLORS["satisfaction"])
        surface.blit(stxt, stxt.get_rect(right=ox + map_w - 10, top=y0))

        # ── carrying indicator ───────────────────
        y1 = y0 + 28
        if shop.player.carrying:
            items = shop.player.carrying
            parts = []
            for c in items[:3]:
                if c["type"] == "order":
                    parts.append(f"주문(T{c['table_id']})")
                elif c["type"] == "food":
                    parts.append(f"음식(T{c['table_id']})")
                elif c["type"] == "drink":
                    parts.append(f"음료(T{c['table_id']})")
            clbl = "운반: " + " | ".join(parts)
            ccol = (200, 200, 100) if items[0]["type"] == "order" else (100, 220, 100)
        else:
            clbl = "운반: -"
            ccol = (120, 120, 120)
        ctxt = self.font_sm.render(clbl, True, ccol)
        surface.blit(ctxt, (ox + 10, y1))

        # ── stats + hints ────────────────────────
        parts2 = [
            f"서빙:{shop.customers_served}",
            f"이탈:{shop.customers_lost}",
            f"요리사:{shop.num_chefs}/{shop.max_chefs}",
            f"조리:{shop.kitchen.num_cooking}/{shop.kitchen.cooking_capacity}",
            f"보관:{len(shop.kitchen.ready)}/{shop.kitchen.storage_capacity}",
            f"테이블:{len(shop.tables)}",
        ]
        if shop.waiting_queue:
            parts2.append(f"대기:{len(shop.waiting_queue)}")
        if shop.employees:
            parts2.append(f"직원:{len(shop.employees)}")
        if shop.bartender_hired:
            parts2.append(f"바:{shop.bar.num_preparing}")
        parts2.append("[U] 상점")
        stat = self.font_sm.render("  ".join(parts2), True, (150, 150, 150))
        surface.blit(stat, (ox + 10, y1 + 18))

        # ── message ──────────────────────────────
        if shop.message:
            mt = self.font_md.render(shop.message, True, (255, 255, 100))
            surface.blit(mt, mt.get_rect(
                centerx=ox + map_w // 2, top=y1 + 40))

    # ═══════════════════════════════════════════════
    #  Upgrade panel overlay
    # ═══════════════════════════════════════════════
    def _draw_upgrade_panel(self, surface, shop, ox, oy):
        if not shop.upgrade_mode:
            return

        map_w = shop.grid_width * TILE_SIZE
        map_h = shop.grid_height * TILE_SIZE

        info = shop.get_upgrade_info()
        panel_w = 380
        panel_h = 90 + len(info) * 44
        panel_x = ox + (map_w - panel_w) // 2
        panel_y = oy + (map_h - panel_h) // 2

        # Background
        overlay = pygame.Surface((panel_w, panel_h), pygame.SRCALPHA)
        overlay.fill((20, 20, 40, 220))
        surface.blit(overlay, (panel_x, panel_y))
        pygame.draw.rect(surface, (100, 100, 150),
                         (panel_x, panel_y, panel_w, panel_h), 2)

        # Title with net profit
        title = self.font_md.render(
            f"상점  ${shop.money}  순이익:${shop.net_profit}",
            True, (255, 215, 0))
        surface.blit(title, title.get_rect(
            centerx=panel_x + panel_w // 2, top=panel_y + 8))

        # Tab bar
        tab_y = panel_y + 32
        tab_names = shop.TAB_NAMES
        for i, name in enumerate(tab_names):
            tx = panel_x + 10 + i * 120
            selected = (i == shop.upgrade_tab)
            col = (255, 255, 100) if selected else (120, 120, 120)
            prefix = ">" if selected else " "
            tt = self.font_sm.render(f"{prefix}[Tab{i+1}] {name}", True, col)
            surface.blit(tt, (tx, tab_y))

        # Entries
        for i, entry in enumerate(info):
            ey = panel_y + 54 + i * 44
            data = entry["data"]
            maxed = entry["maxed"]
            locked = entry.get("locked", False)
            cost = entry["cost"]
            can_afford = entry["can_afford"]
            is_food = entry.get("is_food_unlock", False)

            # Key number
            if locked:
                key_col = (80, 80, 80)
                key_str = "[X]"
            elif maxed:
                key_col = (60, 60, 60)
                key_str = "[v]"
            elif is_food:
                # 자동 해금 — 구매 불가
                key_col = (200, 200, 100)
                key_str = "[~]"
            elif can_afford:
                key_col = (100, 255, 100)
                key_str = f"[{i + 1}]"
            else:
                key_col = (180, 120, 120)
                key_str = f"[{i + 1}]"
            key_txt = self.font_md.render(key_str, True, key_col)
            surface.blit(key_txt, (panel_x + 10, ey))

            # Name + level/status
            if is_food:
                nm_str = f"{data['name']}  ${data['price']}"
                if maxed:
                    nm_str += "  [해금됨]"
                    nm_col = (80, 180, 80)
                elif locked:
                    req = entry.get("unlock_profit_req", data.get("unlock_profit", 0))
                    nm_str += f"  (순이익 ${req} 필요)"
                    nm_col = (100, 100, 100)
                else:
                    nm_str += "  [자동 해금 대기]"
                    nm_col = (200, 200, 100)
            else:
                level = entry["level"]
                if maxed:
                    nm_str = f"{data['name']}  최대"
                    nm_col = (80, 80, 80)
                elif locked:
                    nm_str = f"{data['name']}  (순이익 ${data.get('unlock_profit', 0)} 필요)"
                    nm_col = (100, 100, 100)
                else:
                    nm_str = f"{data['name']}  Lv.{level}/{data['max_level']}"
                    nm_col = (255, 255, 255) if can_afford else (180, 120, 120)

            nm = self.font_md.render(nm_str, True, nm_col)
            surface.blit(nm, (panel_x + 50, ey))

            # Cost (food items are auto-unlocked, no cost shown)
            if not maxed and not locked and not is_food:
                cost_col = (100, 255, 100) if can_afford else (255, 100, 100)
                ct = self.font_sm.render(f"${cost}", True, cost_col)
                surface.blit(ct, (panel_x + 50, ey + 22))

            # Description
            desc_str = data.get("description", "")
            if desc_str:
                desc = self.font_sm.render(desc_str, True, (130, 130, 130))
                surface.blit(desc, (panel_x + 130, ey + 22))

        # Close hint
        hint = self.font_sm.render(
            "[Tab] 탭 전환  |  [1-9] 구매  |  [U/ESC] 닫기",
            True, (120, 120, 120))
        surface.blit(hint, hint.get_rect(
            centerx=panel_x + panel_w // 2,
            top=panel_y + panel_h - 18))

    # ═══════════════════════════════════════════════
    #  Layer 3: Bar counters
    # ═══════════════════════════════════════════════
    def _draw_bar(self, surface, shop, ox, oy):
        bar = shop.bar
        positions = sorted(shop.bar_counter_positions)

        if not shop.bartender_hired:
            return

        # Draw wide bar image spanning all positions
        if positions and self.am.tile_bar_wide:
            left_pos = positions[0]
            bx = left_pos[0] * TILE_SIZE + ox
            by = left_pos[1] * TILE_SIZE + oy
            surface.blit(self.am.tile_bar_wide, (bx, by))
        else:
            for pos in positions:
                rect = pygame.Rect(pos[0] * TILE_SIZE + ox,
                                   pos[1] * TILE_SIZE + oy,
                                   TILE_SIZE, TILE_SIZE)
                if self.am.tile_bar:
                    surface.blit(self.am.tile_bar, rect.topleft)
                else:
                    col = COLORS.get("bar", (100, 60, 140))
                    pygame.draw.rect(surface, col, rect)

        tile_slots: list[tuple[str, dict] | None] = [None] * len(positions)
        idx = 0
        for order in bar.preparing:
            if idx < len(tile_slots):
                tile_slots[idx] = ("preparing", order)
                idx += 1
        for item in bar.ready:
            if idx < len(tile_slots):
                tile_slots[idx] = ("ready", item)
                idx += 1

        for i, pos in enumerate(positions):
            rect = pygame.Rect(pos[0] * TILE_SIZE + ox,
                               pos[1] * TILE_SIZE + oy,
                               TILE_SIZE, TILE_SIZE)

            slot = tile_slots[i] if i < len(tile_slots) else None
            if slot is None:
                continue

            state, data = slot

            if state == "preparing":
                drink_id = data["drink_item"].get("id", "")
                drink_icon = self.am.get_drink_icon(drink_id)
                if drink_icon:
                    surface.blit(drink_icon, (rect.x + 16, rect.y + 2))
                else:
                    name = data["drink_item"]["name"]
                    nm = self.font_sm.render(name[:4], True, (255, 255, 255))
                    surface.blit(nm, nm.get_rect(
                        centerx=rect.centerx, top=rect.y + 4))
                tid = self.font_sm.render(
                    f"T{data['table_id']}", True, (180, 180, 180))
                surface.blit(tid, (rect.right - 22, rect.y + 4))

                total = data["drink_item"]["prep_time"]
                ratio = max(0.0, 1.0 - data["timer"] / total) if total else 1.0
                bw = TILE_SIZE - 8
                bx = rect.x + 4
                by = rect.bottom - 12
                pygame.draw.rect(surface, (60, 60, 60), (bx, by, bw, 8))
                pygame.draw.rect(surface, (160, 100, 220),
                                 (bx, by, int(bw * ratio), 8))
                pct = self.font_sm.render(
                    f"{int(ratio * 100)}%", True, (200, 200, 200))
                surface.blit(pct, pct.get_rect(
                    centerx=rect.centerx, bottom=rect.bottom - 14))

            elif state == "ready":
                done = self.font_sm.render("완성", True, (255, 200, 80))
                surface.blit(done, (rect.x + 2, rect.y + 2))
                drink_id = data["drink_item"].get("id", "")
                drink_icon = self.am.get_drink_icon(drink_id)
                if drink_icon:
                    surface.blit(drink_icon, (rect.x + 16, rect.centery - 16))
                else:
                    name = data["drink_item"]["name"]
                    nm = self.font_sm.render(name[:4], True, (220, 160, 255))
                    surface.blit(nm, nm.get_rect(center=rect.center))
                tid = self.font_sm.render(
                    f"T{data['table_id']}", True, (180, 180, 180))
                surface.blit(tid, tid.get_rect(
                    centerx=rect.centerx, bottom=rect.bottom - 4))

    # ═══════════════════════════════════════════════
    #  Employees (sprite-based with direction detection)
    # ═══════════════════════════════════════════════
    @staticmethod
    def _employee_sprite_key(emp) -> str:
        """Determine sprite key based on employee gender assignment.
        Ratio: 2 male, 3 female (cycle per emp_id).
        """
        return "employee_man" if emp.emp_id % 5 < 2 else "employee_woman"

    @staticmethod
    def _employee_anim_state(emp) -> str:
        from core.employee import Employee
        if emp.state == Employee.IDLE:
            return "idle"
        if emp.state == Employee.ACTING:
            return "down"  # front-facing for actions

        # MOVING: determine direction from movement vector
        if emp.target_x is not None and emp.target_y is not None:
            dx = emp.target_x - (emp.x + TILE_SIZE / 2)
            dy = emp.target_y - (emp.y + TILE_SIZE / 2)
            if abs(dx) > abs(dy):
                return "right" if dx > 0 else "left"
            else:
                return "down" if dy > 0 else "up"
        return "idle"

    def _draw_employees(self, surface, shop, ox, oy):
        for emp in shop.employees:
            ecx = int(emp.x) + ox
            ecy = int(emp.y) + oy
            rect = pygame.Rect(ecx, ecy, TILE_SIZE, TILE_SIZE)

            sprite_key = self._employee_sprite_key(emp)
            anim_state = self._employee_anim_state(emp)

            if self.am.has_sprite(sprite_key, anim_state):
                frame = self.am.get_frame(
                    sprite_key, anim_state, emp.animation_frame)
                surface.blit(frame, rect.topleft)
            elif self.am.has_sprite(sprite_key, "idle"):
                frame = self.am.get_frame(sprite_key, "idle", 0)
                surface.blit(frame, rect.topleft)
            else:
                radius = 14
                cx_px = ecx + TILE_SIZE // 2
                cy_px = ecy + TILE_SIZE // 2
                pygame.draw.circle(surface, emp.color,
                                   (cx_px, cy_px), radius)
                pygame.draw.circle(surface, (0, 0, 0),
                                   (cx_px, cy_px), radius, 1)

            # Carry icon above head
            if emp.carrying:
                carry_type = emp.carrying.get("type", "")
                icon = None
                if carry_type == "food":
                    food_id = emp.carrying.get("menu_item", {}).get("id", "")
                    icon = self.am.get_food_icon(food_id)
                elif carry_type == "drink":
                    drink_id = emp.carrying.get("drink_item", {}).get("id", "")
                    icon = self.am.get_drink_icon(drink_id)
                if icon:
                    surface.blit(icon, icon.get_rect(
                        centerx=ecx + TILE_SIZE // 2,
                        bottom=ecy - 2))
                else:
                    name = self._carry_name(emp.carrying)
                    if name:
                        nm = self.font_sm.render(name, True, (255, 255, 200))
                        surface.blit(nm, nm.get_rect(
                            centerx=ecx + TILE_SIZE // 2,
                            bottom=ecy - 2))

    # ── helper: extract Korean food name from a carry dict ──
    @staticmethod
    def _carry_name(item: dict) -> str:
        t = item.get("type", "")
        if t == "order":
            return item.get("item", {}).get("name", "주문")[:3]
        if t == "food":
            return item.get("menu_item", {}).get("name", "음식")[:3]
        if t == "drink":
            return item.get("drink_item", {}).get("name", "음료")[:3]
        return ""

    # ── carried food/drink icon + label on player ──
    def _draw_carry_labels(self, surface, shop, ox, oy):
        player = shop.player
        if not player.carrying:
            return
        px = player.x + ox
        py = player.y + oy
        for i, item in enumerate(player.carrying[:3]):
            iy = int(py) - 6 - i * 34   # stack upward above head
            cx = int(px + TILE_SIZE // 2)

            # Draw food/drink icon above head
            carry_type = item.get("type", "")
            icon = None
            if carry_type == "food":
                food_id = item.get("menu_item", {}).get("id", "")
                icon = self.am.get_food_icon(food_id)
            elif carry_type == "drink":
                drink_id = item.get("drink_item", {}).get("id", "")
                icon = self.am.get_drink_icon(drink_id)
            if icon:
                surface.blit(icon, icon.get_rect(
                    centerx=cx, bottom=iy))
            else:
                name = self._carry_name(item)
                if name:
                    lbl = self.font_sm.render(name, True, (255, 255, 200))
                    surface.blit(lbl, lbl.get_rect(
                        centerx=cx, bottom=iy))

    # ═══════════════════════════════════════════════
    #  Floating texts (+$X payment labels)
    # ═══════════════════════════════════════════════
    def _draw_floating_texts(self, surface, shop, ox, oy):
        for ft in shop.floating_texts:
            alpha = min(255, int(255 * (ft["timer"] / 1.2)))
            txt = self.font_md.render(ft["text"], True, (50, 255, 50))
            txt.set_alpha(alpha)
            surface.blit(txt, txt.get_rect(
                centerx=int(ft["x"]) + ox,
                bottom=int(ft["y"]) + oy))

    # ═══════════════════════════════════════════════
    #  Trait selection popup
    # ═══════════════════════════════════════════════
    def _draw_trait_popup(self, surface, shop, ox, oy):
        if not shop.trait_selection_active:
            return

        map_w = shop.grid_width * TILE_SIZE
        map_h = shop.grid_height * TILE_SIZE
        choices = shop.trait_choices

        pw = 340
        ph = 60 + len(choices) * 60
        px = ox + (map_w - pw) // 2
        py = oy + (map_h - ph) // 2

        overlay = pygame.Surface((pw, ph), pygame.SRCALPHA)
        overlay.fill((30, 10, 50, 230))
        surface.blit(overlay, (px, py))
        pygame.draw.rect(surface, (200, 160, 255), (px, py, pw, ph), 2)

        title = self.font_md.render(
            f"특성 선택  ({shop.current_day}일차)", True, (255, 200, 100))
        surface.blit(title, title.get_rect(centerx=px + pw // 2, top=py + 10))

        for i, trait in enumerate(choices):
            ty = py + 42 + i * 60
            key_txt = self.font_md.render(f"[{i + 1}]", True, (100, 255, 200))
            surface.blit(key_txt, (px + 12, ty))
            name_txt = self.font_md.render(trait["name"], True, (255, 255, 255))
            surface.blit(name_txt, (px + 50, ty))
            stacks = shop.traits.get(trait["id"], 0)
            max_s = trait.get("max_stacks", 1)
            desc_txt = self.font_sm.render(
                f'{trait["description"]}  ({stacks}/{max_s})',
                True, (180, 180, 180))
            surface.blit(desc_txt, (px + 50, ty + 24))

    # ═══════════════════════════════════════════════
    #  Game-over overlay
    # ═══════════════════════════════════════════════
    def draw_game_over(self, surface: pygame.Surface, shop, *,
                       offset_x: int = 0, offset_y: int = 0,
                       extra_text: str = ""):
        self._ensure_fonts()
        w = shop.grid_width * TILE_SIZE
        h = shop.grid_height * TILE_SIZE + UI_HEIGHT
        overlay = pygame.Surface((w, h), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 140))
        surface.blit(overlay, (offset_x, offset_y))

        cx = offset_x + w // 2
        cy = offset_y + h // 2

        if shop.won:
            title = self.font_lg.render("목표 달성!", True, (255, 215, 0))
        else:
            title = self.font_lg.render("시간 종료!", True, (255, 100, 100))
        surface.blit(title, title.get_rect(center=(cx, cy - 40)))

        rating_mult = 1.0 + shop.shop_rating_stars / 10.0
        score_txt = self.font_md.render(
            f"최종 스코어: {shop.final_score:,.1f}"
            f"  (순이익 ${shop.net_profit:,} × 평점계수 {rating_mult:.2f})",
            True, (255, 255, 150))
        surface.blit(score_txt, score_txt.get_rect(center=(cx, cy - 10)))

        sub = self.font_md.render(
            f"보유금: ${shop.money}  순이익: ${shop.net_profit}"
            f"  평점: {shop.shop_rating_stars:.1f}/5.0★"
            f"   {extra_text}", True, (200, 200, 200))
        surface.blit(sub, sub.get_rect(center=(cx, cy + 15)))

        stat = self.font_sm.render(
            f"서빙: {shop.customers_served}  이탈: {shop.customers_lost}",
            True, (160, 160, 160))
        surface.blit(stat, stat.get_rect(center=(cx, cy + 40)))

        hint = self.font_sm.render("R: 재시작  |  ESC: 종료",
                                   True, (160, 160, 160))
        surface.blit(hint, hint.get_rect(center=(cx, cy + 60)))
