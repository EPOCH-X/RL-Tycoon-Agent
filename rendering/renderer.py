"""Renderer – draws the full game view for one Shop instance.

All Pygame drawing is centralised here so that the core/ modules stay
rendering-free and can be used headlessly by the RL environment.
"""

import pygame
from config.settings import TILE_SIZE, UI_HEIGHT, COLORS
from core.customer import CustomerState
from rendering.asset_manager import AssetManager


class Renderer:
    """Draws a Shop's state onto a Pygame surface."""

    def __init__(self, asset_manager: AssetManager):
        self.am = asset_manager
        self._fonts_ok = False
        self.font_sm: pygame.font.Font | None = None
        self.font_md: pygame.font.Font | None = None
        self.font_lg: pygame.font.Font | None = None

    # ── lazy font init (needs pygame.init first) ─
    def _ensure_fonts(self):
        if not self._fonts_ok:
            self.font_sm = pygame.font.SysFont(None, 16)
            self.font_md = pygame.font.SysFont(None, 24)
            self.font_lg = pygame.font.SysFont(None, 36)
            self._fonts_ok = True

    # ═══════════════════════════════════════════════
    #  Main entry point
    # ═══════════════════════════════════════════════
    def draw(self, surface: pygame.Surface, shop, *,
             offset_x: int = 0, offset_y: int = 0):
        """Draw the complete shop view (map + entities + UI)."""
        self._ensure_fonts()
        self._draw_map(surface, shop, offset_x, offset_y)
        self._draw_tables(surface, shop, offset_x, offset_y)
        self._draw_purchasable_tables(surface, shop, offset_x, offset_y)
        self._draw_kitchen(surface, shop, offset_x, offset_y)
        self._draw_customers(surface, shop, offset_x, offset_y)
        shop.player.render(surface, self.am, offset_x, offset_y)
        self._draw_ui(surface, shop, offset_x, offset_y)
        self._draw_upgrade_panel(surface, shop, offset_x, offset_y)

    # ═══════════════════════════════════════════════
    #  Map tiles
    # ═══════════════════════════════════════════════
    def _draw_map(self, surface, shop, ox, oy):
        for y, row in enumerate(shop.layout):
            for x, tile in enumerate(row):
                rect = pygame.Rect(x * TILE_SIZE + ox,
                                   y * TILE_SIZE + oy,
                                   TILE_SIZE, TILE_SIZE)
                color = COLORS["floor"] if tile == 0 else COLORS["wall"]
                pygame.draw.rect(surface, color, rect)
                pygame.draw.rect(surface, COLORS["grid_line"], rect, 1)

    # ═══════════════════════════════════════════════
    #  Tables (active)
    # ═══════════════════════════════════════════════
    def _draw_tables(self, surface, shop, ox, oy):
        for table in shop.tables:
            rect = pygame.Rect(table.grid_x * TILE_SIZE + ox,
                               table.grid_y * TILE_SIZE + oy,
                               TILE_SIZE, TILE_SIZE)
            col = COLORS["table_occupied"] if table.is_occupied else COLORS["table"]
            pygame.draw.rect(surface, col, rect)
            pygame.draw.rect(surface, COLORS["grid_line"], rect, 1)

            lbl = self.font_sm.render(f"T{table.table_id}", True, COLORS["text"])
            surface.blit(lbl, lbl.get_rect(center=rect.center))

    # ═══════════════════════════════════════════════
    #  Purchasable table ghosts
    # ═══════════════════════════════════════════════
    def _draw_purchasable_tables(self, surface, shop, ox, oy):
        for tdata in shop._purchasable_tables:
            rect = pygame.Rect(tdata["grid_x"] * TILE_SIZE + ox,
                               tdata["grid_y"] * TILE_SIZE + oy,
                               TILE_SIZE, TILE_SIZE)
            ghost = pygame.Surface((TILE_SIZE, TILE_SIZE), pygame.SRCALPHA)
            ghost.fill((139, 90, 43, 50))
            surface.blit(ghost, rect.topleft)
            pygame.draw.rect(surface, (100, 70, 30), rect, 2)
            lbl = self.font_sm.render("$", True, (200, 180, 100))
            surface.blit(lbl, lbl.get_rect(center=rect.center))

    # ═══════════════════════════════════════════════
    #  Kitchen counters
    # ═══════════════════════════════════════════════
    def _draw_kitchen(self, surface, shop, ox, oy):
        kitchen = shop.kitchen
        for pos in shop.kitchen_counter_positions:
            rect = pygame.Rect(pos[0] * TILE_SIZE + ox,
                               pos[1] * TILE_SIZE + oy,
                               TILE_SIZE, TILE_SIZE)
            if kitchen.has_ready:
                col = COLORS["kitchen_ready"]
            elif kitchen.num_cooking > 0:
                col = COLORS["kitchen_cooking"]
            else:
                col = COLORS["kitchen"]
            pygame.draw.rect(surface, col, rect)
            pygame.draw.rect(surface, COLORS["grid_line"], rect, 1)

        # Status label on first kitchen tile
        if shop.kitchen_counter_positions:
            first = min(shop.kitchen_counter_positions)
            rx = first[0] * TILE_SIZE + ox
            ry = first[1] * TILE_SIZE + oy

            if kitchen.num_cooking > 0:
                ct = self.font_sm.render(
                    f"Cook:{kitchen.num_cooking}", True, COLORS["text"])
                surface.blit(ct, (rx + 2, ry + 2))

            if kitchen.has_ready:
                rt = self.font_sm.render(
                    f"Ready:{len(kitchen.ready)}", True, (255, 255, 100))
                surface.blit(rt, (rx + 2, ry + TILE_SIZE - 16))

            bar_x = rx + TILE_SIZE + 4
            for i, order in enumerate(kitchen.cooking):
                if i >= 3:
                    break
                total = order["menu_item"]["cook_time"]
                ratio = 1.0 - order["timer"] / total if total else 1.0
                by = ry + i * 18
                bw = TILE_SIZE - 8
                nm = self.font_sm.render(
                    order["menu_item"]["name"][:6], True, (200, 200, 200))
                surface.blit(nm, (bar_x, by))
                bbx = bar_x + 48
                pygame.draw.rect(surface, (60, 60, 60), (bbx, by + 2, bw, 8))
                pygame.draw.rect(surface, (80, 220, 80),
                                 (bbx, by + 2, int(bw * ratio), 8))

    # ═══════════════════════════════════════════════
    #  Customers (drawn on their table position)
    # ═══════════════════════════════════════════════
    def _draw_customers(self, surface, shop, ox, oy):
        for table in shop.tables:
            cust = table.customer
            if cust is None:
                continue

            cx = cust.pixel_x + ox
            cy = cust.pixel_y + oy
            tile_rect = pygame.Rect(cx, cy, TILE_SIZE, TILE_SIZE)

            margin = TILE_SIZE // 5
            body = tile_rect.inflate(-margin * 2, -margin * 2)
            pygame.draw.rect(surface, cust.color, body)
            pygame.draw.rect(surface, (0, 0, 0), body, 1)

            if cust.state == CustomerState.WAITING_TO_ORDER:
                icon = self.font_sm.render("?!", True, (255, 255, 255))
            elif cust.state == CustomerState.ORDER_TAKEN:
                icon = self.font_sm.render(
                    cust.menu_item["name"][:5], True, (200, 200, 100))
            elif cust.state == CustomerState.EATING:
                icon = self.font_sm.render("nom", True, (100, 255, 100))
            else:
                icon = self.font_sm.render("...", True, (150, 150, 150))
            surface.blit(icon, icon.get_rect(center=body.center))

            if cust.state in (CustomerState.WAITING_TO_ORDER,
                              CustomerState.ORDER_TAKEN):
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
    #  Bottom UI panel
    # ═══════════════════════════════════════════════
    def _draw_ui(self, surface, shop, ox, oy):
        ui_y = shop.grid_height * TILE_SIZE + oy
        map_w = shop.grid_width * TILE_SIZE
        panel = pygame.Rect(ox, ui_y, map_w, UI_HEIGHT)
        pygame.draw.rect(surface, COLORS["ui_bg"], panel)

        y0 = ui_y + 8

        # ── money ────────────────────────────────
        mtxt = self.font_md.render(
            f"Money: ${shop.money} / ${shop.target_money}",
            True, COLORS["money"])
        surface.blit(mtxt, (ox + 10, y0))

        # ── day / timer ──────────────────────────
        rem = shop.time_remaining
        m, s = int(rem) // 60, int(rem) % 60
        dtxt = self.font_md.render(
            f"Day {shop.current_day}/{shop.day_limit}  {m}:{s:02d}",
            True, COLORS["timer"])
        surface.blit(dtxt, dtxt.get_rect(
            centerx=ox + map_w // 2, top=y0))

        # ── satisfaction ─────────────────────────
        rating_pct = int(shop.shop_rating * 100)
        stxt = self.font_md.render(
            f"Rating: {rating_pct}%", True, COLORS["satisfaction"])
        surface.blit(stxt, stxt.get_rect(right=ox + map_w - 10, top=y0))

        # ── carrying indicator ───────────────────
        y1 = y0 + 28
        if shop.player.carrying:
            carry = shop.player.carrying
            if carry["type"] == "order":
                clbl = f"Carrying: ORDER (T{carry['table_id']} {carry['menu_item']['name']})"
                ccol = (200, 200, 100)
            else:
                clbl = f"Carrying: FOOD (T{carry['table_id']} {carry['menu_item']['name']})"
                ccol = (100, 220, 100)
        else:
            clbl = "Carrying: -"
            ccol = (120, 120, 120)
        ctxt = self.font_sm.render(clbl, True, ccol)
        surface.blit(ctxt, (ox + 10, y1))

        # ── stats + upgrade hint ─────────────────
        stat = self.font_sm.render(
            f"Served: {shop.customers_served}  Lost: {shop.customers_lost}"
            f"  Kitchen: {shop.kitchen.num_cooking}/{shop.kitchen.capacity}"
            f"  Tables: {len(shop.tables)}"
            f"   [U] Upgrades",
            True, (150, 150, 150))
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
        panel_w = 360
        panel_h = 50 + len(info) * 48
        panel_x = ox + (map_w - panel_w) // 2
        panel_y = oy + (map_h - panel_h) // 2

        # Background
        overlay = pygame.Surface((panel_w, panel_h), pygame.SRCALPHA)
        overlay.fill((20, 20, 40, 220))
        surface.blit(overlay, (panel_x, panel_y))
        pygame.draw.rect(surface, (100, 100, 150),
                         (panel_x, panel_y, panel_w, panel_h), 2)

        # Title
        title = self.font_md.render(
            f"UPGRADES  (Money: ${shop.money})", True, (255, 215, 0))
        surface.blit(title, title.get_rect(
            centerx=panel_x + panel_w // 2, top=panel_y + 10))

        # Entries
        for i, entry in enumerate(info):
            ey = panel_y + 44 + i * 48
            upg = entry["data"]
            level = entry["level"]
            maxed = entry["maxed"]
            cost = entry["cost"]
            can_afford = entry["can_afford"]

            # Key number
            key_col = (100, 255, 100) if can_afford else (100, 100, 100)
            key_txt = self.font_md.render(f"[{i + 1}]", True, key_col)
            surface.blit(key_txt, (panel_x + 12, ey))

            # Name + level
            if maxed:
                nm_col = (80, 80, 80)
                nm_str = f"{upg['name']}  MAX"
            else:
                nm_col = (255, 255, 255) if can_afford else (180, 120, 120)
                nm_str = f"{upg['name']}  Lv.{level}/{upg['max_level']}"
            nm = self.font_md.render(nm_str, True, nm_col)
            surface.blit(nm, (panel_x + 56, ey))

            # Cost
            if not maxed:
                cost_col = (100, 255, 100) if can_afford else (255, 100, 100)
                ct = self.font_sm.render(f"${cost}", True, cost_col)
                surface.blit(ct, (panel_x + 56, ey + 22))

            # Description
            desc = self.font_sm.render(
                upg.get("description", ""), True, (130, 130, 130))
            surface.blit(desc, (panel_x + 120, ey + 22))

        # Close hint
        hint = self.font_sm.render(
            "Press U or ESC to close", True, (120, 120, 120))
        surface.blit(hint, hint.get_rect(
            centerx=panel_x + panel_w // 2,
            top=panel_y + panel_h - 18))

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
            title = self.font_lg.render("TARGET REACHED!", True, (255, 215, 0))
        else:
            title = self.font_lg.render("TIME'S UP!", True, (255, 100, 100))
        surface.blit(title, title.get_rect(center=(cx, cy - 30)))

        sub = self.font_md.render(
            f"Final Money: ${shop.money}  Rating: {int(shop.shop_rating * 100)}%"
            f"   {extra_text}", True, (200, 200, 200))
        surface.blit(sub, sub.get_rect(center=(cx, cy + 10)))

        stat = self.font_sm.render(
            f"Served: {shop.customers_served}  Lost: {shop.customers_lost}",
            True, (160, 160, 160))
        surface.blit(stat, stat.get_rect(center=(cx, cy + 35)))

        hint = self.font_sm.render("Press R to restart  |  ESC to quit",
                                   True, (160, 160, 160))
        surface.blit(hint, hint.get_rect(center=(cx, cy + 55)))
