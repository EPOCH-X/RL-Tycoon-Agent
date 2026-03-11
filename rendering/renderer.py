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
        self._draw_bar(surface, shop, offset_x, offset_y)
        self._draw_customers(surface, shop, offset_x, offset_y)
        self._draw_employees(surface, shop, offset_x, offset_y)
        shop.player.render(surface, self.am, offset_x, offset_y)
        self._draw_ui(surface, shop, offset_x, offset_y)
        self._draw_upgrade_panel(surface, shop, offset_x, offset_y)
        self._draw_trait_popup(surface, shop, offset_x, offset_y)

    # ═══════════════════════════════════════════════
    #  Map tiles
    # ═══════════════════════════════════════════════
    def _draw_map(self, surface, shop, ox, oy):
        for y, row in enumerate(shop.layout):
            for x, tile in enumerate(row):
                rect = pygame.Rect(x * TILE_SIZE + ox,
                                   y * TILE_SIZE + oy,
                                   TILE_SIZE, TILE_SIZE)
                if tile == 0:
                    color = COLORS["floor"]
                elif tile == 4:
                    color = COLORS.get("bar", (100, 60, 140))
                else:
                    color = COLORS["wall"]
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
                remaining = cust.group_size - cust.food_served_count
                lbl = f"{remaining}x" if remaining > 1 else cust.menu_item["name"][:5]
                icon = self.font_sm.render(lbl, True, (200, 200, 100))
            elif cust.state == CustomerState.EATING:
                icon = self.font_sm.render("nom", True, (100, 255, 100))
            else:
                icon = self.font_sm.render("...", True, (150, 150, 150))
            surface.blit(icon, icon.get_rect(center=body.center))

            # Group size badge (top-left)
            if cust.group_size > 1:
                badge = self.font_sm.render(f"x{cust.group_size}", True, (255, 200, 0))
                surface.blit(badge, (cx + 2, cy + 2))

            # Drink indicator (top-right)
            if cust.drink_item and not cust.drink_served:
                di = self.font_sm.render("D", True, (180, 100, 255))
                surface.blit(di, (cx + TILE_SIZE - 14, cy + 2))

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

        # ── money + net profit ───────────────────
        mtxt = self.font_md.render(
            f"${shop.money}  (Net: ${shop.net_profit})",
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
            items = shop.player.carrying
            parts = []
            for c in items[:3]:
                if c["type"] == "order":
                    parts.append(f"ORD(T{c['table_id']})")
                elif c["type"] == "food":
                    parts.append(f"FOOD(T{c['table_id']})")
                elif c["type"] == "drink":
                    parts.append(f"DRK(T{c['table_id']})")
            clbl = "Carry: " + " | ".join(parts)
            ccol = (200, 200, 100) if items[0]["type"] == "order" else (100, 220, 100)
        else:
            clbl = "Carrying: -"
            ccol = (120, 120, 120)
        ctxt = self.font_sm.render(clbl, True, ccol)
        surface.blit(ctxt, (ox + 10, y1))

        # ── stats + hints ────────────────────────
        parts2 = [
            f"Served:{shop.customers_served}",
            f"Lost:{shop.customers_lost}",
            f"Kitchen:{shop.kitchen.num_cooking}/{shop.kitchen.capacity}",
            f"Tables:{len(shop.tables)}",
        ]
        if shop.employees:
            parts2.append(f"Emp:{len(shop.employees)}")
        if shop.bartender_hired:
            parts2.append(f"Bar:{shop.bar.num_preparing}")
        if shop.delivery_unlocked:
            parts2.append(f"Del:{len(shop.delivery_orders)}")
        parts2.append("[U] Shop")
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
            f"SHOP  ${shop.money}  Net:${shop.net_profit}",
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
                    nm_str += "  [UNLOCKED]"
                    nm_col = (80, 180, 80)
                elif locked:
                    nm_str += f"  (Need net ${data.get('unlock_profit', 0)})"
                    nm_col = (100, 100, 100)
                else:
                    nm_col = (255, 255, 255) if can_afford else (180, 120, 120)
            else:
                level = entry["level"]
                if maxed:
                    nm_str = f"{data['name']}  MAX"
                    nm_col = (80, 80, 80)
                elif locked:
                    nm_str = f"{data['name']}  (Need net ${data.get('unlock_profit', 0)})"
                    nm_col = (100, 100, 100)
                else:
                    nm_str = f"{data['name']}  Lv.{level}/{data['max_level']}"
                    nm_col = (255, 255, 255) if can_afford else (180, 120, 120)

            nm = self.font_md.render(nm_str, True, nm_col)
            surface.blit(nm, (panel_x + 50, ey))

            # Cost
            if not maxed and not locked:
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
            "[Tab] switch  |  [1-9] buy  |  [U/ESC] close",
            True, (120, 120, 120))
        surface.blit(hint, hint.get_rect(
            centerx=panel_x + panel_w // 2,
            top=panel_y + panel_h - 18))

    # ═══════════════════════════════════════════════
    #  Bar counters
    # ═══════════════════════════════════════════════
    def _draw_bar(self, surface, shop, ox, oy):
        bar = shop.bar
        for pos in shop.bar_counter_positions:
            rect = pygame.Rect(pos[0] * TILE_SIZE + ox,
                               pos[1] * TILE_SIZE + oy,
                               TILE_SIZE, TILE_SIZE)
            if shop.bartender_hired:
                if bar.has_ready:
                    col = COLORS.get("bar_ready", (160, 100, 220))
                elif bar.num_preparing > 0:
                    col = COLORS.get("bar", (100, 60, 140))
                else:
                    col = COLORS.get("bar", (100, 60, 140))
            else:
                col = (60, 40, 80)
            pygame.draw.rect(surface, col, rect)
            pygame.draw.rect(surface, COLORS["grid_line"], rect, 1)

        if shop.bar_counter_positions and shop.bartender_hired:
            first = min(shop.bar_counter_positions)
            rx = first[0] * TILE_SIZE + ox
            ry = first[1] * TILE_SIZE + oy

            if bar.num_preparing > 0:
                ct = self.font_sm.render(
                    f"Mix:{bar.num_preparing}", True, COLORS["text"])
                surface.blit(ct, (rx + 2, ry + 2))
            if bar.has_ready:
                rt = self.font_sm.render(
                    f"Ready:{len(bar.ready)}", True, (220, 160, 255))
                surface.blit(rt, (rx + 2, ry + TILE_SIZE - 16))

    # ═══════════════════════════════════════════════
    #  Employees
    # ═══════════════════════════════════════════════
    def _draw_employees(self, surface, shop, ox, oy):
        for emp in shop.employees:
            ecx = emp.x + TILE_SIZE // 2 + ox
            ecy = emp.y + TILE_SIZE // 2 + oy
            radius = 14
            pygame.draw.circle(surface, emp.color, (int(ecx), int(ecy)), radius)
            pygame.draw.circle(surface, (0, 0, 0), (int(ecx), int(ecy)), radius, 1)
            lbl = self.font_sm.render(f"E{emp.emp_id}", True, (255, 255, 255))
            surface.blit(lbl, lbl.get_rect(center=(int(ecx), int(ecy))))

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
            f"Trait Selection  (Day {shop.current_day})", True, (255, 200, 100))
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
            title = self.font_lg.render("TARGET REACHED!", True, (255, 215, 0))
        else:
            title = self.font_lg.render("TIME'S UP!", True, (255, 100, 100))
        surface.blit(title, title.get_rect(center=(cx, cy - 30)))

        sub = self.font_md.render(
            f"Money: ${shop.money}  Net Profit: ${shop.net_profit}"
            f"  Rating: {int(shop.shop_rating * 100)}%"
            f"   {extra_text}", True, (200, 200, 200))
        surface.blit(sub, sub.get_rect(center=(cx, cy + 10)))

        stat = self.font_sm.render(
            f"Served: {shop.customers_served}  Lost: {shop.customers_lost}",
            True, (160, 160, 160))
        surface.blit(stat, stat.get_rect(center=(cx, cy + 35)))

        hint = self.font_sm.render("Press R to restart  |  ESC to quit",
                                   True, (160, 160, 160))
        surface.blit(hint, hint.get_rect(center=(cx, cy + 55)))
