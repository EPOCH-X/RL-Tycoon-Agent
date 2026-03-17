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
            font_name = self._find_korean_font()
            self.font_sm = pygame.font.SysFont(font_name, 16)
            self.font_md = pygame.font.SysFont(font_name, 24)
            self.font_lg = pygame.font.SysFont(font_name, 36)
            self._fonts_ok = True

    @staticmethod
    def _find_korean_font():
        """Find a Korean-supporting system font."""
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
        """Draw the complete shop view (map + entities + UI)."""
        self._ensure_fonts()
        self._draw_map(surface, shop, offset_x, offset_y)
        self._draw_tables(surface, shop, offset_x, offset_y)
        self._draw_purchasable_tables(surface, shop, offset_x, offset_y)
        self._draw_kitchen(surface, shop, offset_x, offset_y)
        self._draw_bar(surface, shop, offset_x, offset_y)
        self._draw_trash_cans(surface, shop, offset_x, offset_y)
        self._draw_customers(surface, shop, offset_x, offset_y)
        self._draw_leaving_customers(surface, shop, offset_x, offset_y)
        self._draw_waiting_queue(surface, shop, offset_x, offset_y)
        self._draw_employees(surface, shop, offset_x, offset_y)
        shop.player.render(surface, self.am, offset_x, offset_y)
        self._draw_carry_labels(surface, shop, offset_x, offset_y)
        self._draw_floating_texts(surface, shop, offset_x, offset_y)
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
                elif tile == 5:
                    color = COLORS.get("trash_can", (120, 100, 80))
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
    #  Kitchen counters (per-tile: chef + food name + gauge)
    # ═══════════════════════════════════════════════
    def _draw_kitchen(self, surface, shop, ox, oy):
        kitchen = shop.kitchen
        positions = sorted(shop.kitchen_counter_positions)

        # Assign items to tiles: cooking first, then ready
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
            slot = tile_slots[i] if i < len(tile_slots) else None

            # Tile background colour
            if slot and slot[0] == "ready":
                col = COLORS["kitchen_ready"]
            elif slot and slot[0] == "cooking":
                col = COLORS["kitchen_cooking"]
            else:
                col = COLORS["kitchen"]
            pygame.draw.rect(surface, col, rect)
            pygame.draw.rect(surface, COLORS["grid_line"], rect, 1)

            if slot is None:
                continue

            state, data = slot

            if state == "cooking":
                # Chef indicator (small yellow circle)
                pygame.draw.circle(
                    surface, (255, 220, 130),
                    (rect.x + 12, rect.y + 12), 7)
                pygame.draw.circle(
                    surface, (180, 140, 60),
                    (rect.x + 12, rect.y + 12), 7, 1)

                # Food name (Korean)
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

                # Percentage text
                pct = self.font_sm.render(
                    f"{int(ratio * 100)}%", True, (200, 200, 200))
                surface.blit(pct, pct.get_rect(
                    centerx=rect.centerx, bottom=rect.bottom - 14))

            elif state == "ready":
                # "완성" label
                done = self.font_sm.render("완성", True, (255, 200, 80))
                surface.blit(done, (rect.x + 2, rect.y + 2))

                # 메뉴 ID (menu.json 의 "id": coffee, sandwich, ...)
                menu_id = data["menu_item"].get("id", "")

                # food 스프라이트가 있으면 아이콘을, 없으면 기존 텍스트 표시
                if menu_id and self.am.has_sprite("food", menu_id):
                    frame = self.am.get_frame("food", menu_id, 0)
                    surface.blit(frame, frame.get_rect(center=rect.center))
                else:
                    # Food name (Korean)
                    name = data["menu_item"]["name"]
                    nm = self.font_sm.render(name[:4], True, (255, 255, 100))
                    surface.blit(nm, nm.get_rect(center=rect.center))
                

                # Table ID
                tid = self.font_sm.render(
                    f"T{data['table_id']}", True, (180, 180, 180))
                surface.blit(tid, tid.get_rect(
                    centerx=rect.centerx, bottom=rect.bottom - 4))

    # ═══════════════════════════════════════════════
    #  Trash cans
    # ═══════════════════════════════════════════════
    def _draw_trash_cans(self, surface, shop, ox, oy):
        for pos in shop.trash_can_positions:
            rect = pygame.Rect(pos[0] * TILE_SIZE + ox,
                               pos[1] * TILE_SIZE + oy,
                               TILE_SIZE, TILE_SIZE)
            col = COLORS.get("trash_can", (120, 100, 80))
            pygame.draw.rect(surface, col, rect)
            pygame.draw.rect(surface, COLORS["grid_line"], rect, 1)
            lbl = self.font_sm.render("🗑", True, COLORS["text"])
            surface.blit(lbl, lbl.get_rect(center=rect.center))

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

            if cust.state == CustomerState.WALKING_TO_TABLE:
                icon = self.font_sm.render("🚶", True, (200, 200, 200))
            elif cust.state == CustomerState.WAITING_TO_ORDER:
                icon = self.font_sm.render("?!", True, (255, 255, 255))
            elif cust.state == CustomerState.ORDER_TAKEN:
                lbl = cust.menu_item["name"][:3]
                icon = self.font_sm.render(lbl, True, (200, 200, 100))
            elif cust.state == CustomerState.EATING:
                icon = self.font_sm.render("냠냠", True, (100, 255, 100))
            else:
                icon = self.font_sm.render("...", True, (150, 150, 150))
            surface.blit(icon, icon.get_rect(center=body.center))

            # Drink indicator (top-right)
            if cust.drink_item and not cust.drink_served:
                di = self.font_sm.render("음", True, (180, 100, 255))
                surface.blit(di, (cx + TILE_SIZE - 14, cy + 2))

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
            margin = TILE_SIZE // 5
            body = pygame.Rect(cx + margin, cy + margin,
                               TILE_SIZE - margin * 2, TILE_SIZE - margin * 2)
            if cust._happy:
                col = (100, 220, 100)  # green tint for happy
            else:
                col = COLORS.get("customer_angry", (220, 80, 50))
            pygame.draw.rect(surface, col, body)
            pygame.draw.rect(surface, (0, 0, 0), body, 1)
            icon_text = "😊" if cust._happy else "😡"
            icon = self.font_sm.render(icon_text, True, (255, 255, 255))
            surface.blit(icon, icon.get_rect(center=body.center))

    def _draw_waiting_queue(self, surface, shop, ox, oy):
        if not shop.waiting_queue:
            return
        ex = int(shop.entrance_x) + ox
        ey = int(shop.entrance_y) + oy
        # Draw each waiting customer in a row near the entrance
        for i, cust in enumerate(shop.waiting_queue):
            # Stack vertically below entrance, offset by index
            cx = ex + (i % 2) * (TILE_SIZE + 4)
            cy = ey + (i // 2) * (TILE_SIZE + 4)
            body = pygame.Rect(cx + 8, cy + 8, TILE_SIZE - 16, TILE_SIZE - 16)
            pygame.draw.rect(surface, cust.color, body)
            pygame.draw.rect(surface, (0, 0, 0), body, 1)
            icon = self.font_sm.render("...", True, (200, 200, 200))
            surface.blit(icon, icon.get_rect(center=body.center))
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
        # Queue count label
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
    #  Bar counters
    # ═══════════════════════════════════════════════
    def _draw_bar(self, surface, shop, ox, oy):
        bar = shop.bar
        positions = sorted(shop.bar_counter_positions)

        if not shop.bartender_hired:
            for pos in positions:
                rect = pygame.Rect(pos[0] * TILE_SIZE + ox,
                                   pos[1] * TILE_SIZE + oy,
                                   TILE_SIZE, TILE_SIZE)
                pygame.draw.rect(surface, (60, 40, 80), rect)
                pygame.draw.rect(surface, COLORS["grid_line"], rect, 1)
            return

        # Assign items to tiles: preparing first, then ready
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

            if slot and slot[0] == "ready":
                col = COLORS.get("bar_ready", (160, 100, 220))
            elif slot and slot[0] == "preparing":
                col = COLORS.get("bar", (100, 60, 140))
            else:
                col = COLORS.get("bar", (100, 60, 140))
            pygame.draw.rect(surface, col, rect)
            pygame.draw.rect(surface, COLORS["grid_line"], rect, 1)

            if slot is None:
                continue

            state, data = slot

            if state == "preparing":
                # Drink name
                name = data["drink_item"]["name"]
                nm = self.font_sm.render(name[:4], True, (255, 255, 255))
                surface.blit(nm, nm.get_rect(
                    centerx=rect.centerx, top=rect.y + 4))

                # Table ID
                tid = self.font_sm.render(
                    f"T{data['table_id']}", True, (180, 180, 180))
                surface.blit(tid, (rect.right - 22, rect.y + 4))

                # Progress bar
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

                name = data["drink_item"]["name"]
                nm = self.font_sm.render(name[:4], True, (220, 160, 255))
                surface.blit(nm, nm.get_rect(center=rect.center))

                tid = self.font_sm.render(
                    f"T{data['table_id']}", True, (180, 180, 180))
                surface.blit(tid, tid.get_rect(
                    centerx=rect.centerx, bottom=rect.bottom - 4))

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
            lbl = self.font_sm.render(f"직{emp.emp_id}", True, (255, 255, 255))
            surface.blit(lbl, lbl.get_rect(center=(int(ecx), int(ecy))))

            # Carry label (food name above employee)
            if emp.carrying:
                name = self._carry_name(emp.carrying)
                if name:
                    nm = self.font_sm.render(
                        name, True, (255, 255, 200))
                    surface.blit(nm, nm.get_rect(
                        centerx=int(ecx), bottom=int(ecy) - radius - 2))

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

    # ── carried food name labels on player ──
    def _draw_carry_labels(self, surface, shop, ox, oy):
        player = shop.player
        if not player.carrying:
            return
        px = player.x + ox
        py = player.y + oy
        for i, item in enumerate(player.carrying[:3]):
            name = self._carry_name(item)
            if not name:
                continue
            lbl = self.font_sm.render(name, True, (255, 255, 200))
            surface.blit(lbl, lbl.get_rect(
                centerx=int(px + TILE_SIZE // 2),
                bottom=int(py) - 10 - i * 14))

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
