"""Customer entity – sits at a table, places an order, waits to be served.

State machine:
  WAITING_TO_ORDER → (player takes order) → ORDER_TAKEN
  ORDER_TAKEN      → (all food served)    → EATING
  EATING           → (after eat_time)     → LEAVING_HAPPY
  Any waiting state → (patience expires)  → LEAVING_ANGRY

Supports:
  - Family groups (multiple food items in one order)
  - Optional drink order (served separately for bonus income)
"""

import random
from core.entity import Entity
from config.settings import COLORS, SATISFACTION_FAST_THRESHOLD


class CustomerState:
    WAITING_TO_ORDER = "waiting_to_order"
    ORDER_TAKEN = "order_taken"
    EATING = "eating"
    LEAVING_HAPPY = "leaving_happy"
    LEAVING_ANGRY = "leaving_angry"


EATING_TIME = 3.0


class Customer(Entity):

    def __init__(self, table_id: int, x: float, y: float,
                 customer_type: dict, menu_items: list[dict],
                 drink_item: dict | None = None,
                 patience_bonus: float = 0.0):
        color_key = customer_type.get("color_key", "customer")
        super().__init__(x, y,
                         color=COLORS.get(color_key, COLORS["customer"]),
                         sprite_key="customer")
        self.table_id = table_id
        self.customer_type = customer_type
        self.menu_items: list[dict] = menu_items
        self.drink_item: dict | None = drink_item

        self.state = CustomerState.WAITING_TO_ORDER
        self.patience = float(customer_type["patience"]) + patience_bonus
        self.max_patience = self.patience
        self.eat_timer = EATING_TIME

        self.wealth_mult = float(customer_type["wealth_mult"])
        self.tip_range = customer_type["tip_range"]
        self.group_size: int = len(menu_items)

        # Tracking
        self.food_served_count: int = 0
        self.drink_served: bool = False
        self.order_claimed: bool = False   # prevent double employee assignment

        self._base_color = self.color

    # ── backward compat ──────────────────────────
    @property
    def menu_item(self) -> dict | None:
        """Primary menu item (first in list)."""
        return self.menu_items[0] if self.menu_items else None

    @property
    def all_food_served(self) -> bool:
        return self.food_served_count >= self.group_size

    # ── patience helpers ─────────────────────────
    @property
    def patience_ratio(self) -> float:
        return max(0.0, self.patience / self.max_patience)

    @property
    def is_done(self) -> bool:
        return self.state in (CustomerState.LEAVING_HAPPY,
                              CustomerState.LEAVING_ANGRY)

    # ── update per tick ──────────────────────────
    def update(self, dt: float):
        if self.is_done:
            return

        if self.state == CustomerState.EATING:
            self.eat_timer -= dt
            if self.eat_timer <= 0:
                self.state = CustomerState.LEAVING_HAPPY
            return

        self.patience -= dt
        if self.patience <= 0:
            self.patience = 0
            self.state = CustomerState.LEAVING_ANGRY

        ratio = self.patience_ratio
        if ratio < 0.35:
            self.color = COLORS["customer_angry"]
        else:
            self.color = self._base_color

    # ── order taken by player ────────────────────
    def take_order(self):
        if self.state == CustomerState.WAITING_TO_ORDER:
            self.state = CustomerState.ORDER_TAKEN

    # ── food served by player (one dish at a time) ─
    def serve_food(self):
        """Serve one food item.  When all served, start eating."""
        if self.state != CustomerState.ORDER_TAKEN:
            return False
        self.food_served_count += 1
        if self.all_food_served:
            self.state = CustomerState.EATING
            self.eat_timer = EATING_TIME
        return True

    # ── drink served ─────────────────────────────
    def serve_drink(self):
        self.drink_served = True

    # ── backward compat alias ────────────────────
    def serve(self):
        self.serve_food()

    # ── payment calculation ──────────────────────
    def calc_payment(self, food_price_bonus: int = 0,
                     tip_bonus_pct: float = 0.0,
                     base_tip_bonus: int = 0) -> int:
        """Money earned from this customer."""
        total = 0
        for item in self.menu_items:
            base = item["price"] + food_price_bonus
            total += int(base * self.wealth_mult)

        satisfaction = self.patience_ratio
        tip_lo, tip_hi = self.tip_range
        raw_tip = random.uniform(tip_lo, tip_hi) * satisfaction
        tip = int((raw_tip + base_tip_bonus) * (1.0 + tip_bonus_pct))

        # Drink income
        drink_income = 0
        if self.drink_served and self.drink_item:
            drink_income = int(self.drink_item["price"] * self.wealth_mult)

        return total + max(0, tip) + drink_income

    def calc_satisfaction(self) -> float:
        if self.state == CustomerState.LEAVING_ANGRY:
            return -1.0
        ratio = self.patience_ratio
        if ratio >= SATISFACTION_FAST_THRESHOLD:
            return 0.5 + 0.5 * ratio
        return 0.2 + 0.3 * ratio
