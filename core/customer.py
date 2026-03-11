"""Customer entity – sits at a table, places an order, waits to be served.

State machine:
  WAITING_TO_ORDER → (player takes order) → ORDER_TAKEN
  ORDER_TAKEN      → (food arrives)       → EATING
  EATING           → (after eat_time)     → LEAVING_HAPPY
  Any waiting state → (patience expires)  → LEAVING_ANGRY
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


EATING_TIME = 3.0   # seconds spent eating before paying


class Customer(Entity):

    def __init__(self, table_id: int, x: float, y: float,
                 customer_type: dict, menu_item: dict):
        color_key = customer_type.get("color_key", "customer")
        super().__init__(x, y,
                         color=COLORS.get(color_key, COLORS["customer"]),
                         sprite_key="customer")
        self.table_id = table_id
        self.customer_type = customer_type
        self.menu_item = menu_item          # what they want to order

        self.state = CustomerState.WAITING_TO_ORDER
        self.patience = float(customer_type["patience"])
        self.max_patience = self.patience
        self.eat_timer = EATING_TIME

        self.wealth_mult = float(customer_type["wealth_mult"])
        self.tip_range = customer_type["tip_range"]

        self._base_color = self.color

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

        # Patience ticks down while waiting (order or food)
        self.patience -= dt
        if self.patience <= 0:
            self.patience = 0
            self.state = CustomerState.LEAVING_ANGRY

        # Visual color shift toward angry
        ratio = self.patience_ratio
        if ratio < 0.35:
            self.color = COLORS["customer_angry"]
        else:
            self.color = self._base_color

    # ── order taken by player ────────────────────
    def take_order(self):
        if self.state == CustomerState.WAITING_TO_ORDER:
            self.state = CustomerState.ORDER_TAKEN

    # ── food served by player ────────────────────
    def serve(self):
        if self.state == CustomerState.ORDER_TAKEN:
            self.state = CustomerState.EATING
            self.eat_timer = EATING_TIME

    # ── payment calculation ──────────────────────
    def calc_payment(self) -> int:
        """Money earned from this customer (base price * wealth + tip)."""
        base = self.menu_item["price"]
        price = int(base * self.wealth_mult)
        tip_lo, tip_hi = self.tip_range
        satisfaction = self.patience_ratio
        tip = int(random.uniform(tip_lo, tip_hi) * satisfaction)
        return price + tip

    def calc_satisfaction(self) -> float:
        """Per-customer satisfaction score in [-1.0, 1.0].

        Positive when served, extra bonus for fast service.
        -1.0 if the customer left angry.
        """
        if self.state == CustomerState.LEAVING_ANGRY:
            return -1.0
        ratio = self.patience_ratio
        if ratio >= SATISFACTION_FAST_THRESHOLD:
            return 0.5 + 0.5 * ratio       # 0.8 – 1.0 range
        return 0.2 + 0.3 * ratio            # 0.2 – 0.5 range
