"""Customer entity – spawns at entrance, walks to table, orders, waits.

State machine:
  WALKING_TO_TABLE → (arrives at table)   → WAITING_TO_ORDER
  WAITING_TO_ORDER → (player takes order) → ORDER_TAKEN
  ORDER_TAKEN      → (food served)        → EATING
  EATING           → (after eat_time)     → LEAVING_HAPPY
  Any waiting state → (patience expires)  → LEAVING_ANGRY

Each customer orders exactly one menu item.
Optional drink order (served separately for bonus income).
"""

import math
import random
from core.entity import Entity
from config.settings import COLORS, SATISFACTION_FAST_THRESHOLD, CUSTOMER_WALK_SPEED, TILE_SIZE, WAITING_PATIENCE


class CustomerState:
    WAITING_OUTSIDE = "waiting_outside"
    WALKING_TO_TABLE = "walking_to_table"
    WAITING_TO_ORDER = "waiting_to_order"
    ORDER_TAKEN = "order_taken"
    EATING = "eating"
    LEAVING_HAPPY = "leaving_happy"
    LEAVING_ANGRY = "leaving_angry"


EATING_TIME = 5.0


class Customer(Entity):

    def __init__(self, table_id: int, x: float, y: float,
                 customer_type: dict, menu_item: dict,
                 drink_item: dict | None = None,
                 patience_bonus: float = 0.0,
                 entrance_x: float | None = None,
                 entrance_y: float | None = None,
                 waiting_outside: bool = False):
        color_key = customer_type.get("color_key", "customer")
        # Start at entrance if given, otherwise at table directly
        start_x = entrance_x if entrance_x is not None else x
        start_y = entrance_y if entrance_y is not None else y
        super().__init__(start_x, start_y,
                         color=COLORS.get(color_key, COLORS["customer"]),
                         sprite_key="customer")
        self.table_id = table_id
        self.customer_type = customer_type
        self.menu_item: dict = menu_item
        self.drink_item: dict | None = drink_item

        # Walking target (table pixel position)
        self.target_x: float = float(x)
        self.target_y: float = float(y)
        self.walk_speed: float = CUSTOMER_WALK_SPEED

        # State initialization
        if waiting_outside:
            self.state = CustomerState.WAITING_OUTSIDE
        elif entrance_x is not None:
            self.state = CustomerState.WALKING_TO_TABLE
        else:
            self.state = CustomerState.WAITING_TO_ORDER

        self.patience = float(customer_type["patience"]) + patience_bonus
        self.max_patience = self.patience
        # Waiting outside uses separate, shorter patience
        self.waiting_patience: float = WAITING_PATIENCE
        self.max_waiting_patience: float = WAITING_PATIENCE
        self.eat_timer = EATING_TIME

        self.wealth_mult = float(customer_type["wealth_mult"])
        self.tip_range = customer_type["tip_range"]

        # Tracking
        self.food_served: bool = False
        self.drink_served: bool = False
        self.order_claimed: bool = False   # prevent double employee assignment

        self._base_color = self.color

    # ── patience helpers ─────────────────────────
    @property
    def patience_ratio(self) -> float:
        return max(0.0, self.patience / self.max_patience)

    @property
    def waiting_patience_ratio(self) -> float:
        return max(0.0, self.waiting_patience / self.max_waiting_patience)

    @property
    def is_done(self) -> bool:
        return self.state in (CustomerState.LEAVING_HAPPY,
                              CustomerState.LEAVING_ANGRY)

    # ── assign table to waiting customer ─────────
    def assign_table(self, table_id: int, table_x: float, table_y: float,
                     entrance_x: float, entrance_y: float):
        """Move a WAITING_OUTSIDE customer to a table."""
        self.table_id = table_id
        self.target_x = table_x
        self.target_y = table_y
        self.x = entrance_x
        self.y = entrance_y
        self.state = CustomerState.WALKING_TO_TABLE

    # ── update per tick ──────────────────────────
    def update(self, dt: float):
        if self.is_done:
            return

        # Waiting outside (separate patience)
        if self.state == CustomerState.WAITING_OUTSIDE:
            self.waiting_patience -= dt
            if self.waiting_patience <= 0:
                self.waiting_patience = 0
                self.state = CustomerState.LEAVING_ANGRY
            ratio = self.waiting_patience_ratio
            if ratio < 0.35:
                self.color = COLORS["customer_angry"]
            else:
                self.color = COLORS.get("customer_waiting", self._base_color)
            return

        # Walking to table (no patience loss during walk)
        if self.state == CustomerState.WALKING_TO_TABLE:
            dx = self.target_x - self.x
            dy = self.target_y - self.y
            dist = math.hypot(dx, dy)
            arrive_dist = TILE_SIZE * 0.3
            if dist <= arrive_dist:
                self.x = self.target_x
                self.y = self.target_y
                self.state = CustomerState.WAITING_TO_ORDER
            else:
                step = self.walk_speed * dt
                if step >= dist:
                    self.x = self.target_x
                    self.y = self.target_y
                    self.state = CustomerState.WAITING_TO_ORDER
                else:
                    self.x += dx / dist * step
                    self.y += dy / dist * step
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

    # ── food served by player ────────────────────
    def serve_food(self):
        """Serve the food item.  Starts eating only when all items served."""
        if self.state != CustomerState.ORDER_TAKEN:
            return False
        self.food_served = True
        self._check_all_served()
        return True

    # ── drink served ─────────────────────────────
    def serve_drink(self):
        if self.state not in (CustomerState.ORDER_TAKEN, CustomerState.EATING):
            return False
        self.drink_served = True
        self._check_all_served()
        return True

    def _check_all_served(self):
        """Transition to EATING when food + drink (if any) are both served."""
        if not self.food_served:
            return
        if self.drink_item and not self.drink_served:
            return
        if self.state != CustomerState.EATING:
            self.state = CustomerState.EATING
            self.eat_timer = EATING_TIME

    # ── backward compat alias ────────────────────
    def serve(self):
        self.serve_food()

    # ── payment calculation ──────────────────────
    def calc_payment(self, food_price_bonus: int = 0,
                     tip_bonus_pct: float = 0.0,
                     base_tip_bonus: int = 0) -> int:
        """Money earned from this customer."""
        base = self.menu_item["price"] + food_price_bonus
        total = int(base * self.wealth_mult)

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
