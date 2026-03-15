"""Station – static interactive objects on the map.

Station types:
  table           – where customers sit; player takes orders / serves food here
  kitchen_counter – player submits orders and picks up cooked food
  bar_counter     – bartender prepares drinks; player picks up and serves
"""

from config.settings import TILE_SIZE


class Table:
    """A dining table. Customers are seated here."""

    def __init__(self, table_id: int, grid_x: int, grid_y: int):
        self.table_id = table_id
        self.grid_x = grid_x
        self.grid_y = grid_y
        self.customer = None        # reference to Customer or None

    @property
    def is_occupied(self) -> bool:
        return self.customer is not None

    @property
    def pixel_x(self) -> float:
        return self.grid_x * TILE_SIZE

    @property
    def pixel_y(self) -> float:
        return self.grid_y * TILE_SIZE

    @property
    def center_x(self) -> float:
        return self.pixel_x + TILE_SIZE / 2

    @property
    def center_y(self) -> float:
        return self.pixel_y + TILE_SIZE / 2


class Kitchen:
    """The centralised kitchen.  Manages a queue of cooking orders.

    Two separate capacities:
      - *cooking_capacity*  = number of chefs (concurrent cooking slots)
      - *storage_capacity*  = number of kitchen tiles (ready-dish storage)

    When a dish finishes cooking but storage is full, the chef stays
    occupied (dish remains in *cooking* with timer ≤ 0) until a storage
    slot opens.  Delivery dishes bypass storage entirely.
    """

    def __init__(self, cooking_capacity: int = 1,
                 storage_capacity: int = 3):
        self.cooking_capacity = cooking_capacity
        self.storage_capacity = storage_capacity
        self.cooking: list[dict] = []
        self.ready: list[dict] = []
        self.delivery_ready: list[dict] = []

    @property
    def can_accept(self) -> bool:
        return len(self.cooking) < self.cooking_capacity

    @property
    def has_ready(self) -> bool:
        return len(self.ready) > 0

    @property
    def num_cooking(self) -> int:
        return len(self.cooking)

    @property
    def has_storage_space(self) -> bool:
        return len(self.ready) < self.storage_capacity

    def submit_order(self, table_id: int, menu_item: dict,
                     *, delivery: bool = False,
                     cook_time_reduction: float = 0.0,
                     cook_time_override: float | None = None) -> bool:
        """Submit an order.  Returns True on success."""
        if not self.can_accept:
            return False
        if cook_time_override is not None:
            timer = max(1.0, cook_time_override)
        else:
            timer = max(1.0, float(menu_item["cook_time"]) - cook_time_reduction)
        self.cooking.append({
            "table_id": table_id,
            "menu_item": menu_item,
            "timer": timer,
            "delivery": delivery,
        })
        return True

    def pick_up(self) -> dict | None:
        """Pick up the first ready dish.  Returns dict or None."""
        if self.ready:
            return self.ready.pop(0)
        return None

    def update(self, dt: float, cook_speed_mult: float = 1.0):
        """Advance all cooking timers by *dt* seconds.

        Finished dishes move to *ready* only if storage has room.
        Otherwise the chef stays occupied (dish stays in *cooking*).
        """
        still_cooking = []
        for order in self.cooking:
            order["timer"] -= dt * cook_speed_mult
            if order["timer"] <= 0:
                if order.get("delivery"):
                    self.delivery_ready.append({
                        "table_id": order["table_id"],
                        "menu_item": order["menu_item"],
                    })
                elif len(self.ready) < self.storage_capacity:
                    self.ready.append({
                        "table_id": order["table_id"],
                        "menu_item": order["menu_item"],
                    })
                else:
                    # Storage full — chef blocked, keep in cooking
                    still_cooking.append(order)
            else:
                still_cooking.append(order)
        self.cooking = still_cooking

    def remove_for_table(self, table_id: int) -> int:
        """Remove all cooking/ready items for *table_id*.

        Returns the number of items removed (for reward/event tracking).
        Called when a customer leaves before their food is served.
        """
        before = len(self.cooking) + len(self.ready)
        self.cooking = [o for o in self.cooking
                        if o["table_id"] != table_id]
        self.ready = [o for o in self.ready
                      if o["table_id"] != table_id]
        after = len(self.cooking) + len(self.ready)
        return before - after

    def reset(self):
        self.cooking.clear()
        self.ready.clear()
        self.delivery_ready.clear()


class BarStation:
    """Bar counter – prepares drinks.  Operated by the bartender (auto-prep).

    Drinks are queued here when a customer orders a drink.  The bartender
    prepares them automatically.  Ready drinks are picked up by the
    player or employee and served to the table.
    """

    def __init__(self, capacity: int = 2):
        self.capacity = capacity
        self.preparing: list[dict] = []
        self.ready: list[dict] = []

    @property
    def can_accept(self) -> bool:
        return len(self.preparing) < self.capacity

    @property
    def has_ready(self) -> bool:
        return len(self.ready) > 0

    @property
    def num_preparing(self) -> int:
        return len(self.preparing)

    def submit_drink(self, table_id: int, drink_item: dict) -> bool:
        if not self.can_accept:
            return False
        self.preparing.append({
            "table_id": table_id,
            "drink_item": drink_item,
            "timer": float(drink_item["prep_time"]),
        })
        return True

    def pick_up(self) -> dict | None:
        if self.ready:
            return self.ready.pop(0)
        return None

    def update(self, dt: float):
        still_preparing = []
        for order in self.preparing:
            order["timer"] -= dt
            if order["timer"] <= 0:
                self.ready.append({
                    "table_id": order["table_id"],
                    "drink_item": order["drink_item"],
                })
            else:
                still_preparing.append(order)
        self.preparing = still_preparing

    def remove_for_table(self, table_id: int) -> int:
        """Remove all preparing/ready drinks for *table_id*.

        Returns the number of items removed.
        """
        before = len(self.preparing) + len(self.ready)
        self.preparing = [o for o in self.preparing
                          if o["table_id"] != table_id]
        self.ready = [o for o in self.ready
                      if o["table_id"] != table_id]
        after = len(self.preparing) + len(self.ready)
        return before - after

    def reset(self):
        self.preparing.clear()
        self.ready.clear()
