"""Station – static interactive objects on the map.

Station types:
  table           – where customers sit; player takes orders / serves food here
  kitchen_counter – player submits orders and picks up cooked food
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
    """The centralised kitchen. Manages a queue of cooking orders.

    Orders submitted by the player are cooked here in parallel
    (up to *capacity* at a time). Completed dishes wait in a ready queue
    for the player to pick up.
    """

    def __init__(self, capacity: int = 3):
        self.capacity = capacity
        # cooking: list of {"table_id", "menu_item", "timer"}
        self.cooking: list[dict] = []
        # ready: list of {"table_id", "menu_item"}
        self.ready: list[dict] = []

    @property
    def can_accept(self) -> bool:
        return len(self.cooking) < self.capacity

    @property
    def has_ready(self) -> bool:
        return len(self.ready) > 0

    @property
    def num_cooking(self) -> int:
        return len(self.cooking)

    def submit_order(self, table_id: int, menu_item: dict) -> bool:
        """Submit an order. Returns True on success."""
        if not self.can_accept:
            return False
        self.cooking.append({
            "table_id": table_id,
            "menu_item": menu_item,
            "timer": float(menu_item["cook_time"]),
        })
        return True

    def pick_up(self) -> dict | None:
        """Pick up the first ready dish. Returns dict or None."""
        if self.ready:
            return self.ready.pop(0)
        return None

    def update(self, dt: float, cook_speed_mult: float = 1.0):
        """Advance all cooking timers by *dt* seconds."""
        still_cooking = []
        for order in self.cooking:
            order["timer"] -= dt * cook_speed_mult
            if order["timer"] <= 0:
                self.ready.append({
                    "table_id": order["table_id"],
                    "menu_item": order["menu_item"],
                })
            else:
                still_cooking.append(order)
        self.cooking = still_cooking

    def reset(self):
        self.cooking.clear()
        self.ready.clear()
