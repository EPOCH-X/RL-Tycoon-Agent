"""Shop – complete state & logic for a single restaurant instance.

Game flow per customer:
  1. Customer spawns at an empty table  →  WAITING_TO_ORDER
  2. Player walks to table & interacts  →  takes order  (player carries "order")
  3. Player walks to kitchen & interacts → submits order (kitchen starts cooking)
  4. Kitchen finishes cooking            → dish moves to ready queue
  5. Player walks to kitchen & interacts → picks up food (player carries "food")
  6. Player walks to table & interacts   → serves food   (customer eats & pays)

Movement is pixel-based (continuous, not grid-snapping).
Interaction uses distance + facing direction.
"""

import math
import random
from collections import deque

from config.settings import (
    TILE_SIZE, STEP_INTERVAL, CUSTOMER_SPAWN_INTERVAL,
    KITCHEN_CAPACITY, DEFAULT_TARGET_MONEY, DEFAULT_DAY_LIMIT,
    DAY_LENGTH, LOST_CUSTOMER_PENALTY, SATISFACTION_HISTORY_LEN,
    PLAYER_SPEED, PLAYER_RADIUS, INTERACT_RANGE,
    ACTION_UP, ACTION_DOWN, ACTION_LEFT, ACTION_RIGHT,
    ACTION_INTERACT, ACTION_NONE, ACTION_BUY_UPGRADE,
    load_json_config,
)
from core.player import Player
from core.customer import Customer, CustomerState
from core.station import Table, Kitchen


class Shop:
    """One self-contained restaurant: map + player + tables + kitchen + money."""

    def __init__(self, *, map_data=None, menu_data=None,
                 customers_data=None, upgrades_data=None,
                 target_money=None, day_limit=None):
        # ── load config ──────────────────────────
        if map_data is None:
            map_data = load_json_config("map_default.json")
        if menu_data is None:
            menu_data = load_json_config("menu.json")
        if customers_data is None:
            customers_data = load_json_config("customers.json")
        if upgrades_data is None:
            upgrades_data = load_json_config("upgrades.json")

        self.map_data = map_data
        self.grid_width: int = map_data["width"]
        self.grid_height: int = map_data["height"]
        self._original_layout: list[list[int]] = [row[:] for row in map_data["layout"]]
        self.layout: list[list[int]] = [row[:] for row in map_data["layout"]]

        self.menu: list[dict] = menu_data
        self.customer_types: list[dict] = customers_data
        self.upgrades_data: list[dict] = upgrades_data

        # ── tables (active) ──────────────────────
        self.tables: list[Table] = []
        for t in map_data.get("tables", []):
            self.tables.append(Table(t["id"], t["grid_x"], t["grid_y"]))
        self._table_positions: dict[tuple[int, int], Table] = {
            (t.grid_x, t.grid_y): t for t in self.tables
        }

        # ── purchasable table slots ──────────────
        self._purchasable_tables: list[dict] = list(
            map_data.get("purchasable_tables", []))
        self.max_tables: int = len(self.tables) + len(self._purchasable_tables)

        # ── kitchen counters ─────────────────────
        self.kitchen_counter_positions: set[tuple[int, int]] = set()
        for kc in map_data.get("kitchen_counters", []):
            self.kitchen_counter_positions.add((kc["grid_x"], kc["grid_y"]))

        self.kitchen = Kitchen(capacity=KITCHEN_CAPACITY)

        # ── player (pixel coords) ────────────────
        ps = map_data["player_start"]
        self.player = Player(
            ps["grid_x"] * TILE_SIZE, ps["grid_y"] * TILE_SIZE)

        # ── game state ───────────────────────────
        self.money: int = 0
        self.time_elapsed: float = 0.0
        self.target_money: int = target_money or DEFAULT_TARGET_MONEY
        self.day_limit: int = day_limit or DEFAULT_DAY_LIMIT
        self.done: bool = False
        self.won: bool = False

        # ── satisfaction ─────────────────────────
        self.satisfaction_history: deque[float] = deque(
            maxlen=SATISFACTION_HISTORY_LEN)
        self.shop_rating: float = 0.5

        # ── customer spawn ───────────────────────
        self._base_weights = [ct["spawn_weight"] for ct in self.customer_types]
        self.customer_spawn_timer: float = 2.0
        self.spawn_rate_mult: float = 1.0
        self.cook_speed_mult: float = 1.0
        self.wealthy_bonus: float = 0.0

        # ── upgrade state ────────────────────────
        self.upgrade_levels: dict[str, int] = {
            u["id"]: 0 for u in self.upgrades_data
        }
        self.upgrade_mode: bool = False

        # ── UI message ───────────────────────────
        self.message: str = ""
        self.message_timer: float = 0.0

        # ── stats ────────────────────────────────
        self.customers_served: int = 0
        self.customers_lost: int = 0

    # ═══════════════════════════════════════════════
    #  Derived properties
    # ═══════════════════════════════════════════════
    @property
    def current_day(self) -> int:
        return min(int(self.time_elapsed / DAY_LENGTH) + 1, self.day_limit)

    @property
    def total_time_limit(self) -> float:
        return self.day_limit * DAY_LENGTH

    @property
    def time_remaining(self) -> float:
        return max(0.0, self.total_time_limit - self.time_elapsed)

    @property
    def num_customers(self) -> int:
        return sum(1 for t in self.tables if t.is_occupied)

    @property
    def max_seated(self) -> int:
        return len(self.tables)

    # ═══════════════════════════════════════════════
    #  Reset
    # ═══════════════════════════════════════════════
    def reset(self):
        self.layout = [row[:] for row in self._original_layout]

        self.tables = []
        for t in self.map_data.get("tables", []):
            self.tables.append(Table(t["id"], t["grid_x"], t["grid_y"]))
        self._table_positions = {
            (t.grid_x, t.grid_y): t for t in self.tables
        }
        self._purchasable_tables = list(
            self.map_data.get("purchasable_tables", []))

        self.kitchen = Kitchen(capacity=KITCHEN_CAPACITY)

        ps = self.map_data["player_start"]
        self.player = Player(
            ps["grid_x"] * TILE_SIZE, ps["grid_y"] * TILE_SIZE)

        self.money = 0
        self.time_elapsed = 0.0
        self.done = False
        self.won = False
        self.message = ""
        self.message_timer = 0.0
        self.customers_served = 0
        self.customers_lost = 0
        self.satisfaction_history.clear()
        self.shop_rating = 0.5
        self.customer_spawn_timer = 2.0
        self.upgrade_mode = False

        for uid in self.upgrade_levels:
            self.upgrade_levels[uid] = 0
        self.cook_speed_mult = 1.0
        self.spawn_rate_mult = 1.0
        self.wealthy_bonus = 0.0

    # ═══════════════════════════════════════════════
    #  Step — RL-compatible (includes movement)
    # ═══════════════════════════════════════════════
    def step(self, action: int) -> float:
        """Full step for RL: movement + game logic.  Returns reward."""
        if self.done:
            return 0.0

        dt = STEP_INTERVAL

        if action in (ACTION_UP, ACTION_DOWN, ACTION_LEFT, ACTION_RIGHT):
            self._move_player_action(action, dt)

        return self._step_inner(action, dt)

    # ═══════════════════════════════════════════════
    #  Step Logic — for human mode (movement handled elsewhere)
    # ═══════════════════════════════════════════════
    def step_logic(self, action: int) -> float:
        """Game-logic-only step.  Movement is handled by tick()."""
        if self.done:
            return 0.0
        return self._step_inner(action, STEP_INTERVAL)

    def _step_inner(self, action: int, dt: float) -> float:
        reward = 0.0

        # 1) Interaction / upgrade
        if action == ACTION_INTERACT:
            reward += self._interact()
        elif action == ACTION_BUY_UPGRADE:
            reward += self._auto_buy_upgrade()

        # 2) Kitchen tick
        self.kitchen.update(dt, self.cook_speed_mult)

        # 3) Customer tick
        for table in self.tables:
            if table.customer is None:
                continue
            cust = table.customer
            cust.update(dt)

            if cust.state == CustomerState.LEAVING_HAPPY:
                payment = cust.calc_payment()
                self.money += payment
                self._record_satisfaction(cust.calc_satisfaction())
                self.customers_served += 1
                reward += float(payment)
                table.customer = None

            elif cust.state == CustomerState.LEAVING_ANGRY:
                self._record_satisfaction(-1.0)
                self.customers_lost += 1
                reward -= LOST_CUSTOMER_PENALTY
                table.customer = None

        # 4) Spawn customers
        self.customer_spawn_timer -= dt
        if self.customer_spawn_timer <= 0:
            self._try_spawn_customer()
            interval = CUSTOMER_SPAWN_INTERVAL / max(0.5, self.spawn_rate_mult)
            self.customer_spawn_timer = max(1.5, interval)

        # 5) Time
        self.time_elapsed += dt

        # 6) Message decay
        if self.message_timer > 0:
            self.message_timer -= dt
            if self.message_timer <= 0:
                self.message = ""

        # 7) Termination
        if self.money >= self.target_money:
            self.done = True
            self.won = True
            reward += 200.0
        elif self.time_elapsed >= self.total_time_limit:
            self.done = True

        return reward

    # ═══════════════════════════════════════════════
    #  Pixel Movement
    # ═══════════════════════════════════════════════
    _ACTION_DIR = {
        ACTION_UP:    (0, -1, Player.FACING_UP),
        ACTION_DOWN:  (0,  1, Player.FACING_DOWN),
        ACTION_LEFT:  (-1, 0, Player.FACING_LEFT),
        ACTION_RIGHT: (1,  0, Player.FACING_RIGHT),
    }

    def _move_player_action(self, action: int, dt: float):
        """RL action → pixel movement with collision."""
        dx, dy, facing = self._ACTION_DIR[action]
        self.player.facing = facing
        self._apply_movement(dx, dy, dt)

    def move_player_continuous(self, dx: float, dy: float, dt: float):
        """Human mode: continuous pixel movement from held keys."""
        if abs(dx) < 0.01 and abs(dy) < 0.01:
            return
        # Normalize diagonal
        length = math.hypot(dx, dy)
        dx /= length
        dy /= length
        # Facing from dominant axis
        if abs(dx) > abs(dy):
            self.player.facing = (Player.FACING_RIGHT if dx > 0
                                  else Player.FACING_LEFT)
        else:
            self.player.facing = (Player.FACING_DOWN if dy > 0
                                  else Player.FACING_UP)
        self._apply_movement(dx, dy, dt)

    def _apply_movement(self, dx: float, dy: float, dt: float):
        move = self.player.speed * dt
        new_x = self.player.x + dx * move
        new_y = self.player.y + dy * move
        # Try full move, then axis-slide
        if self._can_move_to(new_x, new_y):
            self.player.x = new_x
            self.player.y = new_y
        elif self._can_move_to(new_x, self.player.y):
            self.player.x = new_x
        elif self._can_move_to(self.player.x, new_y):
            self.player.y = new_y

    def _can_move_to(self, x: float, y: float) -> bool:
        """Check if player hitbox at (x,y) doesn't overlap solid tiles."""
        cx = x + TILE_SIZE / 2
        cy = y + TILE_SIZE / 2
        r = PLAYER_RADIUS
        for corner_x, corner_y in [(cx - r, cy - r), (cx + r, cy - r),
                                    (cx - r, cy + r), (cx + r, cy + r)]:
            gx = int(corner_x) // TILE_SIZE
            gy = int(corner_y) // TILE_SIZE
            if gx < 0 or gx >= self.grid_width:
                return False
            if gy < 0 or gy >= self.grid_height:
                return False
            if self.layout[gy][gx] != 0:
                return False
        return True

    # ═══════════════════════════════════════════════
    #  Distance-based Interaction
    # ═══════════════════════════════════════════════
    def _interact(self) -> float:
        px = self.player.center_x
        py = self.player.center_y
        fdx, fdy = Player.DIRECTION_VEC[self.player.facing]

        best_type = None
        best_obj = None
        best_dist = INTERACT_RANGE + 1

        # Check tables
        for table in self.tables:
            dist = math.hypot(table.center_x - px, table.center_y - py)
            if dist > INTERACT_RANGE:
                continue
            if dist > 1e-3:
                dirx = (table.center_x - px) / dist
                diry = (table.center_y - py) / dist
                if dirx * fdx + diry * fdy < 0.1:
                    continue
            if dist < best_dist:
                best_type = "table"
                best_obj = table
                best_dist = dist

        # Check kitchen counters
        for kpos in self.kitchen_counter_positions:
            kcx = kpos[0] * TILE_SIZE + TILE_SIZE / 2
            kcy = kpos[1] * TILE_SIZE + TILE_SIZE / 2
            dist = math.hypot(kcx - px, kcy - py)
            if dist > INTERACT_RANGE:
                continue
            if dist > 1e-3:
                dirx = (kcx - px) / dist
                diry = (kcy - py) / dist
                if dirx * fdx + diry * fdy < 0.1:
                    continue
            if dist < best_dist:
                best_type = "kitchen"
                best_obj = kpos
                best_dist = dist

        if best_type is None:
            return 0.0
        if best_type == "table":
            return self._interact_table(best_obj)
        return self._interact_kitchen()

    def _interact_table(self, table: Table) -> float:
        cust = table.customer
        if cust is None:
            self._msg("Empty table")
            return 0.0

        # Take order
        if (self.player.is_idle
                and cust.state == CustomerState.WAITING_TO_ORDER):
            self.player.pick_up_order(table.table_id, cust.menu_item)
            cust.take_order()
            self._msg(f"Order: {cust.menu_item['name']}")
            return 2.0

        # Serve food
        if (self.player.has_food
                and self.player.carrying["table_id"] == table.table_id
                and cust.state == CustomerState.ORDER_TAKEN):
            self.player.drop()
            cust.serve()
            self._msg(f"Served: {cust.menu_item['name']}!")
            return 5.0

        if self.player.has_food:
            self._msg("Wrong table!")
            return -2.0
        elif self.player.has_order:
            self._msg("Deliver order to kitchen first!")
        else:
            self._msg("Already ordered")
        return 0.0

    def _interact_kitchen(self) -> float:
        # Submit order
        if self.player.has_order:
            order = self.player.carrying
            ok = self.kitchen.submit_order(
                order["table_id"], order["menu_item"])
            if ok:
                self.player.drop()
                self._msg(f"Cooking: {order['menu_item']['name']}")
                return 1.0
            else:
                self._msg("Kitchen full!")
            return 0.0

        # Pick up food
        if self.player.is_idle and self.kitchen.has_ready:
            dish = self.kitchen.pick_up()
            self.player.pick_up_food(dish["table_id"], dish["menu_item"])
            self._msg(f"Pick up: {dish['menu_item']['name']}")
            return 1.0

        if self.player.has_food:
            self._msg("Serve the food first!")
        elif not self.kitchen.has_ready:
            self._msg("Nothing ready yet")
        return 0.0

    # ═══════════════════════════════════════════════
    #  Customer spawning
    # ═══════════════════════════════════════════════
    def _try_spawn_customer(self):
        if self.num_customers >= self.max_seated:
            return
        empty_tables = [t for t in self.tables if not t.is_occupied]
        if not empty_tables:
            return

        table = random.choice(empty_tables)
        ctype = self._pick_customer_type()
        menu_item = random.choice(self.menu)
        table.customer = Customer(
            table.table_id,
            table.grid_x * TILE_SIZE,
            table.grid_y * TILE_SIZE,
            ctype, menu_item)

    def _pick_customer_type(self) -> dict:
        weights = list(self._base_weights)
        if self.shop_rating > 0.6:
            bonus = self.shop_rating * (1.0 + self.wealthy_bonus)
            for i, ct in enumerate(self.customer_types):
                if ct["wealth_mult"] >= 1.5:
                    weights[i] += bonus
        return random.choices(self.customer_types, weights=weights, k=1)[0]

    # ═══════════════════════════════════════════════
    #  Satisfaction
    # ═══════════════════════════════════════════════
    def _record_satisfaction(self, value: float):
        self.satisfaction_history.append(value)
        if self.satisfaction_history:
            self.shop_rating = max(0.0, min(1.0,
                sum(self.satisfaction_history) / len(self.satisfaction_history)))

    # ═══════════════════════════════════════════════
    #  Upgrade System
    # ═══════════════════════════════════════════════
    def get_upgrade_info(self) -> list[dict]:
        """Return list of upgrade status dicts for UI rendering."""
        info = []
        for upg in self.upgrades_data:
            uid = upg["id"]
            level = self.upgrade_levels[uid]
            maxed = level >= upg["max_level"]
            cost = (0 if maxed
                    else int(upg["base_cost"] * (upg["cost_multiplier"] ** level)))
            info.append({
                "data": upg,
                "level": level,
                "cost": cost,
                "maxed": maxed,
                "can_afford": not maxed and self.money >= cost,
            })
        return info

    def buy_upgrade(self, upgrade_id: str) -> bool:
        """Buy an upgrade if affordable. Returns True on success."""
        upg = None
        for u in self.upgrades_data:
            if u["id"] == upgrade_id:
                upg = u
                break
        if upg is None:
            return False

        level = self.upgrade_levels[upgrade_id]
        if level >= upg["max_level"]:
            self._msg("Max level!")
            return False

        cost = int(upg["base_cost"] * (upg["cost_multiplier"] ** level))
        if self.money < cost:
            self._msg(f"Need ${cost}!")
            return False

        self.money -= cost
        self.upgrade_levels[upgrade_id] = level + 1
        self._apply_upgrade(upg)
        self._msg(f"{upg['name']} Lv.{level + 1}!")
        return True

    def buy_upgrade_by_index(self, idx: int) -> bool:
        """Buy upgrade by list index (for UI number keys)."""
        if 0 <= idx < len(self.upgrades_data):
            return self.buy_upgrade(self.upgrades_data[idx]["id"])
        return False

    def _apply_upgrade(self, upg: dict):
        etype = upg["effect_type"]
        val = upg["effect_value"]

        if etype == "player_speed":
            self.player.speed += val * PLAYER_SPEED
        elif etype == "cook_speed":
            self.cook_speed_mult += val
        elif etype == "kitchen_capacity":
            self.kitchen.capacity += int(val)
        elif etype == "wealthy_bonus":
            self.wealthy_bonus += val
        elif etype == "buy_table":
            self._activate_next_table()

    def _activate_next_table(self):
        if not self._purchasable_tables:
            return
        tdata = self._purchasable_tables.pop(0)
        table = Table(tdata["id"], tdata["grid_x"], tdata["grid_y"])
        self.tables.append(table)
        self._table_positions[(table.grid_x, table.grid_y)] = table
        self.layout[table.grid_y][table.grid_x] = 2

    def _auto_buy_upgrade(self) -> float:
        """RL action: auto-buy the cheapest affordable upgrade."""
        best_id = None
        best_cost = float('inf')
        for upg in self.upgrades_data:
            uid = upg["id"]
            level = self.upgrade_levels[uid]
            if level >= upg["max_level"]:
                continue
            cost = int(upg["base_cost"] * (upg["cost_multiplier"] ** level))
            if cost <= self.money and cost < best_cost:
                best_id = uid
                best_cost = cost
        if best_id:
            self.buy_upgrade(best_id)
            return 3.0
        return -0.1

    # ═══════════════════════════════════════════════
    #  Helpers
    # ═══════════════════════════════════════════════
    def _msg(self, text: str, duration: float = 1.5):
        self.message = text
        self.message_timer = duration
