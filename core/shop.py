"""Shop – complete state & logic for a single restaurant instance.

Systems:
  - Core loop: customer spawn → take order → cook → serve → payment
  - Chef system: hire chefs (1 chef = 1 dish at a time), limited by kitchen tiles
  - Tip system: satisfaction-based, customer-type-based, trait-based
  - Food unlock: net profit threshold + cost to unlock new menu items
  - Beverage system: separate bar station for drink orders
  - Employee system: auto-waiter AI (take order, submit, pickup, serve)
  - Delivery system: passive income via delivery orders
  - Trait system: periodic offers of permanent bonuses
  - Customer diversity: tourist, critic, VIP, etc.
  - Scoring: net profit tracking
"""

import math
import random
from collections import deque

from config.settings import (
    TILE_SIZE, STEP_INTERVAL, CUSTOMER_SPAWN_INTERVAL,
    KITCHEN_CAPACITY, DEFAULT_TARGET_MONEY, DEFAULT_DAY_LIMIT,
    DAY_LENGTH, SATISFACTION_HISTORY_LEN,
    PLAYER_SPEED, PLAYER_RADIUS, INTERACT_RANGE,
    EMPLOYEE_SPEED, EMPLOYEE_ACTION_DELAY,
    ACTION_UP, ACTION_DOWN, ACTION_LEFT, ACTION_RIGHT,
    ACTION_INTERACT, ACTION_NONE, ACTION_BUY_UPGRADE,
    load_json_config,
)
from core.player import Player
from core.customer import Customer, CustomerState
from core.station import Table, Kitchen, BarStation
from core.employee import Employee


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

        # ── beverage / delivery / trait configs ───
        try:
            self.beverage_config: dict = load_json_config("beverages.json")
        except Exception:
            self.beverage_config = {"unlock_profit": 999999, "items": []}
        try:
            self.delivery_config: dict = load_json_config("delivery.json")
        except Exception:
            self.delivery_config = {"unlock_profit": 999999, "order_interval": 10,
                                     "delivery_time": 12, "price_multiplier": 0.85,
                                     "tip_range": [3, 12]}
        try:
            self.traits_config: dict = load_json_config("traits.json")
        except Exception:
            self.traits_config = {"offer_interval_days": 5, "choices_per_offer": 3, "traits": []}

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

        # ── chef system ──────────────────────────
        self.num_chefs: int = KITCHEN_CAPACITY          # starts with 1 chef
        self.max_chefs: int = len(self.kitchen_counter_positions)  # limited by tiles

        # ── bar counters ─────────────────────────
        self.bar_counter_positions: set[tuple[int, int]] = set()
        for bc in map_data.get("bar_counters", []):
            self.bar_counter_positions.add((bc["grid_x"], bc["grid_y"]))
        self.bar = BarStation(capacity=2)

        # ── trash cans ───────────────────────────
        self.trash_can_positions: set[tuple[int, int]] = set()
        for tc in map_data.get("trash_cans", []):
            self.trash_can_positions.add((tc["grid_x"], tc["grid_y"]))

        # ── entrance (customer spawn point) ──────
        entrance = map_data.get("entrance", map_data["player_start"])
        self.entrance_x: float = entrance["grid_x"] * TILE_SIZE
        self.entrance_y: float = entrance["grid_y"] * TILE_SIZE

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

        # ── economy tracking ─────────────────────
        self.total_earned: int = 0
        self.total_spent: int = 0

        # ── satisfaction ─────────────────────────
        self.satisfaction_history: deque[float] = deque(
            maxlen=SATISFACTION_HISTORY_LEN)
        self.shop_rating: float = 0.5

        # ── customer spawn ───────────────────────
        self._base_weights = [ct["spawn_weight"] for ct in self.customer_types]
        self.customer_spawn_timer: float = CUSTOMER_SPAWN_INTERVAL
        self.spawn_rate_mult: float = 1.0
        self.cook_speed_mult: float = 1.0
        self.wealthy_bonus: float = 0.0

        # ── food unlock ──────────────────────────
        self.unlocked_food: set[str] = set()
        for item in self.menu:
            if item.get("unlock_profit", 0) == 0:
                self.unlocked_food.add(item["id"])

        # ── beverage system ──────────────────────
        self.bartender_hired: bool = False
        self.beverage_menu: list[dict] = []

        # ── employee system ──────────────────────
        self.employees: list[Employee] = []
        self._next_emp_id: int = 1
        self._employee_speed_bonus: float = 0.0

        # ── delivery system ──────────────────────
        self.delivery_unlocked: bool = False
        self.delivery_orders: list[dict] = []
        self.delivery_timer: float = 0.0

        # ── trait system ─────────────────────────
        self.traits: dict[str, int] = {}
        self.trait_selection_active: bool = False
        self.trait_choices: list[dict] = []
        self.next_trait_day: int = self.traits_config.get("offer_interval_days", 5)
        self._last_trait_check_day: int = 0

        # ── trait bonuses (cached for speed) ─────
        self.food_price_bonus: int = 0
        self.cook_time_reduction: float = 0.0
        self.tip_bonus_pct: float = 0.0
        self.base_tip_bonus: int = 0
        self.patience_bonus: float = 0.0

        # ── upgrade state ────────────────────────
        self.upgrade_levels: dict[str, int] = {
            u["id"]: 0 for u in self.upgrades_data
        }
        self.upgrade_mode: bool = False
        self.upgrade_tab: int = 0   # 0=facility, 1=staff, 2=menu

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
    def net_profit(self) -> int:
        return self.total_earned - self.total_spent

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

    @property
    def available_menu(self) -> list[dict]:
        """Menu items currently unlocked for sale."""
        return [m for m in self.menu if m["id"] in self.unlocked_food]

    @property
    def available_beverages(self) -> list[dict]:
        if not self.bartender_hired:
            return []
        np = self.net_profit
        return [b for b in self.beverage_config.get("items", [])
                if b.get("unlock_profit", 0) <= np]

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
        self.num_chefs = KITCHEN_CAPACITY
        self.max_chefs = len(self.kitchen_counter_positions)
        self.bar = BarStation(capacity=2)

        # ── trash cans (reset) ───────────────────
        self.trash_can_positions = set()
        for tc in self.map_data.get("trash_cans", []):
            self.trash_can_positions.add((tc["grid_x"], tc["grid_y"]))

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
        self.customer_spawn_timer = CUSTOMER_SPAWN_INTERVAL
        self.upgrade_mode = False
        self.upgrade_tab = 0

        for uid in self.upgrade_levels:
            self.upgrade_levels[uid] = 0
        self.cook_speed_mult = 1.0
        self.spawn_rate_mult = 1.0
        self.wealthy_bonus = 0.0

        # Economy
        self.total_earned = 0
        self.total_spent = 0

        # Food unlock
        self.unlocked_food = set()
        for item in self.menu:
            if item.get("unlock_profit", 0) == 0:
                self.unlocked_food.add(item["id"])

        # Beverage
        self.bartender_hired = False
        self.beverage_menu = []

        # Employees
        self.employees = []
        self._next_emp_id = 1
        self._employee_speed_bonus = 0.0

        # Delivery
        self.delivery_unlocked = False
        self.delivery_orders = []
        self.delivery_timer = 0.0

        # Traits
        self.traits = {}
        self.trait_selection_active = False
        self.trait_choices = []
        self.next_trait_day = self.traits_config.get("offer_interval_days", 5)
        self._last_trait_check_day = 0
        self.food_price_bonus = 0
        self.cook_time_reduction = 0.0
        self.tip_bonus_pct = 0.0
        self.base_tip_bonus = 0
        self.patience_bonus = 0.0
        self.player.carry_capacity = 1

    # ═══════════════════════════════════════════════
    #  Step — RL-compatible (includes movement)
    # ═══════════════════════════════════════════════
    def step(self, action: int) -> list[tuple[str, float]]:
        """Full step for RL: movement + game logic.

        Returns a list of ``(event_name, value)`` tuples that occurred
        during this step.  The caller (e.g. ``ai.reward.RewardCalculator``)
        converts these events into a scalar reward using configurable weights.
        """
        if self.done:
            return []

        dt = STEP_INTERVAL

        if action in (ACTION_UP, ACTION_DOWN, ACTION_LEFT, ACTION_RIGHT):
            self._move_player_action(action, dt)

        return self._step_inner(action, dt)

    # ═══════════════════════════════════════════════
    #  Step Logic — for human mode (movement handled elsewhere)
    # ═══════════════════════════════════════════════
    def step_logic(self, action: int) -> list[tuple[str, float]]:
        """Game-logic-only step.  Movement is handled by tick()."""
        if self.done:
            return []
        return self._step_inner(action, STEP_INTERVAL)

    def _step_inner(self, action: int, dt: float) -> list[tuple[str, float]]:
        events: list[tuple[str, float]] = []

        # 1) Interaction / upgrade
        if action == ACTION_INTERACT:
            events.extend(self._interact())
        elif action == ACTION_BUY_UPGRADE:
            events.extend(self._auto_buy_upgrade())

        # 2) Kitchen tick
        self.kitchen.update(dt, self.cook_speed_mult)

        # 3) Bar tick
        if self.bartender_hired:
            self.bar.update(dt)

        # 4) Customer tick
        for table in self.tables:
            if table.customer is None:
                continue
            cust = table.customer
            cust.update(dt)

            if cust.state == CustomerState.LEAVING_HAPPY:
                payment = cust.calc_payment(
                    food_price_bonus=self.food_price_bonus,
                    tip_bonus_pct=self.tip_bonus_pct,
                    base_tip_bonus=self.base_tip_bonus,
                )
                self.money += payment
                self.total_earned += payment
                self._record_satisfaction(cust.calc_satisfaction())
                self.customers_served += 1
                events.append(("customer_payment", float(payment)))
                table.customer = None

            elif cust.state == CustomerState.LEAVING_ANGRY:
                self._record_satisfaction(-1.0)
                self.customers_lost += 1
                events.append(("lost_customer", 1.0))
                table.customer = None

        # 5) Delivery-ready food from kitchen
        while self.kitchen.delivery_ready:
            dish = self.kitchen.delivery_ready.pop(0)
            self._start_delivery_transport(dish)

        # 6) Employee AI
        self._update_employees(dt)

        # 7) Delivery system
        if self.delivery_unlocked:
            self._update_delivery(dt)

        # 8) 손님 스폰 (만족도 기반)
        self.customer_spawn_timer -= dt
        if self.customer_spawn_timer <= 0:
            self._try_spawn_customer()
            base_interval = CUSTOMER_SPAWN_INTERVAL / max(0.5, self.spawn_rate_mult)
            # 평점이 높을수록 손님이 자주 방문
            rating_factor = max(0.6, 1.8 - self.shop_rating * 1.2)
            # 초반에는 느리게, 후반에는 점점 빨라짐
            day_factor = max(0.7, 1.3 - (self.current_day - 1) * 0.02)
            adjusted = base_interval * rating_factor * day_factor
            self.customer_spawn_timer = max(3.0, adjusted)

        # 9) Trait system (day-based check)
        self._check_trait_offer()

        # 10) Time
        self.time_elapsed += dt

        # 11) Message decay
        if self.message_timer > 0:
            self.message_timer -= dt
            if self.message_timer <= 0:
                self.message = ""

        # 12) Termination
        if self.money >= self.target_money:
            self.done = True
            self.won = True
            events.append(("win", 1.0))
        elif self.time_elapsed >= self.total_time_limit:
            self.done = True

        return events

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
    def _interact(self) -> list[tuple[str, float]]:
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

        # Check bar counters
        for bpos in self.bar_counter_positions:
            bcx = bpos[0] * TILE_SIZE + TILE_SIZE / 2
            bcy = bpos[1] * TILE_SIZE + TILE_SIZE / 2
            dist = math.hypot(bcx - px, bcy - py)
            if dist > INTERACT_RANGE:
                continue
            if dist > 1e-3:
                dirx = (bcx - px) / dist
                diry = (bcy - py) / dist
                if dirx * fdx + diry * fdy < 0.1:
                    continue
            if dist < best_dist:
                best_type = "bar"
                best_obj = bpos
                best_dist = dist

        # Check trash cans
        for tpos in self.trash_can_positions:
            tcx = tpos[0] * TILE_SIZE + TILE_SIZE / 2
            tcy = tpos[1] * TILE_SIZE + TILE_SIZE / 2
            dist = math.hypot(tcx - px, tcy - py)
            if dist > INTERACT_RANGE:
                continue
            if dist > 1e-3:
                dirx = (tcx - px) / dist
                diry = (tcy - py) / dist
                if dirx * fdx + diry * fdy < 0.1:
                    continue
            if dist < best_dist:
                best_type = "trash"
                best_obj = tpos
                best_dist = dist

        if best_type is None:
            return []
        if best_type == "table":
            return self._interact_table(best_obj)
        if best_type == "bar":
            return self._interact_bar()
        if best_type == "trash":
            return self._interact_trash()
        return self._interact_kitchen()

    def _interact_table(self, table: Table) -> list[tuple[str, float]]:
        cust = table.customer
        if cust is None:
            self._msg("빈 테이블입니다")
            return []

        # Walking customer hasn't arrived yet
        if cust.state == CustomerState.WALKING_TO_TABLE:
            self._msg("손님이 이동 중입니다")
            return []

        # ── Take order ───────────────────────────
        if (self.player.is_idle
                and cust.state == CustomerState.WAITING_TO_ORDER):
            self.player.pick_up_order(table.table_id, cust.menu_item,
                                       drink_item=cust.drink_item)
            cust.take_order()
            self._msg(f"주문: {cust.menu_item['name']}")
            return [("take_order", 1.0)]

        # ── Serve food ───────────────────────────
        if self.player.has_food:
            foods = [c for c in self.player.carrying
                     if c["type"] == "food" and c["table_id"] == table.table_id]
            if foods and cust.state == CustomerState.ORDER_TAKEN:
                f = foods[0]
                if cust.serve_food():
                    self.player.carrying.remove(f)
                    self._msg("서빙 완료!")
                    return [("serve_food", 1.0)]
            elif not foods:
                self._msg("다른 테이블입니다!")
                return [("wrong_table", 1.0)]

        # ── Serve drink ──────────────────────────
        if self.player.has_drink:
            drinks = [c for c in self.player.carrying
                      if c["type"] == "drink" and c["table_id"] == table.table_id]
            if drinks and cust.state in (CustomerState.ORDER_TAKEN,
                                          CustomerState.EATING):
                for d in drinks:
                    cust.serve_drink()
                    self.player.carrying.remove(d)
                self._msg("음료 서빙 완료!")
                return [("serve_drink", 1.0)]

        if self.player.has_order:
            self._msg("먼저 주방에 주문을 전달하세요!")
        elif self.player.has_food:
            self._msg("다른 테이블입니다!")
            return [("wrong_table", 1.0)]
        else:
            self._msg("이미 주문을 받았습니다")
        return []

    def _interact_kitchen(self) -> list[tuple[str, float]]:
        # ── Submit orders ────────────────────────
        if self.player.has_order:
            orders = self.player.drop_orders()
            submitted = 0
            for order in orders:
                item = order["item"]
                cook_time = max(1.0, item["cook_time"] - self.cook_time_reduction)
                ok = self.kitchen.submit_order(
                    order["table_id"], item,
                    cook_time_override=cook_time)
                if ok:
                    submitted += 1

                # Auto-queue drink at bar if bartender hired
                if self.bartender_hired and order.get("drink_item"):
                    self.bar.submit_drink(order["table_id"], order["drink_item"])

            if submitted:
                self._msg(f"{submitted}개 조리 시작")
                return [("submit_kitchen", float(submitted))]
            else:
                self._msg("주방이 가득 찼습니다!")
            return []

        # ── Pick up food ─────────────────────────
        if self.player.can_carry_more and self.kitchen.has_ready:
            while self.player.can_carry_more and self.kitchen.has_ready:
                dish = self.kitchen.pick_up()
                self.player.pick_up_food(dish["table_id"], dish["menu_item"])
            self._msg(f"음식 수거 (x{len([c for c in self.player.carrying if c['type']=='food'])})")
            return [("pickup_food", 1.0)]

        if self.player.has_food or self.player.has_drink:
            self._msg("먼저 서빙을 완료하세요!")
        elif not self.kitchen.has_ready:
            self._msg("아직 완성된 요리가 없습니다")
        return []

    def _interact_bar(self) -> list[tuple[str, float]]:
        """Interact with bar counter to pick up ready drinks."""
        if not self.bartender_hired:
            self._msg("바텐더가 없습니다!")
            return []

        if self.player.can_carry_more and self.bar.has_ready:
            drink = self.bar.pick_up()
            if drink:
                self.player.pick_up_drink(drink["table_id"], drink["drink_item"])
                self._msg(f"{drink['drink_item']['name']} 수거 완료")
                return [("pickup_drink", 1.0)]

        if not self.bar.has_ready:
            self._msg("완성된 음료가 없습니다")
        else:
            self._msg("손이 가득 찼습니다!")
        return []

    def _interact_trash(self) -> list[tuple[str, float]]:
        """Interact with trash can to discard carried items."""
        if not self.player.carrying:
            self._msg("버릴 것이 없습니다")
            return []
        count = len(self.player.carrying)
        self.player.carrying.clear()
        self._msg(f"{count}개 항목 폐기!")
        return [("trash", 1.0)]

    # ═══════════════════════════════════════════════
    #  Customer spawning
    # ═══════════════════════════════════════════════
    def _try_spawn_customer(self):
        if self.num_customers >= self.max_seated:
            return
        empty_tables = [t for t in self.tables if not t.is_occupied]
        if not empty_tables:
            return

        avail = self.available_menu
        if not avail:
            return

        table = random.choice(empty_tables)
        ctype = self._pick_customer_type()

        menu_item = random.choice(avail)

        # Optional drink order
        drink_item = None
        if self.bartender_hired:
            bev = self.available_beverages
            if bev and random.random() < 0.5:
                drink_item = random.choice(bev)

        table.customer = Customer(
            table.table_id,
            table.grid_x * TILE_SIZE,
            table.grid_y * TILE_SIZE,
            ctype, menu_item,
            drink_item=drink_item,
            patience_bonus=self.patience_bonus,
            entrance_x=self.entrance_x,
            entrance_y=self.entrance_y)

    def _pick_customer_type(self) -> dict:
        """Pick a customer type filtered by unlock_rating (satisfaction)."""
        eligible = []
        weights = []
        for i, ct in enumerate(self.customer_types):
            req = ct.get("unlock_rating", 0.0)
            if self.shop_rating >= req:
                eligible.append(ct)
                w = self._base_weights[i]
                # Boost wealthy types based on shop rating + marketing
                if ct["wealth_mult"] >= 1.5 and self.shop_rating > 0.6:
                    w += self.shop_rating * (1.0 + self.wealthy_bonus)
                weights.append(w)
        if not eligible:
            return self.customer_types[0]
        return random.choices(eligible, weights=weights, k=1)[0]

    # ═══════════════════════════════════════════════
    #  Satisfaction
    # ═══════════════════════════════════════════════
    def _record_satisfaction(self, value: float):
        self.satisfaction_history.append(value)
        if self.satisfaction_history:
            self.shop_rating = max(0.0, min(1.0,
                sum(self.satisfaction_history) / len(self.satisfaction_history)))

    # ═══════════════════════════════════════════════
    #  Upgrade System (with tabs and net-profit unlock)
    # ═══════════════════════════════════════════════
    _TAB_CATEGORIES = {
        0: ("facility", "personal", "business"),
        1: ("staff",),
        2: ("menu",),
    }
    TAB_NAMES = ["시설", "직원", "메뉴"]

    def get_upgrade_info(self) -> list[dict]:
        """Return items for the current upgrade tab."""
        cats = self._TAB_CATEGORIES.get(self.upgrade_tab, ())

        if self.upgrade_tab == 2:
            return self._get_food_unlock_info()

        info = []
        for upg in self.upgrades_data:
            if upg.get("category", "") not in cats:
                continue
            uid = upg["id"]
            level = self.upgrade_levels[uid]
            maxed = level >= upg["max_level"]
            cost = (0 if maxed
                    else int(upg["base_cost"] * (upg["cost_multiplier"] ** level)))
            locked = self.net_profit < upg.get("unlock_profit", 0)
            info.append({
                "data": upg,
                "level": level,
                "cost": cost,
                "maxed": maxed,
                "locked": locked,
                "can_afford": not maxed and not locked and self.money >= cost,
            })
        return info

    def _get_food_unlock_info(self) -> list[dict]:
        """Return food items that can be unlocked (menu tab)."""
        info = []
        for item in self.menu:
            req = item.get("unlock_profit", 0)
            cost = item.get("unlock_cost", 0)
            already = item["id"] in self.unlocked_food
            locked = self.net_profit < req
            info.append({
                "data": item,
                "level": 1 if already else 0,
                "cost": cost,
                "maxed": already,
                "locked": locked and not already,
                "can_afford": not already and not locked and self.money >= cost,
                "is_food_unlock": True,
            })
        return info

    def buy_upgrade(self, upgrade_id: str) -> bool:
        upg = None
        for u in self.upgrades_data:
            if u["id"] == upgrade_id:
                upg = u
                break
        if upg is None:
            return False

        level = self.upgrade_levels[upgrade_id]
        if level >= upg["max_level"]:
            self._msg("최대 레벨입니다!")
            return False

        # Chef hiring limited by kitchen tiles
        if upg["effect_type"] == "hire_chef" and self.num_chefs >= self.max_chefs:
            self._msg("주방 칸이 부족합니다! 주방 확장이 필요합니다.")
            return False

        required = upg.get("unlock_profit", 0)
        if self.net_profit < required:
            self._msg(f"순이익 ${required} 필요!")
            return False

        cost = int(upg["base_cost"] * (upg["cost_multiplier"] ** level))
        if self.money < cost:
            self._msg(f"${cost} 필요!")
            return False

        self.money -= cost
        self.total_spent += cost
        self.upgrade_levels[upgrade_id] = level + 1
        self._apply_upgrade(upg)
        self._msg(f"{upg['name']} Lv.{level + 1}!")
        return True

    def buy_upgrade_by_index(self, idx: int) -> bool:
        """Buy from current tab by index."""
        items = self.get_upgrade_info()
        if 0 <= idx < len(items):
            item = items[idx]
            if item.get("is_food_unlock"):
                return self._unlock_food(item["data"])
            return self.buy_upgrade(item["data"]["id"])
        return False

    def _unlock_food(self, food_item: dict) -> bool:
        fid = food_item["id"]
        if fid in self.unlocked_food:
            self._msg("이미 해금되었습니다!")
            return False
        req = food_item.get("unlock_profit", 0)
        if self.net_profit < req:
            self._msg(f"순이익 ${req} 필요!")
            return False
        cost = food_item.get("unlock_cost", 0)
        if self.money < cost:
            self._msg(f"${cost} 필요!")
            return False
        self.money -= cost
        self.total_spent += cost
        self.unlocked_food.add(fid)
        self._msg(f"해금: {food_item['name']}!")
        return True

    def _apply_upgrade(self, upg: dict):
        etype = upg["effect_type"]
        val = upg["effect_value"]

        if etype == "player_speed":
            self.player.speed += val * PLAYER_SPEED
        elif etype == "cook_speed":
            self.cook_speed_mult += val
        elif etype == "kitchen_expand":
            self.max_chefs += int(val)
            self._msg(f"주방 확장! (최대 요리사: {self.max_chefs}명)")
        elif etype == "hire_chef":
            if self.num_chefs < self.max_chefs:
                self.num_chefs += 1
                self.kitchen.capacity = self.num_chefs
                self._msg(f"요리사 고용! ({self.num_chefs}/{self.max_chefs})")
            else:
                self._msg("주방 칸이 부족합니다! 주방 확장이 필요합니다.")
        elif etype == "wealthy_bonus":
            self.wealthy_bonus += val
        elif etype == "buy_table":
            self._activate_next_table()
        elif etype == "hire_waiter":
            self._hire_employee("waiter")
        elif etype == "hire_bartender":
            self.bartender_hired = True
            self._msg("바텐더 고용! 바 영업 시작!")
        elif etype == "hire_delivery":
            self.delivery_unlocked = True
            self.delivery_timer = self.delivery_config.get("order_interval", 10)
            self._msg("배달 서비스 시작!")
        elif etype == "employee_speed":
            bonus = val * EMPLOYEE_SPEED
            for emp in self.employees:
                emp.speed += bonus
            self._employee_speed_bonus = getattr(
                self, '_employee_speed_bonus', 0.0) + bonus

    def _activate_next_table(self):
        if not self._purchasable_tables:
            return
        tdata = self._purchasable_tables.pop(0)
        table = Table(tdata["id"], tdata["grid_x"], tdata["grid_y"])
        self.tables.append(table)
        self._table_positions[(table.grid_x, table.grid_y)] = table
        self.layout[table.grid_y][table.grid_x] = 2

    def _auto_buy_upgrade(self) -> list[tuple[str, float]]:
        """RL action: auto-buy cheapest affordable upgrade or food unlock."""
        best_id = None
        best_cost = float('inf')
        for upg in self.upgrades_data:
            uid = upg["id"]
            level = self.upgrade_levels[uid]
            if level >= upg["max_level"]:
                continue
            req = upg.get("unlock_profit", 0)
            if self.net_profit < req:
                continue
            cost = int(upg["base_cost"] * (upg["cost_multiplier"] ** level))
            if cost <= self.money and cost < best_cost:
                best_id = uid
                best_cost = cost

        # Also consider food unlocks
        for item in self.menu:
            if item["id"] in self.unlocked_food:
                continue
            req = item.get("unlock_profit", 0)
            if self.net_profit < req:
                continue
            cost = item.get("unlock_cost", 0)
            if cost <= self.money and cost < best_cost:
                best_id = f"food:{item['id']}"
                best_cost = cost

        if best_id:
            if best_id.startswith("food:"):
                fid = best_id[5:]
                food = next((m for m in self.menu if m["id"] == fid), None)
                if food:
                    self._unlock_food(food)
            else:
                self.buy_upgrade(best_id)
            return [("buy_upgrade", 1.0)]
        return [("no_upgrade", 1.0)]

    # ═══════════════════════════════════════════════
    #  Employee AI
    # ═══════════════════════════════════════════════
    def _hire_employee(self, emp_type: str):
        ps = self.map_data["player_start"]
        x = ps["grid_x"] * TILE_SIZE
        y = ps["grid_y"] * TILE_SIZE
        emp = Employee(x, y, self._next_emp_id)
        # Apply any existing employee speed upgrades
        emp.speed += getattr(self, '_employee_speed_bonus', 0.0)
        self._next_emp_id += 1
        self.employees.append(emp)
        self._msg(f"종업원 #{emp.emp_id} 고용!")

    def _update_employees(self, dt: float):
        for emp in self.employees:
            if emp.state == Employee.IDLE:
                self._assign_employee_task(emp)
            elif emp.state == Employee.MOVING:
                arrived = emp.move_toward(
                    emp.target_x, emp.target_y, dt,
                    self._can_move_to)
                if arrived:
                    emp.state = Employee.ACTING
                    emp.action_timer = EMPLOYEE_ACTION_DELAY
            elif emp.state == Employee.ACTING:
                emp.action_timer -= dt
                if emp.action_timer <= 0:
                    self._complete_employee_task(emp)
            emp.update_color()

    def _assign_employee_task(self, emp: Employee):
        # 1. If carrying food/drink → go serve (or trash if customer left)
        if emp.carrying and emp.carrying["type"] in ("food", "drink"):
            tid = emp.carrying["table_id"]
            table = self._find_table(tid)
            if table and table.customer:
                emp.assign("serve", table.center_x, table.center_y, tid)
            elif self.trash_can_positions:
                tcx, tcy = self._trash_center()
                emp.assign("discard_trash", tcx, tcy)
            else:
                emp.carrying = None
                emp.finish_task()
            return

        # 2. If carrying order → go to kitchen
        if emp.carrying and emp.carrying["type"] == "order":
            kcx, kcy = self._kitchen_center()
            emp.assign("submit_kitchen", kcx, kcy)
            return

        # 3. Score-based task selection considering distance
        candidates = []

        # Kitchen has ready food → pickup
        if self.kitchen.has_ready:
            if not self._task_claimed_by_other(emp, "pickup_food"):
                kcx, kcy = self._kitchen_center()
                dist = math.hypot(emp.x - kcx, emp.y - kcy)
                candidates.append(("pickup_food", kcx, kcy, None, 8.0 / (dist + 1)))

        # Bar has ready drink → pickup
        if self.bartender_hired and self.bar.has_ready:
            if not self._task_claimed_by_other(emp, "pickup_drink"):
                bcx, bcy = self._bar_center()
                dist = math.hypot(emp.x - bcx, emp.y - bcy)
                candidates.append(("pickup_drink", bcx, bcy, None, 7.0 / (dist + 1)))

        # Waiting customers → take order (urgency from patience)
        for table in self.tables:
            if (table.customer
                    and table.customer.state == CustomerState.WAITING_TO_ORDER
                    and not table.customer.order_claimed
                    and not self._task_claimed_by_other(
                        emp, "take_order", table.table_id)):
                dist = math.hypot(emp.x - table.center_x,
                                  emp.y - table.center_y)
                urgency = 1.0 + (1.0 - table.customer.patience_ratio) * 5.0
                candidates.append((
                    "take_order", table.center_x, table.center_y,
                    table.table_id, urgency * 5.0 / (dist + 1)))

        if candidates:
            candidates.sort(key=lambda c: c[4], reverse=True)
            task, tx, ty, tid, _score = candidates[0]
            if task == "take_order" and tid is not None:
                table = self._find_table(tid)
                if table and table.customer:
                    table.customer.order_claimed = True
            emp.assign(task, tx, ty, tid)

    def _task_claimed_by_other(self, current_emp: Employee,
                               task_type: str,
                               table_id: int | None = None) -> bool:
        """Check if another employee is already assigned this task."""
        for other in self.employees:
            if other.emp_id == current_emp.emp_id:
                continue
            if other.task == task_type:
                if table_id is None or other.target_table_id == table_id:
                    return True
        return False

    def _complete_employee_task(self, emp: Employee):
        task = emp.task

        if task == "take_order":
            table = self._find_table(emp.target_table_id)
            if table and table.customer and table.customer.state == CustomerState.WAITING_TO_ORDER:
                cust = table.customer
                cust.take_order()
                # Carry order to kitchen
                order = {"type": "order", "table_id": table.table_id,
                         "item": cust.menu_item}
                if cust.drink_item:
                    order["drink_item"] = cust.drink_item
                emp.carrying = order
            emp.finish_task()

        elif task == "submit_kitchen":
            if emp.carrying and emp.carrying["type"] == "order":
                order = emp.carrying
                item = order["item"]
                cook_time = max(1.0, item["cook_time"] - self.cook_time_reduction)
                self.kitchen.submit_order(
                    order["table_id"], item, cook_time_override=cook_time)
                if self.bartender_hired and order.get("drink_item"):
                    self.bar.submit_drink(order["table_id"], order["drink_item"])
                emp.carrying = None
            emp.finish_task()

        elif task == "pickup_food":
            if self.kitchen.has_ready:
                dish = self.kitchen.pick_up()
                emp.carrying = {"type": "food", "table_id": dish["table_id"],
                                "menu_item": dish["menu_item"]}
            emp.finish_task()

        elif task == "pickup_drink":
            if self.bar.has_ready:
                drink = self.bar.pick_up()
                if drink:
                    emp.carrying = {"type": "drink", "table_id": drink["table_id"],
                                    "drink_item": drink["drink_item"]}
            emp.finish_task()

        elif task == "serve":
            table = self._find_table(emp.target_table_id)
            if table and table.customer and emp.carrying:
                ct = emp.carrying["type"]
                if ct == "food" and table.customer.state == CustomerState.ORDER_TAKEN:
                    table.customer.serve_food()
                elif ct == "drink":
                    table.customer.serve_drink()
                emp.carrying = None
            emp.finish_task()

        elif task == "discard_trash":
            emp.carrying = None
            emp.finish_task()

        else:
            emp.finish_task()

    def _find_table(self, table_id: int) -> Table | None:
        for t in self.tables:
            if t.table_id == table_id:
                return t
        return None

    def _kitchen_center(self) -> tuple[float, float]:
        if self.kitchen_counter_positions:
            kp = next(iter(self.kitchen_counter_positions))
            return kp[0] * TILE_SIZE + TILE_SIZE / 2, kp[1] * TILE_SIZE + TILE_SIZE / 2
        return TILE_SIZE * 3, TILE_SIZE * 8

    def _bar_center(self) -> tuple[float, float]:
        if self.bar_counter_positions:
            bp = next(iter(self.bar_counter_positions))
            return bp[0] * TILE_SIZE + TILE_SIZE / 2, bp[1] * TILE_SIZE + TILE_SIZE / 2
        return TILE_SIZE * 8, TILE_SIZE * 8

    def _trash_center(self) -> tuple[float, float]:
        if self.trash_can_positions:
            tp = next(iter(self.trash_can_positions))
            return tp[0] * TILE_SIZE + TILE_SIZE / 2, tp[1] * TILE_SIZE + TILE_SIZE / 2
        return TILE_SIZE * 13, TILE_SIZE * 1

    # ═══════════════════════════════════════════════
    #  Delivery System
    # ═══════════════════════════════════════════════
    def _update_delivery(self, dt: float):
        # Spawn new delivery orders
        self.delivery_timer -= dt
        if self.delivery_timer <= 0:
            self._spawn_delivery_order()
            self.delivery_timer = self.delivery_config.get("order_interval", 10)

        # Process active deliveries
        done = []
        for order in self.delivery_orders:
            if order["state"] == "delivering":
                order["timer"] -= dt
                if order["timer"] <= 0:
                    price = int(order["menu_item"]["price"]
                                * self.delivery_config.get("price_multiplier", 0.85))
                    tip_lo, tip_hi = self.delivery_config.get("tip_range", [3, 12])
                    tip = random.randint(tip_lo, tip_hi)
                    payment = price + tip
                    self.money += payment
                    self.total_earned += payment
                    order["state"] = "done"
                    done.append(order)
        for d in done:
            self.delivery_orders.remove(d)

    def _spawn_delivery_order(self):
        avail = self.available_menu
        if not avail:
            return
        item = random.choice(avail)
        cook_time = max(1.0, item["cook_time"] - self.cook_time_reduction)
        ok = self.kitchen.submit_order(0, item, delivery=True,
                                        cook_time_override=cook_time)
        if not ok:
            return  # kitchen full, skip

    def _start_delivery_transport(self, dish: dict):
        """Kitchen finished a delivery order → start delivery timer."""
        self.delivery_orders.append({
            "menu_item": dish["menu_item"],
            "state": "delivering",
            "timer": self.delivery_config.get("delivery_time", 12),
        })

    # ═══════════════════════════════════════════════
    #  Trait System
    # ═══════════════════════════════════════════════
    def _check_trait_offer(self):
        day = self.current_day
        if day <= self._last_trait_check_day:
            return
        self._last_trait_check_day = day
        if day >= self.next_trait_day and not self.trait_selection_active:
            self._offer_traits()

    def _offer_traits(self):
        all_traits = self.traits_config.get("traits", [])
        available = [t for t in all_traits
                     if self.traits.get(t["id"], 0) < t.get("max_stacks", 1)]
        if not available:
            return
        n = min(self.traits_config.get("choices_per_offer", 3), len(available))
        self.trait_choices = random.sample(available, n)
        self.trait_selection_active = True

    def select_trait(self, choice_idx: int) -> bool:
        if not self.trait_selection_active:
            return False
        if choice_idx < 0 or choice_idx >= len(self.trait_choices):
            return False
        trait = self.trait_choices[choice_idx]
        tid = trait["id"]
        self.traits[tid] = self.traits.get(tid, 0) + 1
        self._apply_trait(trait)
        self.trait_selection_active = False
        self.trait_choices = []
        interval = self.traits_config.get("offer_interval_days", 5)
        self.next_trait_day = self.current_day + interval
        self._msg(f"특성: {trait['name']}!")
        return True

    def auto_select_trait(self):
        """RL auto-pick: first available choice."""
        if self.trait_selection_active and self.trait_choices:
            self.select_trait(0)

    def _apply_trait(self, trait: dict):
        effect = trait.get("effect", "")
        val = trait.get("value", 0)
        if effect == "food_price_bonus":
            self.food_price_bonus += int(val)
        elif effect == "cook_time_reduction":
            self.cook_time_reduction += float(val)
        elif effect == "carry_capacity":
            self.player.carry_capacity += int(val)
        elif effect == "tip_bonus":
            self.tip_bonus_pct += float(val)
        elif effect == "speed_bonus":
            self.player.speed += float(val) * PLAYER_SPEED
        elif effect == "spawn_rate":
            self.spawn_rate_mult += float(val)
        elif effect == "patience_bonus":
            self.patience_bonus += float(val)
        elif effect == "base_tip":
            self.base_tip_bonus += int(val)

    # ═══════════════════════════════════════════════
    #  Helpers
    # ═══════════════════════════════════════════════
    def _msg(self, text: str, duration: float = 1.5):
        self.message = text
        self.message_timer = duration

    def get_game_result(self) -> dict:
        """Return game result data for ranking system."""
        return {
            "money": self.money,
            "net_profit": self.net_profit,
            "day_limit": self.day_limit,
            "customers_served": self.customers_served,
            "customers_lost": self.customers_lost,
            "shop_rating": round(self.shop_rating, 2),
            "won": self.won,
        }
