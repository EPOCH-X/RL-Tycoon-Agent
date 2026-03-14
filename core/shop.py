"""Shop – complete state & logic for a single restaurant instance.

Systems:
  - Core loop: customer spawn → take order → cook → serve → payment
  - Chef system: hire chefs (1 chef = 1 dish at a time), limited by kitchen tiles
  - Tip system: satisfaction-based, customer-type-based, trait-based
  - Food unlock: net profit threshold + cost to unlock new menu items
  - Beverage system: separate bar station for drink orders (dual food+drink)
  - Employee system: auto-waiter AI (take order, submit, pickup, serve)
  - Trait system: periodic offers of permanent bonuses
  - Customer diversity: tourist, critic, VIP, etc.
  - Scoring: total sales revenue tracking
"""

import heapq
import math
import random
from collections import deque

from config.settings import (
    TILE_SIZE, STEP_INTERVAL, CUSTOMER_SPAWN_INTERVAL,
    KITCHEN_CAPACITY, DEFAULT_TARGET_MONEY, DEFAULT_DAY_LIMIT,
    DAY_LENGTH, SATISFACTION_HISTORY_LEN,
    PLAYER_SPEED, PLAYER_RADIUS, INTERACT_RANGE,
    EMPLOYEE_SPEED, EMPLOYEE_ACTION_DELAY,
    MAX_WAITING_QUEUE, WAITING_PATIENCE,
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

        # ── beverage / trait configs ─────────────
        try:
            self.beverage_config: dict = load_json_config("beverages.json")
        except Exception:
            self.beverage_config = {"unlock_profit": 999999, "items": []}
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
        self._original_kitchen_counter_positions = set(self.kitchen_counter_positions)

        # ── kitchen expansion slots (floor tiles adjacent to kitchen row) ──
        self._kitchen_expand_slots: list[tuple[int, int]] = []
        max_kx = max((p[0] for p in self.kitchen_counter_positions), default=0)
        ky = min((p[1] for p in self.kitchen_counter_positions), default=1)
        for dx in range(1, 6):
            nx = max_kx + dx
            if 0 < nx < self.grid_width - 1 and self.layout[ky][nx] == 0:
                self._kitchen_expand_slots.append((nx, ky))

        self.kitchen = Kitchen(
            cooking_capacity=KITCHEN_CAPACITY,
            storage_capacity=len(self.kitchen_counter_positions))

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
        self.shop_rating: float = 0.12
        # 초기 관성: 히스토리를 초기 평점값으로 채워서 첫 손님 1명이
        # 평점을 급등시키지 않도록 함
        for _ in range(20):
            self.satisfaction_history.append(self.shop_rating)

        # ── customer spawn ───────────────────────
        self._base_weights = [ct["spawn_weight"] for ct in self.customer_types]
        self.customer_spawn_timer: float = CUSTOMER_SPAWN_INTERVAL
        self.spawn_rate_mult: float = 1.0
        self.cook_speed_mult: float = 1.0
        self.wealthy_bonus: float = 0.0

        # ── waiting queue (밖에서 대기 중인 잠재고객) ──
        self.waiting_queue: list[Customer] = []
        self.max_waiting_queue: int = MAX_WAITING_QUEUE
        self.waiting_customers_seated: int = 0
        self.waiting_customers_left: int = 0

        # ── leaving customers (매장 밖으로 걸어 나가는 중) ──
        self.leaving_customers: list[Customer] = []

        # ── floating texts (결제 금액 등 UI 표시) ──
        self.floating_texts: list[dict] = []

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
        """순이익 = 매장 판매 총 수입 (상점 구매 비용 미차감)."""
        return self.total_earned

    @property
    def shop_rating_stars(self) -> float:
        """평점을 5점 만점 스케일로 반환 (0.0 ~ 5.0)."""
        return round(self.shop_rating * 5.0, 1)

    @property
    def final_score(self) -> float:
        """최종 스코어 = 순이익(판매총액) × (1 + 평점/10).

        평점 0.0 → ×1.0
        평점 4.4 → ×1.44
        평점 5.0 → ×1.5 (최대 보너스)
        """
        return self.net_profit * (1.0 + self.shop_rating_stars / 10.0)

    @property
    def total_time_limit(self) -> float:
        return self.day_limit * DAY_LENGTH

    def _is_walkable_tile(self, grid_x: int, grid_y: int) -> bool:
        return (
            0 <= grid_x < self.grid_width
            and 0 <= grid_y < self.grid_height
            and self.layout[grid_y][grid_x] == 0
        )

    # ── A* pathfinding for employees ─────────────
    def _find_grid_path(
        self, sx: int, sy: int, gx: int, gy: int,
    ) -> list[tuple[int, int]]:
        """A* on tile grid.  Returns path excluding start, inclusive of goal."""
        if (sx, sy) == (gx, gy):
            return []
        counter = 0
        open_set: list[tuple[int, int, int, int]] = [(0, counter, sx, sy)]
        came_from: dict[tuple[int, int], tuple[int, int]] = {}
        g_score: dict[tuple[int, int], int] = {(sx, sy): 0}
        closed: set[tuple[int, int]] = set()

        while open_set:
            _, _, cx, cy = heapq.heappop(open_set)
            if (cx, cy) in closed:
                continue
            closed.add((cx, cy))
            if (cx, cy) == (gx, gy):
                path: list[tuple[int, int]] = []
                nx, ny = cx, cy
                while (nx, ny) != (sx, sy):
                    path.append((nx, ny))
                    nx, ny = came_from[(nx, ny)]
                path.reverse()
                return path
            for dx, dy in ((0, 1), (0, -1), (1, 0), (-1, 0)):
                nx, ny = cx + dx, cy + dy
                if (nx, ny) in closed:
                    continue
                if not self._is_walkable_tile(nx, ny) and (nx, ny) != (gx, gy):
                    continue
                new_g = g_score[(cx, cy)] + 1
                if new_g < g_score.get((nx, ny), 999999):
                    g_score[(nx, ny)] = new_g
                    h = abs(nx - gx) + abs(ny - gy)
                    counter += 1
                    heapq.heappush(open_set, (new_g + h, counter, nx, ny))
                    came_from[(nx, ny)] = (cx, cy)
        return []

    def _set_employee_waypoints(self, emp) -> None:
        """Compute A* path and set intermediate waypoints on employee."""
        if emp.target_x is None or emp.target_y is None:
            emp.waypoints = []
            return
        sgx = int((emp.x + TILE_SIZE / 2) // TILE_SIZE)
        sgy = int((emp.y + TILE_SIZE / 2) // TILE_SIZE)
        ggx = int(emp.target_x // TILE_SIZE)
        ggy = int(emp.target_y // TILE_SIZE)
        path = self._find_grid_path(sgx, sgy, ggx, ggy)
        # Intermediate waypoints only — final target is emp.target_x/y
        if len(path) > 1:
            emp.waypoints = [
                self._tile_center(gx, gy) for gx, gy in path[:-1]
            ]
        else:
            emp.waypoints = []

    def _tile_center(self, grid_x: int, grid_y: int) -> tuple[float, float]:
        return (
            grid_x * TILE_SIZE + TILE_SIZE / 2,
            grid_y * TILE_SIZE + TILE_SIZE / 2,
        )

    def _interaction_anchor_tiles(self, grid_x: int, grid_y: int) -> list[tuple[int, int]]:
        anchors: list[tuple[int, int]] = []
        for dx, dy in ((0, 1), (0, -1), (-1, 0), (1, 0)):
            nx, ny = grid_x + dx, grid_y + dy
            if self._is_walkable_tile(nx, ny):
                anchors.append((nx, ny))
        return anchors

    def _select_anchor_tile(
        self,
        candidates: list[tuple[int, int]],
        reference_x: float | None = None,
        reference_y: float | None = None,
    ) -> tuple[int, int] | None:
        if not candidates:
            return None
        if reference_x is None or reference_y is None:
            return candidates[0]
        return min(
            candidates,
            key=lambda pos: (
                math.hypot(
                    self._tile_center(*pos)[0] - reference_x,
                    self._tile_center(*pos)[1] - reference_y,
                ),
                pos[1],
                pos[0],
            ),
        )

    def get_table_interaction_point(
        self,
        table: Table,
        reference_x: float | None = None,
        reference_y: float | None = None,
    ) -> tuple[float, float]:
        anchor = self._select_anchor_tile(
            self._interaction_anchor_tiles(table.grid_x, table.grid_y),
            reference_x,
            reference_y,
        )
        if anchor is None:
            return table.center_x, table.center_y
        return self._tile_center(*anchor)

    def get_station_interaction_point(
        self,
        positions: set[tuple[int, int]],
        reference_x: float | None = None,
        reference_y: float | None = None,
    ) -> tuple[float, float]:
        ordered_positions = sorted(positions)
        candidates: list[tuple[int, int]] = []
        for grid_x, grid_y in ordered_positions:
            for anchor in self._interaction_anchor_tiles(grid_x, grid_y):
                if anchor not in candidates:
                    candidates.append(anchor)

        selected = self._select_anchor_tile(candidates, reference_x, reference_y)
        if selected is not None:
            return self._tile_center(*selected)
        if ordered_positions:
            return self._tile_center(*ordered_positions[0])
        return self._tile_center(0, 0)

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
        """Menu items currently unlocked based on net profit.

        순이익이 unlock_profit 이상이면 자동 해금.
        """
        np_ = self.net_profit
        return [m for m in self.menu if m.get("unlock_profit", 0) <= np_]

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

        self.kitchen_counter_positions = set(self._original_kitchen_counter_positions)
        # Rebuild kitchen expansion slots
        self._kitchen_expand_slots = []
        max_kx = max((p[0] for p in self.kitchen_counter_positions), default=0)
        ky = min((p[1] for p in self.kitchen_counter_positions), default=1)
        for dx in range(1, 6):
            nx = max_kx + dx
            if 0 < nx < self.grid_width - 1 and self._original_layout[ky][nx] == 0:
                self._kitchen_expand_slots.append((nx, ky))

        self.kitchen = Kitchen(
            cooking_capacity=KITCHEN_CAPACITY,
            storage_capacity=len(self.kitchen_counter_positions))
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
        self.shop_rating = 0.12
        for _ in range(20):
            self.satisfaction_history.append(self.shop_rating)
        self.customer_spawn_timer = CUSTOMER_SPAWN_INTERVAL
        self.upgrade_mode = False
        self.upgrade_tab = 0

        # Waiting queue
        self.waiting_queue = []
        self.waiting_customers_seated = 0
        self.waiting_customers_left = 0
        self.leaving_customers = []
        self.floating_texts = []

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

        사람 모드와의 핵심 차이를 보정합니다:
        - 사람: tick()에서 매 프레임 이동 + update()에서 상호작용 → 동시 가능
        - RL: 매 스텝 행동 1개만 → 이동과 상호작용을 동시에 할 수 없음

        보정: 이동 액션 후 상호작용 가능한 대상이 범위 내에 있으면
        자동으로 상호작용을 시도합니다 (사람이 이동하면서 스페이스를
        누르는 것과 동일한 효과).

        Returns a list of ``(event_name, value)`` tuples that occurred
        during this step.  The caller (e.g. ``ai.reward.RewardCalculator``)
        converts these events into a scalar reward using configurable weights.
        """
        if self.done:
            return []

        dt = STEP_INTERVAL

        if action in (ACTION_UP, ACTION_DOWN, ACTION_LEFT, ACTION_RIGHT):
            self._move_player_action(action, dt)
            # ── 이동 후 자동 상호작용 (사람 모드 동시입력 보정) ──
            # 이동 방향에 상호작용 가능한 대상이 있으면 자동 시도
            auto_events = self._try_auto_interact()
            if auto_events:
                # 이미 상호작용 완료 → ACTION_NONE으로 나머지 로직만 실행
                return self._step_inner(ACTION_NONE, dt, extra_events=auto_events)

        return self._step_inner(action, dt)

    # ═══════════════════════════════════════════════
    #  Step Logic — for human mode (movement handled elsewhere)
    # ═══════════════════════════════════════════════
    def step_logic(self, action: int) -> list[tuple[str, float]]:
        """Game-logic-only step.  Movement is handled by tick()."""
        if self.done:
            return []
        return self._step_inner(action, STEP_INTERVAL)

    def _step_inner(self, action: int, dt: float,
                    extra_events: list[tuple[str, float]] | None = None,
                    ) -> list[tuple[str, float]]:
        events: list[tuple[str, float]] = list(extra_events) if extra_events else []

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
                # +$X 플로팅 텍스트
                self.floating_texts.append({
                    "text": f"+${payment}",
                    "x": table.grid_x * TILE_SIZE + TILE_SIZE // 2,
                    "y": table.grid_y * TILE_SIZE,
                    "timer": 1.2,
                })
                # 떠난 손님 관련 잔여 음식 정리 (안전장치)
                self.kitchen.remove_for_table(table.table_id)
                if self.bartender_hired:
                    self.bar.remove_for_table(table.table_id)
                # 테이블에서 분리 → 밖으로 걸어나가기
                cust.start_exit_walk(self.entrance_x, self.entrance_y)
                self.leaving_customers.append(cust)
                table.customer = None

            elif cust.state == CustomerState.LEAVING_ANGRY:
                self._record_satisfaction(-1.0)
                self.customers_lost += 1
                events.append(("lost_customer", 1.0))
                # ── 고아 음식 정리: 떠난 손님의 주문을 주방/바에서 제거 ──
                orphaned = self.kitchen.remove_for_table(table.table_id)
                if self.bartender_hired:
                    orphaned += self.bar.remove_for_table(table.table_id)
                # 플레이어가 들고 있는 해당 테이블 음식/음료도 제거
                before_carry = len(self.player.carrying)
                self.player.carrying = [
                    c for c in self.player.carrying
                    if c.get("table_id") != table.table_id
                    or c["type"] == "order"  # 주문서는 이미 제출됐을 수 없음
                ]
                orphaned += before_carry - len(self.player.carrying)
                if orphaned > 0:
                    events.append(("orphan_cleared", float(orphaned)))
                # 테이블에서 분리 → 밖으로 걸어나가기
                cust.start_exit_walk(self.entrance_x, self.entrance_y)
                self.leaving_customers.append(cust)
                table.customer = None

        # 4b) Leaving customers walk to exit
        still_leaving: list[Customer] = []
        for cust in self.leaving_customers:
            cust.update(dt)
            if not cust.is_done:
                still_leaving.append(cust)
        self.leaving_customers = still_leaving

        # 4c) Floating texts tick
        for ft in self.floating_texts:
            ft["timer"] -= dt
            ft["y"] -= 30 * dt          # float upward
        self.floating_texts = [ft for ft in self.floating_texts if ft["timer"] > 0]

        # 5) Employee AI
        self._update_employees(dt)

        # 6) 손님 스폰 (평점 기반 — 평점이 높을수록 손님이 많이 옴)
        self.customer_spawn_timer -= dt
        if self.customer_spawn_timer <= 0:
            spawn_events = self._try_spawn_customer()
            events.extend(spawn_events)
            base_interval = CUSTOMER_SPAWN_INTERVAL / max(0.5, self.spawn_rate_mult)
            # 평점이 높을수록 손님이 자주 방문 (5점 → ×0.35, 3점 → ×0.55)
            rating_factor = max(0.35, 1.5 - self.shop_rating * 2.3)
            # 초반에는 느리게, 후반에는 점점 빨라짐
            day_factor = max(0.7, 1.3 - (self.current_day - 1) * 0.02)
            adjusted = base_interval * rating_factor * day_factor
            self.customer_spawn_timer = max(2.5, adjusted)

        # 7) 대기열 업데이트 (대기 → 착석 / 대기 → 이탈)
        events.extend(self._update_waiting_queue(dt))

        # 9) 음식 자동 해금 (순이익 기반)
        self._check_food_unlocks(events)

        # 10) Trait system (day-based check)
        self._check_trait_offer()

        # 10) Time
        self.time_elapsed += dt

        # 11) Message decay
        if self.message_timer > 0:
            self.message_timer -= dt
            if self.message_timer <= 0:
                self.message = ""

        # 12) Termination
        if self.time_elapsed >= self.total_time_limit:
            self.done = True
            # 최종 스코어 = 순이익 × (1 + 평점/10)
            events.append(("game_end", self.final_score))
            if self.money >= self.target_money:
                self.won = True
                events.append(("win", 1.0))

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
    #  Auto-Interact (RL 이동 후 자동 상호작용)
    # ═══════════════════════════════════════════════
    def _try_auto_interact(self) -> list[tuple[str, float]]:
        """이동 후 유효한 상호작용 대상이 범위 내에 있으면 자동 실행.

        사람 모드에서는 이동키를 누른 채 스페이스를 누를 수 있지만,
        RL에서는 매 스텝 액션 1개만 선택 가능하여 불리합니다.

        이 메서드는 이동 방향(facing)에 상호작용 가능한 유효 대상이
        있을 때만 자동으로 상호작용을 시도하며, 의미 없는 상호작용
        (빈 테이블, 이미 주문받은 테이블 등)은 무시합니다.
        """
        px = self.player.center_x
        py = self.player.center_y
        fdx, fdy = Player.DIRECTION_VEC[self.player.facing]

        # 플레이어 상태에 따라 유효한 상호작용인지 사전 검증
        has_order = self.player.has_order
        has_food = self.player.has_food
        has_drink = self.player.has_drink
        is_idle = self.player.is_idle

        # 테이블 확인
        for table in self.tables:
            dist = math.hypot(table.center_x - px, table.center_y - py)
            if dist > INTERACT_RANGE:
                continue
            if dist > 1e-3:
                dirx = (table.center_x - px) / dist
                diry = (table.center_y - py) / dist
                if dirx * fdx + diry * fdy < 0.1:
                    continue
            cust = table.customer
            if cust is None:
                continue
            # 유효한 상호작용만: 주문 대기 + 아이들, 또는 음식/음료 서빙
            if (is_idle and cust.state == CustomerState.WAITING_TO_ORDER):
                return self._interact_table(table)
            if has_food:
                foods = [c for c in self.player.carrying
                         if c["type"] == "food" and c["table_id"] == table.table_id]
                if foods and cust.state == CustomerState.ORDER_TAKEN:
                    return self._interact_table(table)
            if has_drink:
                drinks = [c for c in self.player.carrying
                          if c["type"] == "drink" and c["table_id"] == table.table_id]
                if drinks and cust.state in (CustomerState.ORDER_TAKEN,
                                              CustomerState.EATING):
                    return self._interact_table(table)

        # 주방 카운터 확인 (주문 전달 또는 음식 수거)
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
            # 주문 전달 또는 음식 수거가 가능할 때만
            if has_order or (is_idle and self.kitchen.ready):
                return self._interact_kitchen()
            if not has_order and not has_food and not has_drink and self.kitchen.ready:
                return self._interact_kitchen()

        # 바 카운터 확인 (음료 수거)
        if self.bartender_hired:
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
                if is_idle and self.bar.has_ready:
                    return self._interact_bar()

        # 쓰레기통 확인 (고아 아이템 자동 폐기)
        if self.player.carrying and self._has_orphan_carrying():
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
                return self._interact_trash()

        return []

    def _has_orphan_carrying(self) -> bool:
        """플레이어가 손님이 떠난 테이블의 음식/음료를 들고 있는지 확인."""
        if not self.player.carrying:
            return False
        occupied_table_ids = {
            t.table_id for t in self.tables if t.customer is not None
        }
        return any(
            c.get("table_id", -1) not in occupied_table_ids
            for c in self.player.carrying
            if c["type"] in ("food", "drink")
        )

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
            if not self.kitchen.can_accept:
                self._msg("주방이 가득 찼습니다!")
                return []
            orders = self.player.drop_orders()
            submitted = 0
            rejected = []
            for order in orders:
                item = order["item"]
                cook_time = max(1.0, item["cook_time"] - self.cook_time_reduction)
                ok = self.kitchen.submit_order(
                    order["table_id"], item,
                    cook_time_override=cook_time)
                if ok:
                    submitted += 1
                    if self.bartender_hired and order.get("drink_item"):
                        self.bar.submit_drink(order["table_id"], order["drink_item"])
                else:
                    rejected.append(order)

            # Return rejected orders to player
            for order in rejected:
                self.player.carrying.append(order)

            if submitted:
                self._msg(f"{submitted}개 조리 시작")
                return [("submit_kitchen", float(submitted))]
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
        """Interact with trash can to discard carried items.

        고아 아이템(손님이 떠난 테이블의 음식/음료)은 'trash_orphan'
        이벤트로 분류하여 보상을 다르게 처리합니다.
        """
        if not self.player.carrying:
            self._msg("버릴 것이 없습니다")
            return []

        # 고아 아이템 vs 유효 아이템 분류
        orphan_count = 0
        valid_count = 0
        occupied_table_ids = {
            t.table_id for t in self.tables if t.customer is not None
        }
        for item in self.player.carrying:
            tid = item.get("table_id", -1)
            if tid not in occupied_table_ids:
                orphan_count += 1
            else:
                valid_count += 1

        self.player.carrying.clear()
        total = orphan_count + valid_count
        self._msg(f"{total}개 항목 폐기!")

        events: list[tuple[str, float]] = []
        if orphan_count > 0:
            events.append(("trash_orphan", float(orphan_count)))
        if valid_count > 0:
            events.append(("trash", float(valid_count)))
        return events

    # ═══════════════════════════════════════════════
    #  Customer spawning
    # ═══════════════════════════════════════════════
    def _try_spawn_customer(self) -> list[tuple[str, float]]:
        events: list[tuple[str, float]] = []
        avail = self.available_menu
        if not avail:
            return events

        ctype = self._pick_customer_type()
        menu_item = random.choice(avail)

        # Drink order: 40% chance when bartender is hired (5명 중 2명 꼴)
        drink_item = None
        if self.bartender_hired:
            bev = self.available_beverages
            if bev and random.random() < 0.4:
                drink_item = random.choice(bev)

        # 빈 테이블 확인
        empty_tables = [t for t in self.tables if not t.is_occupied]
        if empty_tables:
            table = random.choice(empty_tables)
            table.customer = Customer(
                table.table_id,
                table.grid_x * TILE_SIZE,
                table.grid_y * TILE_SIZE,
                ctype, menu_item,
                drink_item=drink_item,
                patience_bonus=self.patience_bonus,
                entrance_x=self.entrance_x,
                entrance_y=self.entrance_y)
        elif len(self.waiting_queue) < self.max_waiting_queue:
            # 테이블이 없으면 대기열에 추가
            waiting_cust = Customer(
                -1,  # 아직 테이블 미배정
                self.entrance_x,
                self.entrance_y,
                ctype, menu_item,
                drink_item=drink_item,
                patience_bonus=self.patience_bonus,
                entrance_x=self.entrance_x,
                entrance_y=self.entrance_y,
                waiting_outside=True)
            self.waiting_queue.append(waiting_cust)
            events.append(("customer_waiting", 1.0))
        # else: 대기열도 꽉 참 → 손님 생성 안 됨 (잠재 고객 놓침)
        return events

    def _update_waiting_queue(self, dt: float) -> list[tuple[str, float]]:
        """Update waiting customers: seat if table available, or lose if patience runs out."""
        events: list[tuple[str, float]] = []

        # 1) 빈 테이블에 대기 손님 배정 (FIFO)
        empty_tables = [t for t in self.tables if not t.is_occupied]
        while empty_tables and self.waiting_queue:
            table = empty_tables.pop(0)
            cust = self.waiting_queue.pop(0)
            cust.assign_table(
                table.table_id,
                table.grid_x * TILE_SIZE,
                table.grid_y * TILE_SIZE,
                self.entrance_x,
                self.entrance_y)
            table.customer = cust
            self.waiting_customers_seated += 1
            events.append(("waiting_customer_seated", 1.0))

        # 2) 대기 손님 patience 업데이트
        still_waiting: list[Customer] = []
        for cust in self.waiting_queue:
            cust.update(dt)
            if cust.state == CustomerState.LEAVING_ANGRY:
                self.waiting_customers_left += 1
                self.customers_lost += 1
                # 밖에서 대기 이탈은 평점에 영향 없음
                events.append(("waiting_customer_left", 1.0))
            else:
                still_waiting.append(cust)
        self.waiting_queue = still_waiting

        return events

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
    #  Food Auto-Unlock (순이익 기반 자동 해금)
    # ═══════════════════════════════════════════════
    def _check_food_unlocks(self, events: list[tuple[str, float]]):
        """순이익이 해금 조건을 충족하면 자동으로 메뉴 해금."""
        np_ = self.net_profit
        for item in self.menu:
            fid = item["id"]
            if fid in self.unlocked_food:
                continue
            req = item.get("unlock_profit", 0)
            if np_ >= req:
                self.unlocked_food.add(fid)
                self._msg(f"메뉴 해금: {item['name']}! (순이익 ${np_})")
                events.append(("food_unlock", float(item["price"])))

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
                    else (upg["cost_list"][level] if "cost_list" in upg and level < len(upg["cost_list"])
                          else int(upg["base_cost"] * (upg["cost_multiplier"] ** level))))
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
        """Return food items with auto-unlock status (menu tab).

        음식은 순이익 달성 시 자동 해금. 구매 불필요.
        """
        info = []
        for item in self.menu:
            req = item.get("unlock_profit", 0)
            already = item["id"] in self.unlocked_food
            locked = self.net_profit < req
            info.append({
                "data": item,
                "level": 1 if already else 0,
                "cost": 0,  # 자동 해금 (구매 불필요)
                "maxed": already,
                "locked": locked and not already,
                "can_afford": False,  # 자동 해금이므로 구매 버튼 없음
                "is_food_unlock": True,
                "unlock_profit_req": req,
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

        cost = (upg["cost_list"][level] if "cost_list" in upg and level < len(upg["cost_list"])
               else int(upg["base_cost"] * (upg["cost_multiplier"] ** level)))
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
                # 음식은 자동 해금이므로 구매 불가
                self._msg("음식은 순이익 달성 시 자동 해금됩니다!")
                return False
            return self.buy_upgrade(item["data"]["id"])
        return False

    def _apply_upgrade(self, upg: dict):
        etype = upg["effect_type"]
        val = upg["effect_value"]

        if etype == "player_speed":
            self.player.speed += val * PLAYER_SPEED
        elif etype == "cook_speed":
            self.cook_speed_mult += val
        elif etype == "kitchen_expand":
            # Add a new kitchen counter tile visually
            if self._kitchen_expand_slots:
                new_pos = self._kitchen_expand_slots.pop(0)
                self.kitchen_counter_positions.add(new_pos)
                self.layout[new_pos[1]][new_pos[0]] = 3   # kitchen tile
            self.max_chefs += int(val)
            self.kitchen.storage_capacity += int(val)
            self._msg(f"주방 확장! (보관:{self.kitchen.storage_capacity} 최대요리사:{self.max_chefs}명)")
        elif etype == "hire_chef":
            if self.num_chefs < self.max_chefs:
                self.num_chefs += 1
                self.kitchen.cooking_capacity = self.num_chefs
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

    # ── 업그레이드 우선순위 (ROI 기반) ──
    # 값이 클수록 우선 구매. 게임 진행 단계에 따라 달라짐.
    _UPGRADE_PRIORITY = {
        "speed_shoes":     8,   # 초반 이동속도 = 서빙 효율 핵심
        "hire_chef":       9,   # 요리사 = 처리량 병목 해소
        "buy_table":       7,   # 테이블 = 동시 손님 수
        "cook_speed":      6,   # 조리 속도 = 회전율
        "kitchen_expand":  5,   # 주방 확장 → 요리사 추가 가능
        "marketing":       4,   # 부유한 손님 = 수익 UP
        "hire_waiter":     10,  # 종업원 = 자동화 (최우선)
        "hire_bartender":  3,   # 음료 서비스
        "employee_speed":  3,   # 직원 속도
    }

    def _auto_buy_upgrade(self) -> list[tuple[str, float]]:
        """전략적 업그레이드 구매 (ROI 우선순위 기반).

        음식 해금은 순이익 기반 자동 해금으로 변경되어 여기서 제외.
        대기열에 손님이 있으면 테이블 구매 우선도 동적 상승.
        """
        candidates: list[tuple[float, str, int]] = []  # (priority, id, cost)

        # 대기열 손님 수에 따라 buy_table 동적 우선도 증가
        queue_pressure = len(self.waiting_queue)

        for upg in self.upgrades_data:
            uid = upg["id"]
            level = self.upgrade_levels[uid]
            if level >= upg["max_level"]:
                continue
            # 요리사: 주방 칸 한도 체크
            if upg["effect_type"] == "hire_chef" and self.num_chefs >= self.max_chefs:
                continue
            req = upg.get("unlock_profit", 0)
            if self.net_profit < req:
                continue
            cost = (upg["cost_list"][level] if "cost_list" in upg and level < len(upg["cost_list"])
                   else int(upg["base_cost"] * (upg["cost_multiplier"] ** level)))
            if cost > self.money:
                continue
            priority = self._UPGRADE_PRIORITY.get(uid, 1)
            # 레벨이 높아질수록 우선도 감소 (첫 구매가 가장 가치 있음)
            priority -= level * 0.5
            # 대기열이 있으면 테이블 구매 우선도 대폭 상승
            if uid == "buy_table" and queue_pressure > 0:
                priority += queue_pressure * 3.0
            candidates.append((priority, uid, cost))

        if not candidates:
            return [("no_upgrade", 1.0)]

        # 우선도 높은 것 선택
        candidates.sort(key=lambda c: c[0], reverse=True)
        _, best_id, _ = candidates[0]
        self.buy_upgrade(best_id)
        return [("buy_upgrade", 1.0)]

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
                # Follow waypoints first, then final target
                if emp.waypoints:
                    wp_x, wp_y = emp.waypoints[0]
                    reached = emp.move_toward(
                        wp_x, wp_y, dt, self._can_move_to)
                    if reached:
                        emp.waypoints.pop(0)
                else:
                    arrived = emp.move_toward(
                        emp.target_x, emp.target_y, dt,
                        self._can_move_to)
                    if arrived:
                        emp.state = Employee.ACTING
                        emp.action_timer = EMPLOYEE_ACTION_DELAY
                # Recompute path if stuck too long
                if emp._stuck_timer > 1.5:
                    self._set_employee_waypoints(emp)
                    emp._stuck_timer = 0.0
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
                tx, ty = self.get_table_interaction_point(table, emp.x, emp.y)
                emp.assign("serve", tx, ty, tid)
                self._set_employee_waypoints(emp)
            elif self.trash_can_positions:
                tcx, tcy = self._trash_center(emp.x, emp.y)
                emp.assign("discard_trash", tcx, tcy)
                self._set_employee_waypoints(emp)
            else:
                emp.carrying = None
                emp.finish_task()
            return

        # 2. If carrying order → go to kitchen
        if emp.carrying and emp.carrying["type"] == "order":
            kcx, kcy = self._kitchen_center(emp.x, emp.y)
            emp.assign("submit_kitchen", kcx, kcy)
            self._set_employee_waypoints(emp)
            return

        # 3. Score-based task selection considering distance
        candidates = []

        # Kitchen has ready food → pickup
        if self.kitchen.has_ready:
            if not self._task_claimed_by_other(emp, "pickup_food"):
                kcx, kcy = self._kitchen_center(emp.x, emp.y)
                dist = math.hypot(emp.x - kcx, emp.y - kcy)
                candidates.append(("pickup_food", kcx, kcy, None, 8.0 / (dist + 1)))

        # Bar has ready drink → pickup
        if self.bartender_hired and self.bar.has_ready:
            if not self._task_claimed_by_other(emp, "pickup_drink"):
                bcx, bcy = self._bar_center(emp.x, emp.y)
                dist = math.hypot(emp.x - bcx, emp.y - bcy)
                candidates.append(("pickup_drink", bcx, bcy, None, 7.0 / (dist + 1)))

        # Waiting customers → take order (urgency from patience)
        for table in self.tables:
            if (table.customer
                    and table.customer.state == CustomerState.WAITING_TO_ORDER
                    and not table.customer.order_claimed
                    and not self._task_claimed_by_other(
                        emp, "take_order", table.table_id)):
                tx, ty = self.get_table_interaction_point(table, emp.x, emp.y)
                dist = math.hypot(emp.x - tx, emp.y - ty)
                urgency = 1.0 + (1.0 - table.customer.patience_ratio) * 5.0
                candidates.append((
                    "take_order", tx, ty,
                    table.table_id, urgency * 5.0 / (dist + 1)))

        if candidates:
            candidates.sort(key=lambda c: c[4], reverse=True)
            task, tx, ty, tid, _score = candidates[0]
            if task == "take_order" and tid is not None:
                table = self._find_table(tid)
                if table and table.customer:
                    table.customer.order_claimed = True
            emp.assign(task, tx, ty, tid)
            self._set_employee_waypoints(emp)

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
                ok = self.kitchen.submit_order(
                    order["table_id"], item, cook_time_override=cook_time)
                if ok:
                    if self.bartender_hired and order.get("drink_item"):
                        self.bar.submit_drink(order["table_id"], order["drink_item"])
                    emp.carrying = None
                # If kitchen full, employee keeps carrying and will retry
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

    def _kitchen_center(self, ref_x: float | None = None,
                        ref_y: float | None = None) -> tuple[float, float]:
        if self.kitchen_counter_positions:
            return self.get_station_interaction_point(
                self.kitchen_counter_positions, ref_x, ref_y)
        return TILE_SIZE * 3, TILE_SIZE * 8

    def _bar_center(self, ref_x: float | None = None,
                    ref_y: float | None = None) -> tuple[float, float]:
        if self.bar_counter_positions:
            return self.get_station_interaction_point(
                self.bar_counter_positions, ref_x, ref_y)
        return TILE_SIZE * 8, TILE_SIZE * 8

    def _trash_center(self, ref_x: float | None = None,
                      ref_y: float | None = None) -> tuple[float, float]:
        if self.trash_can_positions:
            return self.get_station_interaction_point(
                self.trash_can_positions, ref_x, ref_y)
        return TILE_SIZE * 13, TILE_SIZE * 1

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

    # ── 특성 가치 평가 (상황 기반) ──
    _TRAIT_BASE_VALUE = {
        "gourmet":          7,   # 음식 가격 +$2 → 꾸준한 수익 증가
        "master_chef":      6,   # 조리 시간 -1초 → 회전율
        "skilled_server":   8,   # 운반 +1 → 효율 대폭 증가
        "charming":         5,   # 팁 +30%
        "efficient":        4,   # 이동 속도 +15%
        "popular":          3,   # 손님 빈도 +20% (후반에만 유용)
        "patient_service":  9,   # 인내심 +5초 → 이탈 방지 (가장 중요)
        "tip_jar":          5,   # 기본 팁 +$3
    }

    def auto_select_trait(self):
        """RL auto-pick: 상황에 맞는 최적 특성 선택."""
        if not self.trait_selection_active or not self.trait_choices:
            return
        best_idx = 0
        best_val = -1
        for i, trait in enumerate(self.trait_choices):
            tid = trait["id"]
            max_stacks = trait.get("max_stacks", 1)
            current = self.traits.get(tid, 0)
            if current >= max_stacks:
                continue
            val = self._TRAIT_BASE_VALUE.get(tid, 1)
            # 이미 보유 중이면 가치 약간 감소 (다양성 유도)
            val -= current * 1.5
            if val > best_val:
                best_val = val
                best_idx = i
        self.select_trait(best_idx)

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
            "shop_rating": round(self.shop_rating, 4),
            "shop_rating_stars": self.shop_rating_stars,
            "final_score": round(self.final_score, 1),
            "won": self.won,
        }
