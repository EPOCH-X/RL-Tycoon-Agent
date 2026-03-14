"""Auto-waiter employee entity with simple state-machine AI.

States:
  IDLE     → scan for tasks
  MOVING   → walk toward target in straight line
  ACTING   → brief action delay at destination (take order / serve / etc.)

After ACTING the task completes and the employee returns to IDLE.
Movement uses direct-line pixel motion with axis-sliding collision.
"""

from __future__ import annotations

import math
from core.entity import Entity
from config.settings import COLORS, EMPLOYEE_SPEED, EMPLOYEE_ACTION_DELAY, TILE_SIZE


class Employee(Entity):
    IDLE = "idle"
    MOVING = "moving"
    ACTING = "acting"

    def __init__(self, x: float, y: float, emp_id: int):
        super().__init__(x, y,
                         color=COLORS["employee"],
                         sprite_key="employee")
        self.emp_id = emp_id
        self.speed: float = EMPLOYEE_SPEED
        self.state: str = self.IDLE

        # Current task
        self.task: str | None = None          # e.g. "take_order", "submit_kitchen", "pickup_food", "serve", "pickup_drink"
        self.target_x: float | None = None
        self.target_y: float | None = None
        self.target_table_id: int | None = None
        self.action_timer: float = 0.0

        # Carrying (simple single-item)
        self.carrying: dict | None = None     # {"type":"order"|"food"|"drink", "table_id":int, ...}

        self._base_color = self.color

        # Stuck detection
        self._stuck_timer: float = 0.0
        self._prev_x: float = x
        self._prev_y: float = y

    # ── movement ──────────────────────────────────
    def move_toward(self, tx: float, ty: float, dt: float,
                    can_move_fn) -> bool:
        """Move toward (tx, ty).  Return True when close enough.

        tx, ty는 타일 중심 좌표이므로, 엔티티의 중심(center)과
        비교하여 거리 계산합니다.  _can_move_to는 top-left 기준이므로
        이동 좌표는 top-left로 변환합니다.

        Stuck detection: 일정 시간 이동 못 하면 축 전환으로 우회 시도.
        """
        # 엔티티 중심 → 타겟 중심 거리
        cx = self.x + TILE_SIZE / 2
        cy = self.y + TILE_SIZE / 2
        dx = tx - cx
        dy = ty - cy
        dist = math.hypot(dx, dy)
        arrive_dist = TILE_SIZE * 0.6

        if dist <= arrive_dist:
            self._stuck_timer = 0.0
            return True

        step = self.speed * dt
        if step >= dist:
            # 정확한 중심 정렬: top-left = target - half_tile
            nx = tx - TILE_SIZE / 2
            ny = ty - TILE_SIZE / 2
        else:
            nx = self.x + dx / dist * step
            ny = self.y + dy / dist * step

        moved = False
        # Try full move, then axis-slide
        if can_move_fn(nx, ny):
            self.x, self.y = nx, ny
            moved = True
        elif can_move_fn(nx, self.y):
            self.x = nx
            moved = True
        elif can_move_fn(self.x, ny):
            self.y = ny
            moved = True

        # Stuck detection: 이동 거리가 극소면 stuck 타이머 증가
        actual_moved = math.hypot(self.x - self._prev_x, self.y - self._prev_y)
        self._prev_x = self.x
        self._prev_y = self.y

        if actual_moved < 0.5:
            self._stuck_timer += dt
        else:
            self._stuck_timer = 0.0

        # Stuck 0.5초 이상이면 수직/수평 우회 시도
        if self._stuck_timer >= 0.5 and not moved:
            # 주 이동 축과 반대 축으로 우회
            jitter = step * 0.8
            if abs(dx) > abs(dy):
                # 가로 이동이 막혔으면 세로로 우회
                for sign in (1, -1):
                    ny2 = self.y + sign * jitter
                    if can_move_fn(self.x, ny2):
                        self.y = ny2
                        self._stuck_timer = 0.0
                        break
            else:
                # 세로 이동이 막혔으면 가로로 우회
                for sign in (1, -1):
                    nx2 = self.x + sign * jitter
                    if can_move_fn(nx2, self.y):
                        self.x = nx2
                        self._stuck_timer = 0.0
                        break

        # 도착 판정도 중심 기준
        cx = self.x + TILE_SIZE / 2
        cy = self.y + TILE_SIZE / 2
        return math.hypot(tx - cx, ty - cy) <= arrive_dist

    # ── assign task ───────────────────────────────
    def assign(self, task: str, target_x: float, target_y: float,
               table_id: int | None = None):
        self.task = task
        self.target_x = target_x
        self.target_y = target_y
        self.target_table_id = table_id
        self.state = self.MOVING

    def finish_task(self):
        self.task = None
        self.target_x = None
        self.target_y = None
        self.target_table_id = None
        self.state = self.IDLE
        self._stuck_timer = 0.0

    # ── visual ────────────────────────────────────
    def update_color(self):
        if self.carrying:
            self.color = COLORS["employee_carry"]
        else:
            self.color = self._base_color

    def reset(self):
        self.state = self.IDLE
        self.task = None
        self.carrying = None
        self.action_timer = 0.0
