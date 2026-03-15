"""MDP (Markov Decision Process) 프레임워크.

경영 시뮬레이션 게임을 MDP로 정형화합니다:
- 상태(State): 현재 자본금, 재고량, 평점, 직원 수 등
- 행동(Action): 생산량 조절, 가격 변경, 업그레이드 구매 등
- 보상(Reward): 이윤(매출 - 비용)
- 전이 확률(Transition): 환경의 다음 상태 확률

이 모듈은 게임 환경의 MDP 구조를 명시적으로 문서화하고,
상태/행동 공간의 분석 도구를 제공합니다.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Any

from config.settings import NUM_ACTIONS


@dataclass
class MDPState:
    """MDP 상태 표현.

    경영 게임의 핵심 상태 변수들을 구조화합니다.
    """
    # 재무
    money: float = 0.0
    net_profit: float = 0.0
    target_money: float = 1500.0

    # 시간
    current_day: int = 1
    day_limit: int = 30
    time_ratio: float = 1.0

    # 매장 상태
    shop_rating: float = 0.5
    tables_active: int = 4
    max_tables: int = 8
    kitchen_capacity: int = 1
    employees: int = 0

    # 손님 통계
    customers_served: int = 0
    customers_lost: int = 0
    customers_waiting: int = 0

    # 플레이어
    player_pos: tuple[float, float] = (0.0, 0.0)
    carrying: str = "none"  # none, order, food, drink

    # 업그레이드
    upgrade_levels: dict[str, int] = field(default_factory=dict)

    def to_vector(self) -> np.ndarray:
        """정규화된 관측 벡터로 변환."""
        return np.array([
            self.money / max(1, self.target_money),
            self.net_profit / max(1, self.target_money),
            self.current_day / max(1, self.day_limit),
            self.time_ratio,
            self.shop_rating,
            self.tables_active / max(1, self.max_tables),
            self.kitchen_capacity / 4.0,
            self.employees / 4.0,
            self.customers_served / 100.0,
            self.customers_lost / 50.0,
            self.customers_waiting / 10.0,
        ], dtype=np.float32)


@dataclass
class MDPAction:
    """MDP 행동 정의."""
    MOVE_UP = 0
    MOVE_DOWN = 1
    MOVE_LEFT = 2
    MOVE_RIGHT = 3
    INTERACT = 4
    WAIT = 5
    BUY_UPGRADE = 6

    @staticmethod
    def describe(action: int) -> str:
        names = {
            0: "위로 이동", 1: "아래로 이동",
            2: "왼쪽 이동", 3: "오른쪽 이동",
            4: "상호작용", 5: "대기",
            6: "업그레이드 구매",
        }
        return names.get(action, f"알 수 없는 행동({action})")


class MDPAnalyzer:
    """MDP 구조를 분석하고 시각화하는 도구.

    Usage:
        analyzer = MDPAnalyzer()
        analyzer.log_transition(state, action, reward, next_state, done)
        report = analyzer.generate_report()
    """

    def __init__(self):
        self.transitions: list[dict] = []
        self.rewards: list[float] = []
        self.states: list[MDPState] = []
        self.action_counts: dict[int, int] = {i: 0 for i in range(NUM_ACTIONS)}

    def log_transition(self, state: MDPState, action: int,
                       reward: float, next_state: MDPState, done: bool):
        self.transitions.append({
            "state": state,
            "action": action,
            "reward": reward,
            "next_state": next_state,
            "done": done,
        })
        self.rewards.append(reward)
        self.states.append(state)
        self.action_counts[action] = self.action_counts.get(action, 0) + 1

    def generate_report(self) -> dict[str, Any]:
        """MDP 분석 리포트를 생성합니다."""
        if not self.transitions:
            return {"error": "No transitions logged"}

        rewards = np.array(self.rewards)
        return {
            "total_transitions": len(self.transitions),
            "reward_stats": {
                "mean": float(rewards.mean()),
                "std": float(rewards.std()),
                "min": float(rewards.min()),
                "max": float(rewards.max()),
                "total": float(rewards.sum()),
            },
            "action_distribution": {
                MDPAction.describe(a): cnt
                for a, cnt in sorted(self.action_counts.items())
            },
            "state_space": {
                "money_range": (
                    min(s.money for s in self.states),
                    max(s.money for s in self.states),
                ),
                "rating_range": (
                    min(s.shop_rating for s in self.states),
                    max(s.shop_rating for s in self.states),
                ),
            },
            "episode_count": sum(
                1 for t in self.transitions if t["done"]
            ),
        }
