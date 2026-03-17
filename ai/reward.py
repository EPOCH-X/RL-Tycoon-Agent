"""Reward calculator – converts game events to RL reward using configurable weights.

Game events are ``(event_name, value)`` tuples emitted by ``Shop.step()``.
The reward for each event is ``weight[event_name] * value``.

Weights are loaded from ``config/train_config.json → reward_shaping``.
Team members can tune reward values **without touching game code**.
"""

# Default reward weights (fallback when config is absent)
#
# 설계 원칙 (타이쿤·경영 시뮬레이션):
#   1. 서비스 체인 완료에 비례하는 보상 (take→submit→pickup→serve)
#   2. 아이들/대기 페널티로 행동을 유도
#   3. time_penalty로 긴박감 부여 (빠른 서비스 = 더 높은 보상)
#   4. lost_customer는 의미있지만 압도적이지 않게
#   5. 보상 스케일을 작게 유지 → 학습 안정성 향상
DEFAULT_WEIGHTS: dict[str, float] = {
    # ── 서비스 체인 (핵심 게임플레이 루프) ──
    # take_order(8) → submit_kitchen(5) → pickup_food(5) → serve_food(15) = 33/고객
    "take_order":         8.0,
    "submit_kitchen":     5.0,
    "pickup_food":        5.0,
    "serve_food":        15.0,
    "pickup_drink":       3.0,
    "serve_drink":        8.0,
    "customer_payment":   1.0,
    # ── 페널티 ──
    "lost_customer":    -15.0,    # 손님 이탈 페널티 (서비스체인 33의 ~45%)
    "wrong_table":       -2.0,
    "trash":             -1.0,    # 유효한 음식 폐기 (강한 페널티)
    "trash_orphan":       0.5,    # 고아 음식 폐기 (올바른 행동 = 소량 보상)
    "orphan_cleared":     0.0,    # 시스템 자동 정리 (정보용, 보상 없음)
    "blocked_move":      -0.1,
    "idle_penalty":      -0.3,    # WAIT 액션 시 매 스텝 (강화)
    "time_penalty":      -0.02,   # 매 스텝 시간 압박
    # ── 업그레이드 ──
    "buy_upgrade":        2.0,
    "food_unlock":        0.3,    # 음식 해금 (value=메뉴가격, 0.3×가격=보상)
    "no_upgrade":         0.0,
    "upgrade_available":  -0.3,   # 업그레이드 가능한데 안 살 때 매 스텝 페널티
    # ── 대기열 (잠재고객) ──
    "customer_waiting":      -0.3,    # 손님이 밖에서 대기 시작 (자리 부족 경고)
    "waiting_customer_seated": 3.0,   # 대기 손님이 착석 (테이블 확보 보상)
    "waiting_customer_left": -8.0,    # 대기 손님 이탈 (잠재 수익 손실)
    # ── 게임 마일스톤 ──
    "win":              200.0,
    "game_end":           0.01,
    # ── 지속적 진행 지표 ──
    "net_profit_delta":   0.2,
    "rating_delta":      10.0,
    "final_score_delta":  0.05,
}


class RewardCalculator:
    """Compute scalar reward from a list of game events.

    Parameters
    ----------
    reward_config : dict | None
        The ``reward_shaping`` section of ``train_config.json``.
        Keys starting with ``_`` (e.g. ``_comment``) are ignored.
        Any key not present falls back to ``DEFAULT_WEIGHTS``.
    """

    def __init__(self, reward_config: dict | None = None):
        self.weights: dict[str, float] = dict(DEFAULT_WEIGHTS)
        if reward_config:
            for key, val in reward_config.items():
                if key.startswith("_"):
                    continue
                self.weights[key] = float(val)

    def __call__(self, events: list[tuple[str, float]]) -> float:
        """Return total reward for a list of ``(event_name, value)``."""
        reward, _, _ = self.details(events)
        return reward

    def details(self, events: list[tuple[str, float]]) -> tuple[float, dict[str, float], dict[str, float]]:
        """Return total reward plus per-event contributions and raw totals."""
        reward = 0.0
        contributions: dict[str, float] = {}
        raw_values: dict[str, float] = {}
        for name, value in events:
            weight = self.weights.get(name, 0.0)
            event_reward = weight * value
            reward += event_reward
            contributions[name] = contributions.get(name, 0.0) + event_reward
            raw_values[name] = raw_values.get(name, 0.0) + value
        return reward, contributions, raw_values
