"""Reward calculator – converts game events to RL reward using configurable weights.

Game events are ``(event_name, value)`` tuples emitted by ``Shop.step()``.
The reward for each event is ``weight[event_name] * value``.

Weights are loaded from ``config/train_config.json → reward_shaping``.
Team members can tune reward values **without touching game code**.
"""

# Default reward weights (fallback when config is absent)
DEFAULT_WEIGHTS: dict[str, float] = {
    "take_order":         8.0,
    "serve_food":        15.0,
    "submit_kitchen":     4.0,
    "pickup_food":        3.0,
    "pickup_drink":       2.0,
    "serve_drink":        6.0,
    "customer_payment":   0.25,
    "lost_customer":    -15.0,
    "wrong_table":       -2.0,
    "buy_upgrade":        0.0,
    "no_upgrade":         0.0,
    "win":              250.0,
    "trash":             -2.0,
    "game_end":           0.05,
    "net_profit_delta":   0.25,
    "rating_delta":      20.0,
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
        reward = 0.0
        for name, value in events:
            weight = self.weights.get(name, 0.0)
            reward += weight * value
        return reward
