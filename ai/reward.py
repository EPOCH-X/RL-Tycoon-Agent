"""Reward calculator – converts game events to RL reward using configurable weights.

Game events are ``(event_name, value)`` tuples emitted by ``Shop.step()``.
The reward for each event is ``weight[event_name] * value``.

Weights are loaded from ``config/train_config.json → reward_shaping``.
Team members can tune reward values **without touching game code**.
"""

# Default reward weights (fallback when config is absent)
DEFAULT_WEIGHTS: dict[str, float] = {
    "take_order":       1.5,
    "serve_food":       8.0,
    "submit_kitchen":   2.0,
    "pickup_food":      2.0,
    "pickup_drink":     1.5,
    "serve_drink":      4.0,
    "customer_payment": 0.05,
    "lost_customer":   -60.0,
    "wrong_table":     -4.0,
    "buy_upgrade":      2.0,
    "no_upgrade":      -0.02,
    "win":            200.0,
    "trash":           -0.5,
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
