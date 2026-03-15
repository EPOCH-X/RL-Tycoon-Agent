"""Reward calculator for RL training."""

DEFAULT_WEIGHTS: dict[str, float] = {
    "take_order": 8.0,
    "submit_kitchen": 5.0,
    "pickup_food": 5.0,
    "serve_food": 15.0,
    "pickup_drink": 3.0,
    "serve_drink": 8.0,
    "customer_payment": 1.0,
    "lost_customer": -15.0,
    "wrong_table": -2.0,
    "trash": -1.0,
    "trash_orphan": 0.5,
    "orphan_cleared": 0.0,
    "stale_carry_cleared": 0.0,
    "blocked_move": -0.1,
    "idle_penalty": -0.3,
    "time_penalty": -0.02,
    "buy_upgrade": 2.0,
    "food_unlock": 0.3,
    "no_upgrade": 0.0,
    "customer_waiting": -0.3,
    "waiting_customer_seated": 3.0,
    "waiting_customer_left": -8.0,
    "win": 200.0,
    "game_end": 0.01,
    "net_profit_delta": 0.2,
    "rating_delta": 10.0,
    "final_score_delta": 0.05,
}


class RewardCalculator:
    def __init__(self, reward_config: dict | None = None):
        self.weights: dict[str, float] = dict(DEFAULT_WEIGHTS)
        if reward_config:
            for key, val in reward_config.items():
                if key.startswith("_"):
                    continue
                self.weights[key] = float(val)

    def __call__(self, events: list[tuple[str, float]]) -> float:
        reward, _, _ = self.details(events)
        return reward

    def details(self, events: list[tuple[str, float]]) -> tuple[float, dict[str, float], dict[str, float]]:
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
