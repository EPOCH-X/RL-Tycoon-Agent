"""Agent interface – abstracts away whether the agent is random, rule-based,
or a trained RL model.  Used by VersusMode and evaluation scripts."""

import numpy as np

from config.settings import NUM_ACTIONS


class RandomAgent:
    """Uniformly random baseline agent."""

    def __init__(self):
        self.num_actions = NUM_ACTIONS

    def predict(self, obs):
        return np.random.randint(0, self.num_actions)


class TrainedAgent:
    """Wraps a Stable-Baselines3 model for deterministic inference."""

    def __init__(self, model_path: str):
        from stable_baselines3 import PPO
        self.model = PPO.load(model_path)

    def predict(self, obs):
        action, _ = self.model.predict(obs, deterministic=True)
        return int(action)


def load_agent(model_path: str | None = None):
    """Factory: return a TrainedAgent if a path is given, else RandomAgent."""
    if model_path:
        return TrainedAgent(model_path)
    return RandomAgent()
