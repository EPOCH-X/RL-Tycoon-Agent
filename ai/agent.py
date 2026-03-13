"""Agent interface – abstracts away whether the agent is random, rule-based,
or a trained RL model.  Used by VersusMode and evaluation scripts."""

import numpy as np

from config.settings import NUM_ACTIONS


class RandomAgent:
    """Uniformly random baseline agent."""

    def __init__(self):
        self.num_actions = NUM_ACTIONS

    def predict(self, obs, action_mask=None):
        if action_mask is not None:
            valid_actions = np.flatnonzero(action_mask)
            if len(valid_actions) > 0:
                return int(np.random.choice(valid_actions))
        return int(np.random.randint(0, self.num_actions))


class TrainedAgent:
    """Wraps a Stable-Baselines3 model for deterministic inference."""

    def __init__(self, model_path: str):
        self.uses_action_mask = False
        try:
            from sb3_contrib import MaskablePPO
            self.model = MaskablePPO.load(model_path)
            self.uses_action_mask = True
        except Exception:
            from stable_baselines3 import PPO
            self.model = PPO.load(model_path)
        self.expected_obs_shape = tuple(self.model.observation_space.shape)

    def _adapt_observation(self, obs):
        """Align live observation size with the model's training-time shape."""
        arr = np.asarray(obs, dtype=np.float32)
        if arr.shape == self.expected_obs_shape:
            return arr

        if arr.ndim != 1 or len(self.expected_obs_shape) != 1:
            return arr

        expected = self.expected_obs_shape[0]
        current = arr.shape[0]
        if current > expected:
            return arr[:expected]
        if current < expected:
            padded = np.zeros(expected, dtype=np.float32)
            padded[:current] = arr
            return padded
        return arr

    def predict(self, obs, action_mask=None):
        obs = self._adapt_observation(obs)
        if self.uses_action_mask:
            action, _ = self.model.predict(
                obs, deterministic=True, action_masks=action_mask)
            return int(action)
        action, _ = self.model.predict(obs, deterministic=True)
        return int(action)


def load_agent(model_path: str | None = None):
    """Factory: return a TrainedAgent if a path is given, else RandomAgent."""
    if model_path:
        return TrainedAgent(model_path)
    return RandomAgent()
