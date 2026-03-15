"""Agent loading helpers for trained and algorithm-specific models."""

import os
import numpy as np

from config.settings import NUM_ACTIONS

_ALGO_PATH_HINTS: dict[str, str] = {
    "a3c": "A3C",
    "sac": "SAC",
    "dqn": "DQN",
    "marl": "MARL",
    "model_based": "ModelBased",
    "modelbased": "ModelBased",
    "ppo": "PPO",
}


def _detect_algo_from_path(model_path: str) -> str | None:
    path_lower = model_path.replace("\\", "/").lower()
    parts = path_lower.replace("/", "_").replace("-", "_").split("_")
    for part in parts:
        if part in _ALGO_PATH_HINTS:
            return _ALGO_PATH_HINTS[part]
    return None


class RandomAgent:
    def __init__(self):
        self.num_actions = NUM_ACTIONS

    def predict(self, obs, action_mask=None):
        if action_mask is not None:
            valid_actions = np.flatnonzero(action_mask)
            if len(valid_actions) > 0:
                return int(np.random.choice(valid_actions))
        return int(np.random.randint(0, self.num_actions))


class TrainedAgent:
    """Wrap SB3 models with observation-shape adaptation."""

    def __init__(self, model_path: str):
        self.uses_action_mask = False
        self.deterministic = True
        try:
            from sb3_contrib import MaskablePPO
            self.model = MaskablePPO.load(model_path)
            self.uses_action_mask = True
        except Exception:
            from stable_baselines3 import PPO
            self.model = PPO.load(model_path)
        self.expected_obs_shape = tuple(self.model.observation_space.shape)

    def _adapt_observation(self, obs):
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
                obs, deterministic=self.deterministic, action_masks=action_mask)
            return int(action)
        action, _ = self.model.predict(obs, deterministic=self.deterministic)
        return int(action)

    def get_action_probs(self, obs):
        import torch as th

        obs = self._adapt_observation(obs)
        obs_t = th.as_tensor(obs).float().unsqueeze(0)
        with th.no_grad():
            dist = self.model.policy.get_distribution(obs_t)
            probs = dist.distribution.probs[0].cpu().numpy()
        return probs


class AlgorithmAgent:
    def __init__(self, algo_name: str, model_path: str):
        from algorithms.registry import get_algorithm

        trainer_class = get_algorithm(algo_name)
        self.trainer = trainer_class()
        self.trainer.build()
        self.trainer.load(model_path)
        self.deterministic = True

    def predict(self, obs, action_mask=None):
        return self.trainer.predict(obs, deterministic=self.deterministic)

    def get_action_probs(self, obs):
        import torch as th

        obs_t = th.as_tensor(obs).float().unsqueeze(0)

        if hasattr(self.trainer, "global_model") and self.trainer.global_model is not None:
            model = self.trainer.global_model
            device = next(model.parameters()).device
            with th.no_grad():
                logits, _ = model(obs_t.to(device))
                probs = th.softmax(logits, dim=-1)
            return probs[0].cpu().numpy()

        if hasattr(self.trainer, "policy") and self.trainer.policy is not None:
            model = self.trainer.policy
            device = next(model.parameters()).device
            with th.no_grad():
                probs = model(obs_t.to(device))
            return probs[0].cpu().numpy()

        return None


def load_agent(model_path: str | None = None, algo_name: str | None = None):
    if model_path is None:
        return RandomAgent()

    if algo_name is None:
        algo_name = _detect_algo_from_path(model_path)

    clean_path = model_path[:-3] if model_path.endswith(".pt") else model_path

    if algo_name and algo_name != "PPO":
        return AlgorithmAgent(algo_name, clean_path)

    if algo_name == "PPO" or model_path.endswith(".zip"):
        return TrainedAgent(model_path)

    try:
        return TrainedAgent(model_path)
    except Exception:
        config_path = os.path.join(os.path.dirname(model_path), "train_config_used.json")
        if os.path.isfile(config_path):
            import json

            with open(config_path, encoding="utf-8") as f:
                cfg = json.load(f)
            detected = cfg.get("algorithm", "").upper()
            if detected and detected != "PPO":
                return AlgorithmAgent(detected, clean_path)
        raise
