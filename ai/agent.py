"""Agent loading helpers for trained and algorithm-specific models."""

import json
import os

import numpy as np

from config.settings import NUM_ACTIONS

_ALGO_PATH_HINTS: dict[str, str] = {
    "a3c": "A3C",
    "qrdqn": "QRDQN",
    "sac": "SAC",
    "dqn": "DQN",
    "marl": "MARL",
    "model_based": "ModelBased",
    "modelbased": "ModelBased",
    "ppo": "PPO",
    "maskableppo": "MaskablePPO",
    "maskable_ppo": "MaskablePPO",
    "discrete_sac": "DiscreteSAC",
    "discretesac": "DiscreteSAC",
    "dreamer": "Dreamer",
    "cross_play": "CrossPlay",
    "crossplay": "CrossPlay",
}


def _detect_algo_from_path(model_path: str) -> str | None:
    path_lower = model_path.replace("\\", "/").lower()
    for hint, algo in sorted(_ALGO_PATH_HINTS.items(), key=lambda x: -len(x[0])):
        if hint in path_lower:
            return algo
    if model_path.endswith(".pt"):
        return None
    parts = path_lower.replace("/", "_").replace("-", "_").split("_")
    for part in parts:
        if part in _ALGO_PATH_HINTS:
            return _ALGO_PATH_HINTS[part]
    return None


def _load_saved_config(model_path: str) -> dict | None:
    cfg_path = os.path.join(os.path.dirname(model_path), "train_config_used.json")
    if not os.path.isfile(cfg_path):
        return None
    with open(cfg_path, encoding="utf-8") as f:
        return json.load(f)


def _detect_algo(model_path: str, algo_name: str | None) -> str | None:
    if algo_name:
        return algo_name
    cfg = _load_saved_config(model_path)
    if cfg:
        configured = cfg.get("algorithm")
        if configured:
            return configured
    return _detect_algo_from_path(model_path)


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

    def __init__(self, model_path: str, algo_name: str | None = None):
        self.uses_action_mask = False
        self.deterministic = True

        normalized_algo = (algo_name or "").strip()
        if normalized_algo == "MaskablePPO":
            from sb3_contrib import MaskablePPO

            self.model = MaskablePPO.load(model_path)
            self.uses_action_mask = True
        else:
            try:
                from stable_baselines3 import PPO

                self.model = PPO.load(model_path)
            except TypeError:
                from sb3_contrib import MaskablePPO

                self.model = MaskablePPO.load(model_path)
                self.uses_action_mask = True

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
                obs,
                deterministic=self.deterministic,
                action_masks=action_mask,
            )
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

        model_dir = os.path.dirname(model_path) or "."
        saved_cfg_path = os.path.join(model_dir, "train_config_used.json")
        cfg = None
        if os.path.isfile(saved_cfg_path):
            with open(saved_cfg_path, encoding="utf-8") as f:
                cfg = json.load(f)
        self.trainer.build(cfg=cfg)
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

    algo_name = _detect_algo(model_path, algo_name)
    clean_path = model_path[:-3] if model_path.endswith(".pt") else model_path

    if model_path.endswith(".zip"):
        return TrainedAgent(model_path, algo_name=algo_name)

    if algo_name and algo_name not in ("PPO", "MaskablePPO"):
        return AlgorithmAgent(algo_name, clean_path)

    try:
        return TrainedAgent(model_path, algo_name=algo_name)
    except Exception:
        cfg = _load_saved_config(model_path)
        if cfg:
            detected = cfg.get("algorithm", "")
            if detected and detected not in ("PPO", "MaskablePPO"):
                return AlgorithmAgent(detected, clean_path)
        raise
