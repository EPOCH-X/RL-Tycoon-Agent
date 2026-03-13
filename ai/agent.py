"""Agent interface – abstracts away whether the agent is random, rule-based,
or a trained RL model.  Used by VersusMode and evaluation scripts.

Supports all algorithm types: PPO, DQN, A3C, SAC, MARL, ModelBased.
"""

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
        self.deterministic = True

    def predict(self, obs):
        action, _ = self.model.predict(obs, deterministic=self.deterministic)
        return int(action)

    def get_action_probs(self, obs):
        """Return action probabilities for the given observation."""
        import torch as th
        obs_t = th.as_tensor(obs).float().unsqueeze(0)
        with th.no_grad():
            dist = self.model.policy.get_distribution(obs_t)
            probs = dist.distribution.probs[0].cpu().numpy()
        return probs


class AlgorithmAgent:
    """알고리즘 레지스트리 기반 에이전트 – 모든 알고리즘을 지원합니다.

    Usage:
        agent = AlgorithmAgent("PPO", "models/ppo/best_model")
        agent = AlgorithmAgent("SAC", "models/sac/best_model")
        agent = AlgorithmAgent("A3C", "models/a3c/final_model")
    """

    def __init__(self, algo_name: str, model_path: str):
        from algorithms.registry import get_algorithm
        TrainerClass = get_algorithm(algo_name)
        self.trainer = TrainerClass()
        self.trainer.build()
        self.trainer.load(model_path)

    def predict(self, obs):
        return self.trainer.predict(obs, deterministic=True)


def load_agent(model_path: str | None = None,
               algo_name: str | None = None):
    """Factory: return an agent based on algorithm name and model path.

    Args:
        model_path: 학습된 모델 경로
        algo_name: 알고리즘 이름 (PPO, DQN, A3C, SAC, MARL, ModelBased)
                   None이면 기존 SB3 PPO 로딩 시도
    """
    if model_path is None:
        return RandomAgent()

    if algo_name and algo_name != "PPO":
        return AlgorithmAgent(algo_name, model_path)

    # 기본: SB3 PPO
    return TrainedAgent(model_path)

