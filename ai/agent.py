"""Agent interface – abstracts away whether the agent is random, rule-based,
or a trained RL model.  Used by VersusMode and evaluation scripts.

Supports all algorithm types: PPO, DQN, A3C, SAC, MARL, ModelBased.
"""

import os
import numpy as np

from config.settings import NUM_ACTIONS

# 알고리즘 ↔ 경로 키워드 매핑 (자동 탐지용)
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
    """모델 경로에서 알고리즘 이름을 자동 탐지합니다.

    경로에 'a3c', 'sac', 'dqn' 등의 키워드가 포함되면 해당 알고리즘 반환.
    .pt 확장자 → custom trainer, .zip 확장자 → SB3 (PPO/DQN).
    """
    path_lower = model_path.replace("\\", "/").lower()
    # 경로 구성 요소에서 알고리즘 이름 탐지
    parts = path_lower.replace("/", "_").replace("-", "_").split("_")
    for part in parts:
        if part in _ALGO_PATH_HINTS:
            return _ALGO_PATH_HINTS[part]
    # 확장자 기반 추론
    if model_path.endswith(".pt"):
        return None  # .pt지만 알고리즘 불명 → 디렉토리의 config 확인
    return None


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
        self.deterministic = True

    def predict(self, obs):
        return self.trainer.predict(obs, deterministic=self.deterministic)

    def get_action_probs(self, obs):
        """Return action probabilities for watch mode debug display.

        Custom trainers에서 softmax 확률을 추출합니다.
        """
        import torch as th
        obs_t = th.as_tensor(obs).float().unsqueeze(0)

        # A3C: ActorCritic.forward → (logits, value)
        if hasattr(self.trainer, 'global_model') and self.trainer.global_model is not None:
            model = self.trainer.global_model
            device = next(model.parameters()).device
            with th.no_grad():
                logits, _ = model(obs_t.to(device))
                probs = th.softmax(logits, dim=-1)
            return probs[0].cpu().numpy()

        # SAC: PolicyNetwork.forward → probs
        if hasattr(self.trainer, 'policy') and self.trainer.policy is not None:
            model = self.trainer.policy
            device = next(model.parameters()).device
            with th.no_grad():
                probs = model(obs_t.to(device))
            return probs[0].cpu().numpy()

        # DQN: Q-network → softmax over Q-values
        if hasattr(self.trainer, 'model') and self.trainer.model is not None:
            try:
                model = self.trainer.model
                action, _ = model.predict(obs, deterministic=False)
                # SB3 DQN doesn't directly expose probs, return uniform
            except Exception:
                pass

        return None


def load_agent(model_path: str | None = None,
               algo_name: str | None = None):
    """Factory: return an agent based on algorithm name and model path.

    Args:
        model_path: 학습된 모델 경로
        algo_name: 알고리즘 이름 (PPO, DQN, A3C, SAC, MARL, ModelBased)
                   None이면 경로에서 자동 탐지 후, SB3 PPO 로딩 시도
    """
    if model_path is None:
        return RandomAgent()

    # 알고리즘 자동 탐지
    if algo_name is None:
        algo_name = _detect_algo_from_path(model_path)

    # .pt 확장자 제거 (AlgorithmAgent.load가 ".pt"를 자동 추가)
    clean_path = model_path
    if clean_path.endswith(".pt"):
        clean_path = clean_path[:-3]

    if algo_name and algo_name not in ("PPO",):
        # Custom trainer (A3C, SAC, MARL, ModelBased 등)
        return AlgorithmAgent(algo_name, clean_path)

    if algo_name == "PPO" or model_path.endswith(".zip"):
        # SB3 PPO
        return TrainedAgent(model_path)

    # 최후 시도: SB3 로딩, 실패 시 .pt로 재시도
    try:
        return TrainedAgent(model_path)
    except Exception:
        # .pt 파일이면 config에서 알고리즘 탐지 시도
        config_path = os.path.join(os.path.dirname(model_path),
                                   "train_config_used.json")
        if os.path.isfile(config_path):
            import json
            with open(config_path, encoding="utf-8") as f:
                cfg = json.load(f)
            detected = cfg.get("algorithm", "").upper()
            if detected and detected != "PPO":
                return AlgorithmAgent(detected, clean_path)
        raise

