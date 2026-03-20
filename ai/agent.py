"""Agent interface – abstracts away whether the agent is random, rule-based,
or a trained RL model.  Used by VersusMode and evaluation scripts.

Supports all algorithm types: PPO, DQN, QRDQN, A3C, SAC, MARL, ModelBased,
MaskablePPO (sb3-contrib).
"""

import os
import numpy as np

from config.settings import NUM_ACTIONS

# 알고리즘 ↔ 경로 키워드 매핑 (자동 탐지용)
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
    """모델 경로에서 알고리즘 이름을 자동 탐지합니다.

    경로에 'a3c', 'sac', 'dqn' 등의 키워드가 포함되면 해당 알고리즘 반환.
    .pt 확장자 → custom trainer, .zip 확장자 → SB3 (PPO/DQN).
    """
    path_lower = model_path.replace("\\", "/").lower()
    # 복합 키워드(예: discrete_sac, cross_play)를 먼저 체크
    for hint, algo in sorted(_ALGO_PATH_HINTS.items(), key=lambda x: -len(x[0])):
        if hint in path_lower:
            return algo
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
    """Wraps a Stable-Baselines3 (or sb3-contrib) model for inference."""

    def __init__(self, model_path: str, algo_name: str | None = None):
        self._is_maskable = False
        if algo_name == "MaskablePPO":
            try:
                from sb3_contrib import MaskablePPO
                self.model = MaskablePPO.load(model_path)
                self._is_maskable = True
            except ImportError:
                raise ImportError(
                    "sb3-contrib 패키지가 필요합니다: pip install sb3-contrib")
        else:
            from stable_baselines3 import PPO
            self.model = PPO.load(model_path)
        self.deterministic = True

        # 모델이 학습된 obs/action 크기를 기억
        self._expected_obs_size: int | None = None
        try:
            self._expected_obs_size = self.model.observation_space.shape[0]
        except Exception:
            pass

    @staticmethod
    def _adapt_observation(obs, expected_size: int | None):
        """Observation 크기가 다를 때 패딩/잘라내기로 맞춥니다."""
        if expected_size is None:
            return obs
        obs = np.asarray(obs, dtype=np.float32)
        if obs.shape[-1] == expected_size:
            return obs
        if obs.shape[-1] > expected_size:
            return obs[..., :expected_size]
        # Zero-pad
        pad_width = expected_size - obs.shape[-1]
        return np.pad(obs, (0, pad_width), constant_values=0.0)

    def predict(self, obs, action_mask=None):
        obs = self._adapt_observation(obs, self._expected_obs_size)
        if self._is_maskable and action_mask is not None:
            action, _ = self.model.predict(
                obs, deterministic=self.deterministic,
                action_masks=action_mask)
        else:
            action, _ = self.model.predict(
                obs, deterministic=self.deterministic)
        action = int(action)
        # 액션 범위 클리핑 (모델의 액션 공간이 더 클 수 있음)
        if action >= NUM_ACTIONS:
            action = NUM_ACTIONS - 1  # ACTION_BUY_UPGRADE or clamp
        return action

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
        import json
        from algorithms.registry import get_algorithm
        TrainerClass = get_algorithm(algo_name)
        self.trainer = TrainerClass()
        # 저장된 학습 설정을 로드하여 네트워크 구조를 정확히 재현
        model_dir = os.path.dirname(model_path) or "."
        saved_cfg_path = os.path.join(model_dir, "train_config_used.json")
        cfg = None
        if os.path.isfile(saved_cfg_path):
            with open(saved_cfg_path, encoding="utf-8") as f:
                cfg = json.load(f)
        self.trainer.build(cfg=cfg)
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
        algo_name: 알고리즘 이름 (PPO, DQN, QRDQN, A3C, SAC, MARL, ModelBased)
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

    # .zip 파일은 항상 SB3(PPO/MaskablePPO)로 로드
    if model_path.endswith(".zip"):
        return TrainedAgent(model_path, algo_name=algo_name)

    if algo_name and algo_name not in ("PPO",):
        # Custom trainer (A3C, SAC, MARL, ModelBased 등)
        return AlgorithmAgent(algo_name, clean_path)

    # 최후 시도: SB3 로딩, 실패 시 .pt로 재시도
    try:
        return TrainedAgent(model_path, algo_name=algo_name)
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

