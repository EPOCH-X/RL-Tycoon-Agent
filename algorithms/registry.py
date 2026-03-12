"""Algorithm registry – 알고리즘 이름으로 트레이너 클래스를 가져옵니다."""

from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from algorithms.base import BaseTrainer

# 지연 임포트를 위한 레지스트리 (import 시간 단축)
ALGORITHM_REGISTRY: dict[str, str] = {
    "PPO":         "algorithms.ppo.trainer.PPOTrainer",
    "DQN":         "algorithms.dqn.trainer.DQNTrainer",
    "A3C":         "algorithms.a3c.trainer.A3CTrainer",
    "SAC":         "algorithms.sac.trainer.SACTrainer",
    "MARL":        "algorithms.marl.trainer.MARLTrainer",
    "ModelBased":  "algorithms.model_based.trainer.ModelBasedTrainer",
}


def get_algorithm(name: str) -> type["BaseTrainer"]:
    """알고리즘 이름으로 트레이너 클래스를 반환합니다."""
    import importlib

    dotted = ALGORITHM_REGISTRY.get(name)
    if dotted is None:
        raise ValueError(
            f"Unknown algorithm '{name}'. "
            f"Available: {list(ALGORITHM_REGISTRY.keys())}"
        )
    module_path, cls_name = dotted.rsplit(".", 1)
    module = importlib.import_module(module_path)
    return getattr(module, cls_name)
