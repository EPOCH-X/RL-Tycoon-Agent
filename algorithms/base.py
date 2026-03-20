"""Base interface for all RL algorithm trainers."""

from abc import ABC, abstractmethod
from typing import Any


class BaseTrainer(ABC):
    """공통 알고리즘 학습 인터페이스.

    모든 알고리즘 트레이너는 이 클래스를 상속해야 합니다.
    """

    name: str = "base"

    @abstractmethod
    def build(self, cfg: dict) -> None:
        """설정(cfg)으로부터 환경, 모델, 콜백 등을 초기화합니다."""

    @abstractmethod
    def train(self, resume_path: str | None = None) -> dict[str, Any]:
        """학습을 실행하고 결과 메트릭을 반환합니다."""

    @abstractmethod
    def save(self, path: str) -> None:
        """모델 가중치를 지정된 경로에 저장합니다."""

    @abstractmethod
    def load(self, path: str) -> None:
        """저장된 가중치를 불러옵니다."""

    @abstractmethod
    def predict(self, obs, deterministic: bool = True) -> int:
        """관측(obs)에 대한 행동을 반환합니다."""


class BaseAgent:
    """학습된 모델을 래핑하여 게임에서 사용하는 에이전트."""

    def __init__(self, trainer: BaseTrainer):
        self.trainer = trainer

    def predict(self, obs) -> int:
        return self.trainer.predict(obs, deterministic=True)
