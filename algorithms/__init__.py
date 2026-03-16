"""algorithms – 강화학습 알고리즘 모음.

각 하위 폴더는 독립된 알고리즘 구현체를 포함합니다:
    ppo/        – Proximal Policy Optimization (SB3)
    dqn/        – Deep Q-Network (SB3)
    a3c/        – Asynchronous Advantage Actor-Critic (PyTorch 직접 구현)
    sac/        – Soft Actor-Critic (SB3, Discrete 래퍼)
    marl/       – Multi-Agent RL (Self-play + Independent PPO)
    model_based/– Model-Based RL (World-Model + MPC)

공통 유틸리티:
    base.py     – 알고리즘 공통 인터페이스
    registry.py – 알고리즘 레지스트리
"""

from algorithms.registry import ALGORITHM_REGISTRY, get_algorithm
