"""통합 학습 런처 – 모든 알고리즘을 통일된 인터페이스로 학습/평가합니다.

Usage:
    # 단일 알고리즘 학습
    python -m algorithms.train_launcher --algo PPO --timesteps 200000

    # DQN 학습 (커스텀 설정)
    python -m algorithms.train_launcher --algo DQN --config algorithms/dqn/config.json

    # 모든 알고리즘 벤치마크 (순차 실행)
    python -m algorithms.train_launcher --benchmark --timesteps 100000

    # MARL Self-play 학습
    python -m algorithms.train_launcher --algo MARL --timesteps 300000

    # 학습된 모델 평가
    python -m algorithms.train_launcher --algo PPO --evaluate --model models/ppo/best_model
"""

import argparse
import json
import os
import time
from typing import Any

from algorithms.registry import get_algorithm, ALGORITHM_REGISTRY


def train_single(algo_name: str, args) -> dict[str, Any]:
    """단일 알고리즘을 학습합니다."""
    TrainerClass = get_algorithm(algo_name)
    trainer = TrainerClass()

    save_path = args.save_path or f"models/{algo_name.lower()}"

    overrides = {}
    if args.timesteps:
        overrides["timesteps"] = args.timesteps
    if args.n_envs:
        overrides["n_envs"] = args.n_envs
    if args.seed is not None:
        overrides["seed"] = args.seed

    print(f"\n{'='*60}")
    print(f"  Training: {algo_name}")
    print(f"  Save path: {save_path}")
    print(f"  Overrides: {overrides}")
    print(f"{'='*60}\n")

    start = time.time()
    trainer.build(config_path=args.config, save_path=save_path, **overrides)
    result = trainer.train()
    elapsed = time.time() - start
    result["wall_time_sec"] = round(elapsed, 1)

    print(f"\n  Wall time: {elapsed:.1f}s")
    return result


def evaluate_model(algo_name: str, model_path: str, n_episodes: int = 20):
    """학습된 모델을 평가합니다."""
    from algorithms.common import make_env

    TrainerClass = get_algorithm(algo_name)
    trainer = TrainerClass()

    # build needed for A3C/SAC/ModelBased to init networks
    trainer.build()
    trainer.load(model_path)

    env = make_env(0, seed=9999)()
    results = []

    for ep in range(n_episodes):
        obs, _ = env.reset()
        done = False
        total_reward = 0.0
        steps = 0
        while not done:
            action = trainer.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            steps += 1
            done = terminated or truncated
        results.append({
            "episode": ep + 1,
            "reward": total_reward,
            "steps": steps,
            "won": info.get("won", False),
            "money": info.get("money", 0),
        })
        print(f"  Episode {ep+1}: reward={total_reward:.1f}, "
              f"money=${info.get('money', 0):.0f}, "
              f"won={info.get('won', False)}")

    env.close()

    import numpy as np
    rewards = [r["reward"] for r in results]
    win_rate = sum(1 for r in results if r["won"]) / len(results)
    print(f"\n  Mean reward: {np.mean(rewards):.1f} ± {np.std(rewards):.1f}")
    print(f"  Win rate: {win_rate*100:.1f}%")
    return results


def benchmark_all(args):
    """모든 알고리즘을 순차 학습하며 벤치마크합니다."""
    results = {}
    for algo_name in ALGORITHM_REGISTRY:
        print(f"\n{'#'*60}")
        print(f"  BENCHMARK: {algo_name}")
        print(f"{'#'*60}")
        try:
            args_copy = argparse.Namespace(**vars(args))
            args_copy.save_path = f"models/benchmark/{algo_name.lower()}"
            result = train_single(algo_name, args_copy)
            results[algo_name] = result
        except Exception as e:
            print(f"  [✗] {algo_name} failed: {e}")
            results[algo_name] = {"error": str(e)}

    # 결과 저장
    os.makedirs("models/benchmark", exist_ok=True)
    with open("models/benchmark/results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)

    print(f"\n\n{'='*60}")
    print("  BENCHMARK RESULTS")
    print(f"{'='*60}")
    for name, res in results.items():
        if "error" in res:
            print(f"  {name:12s}: FAILED - {res['error']}")
        else:
            print(f"  {name:12s}: {res.get('wall_time_sec', '?')}s")
    print(f"\n  Results saved to models/benchmark/results.json")

    return results


def main():
    p = argparse.ArgumentParser(
        description="RL Tycoon – 통합 알고리즘 학습/평가 런처")

    p.add_argument("--algo", type=str, default="PPO",
                   choices=list(ALGORITHM_REGISTRY.keys()),
                   help="학습할 알고리즘 선택")
    p.add_argument("--config", type=str, default=None,
                   help="커스텀 설정 JSON 경로")
    p.add_argument("--timesteps", type=int, default=None,
                   help="총 학습 타임스텝 수")
    p.add_argument("--save-path", type=str, default=None,
                   help="모델 저장 경로")
    p.add_argument("--n-envs", type=int, default=None,
                   help="병렬 환경 수")
    p.add_argument("--seed", type=int, default=None,
                   help="랜덤 시드")

    p.add_argument("--benchmark", action="store_true",
                   help="모든 알고리즘 벤치마크 실행")
    p.add_argument("--evaluate", action="store_true",
                   help="학습된 모델 평가 모드")
    p.add_argument("--model", type=str, default=None,
                   help="평가할 모델 경로")
    p.add_argument("--eval-episodes", type=int, default=20,
                   help="평가 에피소드 수")

    args = p.parse_args()

    if args.benchmark:
        benchmark_all(args)
    elif args.evaluate:
        if not args.model:
            p.error("--evaluate requires --model path")
        evaluate_model(args.algo, args.model, args.eval_episodes)
    else:
        train_single(args.algo, args)


if __name__ == "__main__":
    main()
