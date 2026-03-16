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
import re
import time
from typing import Any

from algorithms.registry import get_algorithm, ALGORITHM_REGISTRY


def _next_version_path(base_path: str) -> str:
    """기존 폴더가 있으면 자동으로 v2, v3, ... 버전을 생성합니다.

    models/ppo 가 이미 존재하면 models/ppo_v2, _v3, ... 중
    존재하지 않는 첫 번째 경로를 반환합니다.
    """
    if not os.path.exists(base_path):
        return base_path

    # 이미 _vN 접미사가 붙어있는 경로인지 확인
    m = re.match(r'^(.+)_v(\d+)$', base_path)
    if m:
        stem, ver = m.group(1), int(m.group(2))
    else:
        stem = base_path
        ver = 1  # 기존 폴더 = v1

    ver += 1
    while os.path.exists(f"{stem}_v{ver}"):
        ver += 1
    return f"{stem}_v{ver}"


# 알고리즘 한글 이름 매핑
ALGO_KR: dict[str, str] = {
    "PPO":        "근위 정책 최적화",
    "DQN":        "심층 Q-네트워크",
    "A3C":        "비동기 어드밴티지 액터-크리틱",
    "SAC":        "소프트 액터-크리틱",
    "MARL":       "멀티에이전트 자기대결",
    "ModelBased": "모델기반 강화학습",
}


def train_single(algo_name: str, args) -> dict[str, Any]:
    """단일 알고리즘을 학습합니다."""
    TrainerClass = get_algorithm(algo_name)
    trainer = TrainerClass()

    days = getattr(args, 'days', None)
    day_suffix = f"_{days}d" if days and days != 30 else ""
    base_path = args.save_path or f"models/{algo_name.lower()}{day_suffix}"
    save_path = _next_version_path(base_path)
    kr = ALGO_KR.get(algo_name, algo_name)

    overrides = {}
    if args.timesteps:
        overrides["timesteps"] = args.timesteps
    if args.n_envs:
        overrides["n_envs"] = args.n_envs
    if args.seed is not None:
        overrides["seed"] = args.seed

    # 사용될 설정 파일 경로 표시
    if days and days != 30:
        config_src = args.config or f"algorithms/{algo_name.lower()}/config_{days}.json"
    else:
        config_src = args.config or f"algorithms/{algo_name.lower()}/config.json"

    print(f"\n{'='*60}")
    print(f"  학습 시작 (Training): {algo_name} ({kr})")
    print(f"  설정 파일 (Config): {config_src}")
    print(f"  저장 경로 (Save path): {save_path}")
    if overrides:
        print(f"  오버라이드 (Overrides): {overrides}")
    print(f"{'='*60}\n")

    start = time.time()
    trainer.build(config_path=args.config, save_path=save_path, days=days,
                  **overrides)
    result = trainer.train()
    elapsed = time.time() - start
    result["wall_time_sec"] = round(elapsed, 1)

    print(f"\n  소요 시간 (Wall time): {elapsed:.1f}s")
    return result


def evaluate_model(algo_name: str, model_path: str, n_episodes: int = 20):
    """학습된 모델을 평가합니다."""
    from algorithms.common import make_env

    kr = ALGO_KR.get(algo_name, algo_name)
    TrainerClass = get_algorithm(algo_name)
    trainer = TrainerClass()

    # build needed for A3C/SAC/ModelBased to init networks
    trainer.build()
    trainer.load(model_path)

    env = make_env(0, seed=9999)()
    results = []

    print(f"\n  모델 평가 (Evaluate): {algo_name} ({kr})")
    print(f"  모델 경로 (Model): {model_path}")
    print(f"  에피소드 수 (Episodes): {n_episodes}\n")

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
        print(f"  에피소드 {ep+1}: 보상(reward)={total_reward:.1f}, "
              f"수익(money)=${info.get('money', 0):.0f}, "
              f"승리(won)={info.get('won', False)}")

    env.close()

    import numpy as np
    rewards = [r["reward"] for r in results]
    win_rate = sum(1 for r in results if r["won"]) / len(results)
    print(f"\n  평균 보상 (Mean reward): {np.mean(rewards):.1f} ± {np.std(rewards):.1f}")
    print(f"  승률 (Win rate): {win_rate*100:.1f}%")
    return results


def benchmark_all(args):
    """모든 알고리즘을 순차 학습하며 벤치마크합니다."""
    algo_list = list(ALGORITHM_REGISTRY.keys())
    print(f"\n{'#'*60}")
    print(f"  벤치마크 시작 (Benchmark Start)")
    print(f"  대상 알고리즘 (Algorithms): {', '.join(algo_list)}")
    print(f"{'#'*60}")

    results = {}
    for idx, algo_name in enumerate(algo_list, 1):
        kr = ALGO_KR.get(algo_name, algo_name)
        print(f"\n{'#'*60}")
        print(f"  [{idx}/{len(algo_list)}] 벤치마크: {algo_name} ({kr})")
        print(f"{'#'*60}")
        try:
            args_copy = argparse.Namespace(**vars(args))
            args_copy.save_path = _next_version_path(
                f"models/benchmark/{algo_name.lower()}")
            # 벤치마크 시 각 알고리즘 전용 config 사용 (--config 무시)
            args_copy.config = None
            result = train_single(algo_name, args_copy)
            results[algo_name] = result
        except Exception as e:
            print(f"  [✗] {algo_name} ({kr}) 실패: {e}")
            results[algo_name] = {"error": str(e)}

    # 결과 저장
    os.makedirs("models/benchmark", exist_ok=True)
    with open("models/benchmark/results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)

    print(f"\n\n{'='*60}")
    print("  벤치마크 결과 (Benchmark Results)")
    print(f"{'='*60}")
    for name, res in results.items():
        kr = ALGO_KR.get(name, name)
        if "error" in res:
            print(f"  {name:12s} ({kr}): 실패 (FAILED) - {res['error']}")
        else:
            print(f"  {name:12s} ({kr}): {res.get('wall_time_sec', '?')}s")
    print(f"\n  결과 저장 완료: models/benchmark/results.json")

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
    p.add_argument("--days", type=int, default=None, choices=[30, 60],
                   help="게임 일수 (30 또는 60, 미지정 시 config 기본값)")

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
