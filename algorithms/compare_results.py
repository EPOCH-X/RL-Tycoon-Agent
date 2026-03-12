"""성능 비교 도구 – 저장된 벤치마크 결과를 시각화합니다.

Usage:
    python -m algorithms.compare_results
    python -m algorithms.compare_results --results-dir models/benchmark
"""

import argparse
import json
import os


def load_results(results_dir: str) -> dict:
    """벤치마크 결과 JSON을 로드합니다."""
    path = os.path.join(results_dir, "results.json")
    if not os.path.isfile(path):
        print(f"결과 파일을 찾을 수 없습니다: {path}")
        print("먼저 벤치마크를 실행하세요:")
        print("  python -m algorithms.train_launcher --benchmark")
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def print_table(results: dict):
    """결과를 테이블로 출력합니다."""
    if not results:
        return

    header = f"{'Algorithm':>12s} | {'Status':>8s} | {'Time(s)':>10s} | {'Timesteps':>12s}"
    print(f"\n{'='*60}")
    print("  알고리즘 벤치마크 결과 비교")
    print(f"{'='*60}")
    print(header)
    print("-" * 60)

    for name, res in results.items():
        if "error" in res:
            print(f"{name:>12s} | {'FAILED':>8s} | {'N/A':>10s} | {'N/A':>12s}")
        else:
            status = "OK"
            wall_time = res.get("wall_time_sec", "?")
            steps = res.get("timesteps", "?")
            print(f"{name:>12s} | {status:>8s} | {wall_time:>10s} | {steps:>12s}")

    print(f"{'='*60}\n")


def try_plot(results: dict):
    """matplotlib이 있으면 차트를 생성합니다."""
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("matplotlib이 설치되어 있지 않아 차트를 건너뜁니다.")
        return

    algos = []
    times = []
    for name, res in results.items():
        if "error" not in res and "wall_time_sec" in res:
            algos.append(name)
            times.append(res["wall_time_sec"])

    if not algos:
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.barh(algos, times, color=[
        "#2196F3", "#4CAF50", "#FF9800", "#E91E63", "#9C27B0", "#00BCD4"
    ][:len(algos)])
    ax.set_xlabel("Wall Time (seconds)")
    ax.set_title("RL Algorithm Training Time Comparison")
    for bar, t in zip(bars, times):
        ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
                f"{t:.1f}s", va="center")
    plt.tight_layout()
    plt.savefig("models/benchmark/comparison.png", dpi=150)
    print("  차트 저장됨: models/benchmark/comparison.png")


def main():
    p = argparse.ArgumentParser(description="벤치마크 결과 비교")
    p.add_argument("--results-dir", type=str, default="models/benchmark",
                   help="결과 디렉토리")
    args = p.parse_args()

    results = load_results(args.results_dir)
    print_table(results)
    try_plot(results)


if __name__ == "__main__":
    main()
