"""RL Tycoon entry point."""

import argparse
import sys


def main():
    parser = argparse.ArgumentParser(description="RL Tycoon Game")
    parser.add_argument(
        "--mode", choices=["human", "ai", "versus", "watch"], default="human",
        help="Game mode (default: human)")
    parser.add_argument(
        "--model", type=str, default=None,
        help="Path to a trained model")
    parser.add_argument(
        "--algo", type=str, default=None,
        help="Algorithm name for watch mode (PPO, DQN, A3C, SAC, etc.)")
    parser.add_argument(
        "--target-money", type=int, default=None,
        help="Target money to win (default: 5000)")
    parser.add_argument(
        "--day-limit", type=int, default=None,
        help="Number of in-game days (default: 30)")
    parser.add_argument(
        "--speed", type=float, default=1.0,
        help="Simulation speed multiplier (default: 1.0)")
    parser.add_argument(
        "--rule-controller", action="store_true",
        help="Enable rule-based live controller overrides for stale carry/upgrades")

    args = parser.parse_args()

    if args.mode == "human":
        from modes.human_mode import HumanMode
        game = HumanMode(
            target_money=args.target_money,
            day_limit=args.day_limit,
            time_scale=args.speed,
        )
    elif args.mode == "ai":
        from modes.ai_mode import AIMode
        game = AIMode(
            model_path=args.model,
            target_money=args.target_money,
            day_limit=args.day_limit,
            time_scale=args.speed,
            use_rule_controller=args.rule_controller,
        )
    elif args.mode == "versus":
        from modes.versus_mode import VersusMode
        game = VersusMode(
            model_path=args.model,
            target_money=args.target_money,
            day_limit=args.day_limit,
            time_scale=args.speed,
            use_rule_controller=args.rule_controller,
        )
    elif args.mode == "watch":
        from modes.watch_mode import WatchMode
        game = WatchMode(
            model_path=args.model,
            algo_name=args.algo,
            target_money=args.target_money,
            day_limit=args.day_limit,
            speed_multiplier=args.speed,
        )
    else:
        print(f"Unknown mode: {args.mode}", file=sys.stderr)
        sys.exit(1)

    game.run()


if __name__ == "__main__":
    main()
