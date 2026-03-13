"""RL Tycoon – entry point.

Launch the game in one of two modes:
    python main.py --mode human          # solo play
    python main.py --mode ai             # AI solo observation
    python main.py --mode versus         # human vs AI
    python main.py --mode versus --model models/best_model.zip
"""

import argparse
import sys


def main():
    parser = argparse.ArgumentParser(description="RL Tycoon Game")
    parser.add_argument(
        "--mode", choices=["human", "ai", "versus"], default="human",
        help="Game mode (default: human)")
    parser.add_argument(
        "--model", type=str, default=None,
        help="Path to a trained SB3 model zip (for versus mode)")
    parser.add_argument(
        "--target-money", type=int, default=None,
        help="Target money to win (default: 5000)")
    parser.add_argument(
        "--day-limit", type=int, default=None,
        help="Number of in-game days (default: 30)")
    parser.add_argument(
        "--speed", type=float, default=1.0,
        help="Simulation speed multiplier for observation (default: 1.0)")
    parser.add_argument(
        "--rule-controller", action="store_true",
        help="Enable rule-based live controller overrides for stale carry/upgrades")

    args = parser.parse_args()

    if args.mode == "human":
        from modes.human_mode import HumanMode
        game = HumanMode(target_money=args.target_money,
                         day_limit=args.day_limit,
                         time_scale=args.speed)
    elif args.mode == "ai":
        from modes.ai_mode import AIMode
        game = AIMode(model_path=args.model,
                      target_money=args.target_money,
                      day_limit=args.day_limit,
                      time_scale=args.speed,
                      use_rule_controller=args.rule_controller)
    elif args.mode == "versus":
        from modes.versus_mode import VersusMode
        game = VersusMode(model_path=args.model,
                          target_money=args.target_money,
                          day_limit=args.day_limit,
                          time_scale=args.speed,
                          use_rule_controller=args.rule_controller)
    else:
        print(f"Unknown mode: {args.mode}", file=sys.stderr)
        sys.exit(1)

    game.run()


if __name__ == "__main__":
    main()
