"""RL Tycoon – entry point.

Launch the game in one of three modes:
    python main.py --mode human          # solo play
    python main.py --mode versus         # human vs AI
    python main.py --mode watch          # spectate trained AI
    python main.py --mode versus --model models/best_model.zip
    python main.py --mode watch  --model models/ppo/best_model.zip --speed 2
"""

import argparse
import sys


def main():
    parser = argparse.ArgumentParser(description="RL Tycoon Game")
    parser.add_argument(
        "--mode", choices=["human", "versus", "watch"], default="human",
        help="Game mode (default: human)")
    parser.add_argument(
        "--model", type=str, default=None,
        help="Path to a trained model (for versus/watch mode)")
    parser.add_argument(
        "--algo", type=str, default=None,
        help="Algorithm name for watch mode (PPO, DQN, A3C, SAC, etc.)")
    parser.add_argument(
        "--speed", type=float, default=1.0,
        help="Speed multiplier for watch mode (default: 1.0)")
    parser.add_argument(
        "--target-money", type=int, default=None,
        help="Target money to win (default: 5000)")
    parser.add_argument(
        "--day-limit", type=int, default=None,
        help="Number of in-game days (default: 30)")

    args = parser.parse_args()

    if args.mode == "human":
        from modes.human_mode import HumanMode
        game = HumanMode(target_money=args.target_money,
                         day_limit=args.day_limit)
    elif args.mode == "versus":
        from modes.versus_mode import VersusMode
        game = VersusMode(model_path=args.model,
                          target_money=args.target_money,
                          day_limit=args.day_limit)
    elif args.mode == "watch":
        from modes.watch_mode import WatchMode
        game = WatchMode(model_path=args.model,
                         algo_name=args.algo,
                         target_money=args.target_money,
                         day_limit=args.day_limit,
                         speed_multiplier=args.speed)
    else:
        print(f"Unknown mode: {args.mode}", file=sys.stderr)
        sys.exit(1)

    game.run()


if __name__ == "__main__":
    main()
