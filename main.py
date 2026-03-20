"""RL Tycoon – entry point.

Launch the game in one of three modes:
    python main.py                           # interactive menu
    python main.py --mode human              # solo play (30 days default)
    python main.py --mode human --days 60    # solo play 60 days
    python main.py --mode versus             # human vs AI
    python main.py --mode versus --days 60   # 60-day versus
    python main.py --mode versus --model models/ppo/best_model.zip
    python main.py --mode watch  --model models/ppo/best_model.zip --speed 2
"""

import argparse
import math
import random
import sys


def _show_menu():
    """Pygame 기반 모드 선택 메뉴. (mode, days, model_path) 튜플 반환."""
    import pygame
    from rendering.asset_manager import AssetManager
    pygame.init()

    WIDTH, HEIGHT = 520, 490
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("RL 타이쿤 – 모드 선택")

    am = AssetManager()
    am.ensure_loaded()

    available = [f.lower() for f in pygame.font.get_fonts()]
    kr_font = None
    for fn in ["malgungothic", "gulim", "dotum", "nanumgothic"]:
        if fn in available:
            kr_font = fn
            break

    title_font = pygame.font.SysFont(kr_font, 36, bold=True)
    btn_font = pygame.font.SysFont(kr_font, 22)
    sub_font = pygame.font.SysFont(kr_font, 16)

    BG = (30, 30, 50)
    BTN_COLOR = (60, 80, 120)
    BTN_HOVER = (80, 110, 170)
    TEXT_COLOR = (255, 255, 255)
    ACCENT = (255, 215, 0)

    buttons = [
        {"label": "[ 솔로 모드 (30일) ]", "mode": "human", "days": 30, "rect": None},
        {"label": "[ 솔로 모드 (60일) ]", "mode": "human", "days": 60, "rect": None},
        {"label": "[ 대결 모드 (30일) ]", "mode": "versus", "days": 30, "rect": None},
        {"label": "[ 대결 모드 (60일) ]", "mode": "versus", "days": 60, "rect": None},
        {"label": "[ 토너먼트 모드 ]", "mode": "tournament", "days": 30, "rect": None},
    ]

    BTN_W, BTN_H = 360, 52
    START_Y = 120
    GAP = 16

    for i, btn in enumerate(buttons):
        x = (WIDTH - BTN_W) // 2
        y = START_Y + i * (BTN_H + GAP)
        btn["rect"] = pygame.Rect(x, y, BTN_W, BTN_H)

    clock = pygame.time.Clock()
    running = True
    result = None
    chef_pool = [img for idx, img in enumerate(am.chef_images) if idx != 5]
    falling_chefs: list[dict] = []

    def spawn_chef(spawn_y: float | None = None):
        if not chef_pool:
            return
        img = random.choice(chef_pool)
        scale = random.uniform(1.35, 2.05)
        falling_chefs.append({
            "image": img,
            "x": random.uniform(20, WIDTH - 20),
            "y": spawn_y if spawn_y is not None else random.uniform(-HEIGHT, -64),
            "speed_y": random.uniform(60.0, 135.0),
            "drift": random.uniform(-30.0, 30.0),
            "angle": random.uniform(0.0, 360.0),
            "spin": random.uniform(-220.0, 220.0),
            "scale": scale,
        })

    for _ in range(10):
        spawn_chef()

    while running:
        dt = clock.tick(30) / 1000.0
        mouse_pos = pygame.mouse.get_pos()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit(0)
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                pygame.quit()
                sys.exit(0)
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                for btn in buttons:
                    if btn["rect"].collidepoint(mouse_pos):
                        result = (btn["mode"], btn["days"], None)
                        running = False
                        break

        if len(falling_chefs) < 12:
            spawn_chef(-80.0)

        for chef in falling_chefs:
            chef["y"] += chef["speed_y"] * dt
            chef["x"] += math.sin(chef["y"] * 0.02) * 8.0 * dt + chef["drift"] * dt
            chef["angle"] = (chef["angle"] + chef["spin"] * dt) % 360.0
            if chef["y"] > HEIGHT + 100:
                chef.update({
                    "x": random.uniform(20, WIDTH - 20),
                    "y": random.uniform(-220, -80),
                    "speed_y": random.uniform(60.0, 135.0),
                    "drift": random.uniform(-30.0, 30.0),
                    "angle": random.uniform(0.0, 360.0),
                    "spin": random.uniform(-220.0, 220.0),
                    "scale": random.uniform(1.35, 2.05),
                })

        falling_chefs.sort(key=lambda item: item["scale"])

        screen.fill(BG)

        for chef in falling_chefs:
            base = chef["image"]
            scaled_size = max(24, int(base.get_width() * chef["scale"]))
            scaled = pygame.transform.smoothscale(base, (scaled_size, scaled_size))
            rotated = pygame.transform.rotozoom(scaled, chef["angle"], 1.0)
            rect = rotated.get_rect(center=(int(chef["x"]), int(chef["y"])))
            screen.blit(rotated, rect)

        # Title
        title_shadow = title_font.render("RL 타이쿤", True, (40, 20, 10))
        screen.blit(title_shadow,
                     (WIDTH // 2 - title_shadow.get_width() // 2 + 2, 32))
        title_surf = title_font.render("RL 타이쿤", True, ACCENT)
        screen.blit(title_surf,
                     (WIDTH // 2 - title_surf.get_width() // 2, 30))
        sub_surf = sub_font.render("모드를 선택하세요", True, (180, 180, 200))
        screen.blit(sub_surf,
                     (WIDTH // 2 - sub_surf.get_width() // 2, 78))

        # Buttons
        for btn in buttons:
            hovered = btn["rect"].collidepoint(mouse_pos)
            color = BTN_HOVER if hovered else BTN_COLOR
            pygame.draw.rect(screen, color, btn["rect"], border_radius=10)
            pygame.draw.rect(screen, (100, 130, 180), btn["rect"],
                             width=2, border_radius=10)
            lbl = btn_font.render(btn["label"], True, TEXT_COLOR)
            screen.blit(lbl, (btn["rect"].centerx - lbl.get_width() // 2,
                              btn["rect"].centery - lbl.get_height() // 2))

        # Footer
        foot = sub_font.render("ESC: 종료  |  대결 모드에서 --model로 AI 모델 지정 가능",
                               True, (120, 120, 140))
        screen.blit(foot, (WIDTH // 2 - foot.get_width() // 2, HEIGHT - 36))

        pygame.display.flip()

    pygame.quit()
    return result


def main():
    parser = argparse.ArgumentParser(description="RL Tycoon Game")
    parser.add_argument(
        "--mode", choices=["human", "versus", "watch", "tournament"], default=None,
        help="Game mode (default: interactive menu)")
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
        help="Target money to win (default: 1500)")
    parser.add_argument(
        "--day-limit", type=int, default=None,
        help="Number of in-game days (default: 30)")
    parser.add_argument(
        "--days", type=int, default=None, choices=[30, 60],
        help="Shorthand for --day-limit (30 or 60)")

    args = parser.parse_args()

    # --days shorthand takes priority if --day-limit not explicitly set
    if args.days and args.day_limit is None:
        args.day_limit = args.days

    # No --mode given → show interactive menu
    if args.mode is None:
        menu_result = _show_menu()
        if menu_result is None:
            sys.exit(0)
        args.mode, menu_days, _ = menu_result
        if args.day_limit is None:
            args.day_limit = menu_days

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
    elif args.mode == "tournament":
        from modes.tournament_mode import TournamentMode
        game = TournamentMode(target_money=args.target_money,
                              day_limit=args.day_limit,
                              speed_multiplier=args.speed)
    else:
        print(f"Unknown mode: {args.mode}", file=sys.stderr)
        sys.exit(1)

    game.run()


if __name__ == "__main__":
    main()
