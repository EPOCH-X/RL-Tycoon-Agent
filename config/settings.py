"""Game-wide constants and configuration helpers."""

import os
import json

# ──────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_DIR = os.path.join(BASE_DIR, "config")
ASSETS_DIR = os.path.join(BASE_DIR, "assets")

# ──────────────────────────────────────────────
# Display
# ──────────────────────────────────────────────
TILE_SIZE = 64          # pixels per grid cell (sprite target size)
FPS = 60                # rendering frame rate
UI_HEIGHT = 120         # pixel height of the bottom UI panel

# ──────────────────────────────────────────────
# Game Tick
# ──────────────────────────────────────────────
STEP_INTERVAL = 0.2     # seconds per game step (5 steps/sec)

# ──────────────────────────────────────────────
# Game Rules – Restaurant Management Sim
# ──────────────────────────────────────────────
DEFAULT_TARGET_MONEY = 1500
DEFAULT_DAY_LIMIT = 30          # in-game days
DAY_LENGTH = 60.0               # real seconds per in-game day

CUSTOMER_SPAWN_INTERVAL = 8.0   # base seconds between spawns
MAX_CUSTOMERS = 4               # max simultaneous seated customers
MAX_WAITING_QUEUE = 6           # max customers waiting outside for a table
WAITING_PATIENCE = 30.0         # seconds a customer will wait outside before leaving
KITCHEN_CAPACITY = 1            # initial number of chefs (1 chef = 1 dish at a time)

# Player movement (distance-based, not grid)
PLAYER_SPEED = 180              # pixels per second
PLAYER_RADIUS = 18              # collision half-width (pixels)
INTERACT_RANGE = 80             # max pixel distance for interaction

# Employee
EMPLOYEE_SPEED = 120            # pixels per second (slower than player)
EMPLOYEE_ACTION_DELAY = 0.8     # seconds to complete an action (take order, etc.)

# Customer
CUSTOMER_WALK_SPEED = 80        # pixels per second (walk to table)

# Satisfaction
SATISFACTION_HISTORY_LEN = 20   # rolling window for shop rating
LOST_CUSTOMER_PENALTY = 10      # money penalty when customer leaves angry
SATISFACTION_FAST_THRESHOLD = 0.6   # patience ratio above this → "fast"
SATISFACTION_SLOW_THRESHOLD = 0.3   # below this → "slow"

# ──────────────────────────────────────────────
# Actions
# ──────────────────────────────────────────────
ACTION_UP = 0
ACTION_DOWN = 1
ACTION_LEFT = 2
ACTION_RIGHT = 3
ACTION_INTERACT = 4
ACTION_NONE = 5
ACTION_BUY_UPGRADE = 6
NUM_ACTIONS = 7

# ──────────────────────────────────────────────
# Colors (Phase 1 placeholder palette)
# ──────────────────────────────────────────────
COLORS = {
    "background":       (40,  40,  40),
    "floor":            (200, 200, 180),
    "wall":             (80,  80,  80),
    "grid_line":        (60,  60,  60),
    "player":           (50,  120, 220),
    "player_carry":     (80,  160, 255),
    "customer":         (220, 180, 50),
    "customer_angry":   (220, 80,  50),
    "customer_wealthy": (180, 120, 220),
    "customer_vip":     (255, 215, 0),
    "customer_tourist": (100, 180, 220),
    "customer_critic":  (220, 50,  50),
    "table":            (139, 90,  43),
    "table_occupied":   (160, 110, 60),
    "kitchen":          (180, 60,  60),
    "kitchen_cooking":  (220, 100, 50),
    "kitchen_ready":    (80,  220, 80),
    "bar":              (100, 60,  140),
    "bar_ready":        (160, 100, 220),
    "employee":         (80,  200, 160),
    "employee_carry":   (120, 240, 190),
    "delivery":         (200, 140, 60),
    "trash_can":        (120, 100, 80),
    "trash_can_active": (160, 140, 100),
    "customer_waiting": (180, 160, 50),
    "text":             (255, 255, 255),
    "ui_bg":            (30,  30,  50),
    "money":            (255, 215, 0),
    "satisfaction":     (100, 220, 100),
    "timer":            (200, 200, 200),
}

# ──────────────────────────────────────────────
# Versus mode
# ──────────────────────────────────────────────
VERSUS_DIVIDER_WIDTH = 4
VERSUS_DIVIDER_COLOR = (200, 200, 200)


def load_json_config(filename: str):
    """Load a JSON configuration file from the config directory."""
    filepath = os.path.join(CONFIG_DIR, filename)
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)
