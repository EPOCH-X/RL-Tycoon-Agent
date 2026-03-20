"""AssetManager – loads all game images from data/images/.

All images are 64×64 PNG files.  Each entity type has multiple states,
and each state may have multiple animation frames (numbered suffix).

Directory layout (data/images/):
  agent/             – walk_front1..4, walk_back1..4, idle, etc.
  background_sample/ – sample1.png .. sample3.png
  customer/{type}/   – sit_eating, sit_wait, stand_wait, walk, exit
  drink/             – water.png, jucie.png, lemonade.png, cocktail.png, wine.png
  employee_man/      – idle, down1-2, up1-3, left1-2, right1-2
  employee_woman/    – same
  food/              – coffee.png, sandwich.png, pasta.png, steak.png, …
  furniture/         – floor, wall, table, chair, bar, trash_can, 주방, 주방벽면1, 주방벽면2
  player/            – idle, down1-2, up1-3, left1-2, right1-2
  요리사/            – 1.png .. 6.png  (chef upgrade order)
"""

import os
import pygame
from config.settings import TILE_SIZE

# ── Image base path ──
_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMAGES_DIR = os.path.join(_BASE_DIR, "data", "images")


def _load_img(path: str, size: tuple[int, int] = (TILE_SIZE, TILE_SIZE)) -> pygame.Surface:
    """Load a single PNG and scale to *size*."""
    img = pygame.image.load(path).convert_alpha()
    if img.get_size() != size:
        img = pygame.transform.smoothscale(img, size)
    return img


def _collect_frames(folder: str, prefix: str,
                    size: tuple[int, int] = (TILE_SIZE, TILE_SIZE)
                    ) -> list[pygame.Surface]:
    """Collect numbered frames: <prefix>1.png, <prefix>2.png, …
    Falls back to <prefix>.png (single frame) if no numbered files exist.
    """
    frames: list[pygame.Surface] = []
    idx = 1
    while True:
        p = os.path.join(folder, f"{prefix}{idx}.png")
        if os.path.isfile(p):
            frames.append(_load_img(p, size))
            idx += 1
        else:
            break
    if not frames:
        p = os.path.join(folder, f"{prefix}.png")
        if os.path.isfile(p):
            frames.append(_load_img(p, size))
    return frames


class AssetManager:
    """Central image cache for the entire game."""

    def __init__(self, assets_dir: str | None = None):
        self.images_dir = assets_dir or IMAGES_DIR

        # ----- Tile images (furniture) -----
        self.tile_floor: pygame.Surface | None = None
        self.tile_wall: pygame.Surface | None = None
        self.tile_table: pygame.Surface | None = None
        self.tile_chair: pygame.Surface | None = None
        self.tile_bar: pygame.Surface | None = None
        self.tile_bar_wide: pygame.Surface | None = None   # 128×64 for hired bartender
        self.tile_trash: pygame.Surface | None = None
        self.tile_kitchen: pygame.Surface | None = None       # 주방.png
        self.tile_kitchen_wall1: pygame.Surface | None = None  # 주방벽면1
        self.tile_kitchen_wall2: pygame.Surface | None = None  # 주방벽면2

        # ----- Backgrounds (per mode) -----
        self.backgrounds: dict[str, pygame.Surface] = {}

        # ----- Character sprites -----
        # {sprite_key: {state_name: [Surface, …]}}
        self.sprites: dict[str, dict[str, list[pygame.Surface]]] = {}

        # ----- Food / Drink icons (32×32) -----
        self.food_icons: dict[str, pygame.Surface] = {}
        self.drink_icons: dict[str, pygame.Surface] = {}

        # ----- Chef (요리사) images 1-6 -----
        self.chef_images: list[pygame.Surface] = []

        self._loaded = False

    # ═══════════════════════════════════════════════
    #  Public API
    # ═══════════════════════════════════════════════
    def ensure_loaded(self):
        """Call after pygame.init() / display set to load all images."""
        if self._loaded:
            return
        self._loaded = True
        self._load_furniture()
        self._load_backgrounds()
        self._load_characters()
        self._load_food_drink()
        self._load_chefs()

    def has_sprite(self, sprite_key: str, state: str = "idle") -> bool:
        sprite_key = sprite_key.lower()
        state = state.lower()
        return (sprite_key in self.sprites
                and state in self.sprites[sprite_key])

    def get_frame(self, sprite_key: str, state: str,
                  frame_index: int) -> pygame.Surface:
        sprite_key = sprite_key.lower()
        state = state.lower()
        frames = self.sprites[sprite_key][state]
        return frames[frame_index % len(frames)]

    def get_background(self, name: str) -> pygame.Surface | None:
        return self.backgrounds.get(name)

    def get_food_icon(self, food_id: str) -> pygame.Surface | None:
        return self.food_icons.get(food_id)

    def get_drink_icon(self, drink_id: str) -> pygame.Surface | None:
        return self.drink_icons.get(drink_id)

    def get_chef_image(self, index: int) -> pygame.Surface | None:
        """Get chef image by 0-based index (upgrade order)."""
        if 0 <= index < len(self.chef_images):
            return self.chef_images[index]
        return None

    # ═══════════════════════════════════════════════
    #  Loaders
    # ═══════════════════════════════════════════════
    def _load_furniture(self):
        d = os.path.join(self.images_dir, "furniture")
        if not os.path.isdir(d):
            return
        mapping = {
            "floor": "floor.png",
            "wall": "wall.png",
            "table": "table.png",
            "chair": "chair.png",
            "bar": "bar.png",
            "trash": "trash_can.png",
            "kitchen": "주방.png",
            "kitchen_wall1": "주방벽면1.png",
            "kitchen_wall2": "주방벽면2.png",
        }
        for attr, fname in mapping.items():
            p = os.path.join(d, fname)
            if os.path.isfile(p):
                setattr(self, f"tile_{attr}", _load_img(p))
        # Wide bar image (128×64) for hired bartender spanning 2 tiles
        bar_path = os.path.join(d, "bar.png")
        if os.path.isfile(bar_path):
            self.tile_bar_wide = _load_img(bar_path, (TILE_SIZE * 2, TILE_SIZE))

    def _load_backgrounds(self):
        d = os.path.join(self.images_dir, "background_sample")
        if not os.path.isdir(d):
            return
        for fname in os.listdir(d):
            if fname.lower().endswith(".png"):
                name = os.path.splitext(fname)[0]
                self.backgrounds[name] = pygame.image.load(
                    os.path.join(d, fname)).convert_alpha()

    def _load_characters(self):
        """Load player, agent, employees, and all customer types."""
        # --- Player (direction-based: idle, down, up, left, right) ---
        player_dir = os.path.join(self.images_dir, "player")
        if os.path.isdir(player_dir):
            states: dict[str, list[pygame.Surface]] = {}
            idle_frames = _collect_frames(player_dir, "idle")
            if idle_frames:
                states["idle"] = idle_frames
            for direction in ("down", "up", "left", "right"):
                frames = _collect_frames(player_dir, direction)
                if frames:
                    states[direction] = frames
            if "down" in states:
                states["action"] = states["down"]
            self.sprites["player"] = states

        # --- Agent (customer-like: walk_front, sit_eating_front, etc.) ---
        agent_dir = os.path.join(self.images_dir, "agent")
        if os.path.isdir(agent_dir):
            states = {}
            _AGENT_PREFIXES = [
                "walk_front", "walk_back",
                "sit_wait_order_front", "sit_wait_food_front",
                "sit_eating_front",
                "stand_wait_order_back",
                "exit_happy_front", "exit_angry_front",
            ]
            for prefix in _AGENT_PREFIXES:
                frames = _collect_frames(agent_dir, prefix)
                if frames:
                    states[prefix] = frames
            # Convenience aliases for directional rendering
            if "walk_front" in states:
                states["down"] = states["walk_front"]
                states["action"] = states["walk_front"]
            if "walk_back" in states:
                states["up"] = states["walk_back"]
            self.sprites["agent"] = states

        # --- Employees (direction-based like player) ---
        for emp_key, folder_name in (("employee_man", "employee_man"),
                                      ("employee_woman", "employee_woman")):
            folder = os.path.join(self.images_dir, folder_name)
            if not os.path.isdir(folder):
                continue
            states = {}
            idle_frames = _collect_frames(folder, "idle")
            if idle_frames:
                states["idle"] = idle_frames
            for direction in ("down", "up", "left", "right"):
                frames = _collect_frames(folder, direction)
                if frames:
                    states[direction] = frames
            if "down" in states:
                states["action"] = states["down"]
            self.sprites[emp_key] = states

        # --- Customers (per type subfolder) ---
        cust_base = os.path.join(self.images_dir, "customer")
        if not os.path.isdir(cust_base):
            return
        _CUSTOMER_PREFIXES = [
            "walk_front", "walk_back",
            "sit_wait_order_front", "sit_wait_food_front",
            "sit_eating_front", "sit_eating_back",
            "stand_wait_order_front", "stand_wait_order_back",
            "exit_happy_front", "exit_angry_front",
        ]
        for type_folder in os.listdir(cust_base):
            type_path = os.path.join(cust_base, type_folder)
            if not os.path.isdir(type_path):
                continue
            sprite_key = f"customer_{type_folder}"
            states = {}
            for prefix in _CUSTOMER_PREFIXES:
                frames = _collect_frames(type_path, prefix)
                if frames:
                    states[prefix] = frames
            self.sprites[sprite_key] = states

    def _load_food_drink(self):
        """Load food and drink icon images (32×32 for overlay)."""
        icon_size = (32, 32)
        food_dir = os.path.join(self.images_dir, "food")
        if os.path.isdir(food_dir):
            for fname in os.listdir(food_dir):
                if not fname.lower().endswith(".png"):
                    continue
                food_id = os.path.splitext(fname)[0].lower()
                self.food_icons[food_id] = _load_img(
                    os.path.join(food_dir, fname), icon_size)

        drink_dir = os.path.join(self.images_dir, "drink")
        if os.path.isdir(drink_dir):
            for fname in os.listdir(drink_dir):
                if not fname.lower().endswith(".png"):
                    continue
                drink_id = os.path.splitext(fname)[0].lower()
                if drink_id == "jucie":
                    drink_id = "juice"
                self.drink_icons[drink_id] = _load_img(
                    os.path.join(drink_dir, fname), icon_size)

    def _load_chefs(self):
        """Load chef images 1.png..6.png in upgrade order."""
        d = os.path.join(self.images_dir, "요리사")
        if not os.path.isdir(d):
            return
        for i in range(1, 7):
            p = os.path.join(d, f"{i}.png")
            if os.path.isfile(p):
                self.chef_images.append(_load_img(p))
