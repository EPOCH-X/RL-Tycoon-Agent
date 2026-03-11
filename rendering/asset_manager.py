"""AssetManager – loads sprite sheets and serves animation frames.

Phase 1: no sprites exist → ``has_sprite()`` always returns False and
         entities fall back to their coloured-rectangle rendering.
Phase 3: drop sprite sheets into ``assets/sprites/<entity>/<state>.png``
         and they are auto-discovered at startup.  Each PNG is a horizontal
         strip that is split into TILE_SIZE × TILE_SIZE frames.
"""

import os
import pygame
from config.settings import TILE_SIZE, ASSETS_DIR


class AssetManager:
    """Discovers, loads, and caches sprite sheets."""

    def __init__(self, assets_dir: str | None = None):
        self.assets_dir = assets_dir or ASSETS_DIR
        # {sprite_key: {state_name: [pygame.Surface, …]}}
        self.sprites: dict[str, dict[str, list[pygame.Surface]]] = {}
        self._load_all()

    # ── public API ───────────────────────────────
    def has_sprite(self, sprite_key: str, state: str = "idle") -> bool:
        return (sprite_key in self.sprites
                and state in self.sprites[sprite_key])

    def get_frame(self, sprite_key: str, state: str,
                  frame_index: int) -> pygame.Surface:
        frames = self.sprites[sprite_key][state]
        return frames[frame_index % len(frames)]

    # ── loader ───────────────────────────────────
    def _load_all(self):
        sprites_dir = os.path.join(self.assets_dir, "sprites")
        if not os.path.isdir(sprites_dir):
            return

        for entity_name in os.listdir(sprites_dir):
            entity_path = os.path.join(sprites_dir, entity_name)
            if not os.path.isdir(entity_path):
                continue

            self.sprites[entity_name] = {}
            for fname in os.listdir(entity_path):
                if not fname.lower().endswith(".png"):
                    continue
                state_name = os.path.splitext(fname)[0]
                full = os.path.join(entity_path, fname)
                sheet = pygame.image.load(full).convert_alpha()
                frames = self._split_sheet(sheet)
                self.sprites[entity_name][state_name] = frames

    @staticmethod
    def _split_sheet(sheet: pygame.Surface) -> list[pygame.Surface]:
        """Split a horizontal sprite sheet into TILE_SIZE frames."""
        w, h = sheet.get_size()
        frames: list[pygame.Surface] = []
        for x in range(0, w, TILE_SIZE):
            fw = min(TILE_SIZE, w - x)
            sub = sheet.subsurface(pygame.Rect(x, 0, fw, h))
            if sub.get_size() != (TILE_SIZE, TILE_SIZE):
                sub = pygame.transform.scale(sub, (TILE_SIZE, TILE_SIZE))
            frames.append(sub)
        return frames if frames else [sheet]
