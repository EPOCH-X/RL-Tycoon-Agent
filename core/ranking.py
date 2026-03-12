"""Ranking system – local JSON storage with future server connection interface.

Tracks game results (money earned within day_limit) and maintains
a local leaderboard. The server API stubs are ready for future integration.
"""

import json
import os
import time

from config.settings import BASE_DIR

RANKING_FILE = os.path.join(BASE_DIR, "data", "rankings.json")


class RankingManager:
    """Manages local rankings and prepares server API interface."""

    def __init__(self):
        self.rankings: list[dict] = self._load()

    # ── persistence ──────────────────────────────
    def _load(self) -> list[dict]:
        if os.path.exists(RANKING_FILE):
            with open(RANKING_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        return []

    def _save(self):
        os.makedirs(os.path.dirname(RANKING_FILE), exist_ok=True)
        with open(RANKING_FILE, "w", encoding="utf-8") as f:
            json.dump(self.rankings, f, ensure_ascii=False, indent=2)

    # ── record & query ───────────────────────────
    def record_result(self, player_name: str, game_result: dict) -> dict:
        """Record a finished game result. Returns the entry."""
        entry = {
            "player_name": player_name,
            "money": game_result["money"],
            "net_profit": game_result["net_profit"],
            "day_limit": game_result["day_limit"],
            "customers_served": game_result["customers_served"],
            "customers_lost": game_result["customers_lost"],
            "shop_rating": game_result["shop_rating"],
            "won": game_result["won"],
            "timestamp": int(time.time()),
        }
        self.rankings.append(entry)
        self._save()
        return entry

    def get_rankings(self, day_limit: int | None = None,
                     top_n: int = 10) -> list[dict]:
        """Get top rankings, optionally filtered by day_limit."""
        filtered = self.rankings
        if day_limit is not None:
            filtered = [r for r in filtered if r["day_limit"] == day_limit]
        filtered = sorted(filtered, key=lambda x: x["money"], reverse=True)
        return filtered[:top_n]

    def get_rank(self, money: int, day_limit: int) -> int:
        """Get the rank position for a given money amount."""
        rankings = self.get_rankings(day_limit=day_limit, top_n=10000)
        for i, r in enumerate(rankings):
            if money >= r["money"]:
                return i + 1
        return len(rankings) + 1

    # ── server API stubs (future integration) ────
    def submit_to_server(self, entry: dict) -> bool:
        """Submit ranking to remote server.

        TODO: Implement HTTP POST to ranking server.
        Expected endpoint: POST /api/rankings
        Body: entry dict (player_name, money, day_limit, etc.)
        Returns: True on success.
        """
        return False

    def fetch_from_server(self, day_limit: int | None = None,
                          top_n: int = 10) -> list[dict]:
        """Fetch global rankings from remote server.

        TODO: Implement HTTP GET from ranking server.
        Expected endpoint: GET /api/rankings?day_limit={day_limit}&top={top_n}
        Returns: list of ranking entries.
        """
        return []

    def sync_with_server(self):
        """Sync local rankings with server (upload unsynced, download global).

        TODO: Implement bidirectional sync.
        1. Upload local entries without 'synced' flag
        2. Download global top rankings
        3. Merge into local display
        """
        pass
