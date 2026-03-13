"""Rule-based control overrides for live AI gameplay."""

from config.settings import (
    ACTION_BUY_UPGRADE,
    ACTION_INTERACT,
    ACTION_UP,
    ACTION_DOWN,
    ACTION_LEFT,
    ACTION_RIGHT,
    INTERACT_RANGE,
)


def decide_override_action(shop):
    """Return a forced action for operational edge cases, or None."""
    if shop.has_stale_carry():
        return move_or_interact_to(shop, *shop._trash_center())

    if shop.should_auto_buy_now():
        return ACTION_BUY_UPGRADE

    return None


def move_or_interact_to(shop, tx: float, ty: float) -> int:
    """Move toward a station and interact once in range."""
    px = shop.player.center_x
    py = shop.player.center_y
    dx = tx - px
    dy = ty - py
    if abs(dx) <= INTERACT_RANGE * 0.5 and abs(dy) <= INTERACT_RANGE * 0.5:
        return ACTION_INTERACT
    if abs(dx) > abs(dy):
        return ACTION_RIGHT if dx > 0 else ACTION_LEFT
    return ACTION_DOWN if dy > 0 else ACTION_UP
