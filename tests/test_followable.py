"""Enforceability oracle (sf2.system1.advice.followable): a rule text-laya can never PLAY at its range is
unfollowable and must never be learned. This is the root cause of chun3's dead rules ('s.* up close' is never
offered -- close stance gives only 'cl.*'). Measured, not guessed."""
from sf2.system1.advice import followable


def test_standing_normal_is_unfollowable_up_close():
    assert followable("s.mk", "close", "chunli") is False        # close offers only cl.* -> dead
    assert followable("s.mk", "mid", "chunli") is True
    assert followable("s.lp", "close", "chunli") is False
    assert followable("s.hk", "close", "chunli") is False        # the stance-inert explore-pool line


def test_range_agnostic_followable_if_offered_anywhere():
    assert followable("s.mk", None, "chunli") is True            # offered at mid/far
    assert followable("throw_F+mp", None, "chunli") is True      # offered up close


def test_close_normal_and_throw_ranges():
    assert followable("cl.hk", "close", "chunli") is True        # the close-valid replacement
    assert followable("throw_F+mp", "close", "chunli") is True
    assert followable("throw_F+mp", "far", "chunli") is False    # throw only up close


def test_movement_and_special_are_followable_grounded():
    assert followable("walk_forward", "close", "chunli") is True
    assert followable("lightning_legs", "mid", "chunli") is True


def test_unknown_move_is_not_followable():
    assert followable("sweep", "mid", "chunli") is False         # not in her menu at all
