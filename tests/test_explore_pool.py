"""The forced-exploration pool (sf2/system2/explore_pool.py): grounded offensive rules the live policy
falls back on when the Coach is silent. pick() must skip lines COVERED by one already in play (not
exact-string) and return a claim (so it enters as `trying`, never `verified`)."""
from sf2.system2 import explore_pool as explore

MOVES = {"s.mk", "walk_forward", "s.hk", "throw_F+hp", "s.mp", "walk_back"}


def test_pick_returns_a_claim_not_a_line():
    c = explore.pick([], MOVES, rotate=0)
    assert isinstance(c, dict) and c.get("kind") and c.get("move")      # a claim dict, ready to admit as trying


def test_pick_skips_a_pool_line_already_covered_in_play():
    # the kit's range-agnostic 'use more s.mk when he stands' COVERS pool[0] 'use more s.mk at mid range ...'
    kit = ["use more s.mk when he stands"]
    c = explore.pick(kit, MOVES, rotate=0)
    assert c is not None and c["move"] != "s.mk"                        # must not re-offer a covered s.mk line


def test_pick_none_when_pool_exhausted():
    in_play = list(explore.EXPLORE_POOL)
    assert explore.pick(in_play, MOVES, rotate=0) is None


def test_pick_rotates_through_distinct_rules():
    seen = {explore.render_line(explore.pick([], MOVES, rotate=r)) for r in range(len(explore.EXPLORE_POOL))}
    assert len(seen) == len(explore.EXPLORE_POOL)                       # every rotation offset yields a distinct rule
