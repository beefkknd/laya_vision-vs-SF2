"""Combos are GROUND-LAUNCHED jump-in macros (sf2.moves_free: kind 'combo', setup 'far', script = jump-in + buttons),
but their move NAMES carry the air prefix 'jf.', so the stance law mis-filed them as air moves -> unreachable for
every policy (decisions are only taken grounded), which is why the 71% table never used a combo. Fix: the `combo`
category is offered GROUNDED (the 'standing' stance = grounded at mid/far, where a jump-in starts), not by its prefix,
and is not stance-voided. Owner 2026-10-05, after the quorum review found the combo bee voted 0/2363 times."""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from sf2.system1.advice import char_categories, followable, moves_in_stance, stance_unreliable  # noqa: E402

COMBOS = {"jf.mk_legs", "jf.hk_s.mp_s.hp"}
CATS = char_categories("chunli")


def test_combos_offered_grounded_at_mid_far_not_air_or_close():
    assert set(moves_in_stance("combo", "standing", CATS)) == COMBOS   # grounded mid/far = where a jump-in starts
    assert moves_in_stance("combo", "close", CATS) == []               # too close for a jump-in
    assert moves_in_stance("combo", "crouch", CATS) == []
    assert moves_in_stance("combo", "air", CATS) == []                 # NOT air moves anymore (the old mis-filing)


def test_combos_are_followable_at_mid_far_not_close():
    assert followable("jf.mk_legs", "far")
    assert followable("jf.hk_s.mp_s.hp", "mid")
    assert not followable("jf.mk_legs", "close")


def test_combos_are_not_stance_voided_but_real_air_normals_still_are():
    assert not stance_unreliable("jf.mk_legs")        # ground-launched -> she IS grounded -> reliable
    assert not stance_unreliable("jf.hk_s.mp_s.hp")
    assert stance_unreliable("j.hk")                  # a genuine air normal still voids
    assert stance_unreliable("c.mk")                  # a crouch normal still voids
