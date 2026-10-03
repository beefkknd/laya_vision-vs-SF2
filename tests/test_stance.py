"""The stance law in one place: a move she can't reliably be in the stance for ON HIS CUE (a crouch
normal c.* or a jump/air attack j./jf.*) voids to block, because a lesson keyed on HIS state can't
guarantee HER posture. Standing (s.*) and close (cl.*) are grounded and reliable. coach_filter and the
book-seed loader both enforce this - test-first for the shared predicate and the book prune.
"""
from sf2.system1.advice import stance_unreliable
from sf2.system2 import seed_rules


def test_crouch_and_air_moves_are_unreliable():
    assert stance_unreliable("c.mk") and stance_unreliable("c.hk")
    assert stance_unreliable("j.hk") and stance_unreliable("jf.lk")


def test_standing_close_special_movement_are_reliable():
    for m in ("s.mk", "cl.mk", "shoryuken_hp", "throw_F+hp", "walk_forward", "block_high"):
        assert not stance_unreliable(m), m


def test_cl_prefix_not_confused_with_crouch():
    assert not stance_unreliable("cl.mp")   # close standing normal, NOT a crouch
    assert stance_unreliable("c.mp")


def test_non_string_is_reliable_by_default():
    assert not stance_unreliable(None)


def test_book_seed_drops_stance_unreliable_rules():
    # the real Guile book carries 'use more c.mk at mid range when he stands' (a crouch normal); the
    # loader must now drop it so the incumbent baseline never carries a rule that voids to block.
    entries = seed_rules.seed_lessons("lessons/book.json", "guile", "chunli")
    bad = [e for e in entries if stance_unreliable(e["claim"].get("move"))]
    assert not bad, "book seed still carries stance-unreliable rules: %s" % [e["line"] for e in bad]
