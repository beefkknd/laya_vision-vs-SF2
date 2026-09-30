"""System 1 fixes after the independent review (docs/reviews/2026-09-29_dr_fable.md, findings 5 and 7), owner-approved.

Block scale: laya-vision scores a block as P(blocked) and an attack as P(hit), and both went through one threshold, so in
13,512 no-advice decisions where he attacked on the ground a block was in the top 3 62% of the time and never rated
above "likely fails". Measured (131k decisions vs Ken/Ryu/Honda): a block scored 0.1-0.2 actually blocked 40%, 0.2+
59-65%; an attack scored 0.5+ actually hit 25-44%. A block is now rated on its own measured scale.

Nothing left: 485 decisions vs Honda where the avoid lessons ruled out every shortlisted move and forward, so text laya
had nothing it was allowed to pick. The shortlist now skips moves an applying avoid lesson rules out.
"""
from sf2.system1 import advice as A
from sf2.system1.advisor import SHORTLIST, shortlist
from sf2.system1.advice import FAILS, FORWARD, MAY, WORKS, answers, read, rating

MOVES = ["sweep", "c.mk", "hp", "mp", "block_high", "block_low"]


def test_a_block_is_rated_on_its_own_scale():
    assert rating(0.25, "block_low") == WORKS and rating(0.15, "block_high") == MAY and rating(0.05, "block_low") == FAILS
    assert rating(0.25, "sweep") == FAILS and rating(0.15) == FAILS


def test_invariant_attack_ratings_are_unchanged():
    for i in range(101):
        p = i / 100
        old = WORKS if p >= 0.5 else MAY if p >= 0.3 else FAILS
        assert rating(p) == old and rating(p, "sweep") == old and rating(p, FORWARD) == old


def test_the_shortlist_uses_the_block_scale():
    scores = {"sweep": 0.35, "c.mk": 0.22, "hp": 0.2, "mp": 0.1, "block_high": 0.05, "block_low": 0.24}
    got = shortlist(scores, [], MOVES, ("mid", "attacking"))
    assert got["block_low"] == WORKS and got["sweep"] == MAY


def test_the_training_data_builder_keeps_the_old_words():
    """scripts/build_advice_data.py calls rating(score) without the move: the dataset rebuild stays byte-identical
    (text laya is not retrained; a future training run should pass the move)."""
    assert rating(0.24) == FAILS


def test_moves_an_applying_avoid_rules_out_leave_the_shortlist():
    scores = {"sweep": 0.9, "c.mk": 0.8, "hp": 0.7, "mp": 0.6, "block_high": 0.05, "block_low": 0.02}
    lessons = ["avoid sweep at mid range", "avoid c.mk at mid range", "avoid hp at mid range", "avoid forward at mid range"]
    got = shortlist(scores, lessons, MOVES, ("mid", "standing"))
    assert "mp" in got and not {"sweep", "c.mk", "hp"} & set(got) and len(got) - 1 == SHORTLIST
    rule, why = answers("mid", "standing", got, [read(t, MOVES + [FORWARD]) for t in lessons])
    assert why != "nothing_left" and rule == ["mp"]


def test_an_avoid_that_does_not_apply_keeps_the_move():
    scores = {"sweep": 0.9, "c.mk": 0.8, "hp": 0.7, "mp": 0.6, "block_high": 0.05, "block_low": 0.02}
    got = shortlist(scores, ["avoid sweep up close"], MOVES, ("mid", "standing"))
    assert "sweep" in got


def test_forward_stays_offered_even_when_avoided():
    """The label rule requires forward in the options; an avoid on forward is judged by the rule, as before."""
    got = shortlist({"sweep": 0.9, "c.mk": 0.8, "hp": 0.7, "mp": 0.6}, ["avoid forward at mid range"], MOVES,
                    ("mid", "standing"))
    assert FORWARD in got and A.FORWARD == FORWARD
