"""The prompts no longer offer forward as a move for lessons (text laya was never trained on lessons naming it,
docs/component_boundaries.md); what she did - forward included - is still shown. ``forward_lessons=True`` is the old
prompt, byte for byte (the golden tests in tests/test_character_prompt.py pin it). Verified book lines are shown as
players' tips."""
import json

from sf2.system2 import character_prompt as C
from sf2.system2 import lesson_prompt as P
from sf2.system2 import lessons as L
from tests.test_character_prompt import GOLDEN, GOLDEN_MOVES, act
from tests.test_verified_lessons import THROW

MOVES = ["sweep", "c.mk", "hp", "block_high", "block_low", "forward"]


def rows():
    return ([act("forward", "mid", taken=3, kind="movement") for _ in range(20)]
            + [act("c.mk", "mid", dealt=4) for _ in range(20)])


def her_moves(system):
    return system.split("Her moves: ", 1)[1].split(". ", 1)[0]


def test_the_character_prompt_does_not_offer_forward_for_lessons():
    sysm, user = (m["content"] for m in C.messages("chunli", "ken", [], rows(), rows(), [], [], MOVES))
    assert her_moves(sysm) == "sweep, c.mk, hp, block_high, block_low"
    assert "never tried there: sweep, hp, block_high, block_low\n" in user + "\n"          # not forward
    assert "- forward: 20 tries" in user                                            # what she did is still shown


def test_the_views_prompt_does_not_offer_forward_for_lessons():
    sysm, user = (m["content"] for m in P.messages("chunli", "ken", [], rows(), rows(), [], [], MOVES))
    assert her_moves(sysm) == "sweep, c.mk, hp, block_high, block_low"
    assert "- forward at mid range: 20 tries" in user                              # the all-games view


def test_the_old_prompts_are_one_flag_away():
    for mod in (C, P):
        sysm = mod.messages("chunli", "ken", [], rows(), rows(), [], [], MOVES, forward_lessons=True)[0]["content"]
        assert her_moves(sysm).endswith("block_low, forward")


def test_on_the_golden_rows_only_the_move_lists_differ():
    with open(GOLDEN) as f:
        g = json.load(f)
    for opp, rs in g["rows"].items():
        for fgc in (False, True):
            new = C.messages("chunli", opp, [], rs, rs[-60:], [], [], GOLDEN_MOVES, fgc=fgc)
            old = C.messages("chunli", opp, [], rs, rs[-60:], [], [], GOLDEN_MOVES, fgc=fgc, forward_lessons=True)
            for n, o in zip(new, old):
                changed = [(a, b) for a, b in zip(n["content"].split("\n"), o["content"].split("\n")) if a != b]
                assert len(n["content"].split("\n")) == len(o["content"].split("\n"))
                for a, b in changed:
                    assert a.startswith(("Her moves:", "- never tried there:")), a
                    assert "forward" in b and "forward" not in a


def test_verified_lines_are_shown_as_players_tips():
    reg = L.from_book([THROW], "lessons/book.json")
    for mod in (C, P):
        user = mod.messages("chunli", "ryu", reg, rows(), rows(), [], [], MOVES)[1]["content"]
        assert "- verified: use more throw up close (players' tip, verified in play: +42 hp/round vs no advice" in user
        assert L.VERIFIED_NOTE in user
    plain = C.messages("chunli", "ryu", [], rows(), rows(), [], [], MOVES)[1]["content"]
    assert L.VERIFIED_NOTE not in plain
