"""The lesson registry (sf2.system2.lessons): Qwen proposes, code verifies and keeps the books.

A claim is judged RELATIVE to what she does now in the same situation: its move's net hit points per decision minus
her average over every other decision there (attacks, walks, blocks). use more / always: kept when clearly better;
avoid: kept when clearly worse. (2026-09-29: a block nets -4.3 per try - "bad" in absolute terms - while her average
when he attacks is about -10.)
"""
import itertools
import random

import pytest

from sf2.system1.advice import read
from sf2.system2 import lessons as L

MOVES = ["sweep", "c.mk", "hp", "mp", "throw", "spinning_bird_kick", "block_low", "back"]


def act(move, rng, net, doing="standing", kind="attack", rnd=0):
    air = doing == "jumping"
    state = {"crouching": "crouch", "attacking": "attack", "stunned": "hit_stun"}.get(doing, "stand")
    return {"action": move, "range": rng, "kind": kind, "dealt": max(net, 0), "taken": max(-net, 0),
            "opp_air": air, "opp_state": state, "round": rnd, "actual": "hit" if net > 0 else "whiff"}


def many(move, rng, mean, n, doing="standing", seed=0, kind="attack"):
    r = random.Random(seed)
    return [act(move, rng, mean + r.randint(-3, 3), doing, kind) for _ in range(n)]


def claim(kind, move, rng=None, when=None):
    return {"kind": kind, "move": move, "range": rng, "when": when}


def world():
    """Up close when he attacks her average is about -10; a block loses 4 there, sweep 18."""
    return (many("sweep", "close", -18, 40, "attacking") + many("block_low", "close", -4, 40, "attacking", 1, "defense")
            + many("c.mk", "close", -10, 40, "attacking", 2) + many("back", "close", -9, 30, "attacking", 3, "movement"))


@pytest.mark.parametrize("kind,rng,when", list(itertools.product(L.KINDS, (None, "close", "mid", "far"),
                                                                 (None,) + tuple(L.WHEN_WORDS))))
def test_every_claim_renders_to_a_line_text_laya_reads_back(kind, rng, when):
    les = read(L.render(claim(kind, "c.mk", rng, when)), MOVES)
    assert (les.move, les.where, les.when) == ("c.mk", rng, when)
    assert les.polarity == {"use_more": "soft", "always": "hard", "avoid": "neg"}[kind]


def test_evidence_is_relative_to_her_average_in_the_same_situation():
    rows = world()
    blk = L.condition_evidence(rows, claim("always", "block_low", "close", "attacking"))
    assert blk["cls"] == "better" and blk["net"] < 0 < blk["diff"]           # negative, yet better than she does
    assert L.condition_evidence(rows, claim("avoid", "sweep", "close", "attacking"))["cls"] == "worse"
    assert L.condition_evidence(rows, claim("avoid", "c.mk", "close", "attacking"))["cls"] == "unclear"
    assert L.condition_evidence(rows, claim("use_more", "hp", "close"))["cls"] == "few"


def test_judging():
    rows = world()
    reg, out = L.propose([], [claim("avoid", "sweep", "close", "attacking"), claim("always", "block_low", "close",
                                                                                  "attacking"),
                              claim("avoid", "c.mk", "close", "attacking")], rows, game=0)
    assert [o["state"] for o in out] == ["registered", "registered", "rejected"]
    reg, out = L.propose([], [claim("use_more", "c.mk", "close", "attacking")], rows, game=0)
    assert out[0]["state"] == "testing"


def test_a_test_is_registered_rejected_or_too_few():
    rows = world()
    reg, _ = L.propose([], [claim("use_more", "c.mk", "close", "attacking")], rows, 0)
    better = rows + many("c.mk", "close", -2, 80, "attacking", 7)
    assert L.review(reg, better, 1)[0]["state"] == "registered"
    reg2, _ = L.propose([], [claim("use_more", "hp", "close", "attacking")], rows + many("hp", "close", 0, 3,
                                                                                         "attacking", 8), 0)
    for g in range(1, L.TEST_GAMES + 1):
        reg2 = L.review(reg2, rows + many("hp", "close", 0, 3 + g, "attacking", 8), g)
    assert reg2[0]["state"] == "rejected" and reg2[0]["why"].startswith("too few")


def test_a_registered_lesson_retires_when_its_evidence_stops_holding():
    rows = world()
    reg, _ = L.propose([], [claim("always", "block_low", "close", "attacking")], rows, 0)
    worse = rows + many("block_low", "close", -30, 200, "attacking", 9, "defense")
    assert L.review(reg, worse, 1)[0]["state"] == "retired"


def test_duplicates_contradictions_and_covered_claims_are_refused():
    rows = world()
    reg, _ = L.propose([], [claim("avoid", "sweep", "close")], rows, 0)
    reg, out = L.propose(reg, [claim("avoid", "sweep", "close"), claim("always", "sweep", "close"),
                               claim("avoid", "sweep", "close", "attacking"), {"kind": "maybe", "move": "x"},
                               claim("use_more", "shoryuken", "close")], rows, 1)
    assert [o["state"] for o in out] == ["refused"] * 5
    assert "already" in out[0]["why"] and "contradicts" in out[1]["why"] and "covered" in out[2]["why"]
    assert "not one of her moves" in out[4]["why"]


def test_in_play_holds_at_most_five_lines_tests_first_then_the_strongest():
    rows = world() + many("mp", "mid", -6, 40, "standing", 4) + many("hp", "mid", -1, 40, "standing", 5) + \
        many("throw", "mid", -9, 40, "standing", 6) + many("c.mk", "mid", -3, 6, "standing", 7)
    claims = [claim("avoid", "sweep", "close", "attacking"), claim("always", "block_low", "close", "attacking"),
              claim("avoid", "throw", "mid"), claim("use_more", "hp", "mid"), claim("avoid", "mp", "mid"),
              claim("use_more", "c.mk", "mid")]
    reg, out = L.propose([], claims, rows, 0)
    lines = L.in_play(reg)
    assert len(lines) <= L.MAX_LINES and lines[0] == "use more c.mk at mid range"          # the test first
    by_line = {r["line"]: r for r in reg}
    strength = [L.strength(by_line[x]) for x in lines if by_line[x]["state"] == "registered"]
    assert strength == sorted(strength, reverse=True)


def test_invariant_no_registered_lesson_contradicts_its_evidence():
    r = random.Random(4)
    rows, reg = [], []
    for g in range(8):
        for _ in range(3):
            rows += many(r.choice(MOVES), r.choice(("close", "mid")), r.randint(-15, 8), 25,
                         r.choice(("standing", "attacking")), seed=r.randint(0, 999))
        claims = [claim(r.choice(L.KINDS), r.choice(MOVES), r.choice((None, "close", "mid")),
                        r.choice((None, "standing", "attacking"))) for _ in range(3)]
        reg, _ = L.propose(reg, claims, rows, game=g)
        reg = L.review(reg, rows, game=g)
        assert L.violations(reg, rows) == []


def test_damage_causes():
    assert L.cause(dict(act("sweep", "mid", 5), taken=8)) == "traded"
    assert L.cause(dict(act("sweep", "mid", -8), actual="whiff")) == "punished"
    assert L.cause(dict(act("sweep", "mid", -8), actual="blocked")) == "punished"
    assert L.cause(dict(act("sweep", "mid", -8), actual="none")) == "stuffed"
    assert L.cause(act("back", "mid", -6, kind="movement")) == "caught"
    assert L.cause(act("sweep", "mid", 5)) is None


def test_a_narrower_opposite_claim_is_an_exception_not_a_contradiction():
    """Ken, two views (2026-09-29): 'avoid hp up close when he jumps' was refused 6 games running as contradicting
    'use more hp up close'. Text laya's rule already lets an applying avoid rule the move out."""
    rows = many("hp", "close", 6, 40, "standing") + many("hp", "close", -14, 30, "jumping", 1) + \
        many("sweep", "close", -4, 40, "standing", 2) + many("sweep", "close", -4, 30, "jumping", 3)
    reg, _ = L.propose([], [claim("use_more", "hp", "close")], rows, 0)
    reg, out = L.propose(reg, [claim("avoid", "hp", "close", "jumping"), claim("avoid", "hp", None)], rows, 1)
    assert out[0]["state"] == "registered"                                 # the exception, judged on its own data
    assert out[1]["state"] == "refused" and "contradicts" in out[1]["why"]   # broader and opposite: a contradiction


def test_a_move_she_never_used_can_be_tried_but_not_avoided():
    """She never blocked (0 of 3853 times he attacked): 'always block_low when he attacks' must be testable."""
    rows = world()[40:]                                                      # no block rows at all
    moves = MOVES
    reg, out = L.propose([], [claim("always", "block_high", "close", "attacking"),
                              claim("avoid", "block_high", "close")], rows, 0, moves=moves + ["block_high"])
    assert out[0]["state"] == "testing"
    assert out[1]["state"] == "refused" and "never" in out[1]["why"]


def test_a_what_if_gets_its_own_test_slot():
    rows = world()
    reg, _ = L.propose([], [claim("use_more", "hp", "close"), claim("use_more", "mp", "close")], rows, 0,
                       moves=MOVES)
    wi = dict(claim("always", "throw", "close", "attacking"), view="what_if")
    reg, out = L.propose(reg, [wi, claim("use_more", "spinning_bird_kick", "close")], rows, 1, moves=MOVES)
    assert out[0]["state"] == "testing" and out[1]["state"] == "refused"      # a normal third test is not taken
    assert L.in_play(reg)[:3] == ["use more hp up close", "use more mp up close", "always throw up close when he attacks"]


def test_the_loop_offers_only_moves_system1_can_choose():
    """Six what-if runs (2026-09-29): 'always back / crouch / jump_back at mid range when he attacks' got 0 tries in
    3 games each: System 1 picks only among laya-vision's rated moves (attacks, blocks) and walking in."""
    from sf2.system1.system1 import choices
    got = choices("chunli")
    assert "block_high" in got and "block_low" in got and "forward" in got and "sweep" in got
    assert not {"back", "crouch", "jump_back", "idle", "jump"} & set(got)
