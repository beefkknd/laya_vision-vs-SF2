"""The lesson registry (sf2.system2.lessons): Qwen proposes, code verifies and keeps the books.

    testing -> registered | rejected       a claim is judged on the rounds inside its own condition
    registered -> retired                  when its evidence stops holding
"""
import itertools
import random

import pytest

from sf2.system1.advice import read
from sf2.system2 import lessons as L

MOVES = ["sweep", "c.mk", "hp", "mp", "throw", "spinning_bird_kick"]


def act(move, rng, net, doing="standing"):
    air = doing == "jumping"
    state = {"crouching": "crouch", "attacking": "attack", "stunned": "hit_stun"}.get(doing, "stand")
    return {"action": move, "range": rng, "kind": "attack", "dealt": max(net, 0), "taken": max(-net, 0),
            "opp_air": air, "opp_state": state, "round": 0}


def many(move, rng, mean, n, doing="standing", seed=0):
    r = random.Random(seed)
    return [act(move, rng, mean + r.randint(-3, 3), doing) for _ in range(n)]


def claim(kind, move, rng=None, when=None):
    return {"kind": kind, "move": move, "range": rng, "when": when}


@pytest.mark.parametrize("kind,rng,when", list(itertools.product(("use_more", "avoid"), (None, "close", "mid", "far"),
                                                            (None,) + tuple(L.WHEN_WORDS))))
def test_every_claim_renders_to_a_line_text_laya_reads_back(kind, rng, when):
    line = L.render(claim(kind, "c.mk", rng, when))
    les = read(line, MOVES)
    assert (les.move, les.where, les.when) == ("c.mk", rng, when)
    assert les.polarity == ("soft" if kind == "use_more" else "neg")


def test_evidence_is_taken_inside_the_claims_condition():
    rows = many("hp", "close", 8, 30, "standing") + many("hp", "close", -12, 30, "jumping", 1) + many("hp", "mid", 5, 5)
    assert L.condition_evidence(rows, claim("use_more", "hp", "close", "standing"))["cls"] == "good"
    assert L.condition_evidence(rows, claim("avoid", "hp", "close", "jumping"))["cls"] == "bad"
    assert L.condition_evidence(rows, claim("use_more", "hp", "close"))["tries"] == 60
    assert L.condition_evidence(rows, claim("use_more", "hp", "mid"))["cls"] == "few"


def test_an_avoid_is_judged_at_once_on_the_data():
    rows = many("sweep", "close", -10, 40) + many("mp", "mid", 0, 40) + many("throw", "close", -20, 3)
    reg, out = L.propose([], [claim("avoid", "sweep", "close"), claim("avoid", "mp", "mid"),
                              claim("avoid", "throw", "close")], rows, game=0)
    assert [o["state"] for o in out] == ["registered", "rejected", "rejected"]
    assert "too few" in out[2]["why"]


def test_a_use_more_is_tried_then_registered_or_rejected():
    rows = many("c.mk", "mid", 1, 10)
    reg, out = L.propose([], [claim("use_more", "c.mk", "mid")], rows, game=0)
    assert out[0]["state"] == "testing" and L.in_play(reg) == ["use more c.mk at mid range"]
    reg = L.review(reg, rows + many("c.mk", "mid", 6, 40, seed=2), game=1)
    assert reg[0]["state"] == "registered"
    reg2, _ = L.propose([], [claim("use_more", "hp", "far")], many("hp", "far", 0, 30), game=0)
    for g in range(1, L.TEST_GAMES + 1):
        reg2 = L.review(reg2, many("hp", "far", 0, 30 + 10 * g, seed=g), game=g)
    assert reg2[0]["state"] == "rejected" and "not shown" in reg2[0]["why"]


def test_a_registered_lesson_retires_when_its_evidence_stops_holding():
    reg, _ = L.propose([], [claim("use_more", "hp", "close")], many("hp", "close", 8, 40), game=0)
    assert reg[0]["state"] == "registered"
    reg = L.review(reg, many("hp", "close", 8, 40) + many("hp", "close", -12, 60, seed=3), game=1)
    assert reg[0]["state"] == "retired"


def test_duplicates_and_contradictions_are_refused():
    rows = many("sweep", "close", -10, 40)
    reg, _ = L.propose([], [claim("avoid", "sweep", "close")], rows, game=0)
    reg, out = L.propose(reg, [claim("avoid", "sweep", "close"), claim("use_more", "sweep", "close"),
                               claim("use_more", "sweep", "close", "jumping"), {"kind": "maybe", "move": "x"}],
                         rows, game=1)
    assert [o["state"] for o in out] == ["refused"] * 4
    assert "already" in out[0]["why"] and "contradicts" in out[1]["why"] and "contradicts" in out[2]["why"]


def test_in_play_holds_at_most_five_lines_tests_first_then_the_strongest_lessons():
    rows = []
    claims = []
    for i, (m, r) in enumerate([("sweep", "close"), ("mp", "mid"), ("hp", "far"), ("throw", "close"),
                                ("spinning_bird_kick", "mid"), ("c.mk", "close")]):
        rows += many(m, r, -4 - 3 * i, 40, seed=i)                   # later ones cost more
        claims.append(claim("avoid", m, r))
    rows += many("c.mk", "mid", 1, 5, seed=9) + many("hp", "mid", 1, 5, seed=10)
    reg, _ = L.propose([], claims + [claim("use_more", "c.mk", "mid"), claim("use_more", "hp", "mid")], rows, 0)
    lines = L.in_play(reg)
    assert len(lines) == L.MAX_LINES and lines[:2] == ["use more c.mk at mid range", "use more hp at mid range"]
    assert lines[2:] == ["avoid c.mk up close", "avoid spinning_bird_kick at mid range", "avoid throw up close"]


def test_invariant_no_registered_lesson_contradicts_its_evidence():
    r = random.Random(4)
    rows = []
    reg = []
    for g in range(8):
        rows += many(r.choice(MOVES), r.choice(("close", "mid", "far")), r.randint(-12, 12), 25, seed=g)
        claims = [claim(r.choice(("use_more", "avoid")), r.choice(MOVES), r.choice((None, "close", "mid")))
                  for _ in range(3)]
        reg, _ = L.propose(reg, claims, rows, game=g)
        reg = L.review(reg, rows, game=g)
        assert L.violations(reg, rows) == []
