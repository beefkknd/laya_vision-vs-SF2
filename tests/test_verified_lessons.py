"""Verified lessons (sf2.system2.lessons.from_book) and the forward refusal.

A verified lesson is a players' tip whose single-line fixed-advice arm helped vs no advice (lessons/book.json,
scripts/book.py). The per-situation check cannot judge it (her history has too few tries), so it is never retired,
it stays in play first, and a claim that repeats, is covered by or contradicts it is refused.
Text laya was never trained on lessons naming forward (docs/component_boundaries.md): such claims are refused.
"""
from sf2.system2 import lessons as L
from tests.test_lessons import claim, many, world

THROW = {"line": "use more throw up close", "claim": {"kind": "use_more", "move": "throw", "range": "close",
                                                     "when": None},
         "arm": "throw", "mean": 42.4, "ci95": [27.0, 59.0], "runs": 8, "seeds": [41001, 41002],
         "batch": "logs/ab/ryu_expert"}
HP = {"line": "use more hp up close when he attacks", "claim": {"kind": "use_more", "move": "hp", "range": "close",
                                                              "when": "attacking"},
      "arm": "hpatt", "mean": 19.8, "ci95": [11.3, 29.7], "runs": 8, "seeds": [42001], "batch": "logs/ab/ken_expert"}
HARD = dict(THROW, line="always throw up close", arm="throwhard", mean=42.4,
            claim={"kind": "always", "move": "throw", "range": "close", "when": None})
NARROW = dict(THROW, line="use more throw up close when he attacks", arm="throwatt", mean=24.4,
              claim={"kind": "use_more", "move": "throw", "range": "close", "when": "attacking"})


def book(*tips):
    return L.from_book(list(tips), "lessons/book.json")


def test_book_lines_enter_the_registry_as_verified_with_their_evidence():
    reg = book(THROW)
    assert [r["state"] for r in reg] == ["verified"] and reg[0]["line"] == "use more throw up close"
    ev = reg[0]["evidence"]
    assert ev["source"] == "players' tip" and ev["book"] == "lessons/book.json" and ev["arm"] == "throw"
    assert ev["mean"] == 42.4 and ev["ci95"] == [27.0, 59.0] and ev["batch"] == "logs/ab/ryu_expert"
    assert reg[0]["why"].startswith("players' tip, verified in play: +42 hp/round vs no advice")


def test_a_book_line_is_rendered_from_its_claim_or_refused():
    bad = dict(THROW, line="use more throw at mid range")                    # line and claim disagree
    try:
        book(bad)
    except ValueError as e:
        assert "does not render" in str(e)
    else:
        raise AssertionError("a book line whose claim renders differently must stop the loop")


def test_a_book_line_covered_by_a_stronger_one_is_left_out():
    """Same move and conditions (use more / always) or a narrower one of the same direction: one slot, not two."""
    assert [r["line"] for r in book(THROW, HARD)] == ["use more throw up close"]      # tie: the softer, owner-named
    assert [r["line"] for r in book(dict(HARD, mean=46.7), THROW)] == ["always throw up close"]
    assert [r["line"] for r in book(NARROW, THROW, HP)] == ["use more throw up close",
                                                          "use more hp up close when he attacks"]


def test_a_verified_lesson_is_never_retired_even_when_her_history_says_worse():
    rows = world() + many("throw", "close", -30, 40, "attacking", 9)          # throw clearly worse there
    assert L.condition_evidence(rows, THROW["claim"])["cls"] == "worse"
    reg = L.review(book(THROW), rows, 5, [0, 0, -200, -200, -200, -200])
    assert reg[0]["state"] == "verified" and reg[0]["evidence"]["mean"] == 42.4
    assert L.violations(reg, rows) == []                                    # the invariant ignores verified


def test_verified_lines_come_first_then_tests_then_registered_at_most_five():
    reg = book(THROW, HP)
    reg, _ = L.propose(reg, [claim("avoid", "sweep", "close", "attacking")], world(), 0)
    reg, _ = L.propose(reg, [claim("use_more", "mp", "mid"), claim("use_more", "c.mk", "far")], world(), 0,
                       moves=["mp", "c.mk"])
    lines = L.in_play(reg)
    assert lines == ["use more throw up close", "use more hp up close when he attacks", "use more mp at mid range",
                     "use more c.mk far away", "avoid sweep up close when he attacks"]
    assert len(lines) == L.MAX_LINES


def test_tests_are_only_taken_while_they_can_be_in_play():
    """3 verified lines leave 2 of the 5 slots: a third test (even a what-if) would never reach text laya."""
    reg = book(THROW, HP, dict(HP, line="use more mk at mid range when he jumps", arm="antiair",
                               claim={"kind": "use_more", "move": "mk", "range": "mid", "when": "jumping"}))
    reg, out = L.propose(reg, [claim("use_more", "mp", "mid"), claim("use_more", "c.mk", "far"),
                               dict(claim("use_more", "lp", "far"), view="what_if")], world(), 0, moves=["lp", "mp", "c.mk"])
    assert [o["state"] for o in out] == ["testing", "testing", "refused"]
    assert "in play" in out[2]["why"]
    assert len(L.in_play(reg)) == L.MAX_LINES


def test_a_claim_repeating_covered_by_or_contradicting_a_verified_lesson_is_refused():
    reg = book(THROW)
    rows = world() + many("throw", "close", 5, 3, "attacking", 9)
    moves = ["throw", "sweep"]
    _, out = L.propose(reg, [claim("use_more", "throw", "close")], rows, 0, moves=moves)
    assert out[0]["state"] == "refused" and out[0]["why"] == "already verified"
    _, out = L.propose(reg, [claim("use_more", "throw", "close", "jumping")], rows, 0, moves=moves)
    assert out[0]["state"] == "refused" and out[0]["why"].startswith("covered by the verified lesson")
    _, out = L.propose(reg, [claim("avoid", "throw")], rows, 0, moves=moves)
    assert out[0]["state"] == "refused" and out[0]["why"].startswith("contradicts the verified lesson")
    _, out = L.propose(reg, [claim("avoid", "throw", "close")], rows, 0, moves=moves)
    assert out[0]["state"] == "refused" and "contradicts" in out[0]["why"]


def test_without_a_book_nothing_changes():
    assert L.from_book([], "lessons/book.json") == []


# ---- forward: text laya was never trained on lessons naming it ----

def test_every_claim_naming_forward_is_refused_with_the_reason():
    rows = world() + many("forward", "mid", 5, 40, "standing", 4, "movement") + many("c.mk", "mid", -5, 40)
    for kind in L.KINDS:
        _, out = L.propose([], [claim(kind, "forward", "mid")], rows, 0, moves=["forward", "c.mk"])
        assert out[0]["state"] == "refused", kind
        assert out[0]["why"] == ("System 1 cannot follow lessons naming forward (text laya untrained, "
                                 "docs/component_boundaries.md)")


def test_forward_can_be_allowed_again_for_an_old_replay():
    rows = world() + many("forward", "mid", 5, 40, "standing", 4, "movement") + many("c.mk", "mid", -5, 40)
    _, out = L.propose([], [claim("use_more", "forward", "mid")], rows, 0, moves=["forward", "c.mk"],
                       unfollowable=())
    assert out[0]["state"] == "registered"
