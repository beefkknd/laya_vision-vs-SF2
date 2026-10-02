"""sf2.system2.seed_rules: the owner's web/AB research (lessons/book.json) as the loop's STARTING playbook.

The loader reads one opponent's VERIFIED lines (``lines``; ``not_verified`` is excluded), validates each against text
laya's play-time parser (sf2.system1.advice.read) so a line that no longer parses to a followable move - or whose
stored claim no longer renders to it - is DROPPED with a logged reason, and returns the survivors as registry entries
(sf2.system2.lessons.from_book) tagged provenance "web".

Seen red: with the module absent the whole file fails to import (ModuleNotFoundError); and with ``validate_tip``
stubbed to never drop, test_a_doctored_line_is_dropped and test_loader_drops_a_doctored_line_and_logs_it both fail.
"""
import json
import logging

import pytest

from sf2.system1 import advice as A
from sf2.system1.system1 import choices
from sf2.system2 import lessons as L
from sf2.system2 import seed_rules as S

BOOK = "lessons/book.json"
MOVES = choices("chunli")

# A clean verified tip, and two doctored twins (known-bad, seeded in-test).
GOOD = {"line": "use more throw up close", "arm": "throw", "mean": 40.0, "ci95": [10.0, 70.0], "runs": 8,
        "seeds": [1, 2], "batch": "logs/ab/ken_expert",
        "claim": {"kind": "use_more", "move": "throw", "range": "close", "when": None}}
DOCTORED_RENDER = dict(GOOD, claim={"kind": "use_more", "move": "throw", "range": "mid", "when": None})  # claim != line
DOCTORED_PARSE = dict(GOOD, line="use more flykick up close",
                      claim={"kind": "use_more", "move": "flykick", "range": "close", "when": None})      # unknown move


def _doc():
    with open(BOOK) as f:
        return json.load(f)


def _verified_tips():
    return [(opp, t) for opp, v in sorted(_doc()["opponents"].items()) for t in v.get("lines", [])]


@pytest.mark.parametrize("opp,tip", _verified_tips(), ids=lambda x: x if isinstance(x, str) else x["line"])
def test_every_verified_line_parses_via_advice_read_and_round_trips(opp, tip):
    """advice.read re-derives exactly the stored claim, and the stored claim renders back to the line."""
    les = A.read(tip["line"], MOVES)
    assert les.move is not None, "%s: advice.read found no move in %r" % (opp, tip["line"])
    got = {"kind": S.POLARITY_KIND[les.polarity], "move": les.move, "range": les.where, "when": les.when}
    assert L.key(got) == L.key(tip["claim"]), "%s: advice.read -> %r, stored %r" % (opp, got, tip["claim"])
    assert L.render(tip["claim"]) == tip["line"]
    ok, why = S.validate_tip(tip, MOVES)
    assert ok, "%s: %s" % (opp, why)


def test_a_doctored_line_is_dropped():
    assert S.validate_tip(GOOD, MOVES)[0]                                   # positive control
    ok_r, why_r = S.validate_tip(DOCTORED_RENDER, MOVES)
    assert not ok_r and "render" in why_r.lower()
    ok_p, why_p = S.validate_tip(DOCTORED_PARSE, MOVES)
    assert not ok_p and why_p                                              # unknown move -> no followable move


def test_loader_drops_a_doctored_line_and_logs_it(tmp_path, caplog):
    doc = {"me": "chunli", "opponents": {"ken": {"lines": [GOOD, DOCTORED_PARSE], "not_verified": []}}}
    p = tmp_path / "book.json"
    p.write_text(json.dumps(doc))
    with caplog.at_level(logging.WARNING):
        reg = S.seed_lessons(str(p), "ken")
    assert [r["line"] for r in reg] == ["use more throw up close"]
    assert any("drop" in m.lower() and "ken" in m and "flykick" in m for m in caplog.messages)


def test_unknown_opponent_returns_no_lessons():
    assert S.seed_lessons(BOOK, "balrog") == []                            # not fought, absent from the book
    assert S.seed_lessons(BOOK, "nobody") == []


def test_not_verified_lines_are_excluded():
    doc = _doc()
    for opp in doc["opponents"]:
        seeded = {r["line"] for r in S.seed_lessons(BOOK, opp)}
        not_verified = {x["line"] for x in doc["opponents"][opp].get("not_verified", [])}
        assert seeded.isdisjoint(not_verified), "%s: a not_verified line leaked in" % opp


def test_a_book_of_only_not_verified_seeds_nothing(tmp_path):
    doc = {"me": "chunli", "opponents": {"ken": {"lines": [],
            "not_verified": [{"line": "use more sweep up close", "why": "hurts"}]}}}
    p = tmp_path / "book.json"
    p.write_text(json.dumps(doc))
    assert S.seed_lessons(str(p), "ken") == []


def test_seed_entries_are_verified_and_tagged_web():
    reg = S.seed_lessons(BOOK, "ryu")
    assert reg, "ryu has verified lines in the book"
    assert all(r["state"] == "verified" for r in reg)
    assert all(r["evidence"]["source"] == "web" for r in reg)              # provenance tag
    assert all(r["claim"]["view"] == "book" for r in reg)                  # the registry-admitted view
    assert all(r["evidence"]["book_sha256"] for r in reg)                  # which bytes it came from
    assert all(r["evidence"]["mean"] and r["evidence"]["batch"] for r in reg)  # the A/B evidence carried through


# Golden: the exact seed set from the real book (regenerate with `python -m sf2.system2.seed_rules`).
GOLDEN = {
    "dhalsim": ["always throw up close"],
    "guile": ["use more throw up close", "use more c.mk at mid range when he stands"],
    "honda": ["use more throw up close", "use more lightning_legs up close when he stands"],
    "ken": ["use more throw up close", "use more hp up close when he attacks"],
    "ryu": ["use more throw up close"],
    "zangief": ["use more throw up close"],   # tie on mean -> the softer line wins the slot (lessons.ORDER)
}


@pytest.mark.parametrize("opp,lines", sorted(GOLDEN.items()))
def test_seed_set_per_opponent_matches_the_golden(opp, lines):
    assert [r["line"] for r in S.seed_lessons(BOOK, opp)] == lines
