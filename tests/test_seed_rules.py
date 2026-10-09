"""sf2.system2.seed_rules: the owner's web/AB research (lessons/book.json) as the loop's STARTING playbook.

The loader reads one opponent's VERIFIED lines (``lines``; ``not_verified`` is excluded), validates each against text
laya's play-time parser (sf2.system1.advice.read) USING THE SAME two-stage MOVE MENU text laya reads at play time
(sf2.system1.advice.char_menu_moves(me)), so a line that no longer parses to a followable move - or whose stored claim
no longer renders to it - is DROPPED with a logged reason, and returns the survivors as registry entries
(sf2.system2.lessons.from_book) tagged provenance "web".

The bug this guards (fixed 2026-10-03): validation used sf2.system1.system1.choices(me), the Stage-1 action vocabulary,
NOT the character's full two-stage menu. choices lacks the two-stage names (hadoken_hp / shoryuken_hp / throw_F+hp), so
a non-Chun-Li seed (Ryu's hadoken/shoryuken) dropped as "no followable move"; and choices carries a bare "hp" that
makes the play parser misread "throw_F+hp" as "hp". char_menu_moves(me) is the menu text laya actually chooses from
(loop_runner, play_loop_screen), so validation now agrees with play.

Seen red: with the module absent the whole file fails to import (ModuleNotFoundError); with ``validate_tip`` stubbed to
never drop, test_a_doctored_line_is_dropped and test_loader_drops_a_doctored_line_and_logs_it fail; and with the
pre-fix ``moves = choices(...)`` restored, test_ryu_book_loads_the_two_stage_lines and
test_throw_F_hp_is_not_misread_as_hp fail (that is the exact bug - verified by stashing the one-line fix).
"""
import json
import logging

import pytest

from sf2.system1 import advice as A
from sf2.system1.system1 import choices
from sf2.system2 import lessons as L
from sf2.system2 import seed_rules as S

BOOK = "lessons/book.json"
# The menu validation uses: Chun-Li's full two-stage move menu (what text laya reads at play time), NOT choices().
MOVES = A.char_menu_moves("chunli")

# A clean verified tip, and two doctored twins (known-bad, seeded in-test). The book stores the generic throw word
# "throw"; the play parser canonicalizes it to the menu's concrete throw move (throw_F+hp) via the throw alias, and
# validate_tip accepts that one reconciliation (see seed_rules._reads_as).
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


# The one book line that is NOT followable under Chun-Li's two-stage menu: the book stored the generic "hp", but the
# two-stage menu has no bare "hp" (only s.hp / cl.hp / c.hp / ...), and "hp" is not a throw word. The old choices-menu
# validation MASKED this (choices has a bare "hp"); at play time read_lesson already returns no move for it, so the
# seed was dead weight. Dropping it is the correct, surfaced behavior - not a regression.
KNOWN_DROPS = {("ken", "use more hp up close when he attacks"): "no followable move",
               # guile's c.mk line: c.* is stance-unreliable (voids to block when keyed on his state) - correct drop
               ("guile", "use more c.mk at mid range when he stands"): "stance-unreliable"}


@pytest.mark.parametrize("opp,tip", _verified_tips(), ids=lambda x: x if isinstance(x, str) else x["line"])
def test_every_verified_line_renders_back_and_validates_or_is_a_known_drop(opp, tip):
    """render(claim) == line for every stored tip (round-trip half 1), and validate_tip admits it against the
    two-stage menu - except the documented unfollowable line, which drops with the expected reason."""
    assert L.render(tip["claim"]) == tip["line"]
    ok, why = S.validate_tip(tip, MOVES)
    expected = KNOWN_DROPS.get((opp, tip["line"]))
    if expected is None:
        assert ok, "%s: %s" % (opp, why)
    else:
        assert not ok and expected in why, "%s: expected drop %r, got (%s, %r)" % (opp, expected, ok, why)


def test_a_doctored_line_is_dropped():
    assert S.validate_tip(GOOD, MOVES)[0]                                   # positive control (throw alias accepted)
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


# Golden: the exact seed set from the real book (me=chunli; regenerate with `python -m sf2.system2.seed_rules`).
GOLDEN = {
    "dhalsim": ["always throw up close"],
    "guile": ["use more throw up close"],   # "use more c.mk ... when he stands" dropped: c.* is stance-unreliable (voids to block keyed on his state)
    "honda": ["use more throw up close", "use more lightning_legs up close when he stands"],
    "ken": ["use more throw up close"],   # "use more hp up close when he attacks" dropped: no bare "hp" in the menu
    "ryu": ["use more throw up close"],
    "zangief": ["use more throw up close"],   # tie on mean -> the softer line wins the slot (lessons.ORDER)
}


@pytest.mark.parametrize("opp,lines", sorted(GOLDEN.items()))
def test_seed_set_per_opponent_matches_the_golden(opp, lines):
    assert [r["line"] for r in S.seed_lessons(BOOK, opp)] == lines


# ---- the bug: a non-Chun-Li character's two-stage seed lines, validated against the WRONG menu ----
# A crafted Ryu book (me="ryu"): three lines in Ryu's two-stage vocabulary. These are DROPPED before the fix (validated
# against choices("ryu"), which has neither hadoken_hp/shoryuken_hp nor throw_F+hp) and KEPT after (char_menu_moves).

def _ryu_tip(line, move, rng, when):
    return {"line": line, "arm": move, "mean": 30.0, "ci95": [5.0, 55.0], "runs": 8, "seeds": [1, 2],
            "batch": "logs/ab/ryu_vs_guile",
            "claim": {"kind": "use_more", "move": move, "range": rng, "when": when}}


RYU_TIPS = [
    _ryu_tip("use more hadoken_hp far away when he stands", "hadoken_hp", "far", "standing"),
    _ryu_tip("use more shoryuken_hp up close when he jumps", "shoryuken_hp", "close", "jumping"),
    _ryu_tip("use more throw_F+hp up close when he stands", "throw_F+hp", "close", "standing"),
]
RYU_LINES = [t["line"] for t in RYU_TIPS]


def _ryu_book(tmp_path):
    doc = {"me": "ryu", "opponents": {"guile": {"lines": RYU_TIPS, "not_verified": []}}}
    p = tmp_path / "ryu_book.json"
    p.write_text(json.dumps(doc))
    return str(p)


@pytest.mark.parametrize("tip", RYU_TIPS, ids=[t["line"] for t in RYU_TIPS])
def test_ryu_lines_drop_under_the_old_choices_menu(tip):
    """RED: the pre-fix menu. Each line fails validation against choices("ryu") - the exact bug the fix removes."""
    ok, _why = S.validate_tip(tip, choices("ryu"))
    assert not ok


def test_throw_F_hp_is_not_misread_as_hp():
    """RED->GREEN: choices("ryu") carries a bare "hp", so the play parser reads "throw_F+hp" as "hp" (wrong move);
    char_menu_moves("ryu") has the concrete throw_F+hp and no bare "hp", so it reads correctly."""
    line = "use more throw_F+hp up close when he stands"
    assert A.read(line, choices("ryu")).move == "hp"                       # the bug, under the wrong menu
    assert A.read(line, A.char_menu_moves("ryu")).move == "throw_F+hp"     # correct, under the two-stage menu


def test_ryu_book_loads_the_two_stage_lines(tmp_path):
    """GREEN: with me="ryu", all three two-stage lines validate and seed (dropped before the fix)."""
    reg = S.seed_lessons(_ryu_book(tmp_path), "guile", me="ryu")
    assert [r["line"] for r in reg] == RYU_LINES
    assert all(r["evidence"]["source"] == "web" and r["state"] == "verified" for r in reg)


def test_chunli_book_still_loads_against_the_two_stage_menu(tmp_path):
    """Regression: me defaults to the book's "chunli"; its generic "throw" lines survive via the throw alias (they
    canonicalize to throw_F+hp, the move System 1 follows), so the Chun-Li seed set is unchanged except the one
    unfollowable "hp" line (KNOWN_DROPS)."""
    for opp, lines in GOLDEN.items():
        assert [r["line"] for r in S.seed_lessons(BOOK, opp)] == lines


def test_unsupported_character_errors_clearly(tmp_path):
    """me is a character with no two-stage move menu (a boss like balrog; the 8 world warriors are all supported):
    a clear ValueError naming the character, not a silent empty seed set or an obscure KeyError."""
    doc = {"me": "balrog", "opponents": {"guile": {"lines": [GOOD], "not_verified": []}}}
    p = tmp_path / "balrog_book.json"
    p.write_text(json.dumps(doc))
    with pytest.raises(ValueError, match="balrog"):
        S.seed_lessons(str(p), "guile")
    with pytest.raises(ValueError, match="balrog"):
        S.seed_lessons(BOOK, "guile", me="balrog")
