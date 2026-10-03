"""The offline rule scorer (sf2.eval.rule_score, README G5): a rule's move is scored against the value table's
cell for its range + condition, versus walking in and the cell's best move.

Seen RED first (how): each assertion was first run against a deliberately wrong scorer and observed to fail --
good/bad: with ``verdict_of`` returning the opposite branch; not_scorable: with the fireball / opponent guards
removed (the scorer then crashed or faked a number); the cell mapping: with ``flags_of`` swapping attacking and
airborne. The faults were reverted once the red was witnessed. (Run the seeded-fault twins in tests/faults if
kept; the witness is recorded in the G5 handback.)

The table used is the committed lessons/value_oracle_v1.json (always present) for the real-data checks, and a
small crafted table for the mapping / no-data checks (so the cell chosen is asserted exactly).
"""
import os

import pytest

from sf2.data.value_oracle import load as load_table
from sf2.eval import rule_score as RS
from sf2.eval.rule_score import Rule, map_move, rule_from_claim, rule_from_line, score_rule, summarize

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TABLE_PATH = os.path.join(REPO, "lessons", "value_oracle_v1.json")
BOOK_PATH = os.path.join(REPO, "lessons", "book.json")


@pytest.fixture(scope="module")
def table():
    return load_table(TABLE_PATH)


# A crafted table: every chunli cell carries the same two moves, so a rule's chosen CELL is asserted exactly and
# values are predictable. "good_move" beats forward by > NET_WORKS_MARGIN; "bad_move" is below forward.
def _crafted():
    out = {}
    for rng in ("close", "mid", "far"):
        for att in (0, 1):
            for air in (0, 1):
                out[("chunli", rng, att, air)] = {"forward": 0.0, "good_move": 9.0, "bad_move": -5.0,
                                                   "mid_move": 1.0}
    return out


# ---------------------------------------------------------------- known good / bad on the real table

def test_best_move_of_cell_scores_good(table):
    """throw is the best move up close in the neutral cell (+10.8 vs walking in 0): a known-good rule."""
    sc = score_rule(Rule(move="throw", range="close"), table)
    assert sc.scorable and sc.verdict == RS.GOOD
    assert sc.cells[0].rank == 1 and sc.cells[0].best_move == "throw"
    assert sc.value >= sc.forward + RS.NET_WORKS_MARGIN


def test_known_bad_move_scores_bad(table):
    """spinning_bird_kick is deeply negative up close (-12.5): below walking in -> bad, ranked last."""
    sc = score_rule(Rule(move="spinning_bird_kick", range="close"), table)
    assert sc.scorable and sc.verdict == RS.BAD
    assert sc.value <= sc.forward
    assert sc.cells[0].rank == sc.cells[0].n_moves          # worst in the cell


# ---------------------------------------------------------------- not_scorable (said, not guessed)

def test_fireball_rule_not_scorable(table):
    sc = score_rule(Rule(move="throw", range="close", fireball=True), table)
    assert not sc.scorable and sc.verdict == RS.NOT_SCORABLE
    assert "fireball" in sc.reason and sc.cells == ()       # no number faked


def test_per_opponent_rule_not_scorable(table):
    sc = score_rule(Rule(move="throw", range="close", opp="dhalsim"), table)
    assert not sc.scorable and sc.verdict == RS.NOT_SCORABLE
    assert "opponent" in sc.reason and sc.cells == ()


def test_move_not_in_vocabulary_not_scorable(table):
    """An aerial / combo is never a value action: not in the table's vocabulary -> not_scorable, not a crash."""
    for move in ("j.mk", "jf.hk_s.mp_s.hp", "walk_back"):
        sc = score_rule(Rule(move=move, range="close"), table)
        assert not sc.scorable and "vocabulary" in sc.reason


def test_move_with_no_data_in_cell_not_scorable():
    """A real move that the required cell never sampled -> not_scorable, not a fabricated 0.0."""
    tbl = _crafted()
    tbl[("chunli", "close", 0, 0)] = {"forward": 0.0, "bad_move": -5.0}   # good_move absent here
    sc = score_rule(Rule(move="good_move", range="close"), tbl)
    assert not sc.scorable and "no data in cell" in sc.reason


# ---------------------------------------------------------------- range + condition -> cell mapping

@pytest.mark.parametrize("rng, when, expected", [
    ("close", "attacking", ("chunli", "close", 1, 0)),
    ("mid", "jumping", ("chunli", "mid", 0, 1)),
    ("far", "standing", ("chunli", "far", 0, 0)),
    ("mid", "crouching", ("chunli", "mid", 0, 0)),
    ("close", "stunned", ("chunli", "close", 0, 0)),
    ("close", None, ("chunli", "close", 0, 0)),
])
def test_condition_maps_to_cell(rng, when, expected):
    sc = score_rule(Rule(move="good_move", range=rng, when=when), _crafted())
    assert sc.scorable and len(sc.cells) == 1 and sc.cells[0].cell == expected


def test_no_range_scores_all_three_ranges():
    sc = score_rule(Rule(move="good_move", when="attacking"), _crafted())
    assert sc.scorable and {c.cell[1] for c in sc.cells} == {"close", "mid", "far"}
    assert all(c.cell[2:] == (1, 0) for c in sc.cells)      # attacking everywhere
    assert sc.verdict == RS.GOOD                            # mean 9.0 vs walk 0.0


# ---------------------------------------------------------------- menu / Lesson name translation

def test_map_move_menu_names(table):
    vocab = RS.char_vocab(table, "chunli")
    assert map_move("throw_F+hp", vocab) == "throw"
    assert map_move("throw_F+mp", vocab) == "throw"
    assert map_move("s.mk", vocab) == "mk"
    assert map_move("cl.hp", vocab) == "hp"
    assert map_move("c.hk", vocab) == "sweep"
    assert map_move("walk_forward", vocab) == "forward"
    assert map_move("c.mk", vocab) == "c.mk"                # oracle-native passes through
    assert map_move("j.mk", vocab) is None                 # aerial: no vocabulary
    assert map_move("jf.mk_legs", vocab) is None


def test_rule_from_rendered_line_parses_and_scores(table):
    """A rendered in-play line ("always throw up close") -> advice.read -> menu move throw_F+hp -> table throw."""
    rule = rule_from_line("always throw up close")
    assert rule.range == "close"
    sc = score_rule(rule, table)
    assert sc.scorable and sc.cells[0].move == "throw" and sc.verdict == RS.GOOD


# ---------------------------------------------------------------- the real book.json

def test_book_json_scores_sensibly(table):
    import json
    book = json.load(open(BOOK_PATH))
    per_opp = {}
    for opp, o in book["opponents"].items():
        scores = [score_rule(rule_from_claim(line["claim"], opp=None), table) for line in o["lines"]]
        per_opp[opp] = summarize(scores)
        # every verified line maps to a real table move (none fall out as not_scorable for vocabulary here)
        assert all(s.scorable for s in scores), [(s.rule.move, s.reason) for s in scores if not s.scorable]
        # "throw up close" is the table's best move up close -> every such line is good
        for s in scores:
            if s.rule.move == "throw" and s.rule.range == "close" and s.rule.when is None:
                assert s.verdict == RS.GOOD, (opp, s.rule.line, s.value, s.forward)
    print("book per-opponent:", {k: {x: v[x] for x in ("rules", "good", "ok", "bad", "not_scorable")}
                                  for k, v in per_opp.items()})
    assert set(per_opp) == {"dhalsim", "guile", "honda", "ken", "ryu", "zangief"}
