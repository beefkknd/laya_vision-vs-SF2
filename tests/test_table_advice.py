"""The lookup table as System 1's ranking WITH text laya (docs/prereg_2x2.md): the table's expected net (hp) becomes
the scores text laya's shortlist is built from, rated in words RELATIVE TO WALKING IN (forward's value in the same table
row): at or below it "likely fails", above it "may work", advice.NET_WORKS_MARGIN or more above it "likely works";
blocks on the same scale. Without an advisor the table's argmax plays as before; the P(hit) path is unchanged."""
import numpy as np
import pytest

from sf2.system1 import advice
from sf2.system1.advice import FAILS, FORWARD, MAY, WORKS, rating
from sf2.system1.advisor import choose, shortlist
from sf2.system1.system1 import System1, _close, choices

IMG = np.zeros((224, 256, 3), np.uint8)
NOTE = "me=chunli dist=close side=left dx=+20 my_bar=full opp_bar=full opp_airborne=0 opp_crouch=0 opp_attacking=0"
CELL = ("chunli", "close", 0, 0)
SIT = ("close", "standing", "full", "full")
TABLE = {CELL: {"throw": 6.0, "lk": 2.5, "block_high": 3.2, "mp": 1.0, "hp": -4.0, "forward": 1.0}}
WALK = 1.0


class FakeAdvisor:
    """Text laya stand-in: picks the first option offered (records what it was asked)."""

    def __init__(self):
        self.asked = []

    def ask(self, text, q):
        self.asked.append((text, q))
        first = next(iter(q["criteria"]))
        return {m: 1.0 if m == first else 0.0 for m in q["criteria"]}


# ---- item 1: ratings on the net scale, relative to walking in ----

@pytest.mark.parametrize("net,walk,want", [(3.0, 0.0, WORKS), (12.0, 0.0, WORKS), (2.99, 0.0, MAY), (0.01, 0.0, MAY),
                                           (0.0, 0.0, FAILS), (-20.0, 0.0, FAILS),
                                           (-1.12, -2.81, MAY), (0.2, -2.81, WORKS), (-2.81, -2.81, FAILS),
                                           (4.0, 2.0, MAY), (5.0, 2.0, WORKS), (1.9, 2.0, FAILS)])
def test_net_scale_words_relative_to_walking_in(net, walk, want):
    assert rating(net, "lk", scale="net", walk=walk) == want


def test_forward_defaults_to_zero():
    assert rating(0.5, "lk", scale="net") == MAY and rating(-0.5, "lk", scale="net") == FAILS


@pytest.mark.parametrize("block", ["block_high", "block_low"])
def test_blocks_use_the_same_net_scale(block):
    assert rating(3.0, block, scale="net", walk=0.0) == WORKS
    assert rating(1.0, block, scale="net", walk=0.0) == MAY
    assert rating(0.25, block, scale="net", walk=0.5) == FAILS     # 0.25 would be "likely works" on P(blocked)


def test_net_margin_is_the_registered_one():
    assert advice.NET_WORKS_MARGIN == 3.0
    assert not hasattr(advice, "NET_WORKS_AT") and not hasattr(advice, "NET_MAY_AT")


def test_p_hit_scale_is_unchanged_and_the_default():
    assert [rating(s) for s in (0.5, 0.49, 0.3, 0.29)] == [WORKS, MAY, MAY, FAILS]
    assert [rating(s, "block_low") for s in (0.2, 0.19, 0.1, 0.09)] == [WORKS, MAY, MAY, FAILS]
    assert rating(0.5, "lk", scale="p_hit") == rating(0.5, "lk")


def test_an_unknown_scale_is_refused():
    with pytest.raises(ValueError):
        rating(1.0, "lk", scale="nett")


def test_shortlist_on_the_net_scale():
    moves = ["throw", "lk", "block_high", "mp", "hp"]
    scores = {m: TABLE[CELL][m] for m in moves}
    got = shortlist(scores, [], moves, ("close", "standing"), scale="net", walk=WALK)
    assert got == {"throw": WORKS, "block_high": MAY, "lk": MAY, FORWARD: None}
    assert shortlist(scores, ["avoid throw up close"], moves, ("close", "standing"), scale="net", walk=WALK) == \
        {"block_high": MAY, "lk": MAY, "mp": FAILS, FORWARD: None}      # mp == forward's value: fails
    assert shortlist(scores, [], moves, ("close", "standing"), scale="net", walk=7.0)["throw"] == FAILS


def test_shortlist_default_is_the_p_hit_scale():
    moves = ["lp", "sweep", "hp", "block_low"]
    scores = {"lp": 0.7, "sweep": 0.35, "hp": 0.1, "block_low": 0.25}
    assert shortlist(scores, [], moves, ("close", "standing")) == \
        {"lp": WORKS, "sweep": MAY, "block_low": WORKS, FORWARD: None}


# ---- item 2: System 1's table mode with an advisor ----

def table_s1(advisor=None, advice_on=True, lessons=()):
    s = System1(None, "chunli", advisor=advisor, oracle=TABLE)
    s.advice_on = advice_on
    s.short = {"lessons": [{"text": t} for t in lessons]}
    return s


def test_without_an_advisor_the_table_argmax_plays_as_before():
    d = table_s1().decide(IMG, IMG, NOTE)
    assert d == {"action": "throw", "best": "throw", "p_hit": None, "predicted": "none", "probs": {},
                 "values": {m: TABLE[CELL].get(m, 0.0) for m in choices("chunli")}}


def test_with_an_advisor_text_laya_picks_from_the_tables_shortlist():
    adv = FakeAdvisor()
    d = table_s1(adv).decide(IMG, IMG, NOTE, SIT)
    assert d["shortlist"] == {"throw": WORKS, "block_high": MAY, "lk": MAY, FORWARD: None}
    assert d["action"] == "throw"                    # the fake picks the first option
    assert d["values"]["throw"] == 6.0 and d["values"]["forward"] == 1.0
    assert d["probs"] == {} and d["p_hit"] is None and d["predicted"] == "none"
    assert d["rule"] == "vision" and d["rule_answers"] == ["throw"]
    assert adv.asked[0][0].endswith("Advice: none.")


def test_a_lesson_names_a_move_onto_the_tables_shortlist():
    adv = FakeAdvisor()
    d = table_s1(adv, lessons=["use more hp up close"]).decide(IMG, IMG, NOTE, SIT)
    assert d["shortlist"]["hp"] == FAILS and "hp" in adv.asked[0][1]["criteria"]
    assert adv.asked[0][0].endswith("Advice: use more hp up close.")


def test_an_avoided_move_is_skipped_from_the_tables_top3():
    d = table_s1(FakeAdvisor(), lessons=["avoid throw up close"]).decide(IMG, IMG, NOTE, SIT)
    assert "throw" not in d["shortlist"] and d["shortlist"]["mp"] == FAILS


def test_the_no_advice_arm_tells_text_laya_advice_none():
    adv = FakeAdvisor()
    d = table_s1(adv, advice_on=False, lessons=["use more hp up close"]).decide(IMG, IMG, NOTE, SIT)
    assert "hp" not in d["shortlist"] and adv.asked[0][0].endswith("Advice: none.")


def test_the_no_advice_arm_matches_the_p_hit_paths_shape():
    """Same keys as the P(hit) path's no-advice arm (plus values), so the log and System 2 read both the same way."""
    adv = FakeAdvisor()
    table = table_s1(adv, advice_on=False).decide(IMG, IMG, NOTE, SIT)
    phit = dict(choose(FakeAdvisor(), SIT, {"lp": 0.6}, [], ["lp"]), best="lp", p_hit=0.6, predicted="hit",
                probs={"lp": 0.6})
    assert set(table) == set(phit) | {"values"}


def test_a_table_decision_logs_values_and_empty_scores():
    d = dict(table_s1(FakeAdvisor()).decide(IMG, IMG, NOTE, SIT), prompt=NOTE)
    r = {"p1_x": 100, "p2_x": 120, "p1_y": 192, "p2_y": 192, "p1_state": 0, "p2_state": 0, "p1_life": 176,
         "p2_life": 176, "p1_react": 0, "p2_react": 0, "timer": 0x99}
    e = _close(0, "chunli", "ryu", (r, [r], d, "hit", 0, None))
    assert e["values"]["throw"] == 6.0 and e["scores"] == {} and e["top3"] == [] and e["p_hit"] is None
    assert e["shortlist"]["throw"] == WORKS and e["rule"] == "vision"


def test_text_laya_walks_in_only_when_nothing_beats_walking_in():
    """The smoke of 2026-09-30 vs Ryu: the absolute scale walked in on 67 of 97 decisions, although forward was never
    the table's best there (e.g. forward -2.81, the best move -1.12)."""
    t = {CELL: {"spinning_bird_kick": -1.12, "mk": -1.5, "forward": -2.81}}
    d = System1(None, "chunli", advisor=FakeAdvisor(), oracle=t).decide(IMG, IMG, NOTE, SIT)
    assert d["rule"] == "vision" and FORWARD not in d["rule_answers"]
    assert set(d["shortlist"].values()) == {MAY, None}
    walk = System1(None, "chunli", advisor=FakeAdvisor(), oracle={CELL: {"lk": -3.0, "forward": 0.5}})
    assert walk.decide(IMG, IMG, NOTE, SIT)["rule"] == "walk"
