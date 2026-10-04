"""The value-table POLICY driving decisions (sf2/system1/value_table.decider + loop_runner.play_round decide=),
Stage 3a. The decider keys a Moment, restricts to the FOLLOWABLE action set at that range, and `choose` picks;
crediting its own choices makes the table converge. Pure (no emulator) via looptools.make_moment."""
import os
import random
import sys

sys.path.insert(0, os.path.dirname(__file__))
from looptools import make_moment                                   # noqa: E402

from sf2.system1 import value_table as VT                           # noqa: E402
from sf2.system1.advice import available_moves, char_categories, stance_of  # noqa: E402

CLOSE = available_moves(stance_of("stand", "close"), char_categories("chunli"))


def _row(action, net):
    return {"range": "close", "opp_state": "stand", "opp_air": False, "opp_shot": False,
            "his_label": "stand", "action": action, "dealt": max(net, 0), "taken": max(-net, 0)}


def test_decider_picks_a_followable_action_and_keys_the_when():
    decide = VT.decider(VT.blank(), "chunli", random.Random(0))
    d = decide(make_moment(dx=36, doing="standing"))               # dx 36 -> close
    assert d["action"] in CLOSE                                    # only a move she can actually play up close
    assert d["when"] == "close|standing|0"
    assert isinstance(d["explored"], bool) and d["category"]


def test_decider_never_proposes_an_unplayable_move():
    # far range: s.* offered, throws NOT -- the followable set already excludes them
    decide = VT.decider(VT.blank(), "chunli", random.Random(1))
    for _ in range(40):
        d = decide(make_moment(dx=150, doing="standing"))          # far
        assert "throw" not in d["action"]                          # throws are not followable far


def test_table_learns_from_the_deciders_own_choices():
    def net(a):
        return 15.0 if a == "throw_F+hp" else -5.0
    t, rng = VT.blank(), random.Random(0)
    for _ in range(300):
        d = VT.decider(t, "chunli", rng)(make_moment(dx=36, doing="standing"))
        t = VT.credit(t, [_row(d["action"], net(d["action"]))])
    assert VT.mean(t["cells"]["close|standing|0"].get("throw_F+hp")) > 0
    assert VT.decider(t, "chunli", rng, eps0=0.0)(make_moment(dx=36, doing="standing"))["action"] == "throw_F+hp"


# ----------------------------------------------------------------- HYBRID policy (text-laya + table override)
def _laya(_m):
    return {"action": "block_high", "category": "block", "source": "should-be-overwritten"}


def test_hybrid_defers_to_text_laya_when_the_table_is_not_confident():
    # empty table -> no confident cell -> text-laya's pick stands (source 'laya')
    decide = VT.hybrid_decider(VT.blank(), "chunli", random.Random(0), _laya, explore=0.0)
    d = decide(make_moment(dx=36, doing="standing"))
    assert d["action"] == "block_high" and d["source"] == "laya" and d["when"] == "close|standing|0"


def test_hybrid_overrides_with_a_confident_good_table_move():
    t = VT.blank()
    for _ in range(VT.MIN_TRIES):                            # the table learns throw is clearly good up close/standing
        t = VT.credit(t, [_row("throw_F+hp", 15)])
    decide = VT.hybrid_decider(t, "chunli", random.Random(0), _laya, explore=0.0)
    d = decide(make_moment(dx=36, doing="standing"))
    assert d["action"] == "throw_F+hp" and d["source"] == "table"   # overrode text-laya's block


def test_hybrid_does_not_override_with_a_known_bad_move():
    t = VT.blank()
    for _ in range(VT.MIN_TRIES):                            # a well-sampled but NEGATIVE move must NOT override
        t = VT.credit(t, [_row("spinning_bird_kick", -12)])
    decide = VT.hybrid_decider(t, "chunli", random.Random(0), _laya, explore=0.0)
    d = decide(make_moment(dx=36, doing="standing"))
    assert d["source"] == "laya"                            # defers; the table's move there is net-negative


def test_hybrid_overrides_sooner_than_the_exploration_threshold():
    # TABLE WEIGHT (owner 2026-10-04: laya is limited, lean on the table): a clearly-good move needs only
    # OVERRIDE_TRIES samples -- fewer than MIN_TRIES -- to override text-laya, so the table speaks with less data.
    assert VT.OVERRIDE_TRIES < VT.MIN_TRIES
    t = VT.blank()
    for _ in range(VT.OVERRIDE_TRIES):                      # fewer samples than the old MIN_TRIES gate required
        t = VT.credit(t, [_row("throw_F+hp", 15)])
    decide = VT.hybrid_decider(t, "chunli", random.Random(0), _laya, explore=0.0)
    d = decide(make_moment(dx=36, doing="standing"))
    assert d["action"] == "throw_F+hp" and d["source"] == "table"   # overrides on OVERRIDE_TRIES, not MIN_TRIES


def test_hybrid_still_floors_on_positive_mean_so_it_never_drifts_defensive():
    # the mean>0 FLOOR stays: in a cell where the table's only sampled move is net-negative, defer to laya's
    # neutral aggression rather than overriding to the least-bad (defensive) move -- guards the pure-table collapse.
    t = VT.blank()
    for _ in range(VT.OVERRIDE_TRIES + 2):
        t = VT.credit(t, [_row("block_high", -5)])
    decide = VT.hybrid_decider(t, "chunli", random.Random(0), _laya, explore=0.0)
    d = decide(make_moment(dx=36, doing="standing"))
    assert d["source"] == "laya"


def test_hybrid_explore_tries_an_under_sampled_move():
    decide = VT.hybrid_decider(VT.blank(), "chunli", random.Random(1), _laya, explore=1.0)
    picks = {decide(make_moment(dx=36, doing="standing"))["source"] for _ in range(20)}
    assert "table-explore" in picks                         # with explore=1 it tries under-sampled moves


def test_hybrid_coverage_target_reopens_exploration_in_a_saturated_cell():
    # gen-2 lesson (owner 2026-10-04): once every move in a cell is past the under-sampled bar, raising the explore
    # RATE does nothing (no move is "under"). Raising the coverage target (min_tries) is what re-opens exploration.
    from sf2.system1.advice import available_moves, stance_of, char_categories
    acts = available_moves(stance_of("stand", "close"), char_categories("chunli"))
    t = VT.blank()
    for a in acts:                                          # saturate EVERY close action to 25 tries (> 20, < 40)
        for _ in range(25):
            t = VT.credit(t, [_row(a, 1)])
    lo = VT.hybrid_decider(t, "chunli", random.Random(1), _laya, min_tries=20, explore=1.0)
    hi = VT.hybrid_decider(t, "chunli", random.Random(1), _laya, min_tries=40, explore=1.0)
    lo_src = {lo(make_moment(dx=36, doing="standing"))["source"] for _ in range(30)}
    hi_src = {hi(make_moment(dx=36, doing="standing"))["source"] for _ in range(30)}
    assert "table-explore" not in lo_src                    # nothing is under-sampled at threshold 20 -> no explore
    assert "table-explore" in hi_src                        # at threshold 40 the 25-try moves are under -> explores
