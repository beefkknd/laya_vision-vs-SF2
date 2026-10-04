"""Pooled per-rule stats (sf2.system2.rule_stats), Step 1B: the fix for 'nothing ever graduates'.

RED first: the block-local scorer sees only one block, so a rule with ~12 tries/block stays "few" forever
(MIN_TRIES=20 on both sides). Pooling the same sums across blocks reaches MIN_TRIES and the SAME verdict the
scorer would give on the pooled rows -- proven by `evidence_from_stats == condition_evidence`.
"""
import pytest

from sf2.system2 import lessons as L
from sf2.system2 import rule_stats as RS
from sf2.system2.move_coach import MIN_TRIES


def _claim(move="s.mk", rng="mid", when="standing", kind="use_more"):
    return {"kind": kind, "move": move, "range": rng, "when": when, "view": None}


def _row(action, dealt, taken, rng="mid", doing="standing"):
    # a decision row shaped like screen_evidence's drows (what condition_evidence consumes)
    return {"range": rng, "action": action, "dealt": dealt, "taken": taken,
            "opp_state": doing, "his_air": False}


def _rows(n_mine, n_rest, mine_net=30, rest_net=-10):
    rows = [_row("s.mk", mine_net + 20, 20) for _ in range(n_mine)]          # dealt-taken = mine_net
    rows += [_row("walk_back", 0, -rest_net) for _ in range(n_rest)]         # dealt-taken = rest_net
    return rows


def test_evidence_from_stats_equals_condition_evidence_on_one_block():
    c = _claim()
    rows = _rows(25, 25)                                                     # >= MIN_TRIES both sides -> a real verdict
    want = L.condition_evidence(rows, c)
    got = RS.evidence_from_stats(RS.tally(None, rows, {"result": "win"}, c))
    assert got["tries"] == want["tries"] and got["others"] == want["others"]
    assert got["cls"] == want["cls"]
    for k in ("net", "base", "diff", "lo", "hi"):
        assert got[k] == pytest.approx(want[k]), k


def test_pooling_across_blocks_reaches_min_tries_where_one_block_cannot():
    c = _claim()
    per_block = _rows(12, 12)                                               # 12 < MIN_TRIES -> "few" per block
    assert L.condition_evidence(per_block, c)["cls"] == "few"              # the bug: one block never judges
    st = RS.tally(None, per_block, {"result": "loss"}, c)
    st = RS.tally(st, per_block, {"result": "win"}, c)                      # a second block, pooled
    ev = RS.evidence_from_stats(st)
    assert ev["tries"] >= MIN_TRIES and ev["cls"] != "few"                  # now it can be judged


def test_round_correlation_counts_fired_vs_idle():
    c = _claim()
    st = RS.tally(None, _rows(3, 1), {"result": "win"}, c)                  # she used s.mk -> fired, won
    st = RS.tally(st, _rows(0, 2), {"result": "loss"}, c)                   # not used -> idle, lost
    assert st["rounds_fired"] == 1 and st["wins_fired"] == 1
    assert st["rounds_idle"] == 1 and st["wins_idle"] == 0


def test_blank_and_immutable():
    c = _claim()
    base = RS.blank()
    out = RS.tally(base, _rows(2, 2), {"result": "win"}, c)
    assert base["n_mine"] == 0 and out["n_mine"] == 2                       # input not mutated
