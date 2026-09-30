"""scripts/factorial_report.py: the 2x2 of docs/prereg_2x2.md - System 1 ranking {runs/all8, table} x advice {none,
Qwen's loop + book} - from lesson-loop runs paired by (opponent, seed): the four cells, both main effects and the
interaction per round, per opponent with the run as the unit and pooled with the opponent as the unit."""
import importlib.util
import json
import os
import sys

import pytest

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LABEL = "character_fgc+book"


def fr():
    if os.path.join(HERE, "scripts") not in sys.path:
        sys.path.insert(0, os.path.join(HERE, "scripts"))
    spec = importlib.util.spec_from_file_location("factorial_report", os.path.join(HERE, "scripts", "factorial_report.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def rounds(hp, taken=None):
    return [{"dealt": max(h, 0) + (taken or 0), "taken": max(-h, 0) + (taken or 0), "result": "win" if h > 0 else "loss"}
            for h in hp]


def acts(close, throws, mid_throws=0):
    return ([{"range": "close", "action": "throw"}] * throws + [{"range": "close", "action": "lk"}] * (close - throws)
            + [{"range": "mid", "action": "throw"}] * mid_throws)


def a_run(root, name, seed, loop, none, table, loop_acts=(), none_acts=(), hold=0.5, registered=2, book=True):
    d = os.path.join(root, name)
    for arm, rs, ac in (("loop", loop, loop_acts), ("none", none, none_acts)):
        os.makedirs(os.path.join(d, arm))
        with open(os.path.join(d, arm, "rounds.jsonl"), "w") as f:
            f.write("".join(json.dumps(r) + "\n" for r in rs))
        with open(os.path.join(d, arm, "actions.jsonl"), "w") as f:
            f.write("".join(json.dumps(a) + "\n" for a in ac))
    v = {"seed": seed, "prompt": "character_fgc", "violations": 0, "failed_jobs": [],
         "qwen_hold_rate": hold, "registered_at_end": ["x"] * registered}
    if book:
        v["book"] = "lessons/book.json"
    if table:
        v.update(oracle="lessons/value_oracle_v1.json", oracle_sha256="abc")
    with open(os.path.join(d, "verdict.json"), "w") as f:
        json.dump(v, f)
    return d


def world(root, opps=("ryu", "ken"), seeds=(1, 2, 3), n=10):
    """A0 = 0, A1 = +10, T0 = +30, T1 = +35 hp per round (T1 taken 5 more per round than T0)."""
    for opp in opps:
        for s in seeds:
            a_run(root, "2026100%d-000000_%s_character_fgc_book" % (s, opp), s, rounds([10] * n), rounds([0] * n),
                  False, loop_acts=acts(10, 0), none_acts=acts(10, 1), hold=0.4, registered=2)
            a_run(root, "2026100%d-000001_%s_character_fgc_book+table" % (s, opp), s, rounds([35] * n, taken=5),
                  rounds([30] * n), True, loop_acts=acts(10, 6, mid_throws=3), none_acts=acts(8, 4), hold=0.8,
                  registered=4)


def test_runs_pair_by_opponent_seed_and_ranking(tmp_path):
    r = str(tmp_path)
    world(r, opps=("ryu",), seeds=(1, 2))
    a_run(r, "20261009-000000_ryu_character_fgc_book", 9, rounds([1] * 10), rounds([0] * 10), False)   # no table twin
    a_run(r, "20261001-000000_ryu_character_fgc", 1, rounds([1] * 10), rounds([0] * 10), False, book=False)  # no book
    prs, unpaired = fr().pairs([r], LABEL)
    assert sorted(prs) == [("ryu", 1), ("ryu", 2)] and unpaired == [("ryu", 9)]
    assert prs[("ryu", 1)]["table"].endswith("+table") and not prs[("ryu", 1)]["all8"].endswith("+table")


def test_the_four_cells_and_the_effects_per_round(tmp_path):
    r = str(tmp_path)
    world(r, opps=("ryu",), seeds=(1,))
    prs, _ = fr().pairs([r], LABEL)
    e = fr().effects(fr().cells(prs[("ryu", 1)]))
    assert e["T0-A0"] == [30] * 10 and e["T1-T0"] == [5] * 10 and e["A1-A0"] == [10] * 10
    assert e["table"] == [27.5] * 10 and e["advice"] == [7.5] * 10
    assert e["interaction"] == [-5] * 10                       # (T1 - T0) - (A1 - A0)


def test_unequal_rounds_refuse_the_pair(tmp_path):
    r = str(tmp_path)
    a_run(r, "1_ryu_character_fgc_book", 1, rounds([10] * 10), rounds([0] * 10), False)
    a_run(r, "2_ryu_character_fgc_book+table", 1, rounds([35] * 11), rounds([30] * 11), True)
    rep = fr().report([r], LABEL)
    assert rep["pairs"] == [] and "rounds" in rep["problems"][0]


def test_report_per_opponent_run_as_unit_and_pooled(tmp_path):
    r = str(tmp_path)
    world(r)
    rep = fr().report([r], LABEL)
    assert len(rep["pairs"]) == 6 and rep["problems"] == []
    ryu = rep["effects"]["interaction"]["per_opp"]["ryu"]
    assert ryu["runs"] == 3 and ryu["rounds"] == 30 and ryu["mean"] == pytest.approx(-5)
    pooled = rep["effects"]["table"]["pooled"]
    assert pooled["opponents"] == 2 and pooled["mean"] == pytest.approx(27.5) and pooled["verdict"] == "HELPS"
    assert rep["effects"]["advice"]["pooled"]["mean"] == pytest.approx(7.5)
    assert rep["effects"]["interaction"]["pooled"]["verdict"] == "HURTS"


def test_cell_stats_hp_taken_won_and_throws_per_close_decision(tmp_path):
    r = str(tmp_path)
    world(r)
    c = fr().report([r], LABEL)["cells"]
    assert c["A0"]["hp"] == 0 and c["T1"]["hp"] == pytest.approx(35) and c["T0"]["hp"] == pytest.approx(30)
    assert c["T1"]["taken"] == pytest.approx(5) and c["T0"]["taken"] == 0
    assert c["T1"]["won"] == 60 and c["A0"]["won"] == 0 and c["T1"]["rounds"] == 60
    assert c["T1"]["throws_per_close"] == pytest.approx(0.6)      # the mid-range throws do not count
    assert c["T0"]["throws_per_close"] == pytest.approx(0.5) and c["A1"]["throws_per_close"] == 0
    assert c["A0"]["throws_per_close"] == pytest.approx(0.1)


def test_qwen_per_ranking(tmp_path):
    r = str(tmp_path)
    world(r)
    q = fr().report([r], LABEL)["qwen"]
    assert q["all8"] == {"runs": 6, "registered_mean": 2, "hold_rate_mean": pytest.approx(0.4)}
    assert q["table"] == {"runs": 6, "registered_mean": 4, "hold_rate_mean": pytest.approx(0.8)}


def test_no_pairs_no_verdict(tmp_path):
    rep = fr().report([str(tmp_path)], LABEL)
    assert rep["pairs"] == [] and rep["effects"]["table"]["pooled"]["verdict"] == "NO VERDICT"
