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


def rounds(hp, taken=None, per_game=3):
    return [{"dealt": max(h, 0) + (taken or 0), "taken": max(-h, 0) + (taken or 0), "result": "win" if h > 0 else "loss",
             "round": i, "game": i // per_game} for i, h in enumerate(hp)]


def acts(close, throws, mid_throws=0):
    return ([{"range": "close", "action": "throw"}] * throws + [{"range": "close", "action": "lk"}] * (close - throws)
            + [{"range": "mid", "action": "throw"}] * mid_throws)


def a_run(root, name, seed, loop, none, table, loop_acts=(), none_acts=(), hold=0.5, registered=2, book=True,
          extra=None, run=None):
    d = os.path.join(root, name)
    for arm, rs, ac in (("loop", loop, loop_acts), ("none", none, none_acts)):
        os.makedirs(os.path.join(d, arm))
        with open(os.path.join(d, arm, "rounds.jsonl"), "w") as f:
            f.write("".join(json.dumps(r) + "\n" for r in rs))
        with open(os.path.join(d, arm, "actions.jsonl"), "w") as f:
            f.write("".join(json.dumps(a) + "\n" for a in ac))
        if run is not None:
            with open(os.path.join(d, arm, "run.json"), "w") as f:
                json.dump(dict(run, arm=arm), f)
    v = {"seed": seed, "prompt": "character_fgc", "violations": 0, "failed_jobs": [],
         "qwen_hold_rate": hold, "registered_at_end": ["x"] * registered}
    if book:
        v["book"] = "lessons/book.json"
    if table:
        v.update(oracle="lessons/value_oracle_v1.json", oracle_sha256="abc")
    v.update(extra or {})
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
    prs, unpaired = fr().pairs([r], LABEL, min_rounds=10)
    assert sorted(prs) == [("ryu", 1), ("ryu", 2)] and unpaired == [("ryu", 9)]
    assert prs[("ryu", 1)]["table"].endswith("+table") and not prs[("ryu", 1)]["all8"].endswith("+table")


def test_the_four_cells_and_the_effects_per_round(tmp_path):
    r = str(tmp_path)
    world(r, opps=("ryu",), seeds=(1,))
    prs, _ = fr().pairs([r], LABEL, min_rounds=10)
    e = fr().effects(fr().cells(prs[("ryu", 1)]))
    assert e["T0-A0"] == [30] * 10 and e["T1-T0"] == [5] * 10 and e["A1-A0"] == [10] * 10
    assert e["table"] == [27.5] * 10 and e["advice"] == [7.5] * 10
    assert e["interaction"] == [-5] * 10                       # (T1 - T0) - (A1 - A0)


def test_unequal_rounds_refuse_the_pair(tmp_path):
    r = str(tmp_path)
    a_run(r, "1_ryu_character_fgc_book", 1, rounds([10] * 10), rounds([0] * 10), False)
    a_run(r, "2_ryu_character_fgc_book+table", 1, rounds([35] * 11), rounds([30] * 11), True)
    rep = fr().report([r], LABEL, min_rounds=10)
    assert rep["pairs"] == [] and "rounds" in rep["problems"][0]


def test_report_per_opponent_run_as_unit_and_pooled(tmp_path):
    r = str(tmp_path)
    world(r)
    rep = fr().report([r], LABEL, min_rounds=10)
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
    c = fr().report([r], LABEL, min_rounds=10)["cells"]
    assert c["A0"]["hp"] == 0 and c["T1"]["hp"] == pytest.approx(35) and c["T0"]["hp"] == pytest.approx(30)
    assert c["T1"]["taken"] == pytest.approx(5) and c["T0"]["taken"] == 0
    assert c["T1"]["won"] == 60 and c["A0"]["won"] == 0 and c["T1"]["rounds"] == 60
    assert c["T1"]["throws_per_close"] == pytest.approx(0.6)      # the mid-range throws do not count
    assert c["T0"]["throws_per_close"] == pytest.approx(0.5) and c["A1"]["throws_per_close"] == 0
    assert c["A0"]["throws_per_close"] == pytest.approx(0.1)


def test_qwen_per_ranking(tmp_path):
    r = str(tmp_path)
    world(r)
    q = fr().report([r], LABEL, min_rounds=10)["qwen"]
    assert q["all8"] == {"runs": 6, "registered_mean": 2, "hold_rate_mean": pytest.approx(0.4)}
    assert q["table"] == {"runs": 6, "registered_mean": 4, "hold_rate_mean": pytest.approx(0.8)}


def test_no_pairs_no_verdict(tmp_path):
    rep = fr().report([str(tmp_path)], LABEL, min_rounds=10)
    assert rep["pairs"] == [] and rep["effects"]["table"]["pooled"]["verdict"] == "NO VERDICT"


# ---- which run of a (opponent, seed, ranking), and whether a pair may be compared (blind review 2026-09-30) ----

N = 30          # the registered count: qwen_lessons' defaults, 10 games x 3 rounds per arm


def one_pair(r, all8=None, table=None, all8_run=None, table_run=None, seed=1, opp="ryu", stamp="20261001"):
    a_run(r, "%s-000000_%s_character_fgc_book" % (stamp, opp), seed, rounds([10] * N), rounds([0] * N), False,
          extra=all8, run=all8_run)
    a_run(r, "%s-000001_%s_character_fgc_book+table" % (stamp, opp), seed, rounds([35] * N), rounds([30] * N), True,
          extra=table, run=table_run)


def test_the_registered_round_count_is_the_default(tmp_path):
    assert fr().MIN_ROUNDS == 30
    r = str(tmp_path)
    world(r, opps=("ryu",), seeds=(1,))                   # 10-round runs: smoke-sized under the default
    prs, _ = fr().pairs([r], LABEL)
    assert prs == {}


def test_the_newest_complete_run_wins_and_the_others_are_listed(tmp_path):
    r = str(tmp_path)
    one_pair(r)
    a_run(r, "20261002-000000_ryu_character_fgc_book", 1, rounds([50] * N), rounds([0] * N), False)   # a good rerun
    a_run(r, "20261003-000000_ryu_character_fgc_book", 1, rounds([99] * N), rounds([0] * 12), False)  # crashed none arm
    a_run(r, "20261004-000000_ryu_character_fgc_book", 1, rounds([99] * 3), rounds([0] * 3), False)   # a smoke run
    a_run(r, "20261005-000000_ryu_character_fgc_book", 1, rounds([99] * 33), rounds([0] * N), False)  # unequal arms
    rep = fr().report([r], LABEL)
    assert [os.path.basename(p[2]) for p in rep["pairs"]] == ["20261002-000000_ryu_character_fgc_book"]
    passed = {os.path.basename(d): why for d, why in rep["passed_over"]}
    assert set(passed) == {"20261001-000000_ryu_character_fgc_book", "20261003-000000_ryu_character_fgc_book",
                           "20261004-000000_ryu_character_fgc_book", "20261005-000000_ryu_character_fgc_book"}
    assert "33 vs 30" in passed["20261005-000000_ryu_character_fgc_book"]
    assert "newer" in passed["20261001-000000_ryu_character_fgc_book"]
    assert "none 12" in passed["20261003-000000_ryu_character_fgc_book"]
    assert "rounds" in passed["20261004-000000_ryu_character_fgc_book"]
    assert rep["effects"]["A1-A0"]["per_opp"]["ryu"]["mean"] == 50


@pytest.mark.parametrize("field,a,b", [("book_sha256", "ee8a", "ffff"), ("forward_lessons", False, True),
                                       ("lock", None, "lesson_loop_v1")])
def test_a_pair_whose_verdicts_differ_is_refused(tmp_path, field, a, b):
    r = str(tmp_path)
    one_pair(r, all8={field: a}, table={field: b})
    rep = fr().report([r], LABEL)
    assert rep["pairs"] == [] and field in rep["problems"][0]


@pytest.mark.parametrize("field,a,b", [("advisor", "runs/text_laya/advice_v1", "runs/text_laya/other"),
                                       ("commit", "abc", "def"), ("games", 10, 5), ("rounds", 3, 6)])
def test_a_pair_whose_run_files_differ_is_refused(tmp_path, field, a, b):
    r = str(tmp_path)
    one_pair(r, all8_run={field: a}, table_run={field: b})
    rep = fr().report([r], LABEL)
    assert rep["pairs"] == [] and field in rep["problems"][0]


def test_rounds_per_game_are_read_from_the_rounds_when_not_recorded(tmp_path):
    r = str(tmp_path)
    a_run(r, "1_ryu_character_fgc_book", 1, rounds([10] * N), rounds([0] * N), False)
    a_run(r, "2_ryu_character_fgc_book+table", 1, rounds([35] * N, per_game=6), rounds([30] * N, per_game=6), True)
    rep = fr().report([r], LABEL)
    assert rep["pairs"] == [] and "games" in rep["problems"][0]


def test_every_table_run_uses_the_same_table_and_every_all8_run_the_same_model(tmp_path):
    r = str(tmp_path)
    one_pair(r, seed=1, stamp="20261001", all8_run={"model": "runs/all8/best"}, table={"oracle_sha256": "abc"})
    one_pair(r, seed=2, stamp="20261002", all8_run={"model": "runs/all8/best"}, table={"oracle_sha256": "zzz"})
    rep = fr().report([r], LABEL)
    assert any("oracle_sha256" in p for p in rep["problems"])
    r2 = str(tmp_path / "b")
    one_pair(r2, seed=1, stamp="20261001", all8_run={"model": "runs/all8/best"})
    one_pair(r2, seed=2, stamp="20261002", all8_run={"model": "runs/all8/other"})
    rep2 = fr().report([r2], LABEL)
    assert any("model" in p for p in rep2["problems"])


def test_a_clean_pair_has_no_problems_and_main_exits_0(tmp_path):
    r = str(tmp_path)
    same = {"advisor": "a", "commit": "c", "games": 10, "rounds": 3}
    one_pair(r, all8_run=dict(same, model="runs/all8/best"), table_run=dict(same, model=None))
    rep = fr().report([r], LABEL)
    assert len(rep["pairs"]) == 1 and rep["problems"] == []
    assert fr().main([r]) == 0
    one_pair(str(tmp_path / "bad"), all8={"book_sha256": "x"}, table={"book_sha256": "y"})
    assert fr().main([str(tmp_path / "bad")]) == 1


def test_another_prompt_never_pairs(tmp_path):
    r = str(tmp_path)
    one_pair(r, table={"prompt": "character"})              # its label is character+book+table: not this report's
    rep = fr().report([r], LABEL)
    assert rep["pairs"] == [] and rep["unpaired"] == [["ryu", 1]]
