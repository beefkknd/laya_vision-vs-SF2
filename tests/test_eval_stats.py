"""The A/B statistics: pairs must line up, the opponent is the unit when pooling, a failed job means no verdict."""
import random

import pytest

from sf2.eval.stats import ci, paired, pooled, summarize


def rounds(hps):
    return [{"dealt": max(h, 0), "taken": max(-h, 0), "result": "win" if h > 0 else "loss"} for h in hps]


def noisy(mean, n, seed):
    r = random.Random(seed)
    return [round(mean + r.gauss(0, 5)) for _ in range(n)]


def test_pairs_must_line_up():
    with pytest.raises(ValueError, match="3 rounds vs 2"):
        paired(rounds([1, 2, 3]), rounds([1, 2]))
    assert paired(rounds([5, 0]), rounds([1, 2])) == [4, -2]


def test_ci_needs_two_rounds():
    m, lo, hi = ci([3.0])
    assert m == 3.0 and lo != lo and hi != hi                              # nan: no interval from one round


def test_one_strong_opponent_does_not_make_a_pooled_win():
    # 100 rounds each; ryu +20 per round, ken -2. Pooling rounds as independent says HELPS (round-level CI far
    # above 0); with the opponent as the unit, two opponents cannot show it.
    d = {"ryu": noisy(20, 100, 1), "ken": noisy(-2, 100, 2)}
    naive = ci(d["ryu"] + d["ken"])
    assert naive[1] > 0
    p = pooled(d)
    assert p["opponents"] == 2 and p["verdict"] == "NOT SHOWN" and p["ci95"][0] <= 0


def test_consistent_gain_over_many_opponents_helps():
    d = {o: noisy(8, 30, i) for i, o in enumerate(["ryu", "ken", "honda", "zangief", "dhalsim", "guile", "blanka"])}
    assert pooled(d)["verdict"] == "HELPS"


def test_summary_with_a_failed_job_or_short_arm_gives_no_verdict():
    base = rounds(noisy(0, 30, 3))
    data = {"ryu": {"none": base, "code_short": rounds(noisy(10, 30, 4))},
            "ken": {"none": base, "code_short": rounds(noisy(10, 29, 5))}}          # one round short
    s = summarize(data, ["none", "code_short"])
    assert "vs_none" not in s["per_opp"]["ken"]["code_short"]
    assert s["per_opp"]["ken"]["code_short"]["error"].startswith("29 rounds vs 30")
    assert s["pooled"]["code_short"]["verdict"] == "NO VERDICT"
    ok = {"ryu": data["ryu"], "ken": {"none": base, "code_short": rounds(noisy(10, 30, 6))}}
    assert summarize(ok, ["none", "code_short"])["pooled"]["code_short"]["verdict"] != "NO VERDICT"
    assert summarize(ok, ["none", "code_short"], failed=[("ryu", "code_short")])["pooled"]["code_short"][
        "verdict"] == "NO VERDICT"


def test_too_little_evidence_gets_no_verdict():             # a 1-round smoke once printed "HURTS" (2026-09-29)
    one = pooled({"ryu": [-129]})
    assert one["verdict"] == "TOO FEW" and "rounds" in one["why"]
    assert pooled({"ryu": noisy(-20, 30, 1)})["verdict"] == "TOO FEW"          # one opponent is not "in general"
    assert pooled({"ryu": noisy(-20, 30, 1), "ken": noisy(-20, 30, 2)})["verdict"] == "HURTS"


def test_pooled_names_opponents_an_arm_did_not_play():         # e.g. qwen skipped: no memory for that opponent
    base = rounds(noisy(0, 30, 3))
    data = {"ryu": {"none": base, "qwen": rounds(noisy(5, 30, 4))}, "ken": {"none": base},
            "honda": {"none": base, "qwen": rounds(noisy(5, 30, 5))}}
    assert summarize(data, ["none", "qwen"])["pooled"]["qwen"]["missing"] == ["ken"]


def test_slope_per_round():
    from sf2.eval.stats import slope
    assert slope([1, 2, 3, 4]) == 1.0 and slope([5]) == 0.0 and slope([3, 1]) == -2.0


def test_runs_pool_by_appending_each_runs_pairs(tmp_path):
    """Several A/B runs (different seeds) pool per opponent; a run where an arm is short makes it unpaired."""
    import json
    import os
    from sf2.eval.stats import load_runs

    def write(root, opp, arm, hps):
        os.makedirs(os.path.join(root, "%s_%s" % (opp, arm)))
        with open(os.path.join(root, "%s_%s" % (opp, arm), "rounds.jsonl"), "w") as f:
            f.writelines(json.dumps(r) + "\n" for r in rounds(hps))
    a, b = str(tmp_path / "a"), str(tmp_path / "b")
    write(a, "ryu", "none", [0, 0])
    write(a, "ryu", "code_short", [5, 5])
    write(b, "ryu", "none", [1])
    write(b, "ryu", "code_short", [9])
    data = load_runs([a, b], ["ryu"], ["none", "code_short"])
    assert paired(data["ryu"]["code_short"], data["ryu"]["none"]) == [5, 5, 8]
    write(b, "ken", "none", [1, 1])
    write(b, "ken", "code_short", [9])
    with pytest.raises(ValueError, match="ken code_short"):
        load_runs([a, b], ["ryu", "ken"], ["none", "code_short"])


# --- the run as the unit (2026-09-29 review, finding 1): every round of a run shares its lessons ---

from sf2.eval.stats import MIN_RUNS, run_level  # noqa: E402


def test_run_level_mean_is_the_mean_of_run_means():
    r = run_level([[10, 10], [0, 0, 0, 0], [20]])
    assert r["mean"] == 10.0 and (r["runs"], r["rounds"]) == (3, 7)


def test_runs_that_disagree_widen_the_interval_rounds_as_unit_would_hide():
    # 6 runs x 30 rounds: each run's rounds sit tightly around its own mean, the runs disagree (-10 .. +30).
    runs = [noisy(m, 30, i) for i, m in enumerate([-10, 0, 10, 20, 30, 5])]
    flat = ci([x for r in runs for x in r])
    assert flat[1] > 0                                             # rounds as unit: "HELPS"
    r = run_level(runs)
    assert r["ci95"][0] < 0 < r["ci95"][1] and r["verdict"] == "NOT SHOWN"


def test_consistent_runs_help():
    r = run_level([noisy(20, 30, i) for i in range(5)])
    assert r["verdict"] == "HELPS" and r["ci95"][0] > 10


def test_fewer_than_min_runs_is_too_few():
    r = run_level([noisy(50, 30, i) for i in range(MIN_RUNS - 1)])
    assert r["verdict"] == "TOO FEW" and "runs" in r["why"] and r["runs"] == MIN_RUNS - 1


def test_run_level_ignores_empty_runs_and_handles_none():
    assert run_level([])["verdict"] == "NO VERDICT"
    r = run_level([[], [1], [2], [3]])
    assert r["runs"] == 3 and r["mean"] == 2.0
