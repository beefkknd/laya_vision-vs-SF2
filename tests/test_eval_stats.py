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
