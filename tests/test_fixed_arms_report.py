"""scripts/fixed_arms_report.py: fixed-advice arms vs no advice over a batch of ab_memory runs, the seed as the unit."""
import importlib.util
import json
import os
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def mod():
    sys.path.insert(0, os.path.join(HERE, "scripts"))
    spec = importlib.util.spec_from_file_location("far", os.path.join(HERE, "scripts", "fixed_arms_report.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def batch(tmp, seeds):
    logs = os.path.join(tmp, "logs")
    os.makedirs(logs)
    for s, (throw, none) in enumerate(seeds):
        root = os.path.join(tmp, "run%d" % s)
        for arm, hps in (("none", none), ("throw", throw), ("cmk", none)):
            os.makedirs(os.path.join(root, "ryu_" + arm))
            with open(os.path.join(root, "ryu_" + arm, "rounds.jsonl"), "w") as f:
                f.write("".join(json.dumps({"dealt": max(h, 0), "taken": max(-h, 0),
                                            "result": "win" if h > 0 else "loss"}) + "\n" for h in hps))
        with open(os.path.join(logs, "seed_%d.log" % s), "w") as f:
            f.write("9 headless runs\nsaved %s\n" % root)
    return logs


def test_arms_are_paired_per_seed_with_the_seed_as_unit(tmp_path):
    logs = batch(str(tmp_path), [([30, 10], [0, 0]), ([50, 30], [0, 0]), ([40, 20], [0, 0])])
    r = mod().report(logs, "ryu", ["throw", "cmk"])
    assert r["seeds"] == 3 and r["rounds_per_arm"] == 6
    assert r["vs_none"]["throw"]["mean"] == 30 and r["vs_none"]["throw"]["runs"] == 3
    assert r["vs_none"]["cmk"]["mean"] == 0 and r["throw_minus_cmk"]["mean"] == 30
    assert r["wins"] == {"throw": 6, "cmk": 0, "none": 0}


def test_a_log_without_a_saved_run_stops_the_report(tmp_path):
    logs = batch(str(tmp_path), [([30, 10], [0, 0])])
    with open(os.path.join(logs, "seed_9.log"), "w") as f:
        f.write("crashed\n")
    try:
        mod().report(logs, "ryu", [])
    except SystemExit as e:
        assert "no run saved" in str(e)
    else:
        raise AssertionError("a failed seed must not be silently dropped")
