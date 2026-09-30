"""scripts/value_inplay.py: the pre-registered in-play test's report (docs/prereg_lv_value.md) on tiny fixtures."""
import importlib.util
import json
import os
import sys

import pytest

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, "..", "scripts"))
spec = importlib.util.spec_from_file_location("value_inplay", os.path.join(HERE, "..", "scripts", "value_inplay.py"))
vi = importlib.util.module_from_spec(spec)
spec.loader.exec_module(vi)


def write(root, arm, opp, seed, hps, throws=0, close=4):
    d = os.path.join(root, arm, "%s_s%d" % (opp, seed), "chunli")
    os.makedirs(d)
    with open(os.path.join(d, "games.jsonl"), "w") as f:
        for i, h in enumerate(hps):
            f.write(json.dumps({"game": i, "dealt": max(h, 0) + 50, "taken": 50 - min(h, 0)}) + "\n")
    with open(os.path.join(d, "actions.jsonl"), "w") as f:
        for i in range(close):
            f.write(json.dumps({"range": "close", "action": "throw" if i < throws else "lk"}) + "\n")


def test_report_pairs_by_seed_and_pools(tmp_path):
    root = str(tmp_path)
    for opp in ("ryu", "ken"):
        for s in (1, 2, 3):
            write(root, "old", opp, s, [0] * 10)
            write(root, "new", opp, s, [10] * 10, throws=2)
    rep = vi.report(root, "new", "old")
    assert rep["per_opp"]["ryu"]["mean"] == 10 and rep["per_opp"]["ryu"]["runs"] == 3
    assert rep["pooled"]["mean"] == 10 and rep["pooled"]["verdict"] == "HELPS"
    assert rep["throw_close"]["new"]["ryu"] == 0.5 and rep["throw_close"]["old"]["ryu"] == 0.0
    assert rep["success"] is True


def test_one_opponent_clearly_worse_is_not_success(tmp_path):
    root = str(tmp_path)
    for opp, d in (("ryu", 60), ("ken", 60), ("guile", 60), ("zangief", 60), ("dhalsim", 60), ("honda", -5)):
        for s in (1, 2, 3):
            write(root, "old", opp, s, [0] * 10)
            write(root, "new", opp, s, [d + (s - 2)] * 10, throws=1)
    rep = vi.report(root, "new", "old")
    assert rep["pooled"]["verdict"] == "HELPS"          # isolates the per-opponent rule
    assert rep["per_opp"]["honda"]["verdict"] == "HURTS" and rep["success"] is False


def test_unpaired_seed_refused(tmp_path):
    root = str(tmp_path)
    write(root, "old", "ryu", 1, [0] * 10)
    write(root, "new", "ryu", 1, [0] * 9)
    with pytest.raises(ValueError):
        vi.report(root, "new", "old")
    write(root, "new", "ken", 2, [0] * 10)
    with pytest.raises(ValueError):
        vi.report(root, "new", "old")
