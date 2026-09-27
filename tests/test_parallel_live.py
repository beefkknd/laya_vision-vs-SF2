"""parallel.py copies every worker's output into its own log and one shared, tail-able log (out/live.log)."""
import io
import os
import sys
import threading

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))
import parallel  # noqa: E402


def test_pump_writes_each_line_to_the_worker_log_and_the_live_log_with_its_name():
    class Proc:
        stdout = iter(["match 0 round 0: me\n", "done\n"])

    own, live = io.StringIO(), io.StringIO()
    parallel.pump(Proc(), own, "ryu_A_dev_w1", live, threading.Lock())
    assert own.getvalue() == "match 0 round 0: me\ndone\n"
    lines = live.getvalue().splitlines()
    assert len(lines) == 2 and all("[ryu_A_dev_w1] " in x for x in lines)
    assert lines[0].endswith("match 0 round 0: me") and lines[0][2] == ":"  # HH:MM:SS first


def test_every_finished_run_appends_one_row_to_the_results_ledger(tmp_path):
    path = tmp_path / "results.jsonl"
    gate = {"rounds": 43, "round_win_rate": 0.93, "net_damage_per_round": 93.8, "net_damage_se": 7.6,
            "matches": 20, "distinct_matches": 20, "action_mix": {"hk": 0.3}}
    argv = ["--savestate", "states/arcade_chunli_vs_ryu.state", "--me", "chunli", "--opp", "ryu"]
    for name in ("ryu_T_dev", "ryu_A_dev"):
        parallel.record_result(str(path), name, "play_student", "runs/r0/best", "openings/dev.txt", 4, argv, gate)
    import json

    rows = [json.loads(x) for x in path.read_text().splitlines()]
    assert [r["name"] for r in rows] == ["ryu_T_dev", "ryu_A_dev"]  # appended, never overwritten
    r = rows[0]
    assert r["opp"] == "ryu" and r["savestate"].endswith("ryu.state") and r["openings"] == "openings/dev.txt"
    assert r["net_damage_per_round"] == 93.8 and r["rounds"] == 43 and "time" in r and "action_mix" not in r

