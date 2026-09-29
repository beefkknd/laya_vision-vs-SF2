"""scripts/compare_runs.py: two runs of the same lock and seed, compared file by file - identical, or where they part."""
import importlib.util
import json
import os
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def cmp():
    if os.path.join(HERE, "scripts") not in sys.path:
        sys.path.insert(0, os.path.join(HERE, "scripts"))
    spec = importlib.util.spec_from_file_location("compare_runs", os.path.join(HERE, "scripts", "compare_runs.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def write(d, name, rows):
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, name), "w") as f:
        f.write("".join(json.dumps(r) + "\n" for r in rows))


def test_identical_files_pass_and_the_first_difference_is_named(tmp_path):
    a, b = str(tmp_path / "a"), str(tmp_path / "b")
    for d in (a, b):
        write(os.path.join(d, "none"), "rounds.jsonl", [{"round": 0, "dealt": 5}, {"round": 1, "dealt": 7}])
    write(os.path.join(a, "loop"), "rounds.jsonl", [{"round": 0, "dealt": 5}, {"round": 1, "dealt": 7}])
    write(os.path.join(b, "loop"), "rounds.jsonl", [{"round": 0, "dealt": 5}, {"round": 1, "dealt": 9}])
    rows = {(r["arm"], r["file"]): r for r in cmp().compare(a, b, files=("rounds.jsonl",))}
    assert rows[("none", "rounds.jsonl")]["same"] and rows[("none", "rounds.jsonl")]["lines"] == (2, 2)
    loop = rows[("loop", "rounds.jsonl")]
    assert not loop["same"] and loop["first_diff"] == 1 and loop["where"] == {"round": 1} and loop["keys"] == ["dealt"]


def test_a_missing_file_is_a_difference(tmp_path):
    a, b = str(tmp_path / "a"), str(tmp_path / "b")
    write(os.path.join(a, "none"), "rounds.jsonl", [{"round": 0}])
    os.makedirs(os.path.join(b, "none"))
    r = cmp().compare(a, b, arms=("none",), files=("rounds.jsonl",))[0]
    assert not r["same"] and r["lines"] == (1, 0)
