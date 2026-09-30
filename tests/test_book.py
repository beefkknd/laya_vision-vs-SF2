"""scripts/book.py: the book of verified players' tips (lessons/book.json), built from pre-registered fixed-advice
batches (scripts/fixed_arms_report.py over logs/ab/<opp>_expert). A tip is verified when its SINGLE-line arm helps vs
no advice (run-level 95% interval above 0); a multi-line set never enters, nor a line naming forward (text laya
cannot follow it). Deterministic: the same batches give the same bytes."""
import hashlib
import importlib.util
import json
import os
import sys

import pytest

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.environ.get("SF2_DATA", HERE)


def mod():
    sys.path.insert(0, os.path.join(HERE, "scripts"))
    spec = importlib.util.spec_from_file_location("book", os.path.join(HERE, "scripts", "book.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


ARMS = {"throw": ["use more throw up close"], "cmk": ["use more c.mk at mid range when he stands"],
        "expert": ["use more throw up close", "use more sweep when he attacks"],
        "walk": ["use more forward at mid range"], "hurt": ["avoid spinning_bird_kick"]}
GAIN = {"throw": 40, "cmk": 0, "expert": 30, "walk": 20, "hurt": -30}


def batch(data, opp="ryu", seeds=(41001, 41002, 41003), arms=ARMS):
    """A fake batch under ``data``: logs/ab/<opp>_expert/seed_S.log -> rollouts/ab/rS (relative, as the real logs)."""
    logs = os.path.join(data, "logs", "ab", "%s_expert" % opp)
    os.makedirs(logs)
    for i, s in enumerate(seeds):
        rel = os.path.join("rollouts", "ab", "r%d_%s" % (s, opp))
        root = os.path.join(data, rel)
        os.makedirs(root)
        with open(os.path.join(root, "run.json"), "w") as f:
            json.dump({"seed": s, "arms": ["none"] + sorted(arms)}, f)
        for arm in ["none"] + sorted(arms):
            os.makedirs(os.path.join(root, "%s_%s" % (opp, arm)))
            hps = [GAIN.get(arm, 0) + d + i for d in (-5, 0, 5)] if arm != "none" else [0, 0, 0]
            with open(os.path.join(root, "%s_%s" % (opp, arm), "rounds.jsonl"), "w") as f:
                f.write("".join(json.dumps({"dealt": max(h, 0), "taken": max(-h, 0),
                                            "result": "win" if h > 0 else "loss"}) + "\n" for h in hps))
            if arm != "none":
                with open(os.path.join(root, "memory_%s_%s.json" % (opp, arm)), "w") as f:
                    json.dump({"me": "chunli", "opp": opp, "source": "fixed",
                               "lessons": [{"text": t} for t in arms[arm]]}, f)
        with open(os.path.join(logs, "seed_%d.log" % s), "w") as f:
            f.write("8 headless runs\nsaved %s\n" % rel)
    return "logs/ab/%s_expert" % opp


def test_only_single_lines_that_help_are_verified(tmp_path):
    b = mod()
    d = str(tmp_path)
    batch(d)
    doc = b.build(d, {"ryu": "logs/ab/ryu_expert"})
    o = doc["opponents"]["ryu"]
    assert [x["line"] for x in o["lines"]] == ["use more throw up close"]
    t = o["lines"][0]
    assert t["claim"] == {"kind": "use_more", "move": "throw", "range": "close", "when": None}
    assert t["arm"] == "throw" and t["mean"] == pytest.approx(41) and t["ci95"][0] > 0 and t["runs"] == 3
    assert t["seeds"] == [41001, 41002, 41003] and t["batch"] == "logs/ab/ryu_expert"
    why = {x["arm"]: x["why"] for x in o["not_verified"]}
    assert why["cmk"].startswith("not shown") and why["hurt"].startswith("hurts")
    assert why["walk"].startswith("names forward")                   # helped, but System 1 cannot follow it
    assert why["expert"].startswith("2 lines")                       # a set is not a tip


def test_the_rebuild_is_byte_identical_and_check_catches_a_drift(tmp_path):
    b = mod()
    d = str(tmp_path)
    batch(d)
    out = os.path.join(d, "book.json")
    b.write(b.build(d, {"ryu": "logs/ab/ryu_expert"}), out)
    first = open(out, "rb").read()
    b.write(b.build(d, {"ryu": "logs/ab/ryu_expert"}), out)
    assert open(out, "rb").read() == first
    assert b.check(d, {"ryu": "logs/ab/ryu_expert"}, out) == []
    doc = json.loads(first)
    doc["opponents"]["ryu"]["lines"][0]["mean"] = 99.0
    with open(out, "w") as f:
        json.dump(doc, f)
    assert b.check(d, {"ryu": "logs/ab/ryu_expert"}, out)            # hand-edited: refused


def test_seeds_that_disagree_on_an_arms_lines_stop_the_build(tmp_path):
    b = mod()
    d = str(tmp_path)
    batch(d)
    with open(os.path.join(d, "rollouts", "ab", "r41002_ryu", "memory_ryu_throw.json"), "w") as f:
        json.dump({"lessons": [{"text": "always throw up close"}]}, f)
    with pytest.raises(SystemExit, match="differ between seeds"):
        b.build(d, {"ryu": "logs/ab/ryu_expert"})


def test_a_line_the_grammar_does_not_render_back_stops_the_build(tmp_path):
    b = mod()
    d = str(tmp_path)
    batch(d, arms={"throw": ["throw him up close a lot"]})
    with pytest.raises(SystemExit, match="does not render"):
        b.build(d, {"ryu": "logs/ab/ryu_expert"})


def test_the_committed_book_is_what_the_named_command_builds():
    """lessons/book.json == scripts/book.py build over the real batches (skipped where they are absent)."""
    if not os.path.isdir(os.path.join(DATA, "logs", "ab", "ryu_expert")):
        pytest.skip("no fixed-advice batches under %s" % DATA)
    b = mod()
    assert b.check(DATA, b.BATCHES, os.path.join(HERE, "lessons", "book.json")) == []


def test_the_real_book_has_the_throw_for_all_six():
    path = os.path.join(HERE, "lessons", "book.json")
    with open(path) as f:
        doc = json.load(f)
    assert sorted(doc["opponents"]) == ["dhalsim", "guile", "honda", "ken", "ryu", "zangief"]
    for opp, o in doc["opponents"].items():
        assert "use more throw up close" in [x["line"] for x in o["lines"]], opp
        for x in o["lines"]:
            assert x["ci95"][0] > 0 and x["claim"]["move"] != "forward" and x["batch"] == "logs/ab/%s_expert" % opp
    assert hashlib.sha256(open(path, "rb").read()).hexdigest()
