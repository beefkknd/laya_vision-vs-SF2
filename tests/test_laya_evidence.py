"""scripts/laya_evidence.py: from finished lesson-loop runs, does System 1 follow the lessons in play (compliance vs
the no-advice arm in the same situation), and did laya-vision rate the named move into its top 3?"""
import importlib.util
import json
import os
import sys

import pytest

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load():
    if os.path.join(HERE, "scripts") not in sys.path:
        sys.path.insert(0, os.path.join(HERE, "scripts"))
    spec = importlib.util.spec_from_file_location("laya_evidence", os.path.join(HERE, "scripts", "laya_evidence.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


E = load()


def act(action, rng="close", doing="standing", top3=("hp", "mp", "lp"), rnd=0):
    air = doing == "jumping"
    state = {"crouching": "crouch", "attacking": "attack"}.get(doing, "jump" if air else "stand")
    return {"action": action, "range": rng, "opp_state": state, "opp_air": air, "round": rnd, "game": rnd,
            "top3": [[m, 0.3] for m in top3], "kind": "attack"}


def jsonl(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write("".join(json.dumps(r) + "\n" for r in rows))


def make_run(root, name, loop, none, lines, prompt=None, ledger=(), opp="ken"):
    """One finished run: both arms, one round (0) with ``lines`` in play, a verdict."""
    d = os.path.join(str(root), name)
    rounds = sorted({a["round"] for a in loop + none}) or [0]
    for arm, acts in (("loop", loop), ("none", none)):
        jsonl(os.path.join(d, arm, "actions.jsonl"), acts)
        jsonl(os.path.join(d, arm, "rounds.jsonl"),
              [{"round": r, "game": r, "arm": arm, "lines": lines if arm == "loop" else lines} for r in rounds])
        run = {"arm": arm, "opp": opp}
        if prompt:
            run["prompt"] = prompt
        with open(os.path.join(d, arm, "run.json"), "w") as f:
            json.dump(run, f)
    jsonl(os.path.join(d, "loop", "ledger.jsonl"), list(ledger))
    with open(os.path.join(d, "verdict.json"), "w") as f:
        json.dump({"updates": 1}, f)
    return d


def lesson(res, line):
    [x] = [les for les in res["lessons"] if les["line"] == line]
    return x


def test_soft_lesson_followed_and_not(tmp_path):
    line = "use more hp up close"
    loop = [act("hp"), act("hp"), act("hp"), act("mp")]
    none = [act("hp"), act("mp"), act("mp"), act("lp")]
    res = E.run_evidence(make_run(tmp_path, "r_ken", loop, none, [line]))
    x = lesson(res, line)
    assert (x["polarity"], x["kind"], x["n"], x["followed"]) == ("soft", "attack", 4, 3)
    assert x["compliance"] == pytest.approx(0.75)
    assert (x["base_n"], x["base_followed"]) == (4, 1)
    assert x["base"] == pytest.approx(0.25)
    assert x["effect"] == pytest.approx(0.5)


def test_avoid_lesson_counts_not_picking(tmp_path):
    line = "avoid sweep up close"
    loop = [act("sweep"), act("hp"), act("mp"), act("hp")]
    none = [act("sweep"), act("sweep"), act("hp"), act("sweep")]
    x = lesson(E.run_evidence(make_run(tmp_path, "r_ken", loop, none, [line])), line)
    assert x["polarity"] == "neg"
    assert x["compliance"] == pytest.approx(0.75)      # 3 of 4 did NOT pick sweep
    assert x["base"] == pytest.approx(0.25)            # the none arm avoided it once in 4


def test_lesson_that_does_not_apply_is_not_counted(tmp_path):
    line = "use more hp up close when he stands"
    loop = [act("hp"), act("mp", rng="mid"), act("mp", doing="jumping"), act("mp", doing="crouching")]
    none = [act("mp"), act("hp", rng="far"), act("hp", doing="jumping")]
    x = lesson(E.run_evidence(make_run(tmp_path, "r_ken", loop, none, [line])), line)
    assert (x["n"], x["followed"]) == (1, 1)
    assert (x["base_n"], x["base_followed"]) == (1, 0)     # only the close/standing decision of the none arm


def test_lesson_counts_only_rounds_where_it_was_in_play(tmp_path):
    d = make_run(tmp_path, "r_ken", [act("hp", rnd=0), act("mp", rnd=1)], [act("mp", rnd=0), act("hp", rnd=1)],
                 ["use more hp up close"])
    jsonl(os.path.join(d, "loop", "rounds.jsonl"), [{"round": 0, "game": 0, "lines": ["use more hp up close"]},
                                                     {"round": 1, "game": 1, "lines": []}])
    x = lesson(E.run_evidence(d), "use more hp up close")
    assert (x["n"], x["followed"], x["base_n"], x["base_followed"]) == (1, 1, 1, 0)


def test_top3_share(tmp_path):
    line = "use more c.mk up close"
    loop = [act("mp", top3=("c.mk", "hp", "mp")), act("mp"), act("mp"), act("c.mk", top3=("c.mk", "mp", "lp"))]
    x = lesson(E.run_evidence(make_run(tmp_path, "r_ken", loop, [act("mp")], [line])), line)
    assert x["top3"] == pytest.approx(0.5)
    assert x["kind"] == "attack"
    assert (x["n_in_top3"], x["followed_in_top3"]) == (2, 1)       # followed when laya-vision rated it in
    assert (x["n_out_top3"], x["followed_out_top3"]) == (2, 0)
    assert E.pool([x])["compliance_in_top3"] == pytest.approx(0.5)


def test_block_and_forward_kinds_and_no_move_lines(tmp_path):
    lines = ["always block_low up close when he attacks", "he jumps a lot"]
    loop = [act("block_low", doing="attacking"), act("hp", doing="attacking")]
    res = E.run_evidence(make_run(tmp_path, "r_ken", loop, [act("hp", doing="attacking")], lines))
    x = lesson(res, lines[0])
    assert (x["polarity"], x["kind"], x["n"], x["followed"]) == ("hard", "block", 2, 1)
    assert res["no_move_lines"] == ["he jumps a lot"]


def test_aggregate_pools_decisions_by_polarity_kind_prompt(tmp_path):
    a = make_run(tmp_path, "a_ken", [act("hp"), act("hp")], [act("mp"), act("mp")], ["use more hp up close"])
    b = make_run(tmp_path, "b_ken", [act("sweep"), act("hp")], [act("sweep"), act("sweep")],
                 ["avoid sweep up close"], prompt="character")
    doc = E.collect([str(tmp_path)])
    assert doc["failures"] == []
    assert {r["run"] for r in doc["runs"]} == {a, b}
    assert doc["by_polarity"]["soft"]["n"] == 2 and doc["by_polarity"]["soft"]["compliance"] == 1.0
    assert doc["by_polarity"]["neg"]["compliance"] == 0.5 and doc["by_polarity"]["neg"]["base"] == 0.0
    assert doc["by_prompt"]["views"]["lessons"] == 1 and doc["by_prompt"]["character"]["lessons"] == 1
    assert doc["by_kind"]["attack"]["n"] == 4
    assert doc["all"]["effect"] == pytest.approx(0.75 - 0.0)


def test_unreadable_run_is_reported(tmp_path):
    make_run(tmp_path, "good_ken", [act("hp")], [act("mp")], ["use more hp up close"])
    bad = make_run(tmp_path, "bad_ken", [act("hp")], [act("mp")], ["use more hp up close"])
    with open(os.path.join(bad, "none", "actions.jsonl"), "w") as f:
        f.write("{not json\n")
    unfinished = os.path.join(str(tmp_path), "running_ken", "loop")
    os.makedirs(unfinished)
    doc = E.collect([str(tmp_path)])
    assert [f["run"] for f in doc["failures"]] == [bad]
    assert [r["run"] for r in doc["runs"]] == [os.path.join(str(tmp_path), "good_ken")]
    assert doc["skipped_unfinished"] == [os.path.dirname(unfinished)]


def test_missing_none_arm_is_a_failure(tmp_path):
    d = make_run(tmp_path, "x_ken", [act("hp")], [act("mp")], ["use more hp up close"])
    os.remove(os.path.join(d, "none", "actions.jsonl"))
    assert [f["run"] for f in E.collect([str(tmp_path)])["failures"]] == [d]


def test_opponent_move_mentions(tmp_path):
    ledger = [{"game": 0, "problems": ["bad claim", "another"],
               "claims": [{"move": "block_high", "why": "He throws a Fireball at far range; block the hadoken."},
                          {"move": "hp", "why": "his Dragon Punch beats jump-ins"},
                          {"move": "spinning_bird_kick", "why": "spinning bird kick is her best"}]},
              {"game": 1, "problems": [], "claims": [{"move": "c.mk", "why": "hurricane kick whiffs", "when": "x"}]}]
    make_run(tmp_path, "m_ken", [act("hp")], [act("mp")], ["use more hp up close"], ledger=ledger)
    doc = E.collect([str(tmp_path)])
    m = doc["opponent_moves"]
    assert m["counts"]["ken"] == {"fireball": 1, "hadoken": 1, "dragon punch": 1, "hurricane": 1}
    assert m["claims_mentioning"] == 3 and m["claims"] == 4
    assert len(m["examples"]) == 3 and "Fireball" in m["examples"][0]
    assert doc["ledger_problems"] == {"total": 2, "by_opp": {"ken": 2}}


def test_main_writes_json(tmp_path):
    make_run(tmp_path, "r_ken", [act("hp")], [act("mp")], ["use more hp up close"])
    out = os.path.join(str(tmp_path), "o", "ev.json")
    assert E.main(["--root", str(tmp_path), "--out", out]) == 0
    assert json.load(open(out))["all"]["n"] == 1
