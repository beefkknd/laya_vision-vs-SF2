"""Tests for the G7 per-game loop report (sf2.eval.loop_report).

Admission (testing doctrine): each check was SEEN RED first by running it against the code before it existed /
against a deliberately wrong expectation -- noted per test. They cover the trend extraction, follows_rule-per-rule
(counts only where the rule APPLIES), the G5 wiring (injected scorer, so the diagnosis is testable without the
table), the three diagnosis flags (a)/(b)/(c) on crafted known-bad games, and reader-vs-RAM = n/a with no replay.
"""
import json
import os

import pytest

from sf2.eval import loop_report as LR


# ---------- helpers: craft a tiny run dir ----------

def _decision(game, rnd, k, situation, action, follows, lines, category="special", rule="default"):
    return {"event": "decision", "game": game, "round": rnd, "k": k, "situation": situation,
            "action": action, "category": category, "rule": rule, "follows_rule": follows, "lines": lines}


def _round(game, rnd, result, hp, dealt=0, taken=0):
    return {"event": "round", "game": game, "round": rnd, "result": result, "hp": hp,
            "dealt": dealt, "taken": taken, "source": "screen"}


def _qwen(game, added=(), removed=(), in_play_after=()):
    return {"event": "qwen", "game": game, "added": list(added), "removed": list(removed),
            "in_play_after": list(in_play_after)}


# ---------- 1. trend (hp per game) ----------

def test_trend_mean_hp_per_game():
    """Seen RED: before rounds_of_game existed the report had no mean_hp_per_round. Crafted: game0 rounds
    -72,+64 -> mean -4; game1 rounds -106,-80 -> mean -93 (matches the real 221402 verdict game_hp)."""
    trace = [
        {"event": "seed", "lines": ["use more lightning_legs up close when he stands"]},
        _round(0, 0, "loss", -72), _round(0, 1, "win", 64),
        _round(1, 0, "loss", -106), _round(1, 1, "loss", -80),
    ]
    run = LR.load_run(_write_run_dir(trace))
    g0, g1 = LR.rounds_of_game(run, 0), LR.rounds_of_game(run, 1)
    assert LR._mean([r["hp"] for r in g0]) == pytest.approx(-4.0)
    assert LR._mean([r["hp"] for r in g1]) == pytest.approx(-93.0)


def _write_run_dir(trace, **kw):
    import tempfile
    d = tempfile.mkdtemp()
    with open(os.path.join(d, "run.json"), "w") as f:
        json.dump(kw.get("run", {"me": "chunli", "opp": "honda", "qwen": "on"}), f)
    with open(os.path.join(d, "trace.jsonl"), "w") as f:
        f.write("\n".join(json.dumps(e) for e in trace))
    for name in ("verdict", "score"):
        if name in kw:
            with open(os.path.join(d, name + ".json"), "w") as f:
                json.dump(kw[name], f)
    return d


def test_improved_flag_follows_trend():
    """Seen RED against a wrong expectation (improved True for game1): -93 < -4, so game1 did NOT improve."""
    trace = [_round(0, 0, "loss", -72), _round(0, 1, "win", 64),
             _round(1, 0, "loss", -106), _round(1, 1, "loss", -80)]
    rep = LR.build_report(_write_run_dir(trace), scorer=None)
    assert rep["games"][0]["improved"] is None      # first game: no baseline
    assert rep["games"][1]["improved"] is False      # got worse


# ---------- 2. follows_rule per rule: counts only where the rule APPLIES ----------

LINE_LEGS = "use more lightning_legs up close when he stands"


def test_follows_per_rule_only_when_applies():
    """Seen RED against a wrong expectation (share over ALL decisions). The rule applies only close+standing.
    3 applying decisions, legs picked on 2 -> 2/3; the mid+attacking decision must NOT count."""
    decs = [
        _decision(0, 0, 1, ["close", "standing", "full", "full"], "lightning_legs", True, [LINE_LEGS]),
        _decision(0, 0, 2, ["close", "standing", "full", "full"], "lightning_legs", True, [LINE_LEGS]),
        _decision(0, 0, 3, ["close", "standing", "full", "full"], "block_high", False, [LINE_LEGS]),
        _decision(0, 0, 4, ["mid", "attacking", "full", "full"], "block_high", False, [LINE_LEGS]),
    ]
    fr = LR.follows_per_rule(decs, [LINE_LEGS])[LINE_LEGS]
    assert fr["applies"] == 3 and fr["followed"] == 2
    assert fr["share"] == pytest.approx(2 / 3)


def test_follows_per_rule_no_applicable_is_none():
    """Seen RED against expecting 0.0: a rule that never applies has share None, not 0 (don't invent a denominator)."""
    decs = [_decision(0, 0, 1, ["far", "jumping", "full", "full"], "block_high", False, [LINE_LEGS])]
    fr = LR.follows_per_rule(decs, [LINE_LEGS])[LINE_LEGS]
    assert fr["applies"] == 0 and fr["share"] is None


# ---------- 3. G5 wired per in-play rule (injected scorer) ----------

def test_g5_wired_per_rule_injected():
    """Seen RED: with scorer=None every rule reads 'unavailable'; an injected scorer must flow its verdict through
    per line. (Decouples the diagnosis from the real table.)"""
    lines = ["rule A", "rule B"]
    assert LR.g5_of_rules(lines, None)["rule A"]["verdict"] == "unavailable"
    fake = {"rule A": "bad", "rule B": "good"}
    scorer = lambda ln: {"verdict": fake[ln]}
    got = LR.g5_of_rules(lines, scorer)
    assert got["rule A"]["verdict"] == "bad" and got["rule B"]["verdict"] == "good"


# ---------- 4. diagnosis flags (a)/(b)/(c) fire on crafted cases ----------

def test_diagnosis_a_bad_and_followed():
    """(a): a G5-BAD rule she still FOLLOWS (share >= FOLLOW_HI). Seen RED before diagnose_game existed."""
    g5 = {"bad rule": {"verdict": "bad"}}
    follows = {"bad rule": {"share": 0.9}}
    d = LR.diagnose_game(improved=False, g5=g5, follows=follows, agreement=1.0)
    assert "a_bad_rules" in d["flags"]
    assert "b_not_following" not in d["flags"] and "c_bad_facts" not in d["flags"]


def test_diagnosis_b_good_but_not_followed():
    """(b): a G5-GOOD rule with LOW follows (< FOLLOW_LO)."""
    g5 = {"good rule": {"verdict": "good"}}
    follows = {"good rule": {"share": 0.1}}
    d = LR.diagnose_game(improved=False, g5=g5, follows=follows, agreement=1.0)
    assert d["flags"] == ["b_not_following"]


def test_diagnosis_c_bad_facts():
    """(c): reader-vs-RAM agreement below AGREE_LO."""
    d = LR.diagnose_game(improved=False, g5={}, follows={}, agreement=0.6667)
    assert d["flags"] == ["c_bad_facts"]


def test_diagnosis_a_not_fired_when_bad_rule_ignored():
    """A BAD rule she does NOT follow is not cause (a) (she isn't acting on the bad advice). Seen RED against a
    verdict-only rule that ignored follows_rule."""
    g5 = {"bad rule": {"verdict": "bad"}}
    follows = {"bad rule": {"share": 0.1}}
    d = LR.diagnose_game(improved=False, g5=g5, follows=follows, agreement=1.0)
    assert "a_bad_rules" not in d["flags"]


def test_diagnosis_silent_when_improved():
    """No cause is flagged when hp improved (or on the baseline game)."""
    g5 = {"bad rule": {"verdict": "bad"}}
    follows = {"bad rule": {"share": 0.9}}
    assert LR.diagnose_game(True, g5, follows, 0.5)["flags"] == []
    assert LR.diagnose_game(None, g5, follows, 0.5)["flags"] == []


def test_diagnosis_notes_g5_unavailable():
    """When G5 is unavailable the (a)/(b) legs are explicitly not-evaluated, not silently 'no cause'."""
    g5 = {"rule": {"verdict": "unavailable"}}
    d = LR.diagnose_game(False, g5, {"rule": {"share": 0.9}}, None)
    assert "G5 unavailable" in d["note"] and "reader-vs-RAM n/a" in d["note"]


# ---------- 5. reader-vs-RAM = n/a when no replay ----------

def test_reader_vs_ram_na_without_score():
    """Seen RED against expecting a number: a run with no score.json has agreement None (n/a), not 0."""
    run = LR.load_run(_write_run_dir([_round(0, 0, "loss", -10)]))
    rr = LR.reader_vs_ram_of_game(run, 0)
    assert rr["agreement"] is None and rr["rounds"] == 0


def test_reader_vs_ram_from_score_json():
    """With score.json replay agreement present, it is averaged per game."""
    score = {"rounds": [
        {"rec": "x/g00_r0", "result": "loss", "hp": -10, "dealt": 0, "taken": 10,
         "agreement": {"table_cell": 0.6, "can_act": 0.8}},
        {"rec": "x/g00_r1", "result": "loss", "hp": -20, "dealt": 0, "taken": 20,
         "agreement": {"table_cell": 0.8, "can_act": 1.0}},
    ]}
    run = LR.load_run(_write_run_dir([{"event": "seed", "lines": []}], score=score))
    rr = LR.reader_vs_ram_of_game(run, 0)
    assert rr["agreement"] == pytest.approx(0.7) and rr["rounds"] == 2


def test_rounds_fall_back_to_score_when_trace_has_no_round():
    """A 'broken' run whose trace emitted only a seed still gets hp/result from score.json (the real 205940 shape)."""
    score = {"rounds": [{"rec": "x/g00_r0", "result": "loss", "hp": -170, "dealt": 0, "taken": 170,
                         "agreement": {"table_cell": 0.6667}}]}
    run = LR.load_run(_write_run_dir([{"event": "seed", "lines": []}], score=score))
    rounds = LR.rounds_of_game(run, 0)
    assert len(rounds) == 1 and rounds[0]["hp"] == -170 and rounds[0]["source"] == "replay"


# ---------- 6. qwen churn + end to end on a crafted run ----------

def test_qwen_churn_and_full_report():
    """Qwen added a rule in game1; cumulative in-play count grows. End-to-end build_report over 2 games."""
    trace = [
        {"event": "seed", "lines": [LINE_LEGS]},
        _decision(0, 0, 1, ["close", "standing", "full", "full"], "lightning_legs", True, [LINE_LEGS]),
        _round(0, 0, "win", 10),
        _qwen(0, in_play_after=[LINE_LEGS]),
        _decision(1, 0, 1, ["close", "standing", "full", "full"], "block_high", False, [LINE_LEGS]),
        _round(1, 0, "loss", -50),
        _qwen(1, added=["always block_low at mid range when he attacks"],
              in_play_after=[LINE_LEGS, "always block_low at mid range when he attacks"]),
    ]
    scorer = lambda ln: {"verdict": "good"} if "lightning_legs" in ln else {"verdict": "ok"}
    rep = LR.build_report(_write_run_dir(trace), scorer=scorer)
    assert rep["games"][1]["qwen"]["added"] == ["always block_low at mid range when he attacks"]
    assert rep["games"][1]["qwen"]["in_play_count"] == 2
    # game1 did not improve, lightning_legs is G5-good but followed 0/1 -> flag (b)
    assert rep["games"][1]["improved"] is False
    assert "b_not_following" in rep["games"][1]["diagnosis"]["flags"]
    assert LR.render_text(rep)  # renders without error


def test_make_g5_scorer_none_when_table_missing(tmp_path):
    """make_g5_scorer returns None (not a crash) when the table file is absent."""
    assert LR.make_g5_scorer(str(tmp_path / "nope.json")) is None
