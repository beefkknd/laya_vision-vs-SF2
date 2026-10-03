"""Tests for the PURE monitor data layer (``sf2.eval.tui_model``).

These test the data layer only -- no ``rich``, no terminal. They parse the REAL overnight run dirs
and crafted tiny dirs. The one-frame render smoke (``--once``) is exercised by invoking
``scripts/monitor_tui.py`` as a subprocess so the renderer stays out of this pure-layer suite.

Seen-RED provenance: each assertion was first run against a deliberately broken twin of the code to
confirm it can fail -- see the module docstring in the report. e.g. hp_bar rounding, the me/opp health
mapping, and the thinking inference were each checked against an inverted implementation and went red.
"""
import json
import os
import subprocess
import sys

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from sf2.eval import tui_model as T   # noqa: E402

HONDA = os.path.join(REPO, "rollouts", "loop_screen", "overnight_chunli_honda")
KEN = os.path.join(REPO, "rollouts", "loop_screen", "overnight_chunli_ken")


# --------------------------------------------------------------------------- hp_bar (pure)

def test_hp_bar_full_half_empty():
    assert T.hp_bar(1.0, 10) == "#" * 10
    assert T.hp_bar(0.0, 10) == "-" * 10
    assert T.hp_bar(0.5, 10) == "#####-----"


def test_hp_bar_clamps_out_of_range():
    assert T.hp_bar(2.0, 8) == "#" * 8
    assert T.hp_bar(-1.0, 8) == "-" * 8
    assert T.hp_bar("bad", 8) == "-" * 8  # type: ignore[arg-type]


def test_parse_round_dir():
    assert T.parse_round_dir("g03_r1") == (3, 1)
    assert T.parse_round_dir("g10_r0") == (10, 0)
    assert T.parse_round_dir("loop_report.json") is None
    assert T.parse_round_dir("trace.jsonl") is None


# --------------------------------------------------------------------------- real overnight dirs

@pytest.mark.skipif(not os.path.isdir(HONDA), reason="overnight honda dir not present")
def test_real_honda_metadata_and_hp():
    m = T.build_model(HONDA)
    assert m.me == "chunli"
    assert m.opp == "honda"
    assert m.games == 10 and m.rounds == 2
    # HP fractions map to 0..176 hp units
    assert 0 <= m.me_hp <= T.HP_MAX
    assert 0 <= m.opp_hp <= T.HP_MAX
    assert abs(m.me_hp - round(m.me_hp_frac * T.HP_MAX)) <= 1


@pytest.mark.skipif(not os.path.isdir(HONDA), reason="overnight honda dir not present")
def test_real_honda_pipeline_decisions():
    m = T.build_model(HONDA)
    assert len(m.decisions) >= 1
    d = m.decisions[-1]
    assert d.category  # a category was chosen
    assert 0.0 <= d.cat_prob <= 1.0
    assert 0.0 <= d.move_prob <= 1.0
    assert d.rng in ("close", "mid", "far", "?")
    # decisions belong to the current (newest) round
    assert all(dv.game == m.cur_game and dv.round == m.cur_round for dv in m.decisions)


@pytest.mark.skipif(not os.path.isdir(HONDA), reason="overnight honda dir not present")
def test_real_honda_in_play_from_last_qwen():
    m = T.build_model(HONDA)
    # honda has qwen events; in_play must equal the LAST qwen event's in_play_after
    trace = T._read_jsonl(os.path.join(HONDA, "trace.jsonl"))
    qwen = [e for e in trace if e.get("event") == "qwen"]
    assert qwen, "fixture should have qwen events"
    assert list(m.in_play) == list(qwen[-1]["in_play_after"])
    assert len(m.in_play) >= 1


@pytest.mark.skipif(not os.path.isdir(HONDA), reason="overnight honda dir not present")
def test_real_honda_qwen_churn_and_grading():
    m = T.build_model(HONDA)
    assert len(m.qwen) >= 2
    # at least one game added a rule, and the grader (if present) tagged it good/ok/bad/not_scorable
    added = [gr for qv in m.qwen for gr in qv.added]
    assert added, "some game should have added a rule"
    if T.get_grader() is not None:
        verdicts = {gr.verdict for gr in added}
        assert verdicts & {"good", "ok", "bad", "not_scorable"}


@pytest.mark.skipif(not os.path.isdir(HONDA), reason="overnight honda dir not present")
def test_real_honda_follows_rule_pct_and_status():
    m = T.build_model(HONDA)
    assert 0.0 <= m.stats.follows_rule_pct <= 100.0
    assert m.stats.dealt >= 0 and m.stats.taken >= 0
    # a complete overnight run with all round dirs present is done, not thinking
    assert m.status in ("done", "running")
    assert m.qwen_thinking is False


@pytest.mark.skipif(not os.path.isdir(KEN), reason="overnight ken dir not present")
def test_real_ken_parses():
    m = T.build_model(KEN)
    assert m.me == "chunli" and m.opp == "ken"
    assert len(m.decisions) >= 1
    assert len(m.in_play) >= 1


# --------------------------------------------------------------------------- crafted tiny dirs

def _write_run(tmp, games=2, rounds=2):
    os.makedirs(tmp, exist_ok=True)
    with open(os.path.join(tmp, "run.json"), "w") as fh:
        json.dump({"me": "chunli", "opp": "honda", "games": games, "rounds": rounds}, fh)


def _decision_record(k, hp_left, hp_right, cat="throw", act="throw"):
    return {
        "k": k,
        "facts": {"left": {"character": "chunli", "health": hp_left},
                  "right": {"character": "honda", "health": hp_right}},
        "moment": {"doing": "standing", "his_label": "stand", "dx": 40, "fireball": False},
        "situation": ["close", "standing", "full", "full"],
        "category": cat, "action": act,
        "cat_probs": {cat: 0.8}, "move_probs": {act: 0.9},
        "rule": "soft", "follows_rule": True,
        "pressed": ["a", "-", "a", "-"],
    }


def _write_round(tmp, g, r, records):
    d = os.path.join(tmp, "g%02d_r%d" % (g, r))
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "decisions.jsonl"), "w") as fh:
        for rec in records:
            fh.write(json.dumps(rec) + "\n")


def _write_trace(tmp, events):
    with open(os.path.join(tmp, "trace.jsonl"), "w") as fh:
        for e in events:
            fh.write(json.dumps(e) + "\n")


def test_crafted_hp_from_facts(tmp_path):
    tmp = str(tmp_path / "run")
    _write_run(tmp)
    _write_round(tmp, 0, 0, [_decision_record(10, 1.0, 1.0),
                             _decision_record(20, 0.5, 0.25)])
    _write_trace(tmp, [{"event": "seed", "lines": ["use more throw up close"]}])
    m = T.build_model(tmp, grade_qwen=False)
    assert m.me_hp == T.HP_MAX // 2          # 0.5 -> 88
    assert m.opp_hp == T.HP_MAX // 4         # 0.25 -> 44
    assert list(m.in_play) == ["use more throw up close"]   # seed used when no qwen event


def test_crafted_me_opp_mapping_by_character(tmp_path):
    """If 'me' is on the RIGHT side, my HP must still track me, not left."""
    tmp = str(tmp_path / "run")
    _write_run(tmp)
    rec = {
        "k": 1,
        "facts": {"left": {"character": "honda", "health": 1.0},
                  "right": {"character": "chunli", "health": 0.25}},
        "moment": {"doing": "standing", "dx": -40, "fireball": False},
        "situation": ["close", "standing", "full", "full"],
        "category": "throw", "action": "throw",
        "cat_probs": {"throw": 0.8}, "move_probs": {"throw": 0.9},
        "rule": "soft", "follows_rule": True, "pressed": [],
    }
    _write_round(tmp, 0, 0, [rec])
    _write_trace(tmp, [])
    m = T.build_model(tmp, grade_qwen=False)
    assert m.me_hp == T.HP_MAX // 4          # chunli (me) is on the right at 0.25


def test_crafted_thinking_inferred(tmp_path):
    """Game 0 (both rounds) finished with a round event for the final round, game 1 not started:
    qwen is reflecting -> thinking True."""
    tmp = str(tmp_path / "run")
    _write_run(tmp, games=2, rounds=2)
    _write_round(tmp, 0, 0, [_decision_record(1, 1.0, 1.0)])
    _write_round(tmp, 0, 1, [_decision_record(2, 0.6, 0.2)])
    _write_trace(tmp, [
        {"event": "seed", "lines": ["use more throw up close"]},
        {"event": "round", "game": 0, "round": 0, "result": "win", "hp": 50, "dealt": 100, "taken": 50},
        {"event": "round", "game": 0, "round": 1, "result": "win", "hp": 40, "dealt": 90, "taken": 50},
    ])
    m = T.build_model(tmp, grade_qwen=False)
    assert m.qwen_thinking is True
    assert m.status == "thinking"


def test_crafted_not_thinking_mid_game(tmp_path):
    """Only round 0 of game 0 done (round 1 not reached): qwen does NOT run between rounds."""
    tmp = str(tmp_path / "run")
    _write_run(tmp, games=2, rounds=2)
    _write_round(tmp, 0, 0, [_decision_record(1, 1.0, 1.0)])
    _write_trace(tmp, [
        {"event": "round", "game": 0, "round": 0, "result": "win", "hp": 50, "dealt": 100, "taken": 50},
    ])
    m = T.build_model(tmp, grade_qwen=False)
    assert m.qwen_thinking is False


def test_crafted_not_thinking_next_game_started(tmp_path):
    """Game 1 decisions have appeared: qwen already finished -> running, not thinking."""
    tmp = str(tmp_path / "run")
    _write_run(tmp, games=2, rounds=2)
    _write_round(tmp, 0, 0, [_decision_record(1, 1.0, 1.0)])
    _write_round(tmp, 0, 1, [_decision_record(2, 0.6, 0.2)])
    _write_round(tmp, 1, 0, [_decision_record(3, 1.0, 1.0)])
    _write_trace(tmp, [
        {"event": "round", "game": 0, "round": 0, "result": "win", "hp": 50, "dealt": 100, "taken": 50},
        {"event": "round", "game": 0, "round": 1, "result": "win", "hp": 40, "dealt": 90, "taken": 50},
    ])
    m = T.build_model(tmp, grade_qwen=False)
    assert m.qwen_thinking is False
    assert m.status == "running"
    assert m.cur_game == 1 and m.cur_round == 0


def test_crafted_done_after_final_game(tmp_path):
    tmp = str(tmp_path / "run")
    _write_run(tmp, games=1, rounds=1)
    _write_round(tmp, 0, 0, [_decision_record(1, 0.3, 0.0)])
    _write_trace(tmp, [
        {"event": "round", "game": 0, "round": 0, "result": "win", "hp": 60, "dealt": 120, "taken": 30},
    ])
    m = T.build_model(tmp, grade_qwen=False)
    assert m.status == "done"
    assert m.qwen_thinking is False


def test_build_model_never_raises_on_garbage(tmp_path):
    tmp = str(tmp_path / "empty")
    os.makedirs(tmp, exist_ok=True)
    m = T.build_model(tmp, grade_qwen=False)   # no run.json, no trace, no round dirs
    assert m.status == "empty"
    assert m.error is not None


def test_partial_trailing_jsonl_line_ignored(tmp_path):
    """A half-flushed final line (loop still writing) must not crash the parse."""
    tmp = str(tmp_path / "run")
    _write_run(tmp)
    d = os.path.join(tmp, "g00_r0")
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "decisions.jsonl"), "w") as fh:
        fh.write(json.dumps(_decision_record(1, 1.0, 1.0)) + "\n")
        fh.write('{"k": 2, "facts": {"left": {"heal')   # truncated mid-write
    _write_trace(tmp, [])
    m = T.build_model(tmp, grade_qwen=False)
    assert len(m.decisions) == 1   # the good record; the partial one skipped


# --------------------------------------------------------------------------- --once render smoke

@pytest.mark.skipif(not os.path.isdir(HONDA), reason="overnight honda dir not present")
def test_once_render_subprocess_real_dir():
    """scripts/monitor_tui.py --once --watch <honda> renders one frame and exits 0."""
    script = os.path.join(REPO, "scripts", "monitor_tui.py")
    out = subprocess.run([sys.executable, script, "--once", "--watch", HONDA],
                         capture_output=True, text=True, timeout=60)
    assert out.returncode == 0, out.stderr
    text = out.stdout
    assert "chunli" in text and "honda" in text
    assert "SHORT MEMORY" in text or "MEMORY" in text
    assert "QWEN" in text


def test_once_render_subprocess_crafted(tmp_path):
    tmp = str(tmp_path / "run")
    _write_run(tmp)
    _write_round(tmp, 0, 0, [_decision_record(1, 1.0, 0.5)])
    _write_trace(tmp, [{"event": "seed", "lines": ["use more throw up close"]}])
    script = os.path.join(REPO, "scripts", "monitor_tui.py")
    out = subprocess.run([sys.executable, script, "--once", "--watch", tmp],
                         capture_output=True, text=True, timeout=60)
    assert out.returncode == 0, out.stderr
    assert "chunli" in out.stdout
