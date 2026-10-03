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
import re
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


# --------------------------------------------------------------------------- shaded_bar (pure, NEW)

_TAG = re.compile(r"\[/?[a-z ]+\]")      # strip rich markup tags -> raw glyphs only


def _glyphs(markup: str) -> str:
    return _TAG.sub("", markup)


def test_shaded_bar_levels_glyphs():
    """Full '█', partial eighth-ramp cell, dim '░' track -- exact glyphs at the key fractions.

    Seen RED first: run against the old codebase (no shaded_bar) -> AttributeError, and against a
    twin that used int(round(f*width)) full cells with no partial ramp -> the 0.3125 case is
    '██░░░░░░' not '██▌░░░░░', so the partial-cell assertion fails.
    """
    assert _glyphs(T.shaded_bar(1.0, 8)) == "████████"
    assert _glyphs(T.shaded_bar(0.0, 8)) == "░░░░░░░░"
    assert _glyphs(T.shaded_bar(0.5, 8)) == "████░░░░"
    assert _glyphs(T.shaded_bar(0.25, 8)) == "██░░░░░░"
    # partial-cell boundary: 0.3125*8*8 = 20 eighths -> 2 full + 4/8 ('▌') + 5 track
    assert _glyphs(T.shaded_bar(0.3125, 8)) == "██▌░░░░░"


def test_shaded_bar_color_by_health():
    assert "[green]" in T.shaded_bar(0.9, 8)     # >0.6
    assert "[yellow]" in T.shaded_bar(0.45, 8)   # 0.3..0.6
    assert "[red]" in T.shaded_bar(0.1, 8)       # <0.3
    assert "[dim]" in T.shaded_bar(0.5, 8)       # empty track is dim
    # explicit color override (win-rate meter forces green)
    assert "[green]" in T.shaded_bar(0.5, 8, color="green")
    assert "[yellow]" not in T.shaded_bar(0.5, 8, color="green")


def test_shaded_bar_markup_is_valid():
    """Every produced bar must be parseable rich markup (balanced tags, no stray brackets)."""
    from rich.text import Text as _RT
    for f in (0.0, 0.17, 0.25, 0.3, 0.5, 0.6, 0.8125, 1.0, 2.0, -1.0):
        _RT.from_markup(T.shaded_bar(f, 12))   # raises on malformed markup


def test_shaded_bar_width_invariant():
    """Rendered glyph width always equals the requested width (partial cell counts as one)."""
    for f in (0.0, 0.1, 0.3125, 0.49, 0.5, 0.99, 1.0):
        for w in (1, 7, 12, 28):
            assert len(_glyphs(T.shaded_bar(f, w))) == w


# --------------------------------------------------------------------------- per-game W/L (pure, NEW)

def _rr(game, rnd, result):
    return T.RoundResult(game=game, round=rnd, result=result, hp=0, dealt=0, taken=0)


def test_per_game_wl_three_wins_two_losses():
    """3 wins then 2 losses, one decisive round each -> ('W','W','W','L','L') in game order.

    Seen RED first: a twin that keyed the dict by round instead of game collapsed all into
    two buckets and returned the wrong length/order.
    """
    results = (_rr(0, 0, "win"), _rr(1, 0, "win"), _rr(2, 0, "win"),
               _rr(3, 0, "loss"), _rr(4, 0, "loss"))
    assert T.per_game_wl(results) == ("W", "W", "W", "L", "L")


def test_per_game_wl_multiround_and_tie():
    results = (
        _rr(0, 0, "loss"), _rr(0, 1, "win"),    # 1-1 -> tie
        _rr(1, 0, "win"), _rr(1, 1, "win"),     # 2-0 -> win
        _rr(2, 0, "loss"), _rr(2, 1, "loss"),   # 0-2 -> loss
    )
    assert T.per_game_wl(results) == ("T", "W", "L")


def test_win_rate_number():
    assert T.win_rate(("W", "W", "W", "L", "L")) == (3, 5, 60)
    assert T.win_rate(()) == (0, 0, 0)
    assert T.win_rate(("W", "T", "L", "T")) == (1, 4, 25)


@pytest.mark.skipif(not os.path.isdir(HONDA), reason="overnight honda dir not present")
def test_real_honda_per_game_wl():
    m = T.build_model(HONDA)
    assert len(m.per_game_wl) == m.games                 # one verdict per game played
    assert set(m.per_game_wl) <= {"W", "L", "T"}
    wins, played, pct = T.win_rate(m.per_game_wl)
    assert 0 <= wins <= played == len(m.per_game_wl)
    assert 0 <= pct <= 100


@pytest.mark.skipif(not os.path.isdir(HONDA), reason="overnight honda dir not present")
def test_once_render_has_new_visual_elements():
    """--once on the real honda dir renders the shaded bar (block glyph) and the win-rate meter."""
    script = os.path.join(REPO, "scripts", "monitor_tui.py")
    out = subprocess.run([sys.executable, script, "--once", "--watch", HONDA],
                         capture_output=True, text=True, timeout=60)
    assert out.returncode == 0, out.stderr
    assert "█" in out.stdout            # a full block glyph from a shaded bar
    assert "win rate" in out.stdout.lower()


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


@pytest.mark.skipif(not os.path.isdir(HONDA), reason="overnight honda dir not present")
def test_save_svg_frame(tmp_path):
    """--save <path>.svg writes a non-empty SVG containing a block glyph, headless, exit 0.

    Seen RED first: before --save existed, argparse rejected the flag (exit 2) and no file was
    written, so the file-exists / block-glyph assertions failed.
    """
    out_svg = str(tmp_path / "frame.svg")
    script = os.path.join(REPO, "scripts", "monitor_tui.py")
    out = subprocess.run([sys.executable, script, "--save", out_svg, "--watch", HONDA],
                         capture_output=True, text=True, timeout=60)
    assert out.returncode == 0, out.stderr
    assert os.path.isfile(out_svg)
    data = open(out_svg, encoding="utf-8").read()
    assert len(data) > 0
    assert data.lstrip().startswith("<svg") or "<svg" in data[:2000]
    assert "█" in data                   # a full block glyph rendered into the SVG


@pytest.mark.skipif(not os.path.isdir(HONDA), reason="overnight honda dir not present")
def test_save_html_frame(tmp_path):
    out_html = str(tmp_path / "frame.html")
    script = os.path.join(REPO, "scripts", "monitor_tui.py")
    out = subprocess.run([sys.executable, script, "--save", out_html, "--watch", HONDA],
                         capture_output=True, text=True, timeout=60)
    assert out.returncode == 0, out.stderr
    assert os.path.getsize(out_html) > 0
    assert "█" in open(out_html, encoding="utf-8").read()


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


# --------------------------------------------------------------------------- session (multi-round) layer (NEW)

def _round_event(game, rnd, result):
    return {"event": "round", "game": game, "round": rnd, "result": result,
            "hp": 10, "dealt": 50, "taken": 40}


def _write_session_round(session, num, opp, game_results, games=None, rounds=1):
    """A session round subdir 'round_<NN>_<opp>/' that is itself a normal run dir. ``game_results`` is one
    result ('win'/'loss') per game (one SF2 round per game here), driving per_game_wl for that round."""
    rdir = os.path.join(session, "round_%02d_%s" % (num, opp))
    os.makedirs(rdir, exist_ok=True)
    with open(os.path.join(rdir, "run.json"), "w") as fh:
        json.dump({"me": "chunli", "opp": opp, "games": games or len(game_results), "rounds": rounds}, fh)
    events = [{"event": "seed", "lines": ["use more throw up close"]}]
    for g, res in enumerate(game_results):
        _write_round(rdir, g, 0, [_decision_record(1, 1.0, 0.5 if res == "win" else 0.9)])
        events.append(_round_event(g, 0, res))
    with open(os.path.join(rdir, "trace.jsonl"), "w") as fh:
        for e in events:
            fh.write(json.dumps(e) + "\n")
    return rdir


def _crafted_session(tmp_path):
    """Round 0 vs honda: 2 games 1W/1L. Round 1 vs ken: 2 games 2W. -> cumulative series W,L,W,W (win rate 3/4)."""
    session = str(tmp_path / "session")
    os.makedirs(session, exist_ok=True)
    with open(os.path.join(session, "session.json"), "w") as fh:
        json.dump({"rounds": 2, "opp_order": ["honda", "ken"]}, fh)
    _write_session_round(session, 0, "honda", ["win", "loss"])
    _write_session_round(session, 1, "ken", ["win", "win"])
    return session


def test_session_model_aggregates_two_rounds(tmp_path):
    """Cumulative per-game W/L across both rounds is [W,L,W,W]; win rate 3/4; per-round cum% is 50 then 75.

    Seen RED first: against a twin of build_session_model that reset the cumulative series each round (used only
    the last round's per_game_wl) -> per_game_wl == ('W','W') and win_rate 2/2, so the [W,L,W,W] / 3-of-4 asserts
    fail. Also red against the pre-change module (build_session_model does not exist -> AttributeError).
    """
    session = _crafted_session(tmp_path)
    sm = T.build_session_model(session, grade_qwen=False)

    assert sm.rounds_planned == 2
    assert sm.opp_order == ("honda", "ken")
    assert sm.per_game_wl == ("W", "L", "W", "W")
    assert (sm.cum_wins, sm.cum_played, sm.cum_pct) == (3, 4, 75)

    assert len(sm.rounds) == 2
    r0, r1 = sm.rounds
    assert (r0.num, r0.opp, r0.wins, r0.losses) == (0, "honda", 1, 1)
    assert r0.cum_pct == 50 and r0.cum_played == 2
    assert (r1.num, r1.opp, r1.wins, r1.losses) == (1, "ken", 2, 0)
    assert r1.cum_pct == 75 and r1.cum_played == 4


def test_session_active_is_latest_round(tmp_path):
    """The live view is the newest round dir's single-run model (round 1 vs ken here)."""
    session = _crafted_session(tmp_path)
    sm = T.build_session_model(session, grade_qwen=False)
    assert sm.active_num == 1 and sm.active_opp == "ken"
    assert sm.active is not None
    assert sm.active.opp == "ken"
    assert sm.active.run_dir.endswith("round_01_ken")


def test_session_tolerates_missing_session_json(tmp_path):
    """No session.json: opp_order empty, rounds_planned falls back to the count of round dirs present."""
    session = str(tmp_path / "s2")
    os.makedirs(session, exist_ok=True)
    _write_session_round(session, 0, "honda", ["win", "win"])
    sm = T.build_session_model(session, grade_qwen=False)
    assert sm.opp_order == ()
    assert sm.rounds_planned == 1
    assert sm.per_game_wl == ("W", "W")


def test_session_empty_dir(tmp_path):
    session = str(tmp_path / "empty_session")
    os.makedirs(session, exist_ok=True)
    sm = T.build_session_model(session, grade_qwen=False)
    assert sm.status == "empty"
    assert sm.active is None and sm.per_game_wl == ()
    assert sm.error is not None


def test_parse_session_round_dir():
    assert T.parse_session_round_dir("round_00_honda") == (0, "honda")
    assert T.parse_session_round_dir("round_12_ken") == (12, "ken")
    assert T.parse_session_round_dir("g00_r0") is None
    assert T.parse_session_round_dir("session.json") is None


def test_session_once_renders(tmp_path):
    """monitor_tui.py --session --once renders the growing win-rate meter (block glyph + 'win rate') headless.

    Seen RED first: before --session existed, argparse had no such flag, so the process either errored (no
    run dir resolved) or never showed 'win rate' from the SESSION view -> the grep asserts fail.
    """
    session = _crafted_session(tmp_path)
    script = os.path.join(REPO, "scripts", "monitor_tui.py")
    out = subprocess.run([sys.executable, script, "--session", session, "--once"],
                         capture_output=True, text=True, timeout=60)
    assert out.returncode == 0, out.stderr
    assert "█" in out.stdout                      # the history strip / win-rate meter block glyph
    assert "win rate" in out.stdout.lower()
    assert "SESSION" in out.stdout
    assert "ken" in out.stdout                    # the live round's opponent


def test_session_save_svg(tmp_path):
    """--session with --save writes an SVG with a block glyph (the export is reused for the session view)."""
    session = _crafted_session(tmp_path)
    out_svg = str(tmp_path / "session.svg")
    script = os.path.join(REPO, "scripts", "monitor_tui.py")
    out = subprocess.run([sys.executable, script, "--session", session, "--save", out_svg],
                         capture_output=True, text=True, timeout=60)
    assert out.returncode == 0, out.stderr
    assert os.path.isfile(out_svg)
    assert "█" in open(out_svg, encoding="utf-8").read()
