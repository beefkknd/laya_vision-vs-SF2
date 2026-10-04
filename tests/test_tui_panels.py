"""The TUI actually renders the NEW panels (regression guard for 'I still see the old TUI').

Renders render() and render_session() to text and asserts the new sections are present: SHORT MEMORY
(roomy, listing the in-play rules), the TREND bar graph, and the cross-system DATA FLOW. If someone runs
a stale build, or a layout change drops a panel, this goes RED.
"""
import importlib.util
import json
import os

from rich.console import Console

from sf2.eval import tui_model as T

_MT = importlib.util.spec_from_file_location(
    "monitor_tui", os.path.join(os.path.dirname(__file__), "..", "scripts", "monitor_tui.py"))
mt = importlib.util.module_from_spec(_MT)
_MT.loader.exec_module(mt)


def _to_text(renderable) -> str:
    con = Console(record=True, width=160, file=open(os.devnull, "w"))
    con.print(renderable)
    return con.export_text()


def _single_run(tmp):
    os.makedirs(os.path.join(tmp, "g00_r0"), exist_ok=True)
    open(os.path.join(tmp, "run.json"), "w").write(json.dumps({"me": "chunli", "opp": "honda", "games": 2, "rounds": 1}))
    rules = ["use more s.mk at mid range when he stands", "use more throw up close"]
    with open(os.path.join(tmp, "trace.jsonl"), "w") as f:
        f.write(json.dumps({"event": "seed", "opp": "honda", "lines": rules}) + "\n")
        f.write(json.dumps({"event": "round", "game": 0, "round": 0, "result": "win", "hp": 40, "dealt": 80, "taken": 40}) + "\n")
    return tmp


def test_single_run_view_has_the_new_panels(tmp_path):
    m = T.build_model(_single_run(str(tmp_path)), grade_qwen=False)
    txt = _to_text(mt.render(m, frame=2))
    for tag in ("LIVE GAMEPLAY", "SHORT MEMORY", "QWEN", "TREND", "DATA FLOW"):
        assert tag in txt, "missing panel: %s" % tag
    assert "use more s.mk at mid range when he stands" in txt     # in-play rules actually listed
    assert "VISION" in txt and "CONTROL" in txt                   # the data-flow nodes


def test_session_view_has_career_trend_and_data_flow(tmp_path):
    # a 1-opponent session dir
    rd = os.path.join(str(tmp_path), "round_00_honda")
    _single_run(rd)
    sm = T.build_session_model(str(tmp_path), grade_qwen=False)
    txt = _to_text(mt.render_session(sm, frame=2))
    for tag in ("SESSION", "CAREER TREND", "DATA FLOW", "SHORT MEMORY"):
        assert tag in txt, "missing panel: %s" % tag
