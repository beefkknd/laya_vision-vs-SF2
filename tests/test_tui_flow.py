"""The bottom DATA FLOW strip shows the CURRENT architecture, not a retired one.

The strip used to hard-code the old rules/text-laya pipeline (VISION -> TEXT -> CATEGORY -> MOVE ->
CONTROL, and laya_text -> QWEN -> PLAYBOOK -> MEMORY, "MEMORY feeds back into MOVE"). The live System 1
is the bee-quorum + value table: a frame becomes words, three bees vote, the table decides, the gamepad
acts; between rounds Qwen writes one rule that re-weights the table. These tests pin the strip to that
reality so a stale build goes RED. Seen RED against the old node constants before the fix.
"""
import importlib.util
import json
import os
import types

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


# the retired nodes/labels that must NOT reappear in the strip (the full old PLAY + LEARN node sets)
_RETIRED = ("VISION", "TEXT ", "CATEGORY", "MOVE", "CONTROL", "PLAYBOOK", "MEMORY",
            "laya_text", "feeds back into MOVE")


def _run_with_source(tmp: str, source: str) -> str:
    """A tiny --policy quorum run whose single decision was chosen by ``source`` (a voter name), so a
    rendered dashboard must surface that voter in the DATA FLOW strip."""
    os.makedirs(os.path.join(tmp, "g00_r0"), exist_ok=True)
    with open(os.path.join(tmp, "run.json"), "w") as f:
        json.dump({"me": "ryu", "opp": "ken", "games": 1, "rounds": 1, "policy": "quorum"}, f)
    dec = {"i": 0, "k": 12, "action": "hadouken", "category": "special",
           "when": "mid|standing|0", "explored": False, "source": source,
           "situation": ["mid", "standing", "high", "full"],
           "moment": {"dx": 70, "doing": "standing", "fireball": False}, "pressed": ["down", "right", "y"]}
    with open(os.path.join(tmp, "g00_r0", "decisions.jsonl"), "w") as f:
        f.write(json.dumps(dec) + "\n")
    with open(os.path.join(tmp, "trace.jsonl"), "w") as f:
        f.write(json.dumps({"event": "seed", "opp": "ken", "lines": []}) + "\n")
    return tmp


def test_play_path_is_the_quorum_table_pipeline():
    txt = _to_text(mt._pipeline_panel(False, 0, source="frontier"))
    for node in ("FRAME", "EYE", "BEES", "TABLE", "GAMEPAD"):
        assert node in txt, "play path missing node: %s" % node
    for dead in _RETIRED:
        assert dead not in txt, "retired label still shown: %r" % dead


def test_learn_path_is_log_qwen_rule_table():
    txt = _to_text(mt._pipeline_panel(True, 0))
    for node in ("LOG", "QWEN", "RULE", "TABLE"):
        assert node in txt, "learn path missing node: %s" % node
    for dead in _RETIRED:
        assert dead not in txt, "retired label still shown: %r" % dead


def test_footer_describes_the_rule_reweighting_the_table():
    txt = _to_text(mt._pipeline_panel(False, 0))
    assert "re-weight" in txt.lower() and "TABLE" in txt
    assert "feeds back into MOVE" not in txt


def test_latest_voter_source_is_surfaced():
    """When a real decision exists, the strip grounds the pulse in the voter that actually chose,
    rather than only a cosmetic frame%%N march."""
    txt = _to_text(mt._pipeline_panel(False, 7, source="pressure"))
    assert "pressure" in txt


def test_latest_source_helper_handles_missing_and_reads_the_last_decision():
    """_latest_source: '' for no model / no decisions, else the newest decision's source. A regression
    that always returned '' (or read the wrong end) would drop the voter from the strip."""
    assert mt._latest_source(None) == ""
    assert mt._latest_source(types.SimpleNamespace(decisions=())) == ""
    d = lambda s: types.SimpleNamespace(source=s)
    assert mt._latest_source(types.SimpleNamespace(decisions=(d("laya"), d("frontier")))) == "frontier"
    assert mt._latest_source(types.SimpleNamespace(decisions=(d(None),))) == ""


def test_rendered_dashboard_surfaces_the_real_voter_in_the_flow(tmp_path):
    """End-to-end: build a real model from a run whose decision was chosen by 'frontier', render the
    single-run dashboard, and assert the DATA FLOW strip names that voter. Exercises _latest_source AND
    the render() call site (not just _pipeline_panel in isolation)."""
    m = T.build_model(_run_with_source(str(tmp_path), "frontier"), grade_qwen=False)
    txt = _to_text(mt.render(m, frame=3))
    assert "frontier" in txt and "table fires" in txt
