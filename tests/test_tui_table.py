"""The loop monitor (overlay) must show WHAT THE VALUE TABLE / QUORUM decided, not just a rules-policy
pipeline. For --policy table/hybrid/quorum a decision row carries `when` (the table cell), `source`
(which voter/path chose the move) and `explored` (an exploration pick) instead of cat_probs/move_probs/
follows_rule. The pure model must surface those, and build_model must read the run's `policy`.

Seen RED provenance: run against the pre-change tui_model (no DecisionView.cell/source/explored and no
DashboardModel.policy) these assertions fail (AttributeError / missing attr). The panel test was first
run against the old renderer, which showed no cell/source for a table decision.
"""
import json
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from sf2.eval import tui_model as T   # noqa: E402


def _table_run(tmp: str) -> str:
    """A tiny --policy table run dir: run.json{policy:table} + one round with table-style decisions."""
    os.makedirs(os.path.join(tmp, "g00_r0"), exist_ok=True)
    with open(os.path.join(tmp, "run.json"), "w") as f:
        json.dump({"me": "ken", "opp": "ryu", "games": 1, "rounds": 1, "policy": "table"}, f)
    decs = [
        {"i": 0, "k": 10, "action": "cl.lk", "category": "kick",
         "when": "close|standing|0", "explored": False, "source": "table",
         "situation": ["close", "standing", "high", "full"],
         "moment": {"dx": -32, "doing": "standing", "fireball": False},
         "pressed": ["down", "b"]},
        {"i": 1, "k": 44, "action": "jump", "category": "move",
         "when": "far|standing|0", "explored": True, "source": "explore",
         "situation": ["far", "standing", "high", "full"],
         "moment": {"dx": 120, "doing": "standing", "fireball": False},
         "pressed": ["up"]},
    ]
    with open(os.path.join(tmp, "g00_r0", "decisions.jsonl"), "w") as f:
        for d in decs:
            f.write(json.dumps(d) + "\n")
    with open(os.path.join(tmp, "trace.jsonl"), "w") as f:
        f.write(json.dumps({"event": "seed", "opp": "ryu", "lines": []}) + "\n")
    return tmp


def _quorum_run(tmp: str) -> str:
    os.makedirs(os.path.join(tmp, "g00_r0"), exist_ok=True)
    with open(os.path.join(tmp, "run.json"), "w") as f:
        json.dump({"me": "ryu", "opp": "ken", "games": 1, "rounds": 1, "policy": "quorum"}, f)
    d = {"i": 0, "k": 12, "action": "hadouken", "category": "special",
         "when": "mid|standing|0", "explored": False, "source": "laya-vision",
         "quorum": {"quorum_move": "hadouken", "votes": {"laya-vision": "hadouken"}},
         "situation": ["mid", "standing", "high", "full"],
         "moment": {"dx": 70, "doing": "standing", "fireball": False}, "pressed": ["down", "right", "y"]}
    with open(os.path.join(tmp, "g00_r0", "decisions.jsonl"), "w") as f:
        f.write(json.dumps(d) + "\n")
    with open(os.path.join(tmp, "trace.jsonl"), "w") as f:
        f.write(json.dumps({"event": "seed", "opp": "ken", "lines": []}) + "\n")
    return tmp


def test_build_model_reads_policy(tmp_path):
    m = T.build_model(_table_run(str(tmp_path)), grade_qwen=False)
    assert m.policy == "table"


def test_table_decisions_surface_cell_source_explored(tmp_path):
    m = T.build_model(_table_run(str(tmp_path)), grade_qwen=False)
    assert len(m.decisions) == 2
    d0, d1 = m.decisions
    assert d0.action == "cl.lk" and d0.category == "kick"
    assert d0.cell == "close|standing|0"
    assert d0.source == "table"
    assert d0.explored is False
    assert d1.cell == "far|standing|0"
    assert d1.source == "explore"
    assert d1.explored is True


def test_quorum_decision_surfaces_source(tmp_path):
    m = T.build_model(_quorum_run(str(tmp_path)), grade_qwen=False)
    assert m.policy == "quorum"
    assert m.decisions[0].source == "laya-vision"
    assert m.decisions[0].cell == "mid|standing|0"


def test_panel_shows_table_cell_and_source(tmp_path):
    """The rendered gameplay panel shows the table cell + source for a table run (owner records this)."""
    import importlib.util
    from rich.console import Console
    spec = importlib.util.spec_from_file_location(
        "monitor_tui", os.path.join(REPO, "scripts", "monitor_tui.py"))
    mt = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mt)
    m = T.build_model(_table_run(str(tmp_path)), grade_qwen=False)
    con = Console(record=True, width=160, file=open(os.devnull, "w"))
    con.print(mt.render(m, frame=2))
    txt = con.export_text()
    assert "close|standing|0" in txt      # the table cell the move was chosen in
    assert "table" in txt                  # the source
    assert "explore" in txt                # the exploration pick on the second decision


# ---- the bee-quorum vote ("table fire") is surfaced from the latest decision's quorum record ----
def _quorum_vote_run(tmp: str) -> str:
    os.makedirs(os.path.join(tmp, "g00_r0"), exist_ok=True)
    with open(os.path.join(tmp, "run.json"), "w") as f:
        json.dump({"me": "ken", "opp": "ryu", "games": 1, "rounds": 1, "policy": "quorum"}, f)
    dec = {"i": 0, "k": 10, "action": "shoryuken_hp", "category": "special", "when": "mid|attacking|0",
           "source": "quorum", "situation": ["mid", "attacking", "half", "half"],
           "moment": {"dx": 60, "doing": "attacking", "fireball": True}, "pressed": ["f", "d", "hp"],
           "quorum": {"mode": "vote", "share": 0.78, "theta": 0.5, "top": "shoryuken_hp",
                      "quorum_move": "shoryuken_hp", "laya_move": "block_high",
                      "proposals": [["laya", "block_high", 0.9, 1.0], ["table", "shoryuken_hp", 0.8, 1.3],
                                    ["fireball", "jump_in", 1.0, 1.0]],
                      "scores": {"block_high": 0.9, "shoryuken_hp": 1.84, "jump_in": 1.0}}}
    with open(os.path.join(tmp, "g00_r0", "decisions.jsonl"), "w") as f:
        f.write(json.dumps(dec) + "\n")
    with open(os.path.join(tmp, "trace.jsonl"), "w") as f:
        f.write(json.dumps({"event": "table", "game": 0, "round": 0, "cells": 25}) + "\n")
        f.write(json.dumps({"event": "scout", "game": 0, "round": 0, "verdict": "winning"}) + "\n")
    return tmp


def test_quorum_view_parsed_from_the_latest_decision_vote(tmp_path):
    m = T.build_model(_quorum_vote_run(str(tmp_path / "q")), grade_qwen=False)
    q = m.quorum
    assert q is not None
    assert q.quorum_move == "shoryuken_hp" and q.share == 0.78
    assert q.cells == 25 and q.verdict == "winning"
    assert {p[0] for p in q.proposals} == {"laya", "table", "fireball"}
    assert q.scores[0][0] == "shoryuken_hp"                     # sorted best-first -> the winner leads


def test_quorum_panel_renders_the_vote():
    import io
    sys.path.insert(0, os.path.join(REPO, "scripts"))
    import monitor_tui as MT
    from rich.console import Console
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        m = T.build_model(_quorum_vote_run(os.path.join(d, "q")), grade_qwen=False)
        con = Console(file=io.StringIO(), width=100, legacy_windows=False)
        con.print(MT._quorum_panel(m))
        out = con.file.getvalue()
    assert "shoryuken_hp" in out and "fires" in out            # the winner + that it fired
    assert "25 cells" in out and "winning" in out              # table size + the scout read
    assert "laya" in out and "table" in out and "fireball" in out   # the voters
