"""The P(hit)-based eval tools refuse lookup-table runs (docs/prereg_2x2.md): a table run logs no laya-vision score
(top3 [] , scores {}) and rates its shortlist on the net scale, so boundary's vision stage, laya_evidence's top-3 and
gaps' "is 'likely works' really likely" would report nonsense on it. A table run: run.json / verdict.json names an
"oracle", or the run's name ends "+table" (sf2.eval.logs.table_run); a table decision: "values" and no top3."""
import json
import os

import pytest

from sf2.eval import boundary as b
from sf2.eval import gaps
from sf2.eval.logs import table_run, table_rows
from tests.test_boundary import MOVES, THROW, _cli, row
from tests.test_laya_evidence import E, act, make_run


def table_row(**kw):
    return dict(row(**kw), top3=[], scores={}, values={"throw": 6.0, "forward": 1.0})


# ---- the helper ----

def test_table_run_by_name_run_file_or_verdict(tmp_path):
    plain = tmp_path / "20260930-000000_ryu"
    (plain / "loop").mkdir(parents=True)
    assert table_run(str(plain)) is None and table_run(str(plain / "loop")) is None
    named = tmp_path / "20260930-000001_ryu_book+table"
    named.mkdir()
    assert "+table" in table_run(str(named))
    arm = tmp_path / "x" / "none"
    arm.mkdir(parents=True)
    (arm / "run.json").write_text(json.dumps({"arm": "none", "oracle": "lessons/value_oracle_v1.json"}))
    assert "oracle" in table_run(str(arm)) and "oracle" in table_run(str(tmp_path / "x"))     # the arm's run file
    ab = tmp_path / "20260930_s5"
    (ab / "ryu_none").mkdir(parents=True)
    (ab / "run.json").write_text(json.dumps({"model": None, "oracle": "t.json"}))
    assert "oracle" in table_run(str(ab / "ryu_none"))                                          # the batch's run.json
    v = tmp_path / "v"
    v.mkdir()
    (v / "verdict.json").write_text(json.dumps({"oracle": "t.json"}))
    assert table_run(str(v))


def test_table_rows_counts_decisions_without_laya_vision():
    assert table_rows([row(), row()]) == 0
    assert table_rows([row(), table_row()]) == 1
    assert table_rows([dict(row(), values={"lk": 1.0})]) == 0      # a value checkpoint: values AND laya-vision's top3


# ---- boundary ----

def test_boundary_trace_refuses_table_rows():
    arms = {"none": {"lines": [], "rows": [table_row()] * 3, "rounds": []},
            "throw": {"lines": [THROW], "rows": [row(action="throw")] * 3, "rounds": []}}
    with pytest.raises(ValueError, match="lookup-table"):
        b.trace(THROW, MOVES, arms, [], [])


def test_boundary_cli_refuses_a_table_batch(tmp_path):
    cli = _cli()
    root = tmp_path / "rollouts" / "20260930_s5+table"
    (root / "ryu_none").mkdir(parents=True)
    logs = tmp_path / "logs"
    logs.mkdir()
    (logs / "a.log").write_text("saved rollouts/20260930_s5+table\n")
    with pytest.raises(SystemExit, match="lookup-table"):
        cli.roots(str(logs), str(tmp_path))


def test_boundary_cli_leaves_table_ledgers_out(tmp_path):
    cli = _cli()
    for name in ("1_ryu", "2_ryu+table"):
        d = tmp_path / "rollouts" / "qwen_lessons" / name / "loop"
        d.mkdir(parents=True)
        (d / "ledger.jsonl").write_text(json.dumps({"game": 0, "outcome": []}) + "\n")
    assert len(cli.ledgers(str(tmp_path), "ryu")) == 1


# ---- laya_evidence ----

def test_laya_evidence_skips_table_runs_with_the_reason(tmp_path):
    line = "use more hp up close"
    make_run(tmp_path, "1_ken", [act("hp")], [act("mp")], [line])
    make_run(tmp_path, "2_ken_book+table", [act("hp")], [act("mp")], [line])
    res = E.collect([str(tmp_path)])
    assert res["runs_read"] == 1 and res["failures"] == []
    assert [os.path.basename(d) for d, _ in res["skipped_table"]] == ["2_ken_book+table"]


# ---- gaps ----

def test_gaps_refuses_table_decisions():
    acts = [dict(table_row(), opp="ryu", kind="attack", actual="hit", i_was_hit=False, opp_attacked=False)]
    with pytest.raises(ValueError, match="lookup-table"):
        gaps.report("s", acts, [], [])


def test_report_cli_refuses_a_table_session(tmp_path, monkeypatch):
    import importlib.util
    import sys
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, os.path.join(here, "scripts"))
    spec = importlib.util.spec_from_file_location("report_cli", os.path.join(here, "scripts", "report.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    s = tmp_path / "sess+table"
    s.mkdir()
    monkeypatch.setattr(sys, "argv", ["report.py", "gaps", "--session", str(s)])
    with pytest.raises(SystemExit, match="lookup-table"):
        m.main()
