import pytest

from sf2.data.action_common import filter_row, filter_rows, rare_codes


def _row(actor="ken", code=30, key="act", answer="act30"):
    crit = {"act08": "", "act30": "", "act61": "", "act70": ""}
    return {"id": "x", "actor": actor, "code": code, "key": key, "answer": answer,
            "question": {"type": "choice", "instructions": "q", "criteria": crit},
            "label": list(crit).index(answer) if answer in crit else 0}


def test_rare_codes_threshold():
    table = {"ken act08": {"episodes_seen": 9}, "ken act30": {"episodes_seen": 10}, "chunli act08": {"episodes_seen": 1}}
    assert rare_codes(table) == {("ken", 8), ("chunli", 8)}


def test_rare_codes_bad_key():
    with pytest.raises(ValueError):
        rare_codes({"ken 30": {"episodes_seen": 1}})


def test_option_removed_and_label_reindexed():
    out = filter_row(_row(answer="act30"), {("ken", 8)})
    assert list(out["question"]["criteria"]) == ["act30", "act61", "act70"]
    assert out["label"] == 0 and out["answer"] == "act30"


def test_other_actor_rare_code_untouched():
    out = filter_row(_row(), {("chunli", 8)})
    assert "act08" in out["question"]["criteria"] and out["label"] == 1


def test_rare_row_dropped_act_and_stage():
    rare = {("ken", 30)}
    assert filter_row(_row(), rare) == {}
    assert filter_row(_row(key="stage", answer="stg1"), rare) == {}


def test_stage_row_kept_unchanged():
    row = _row(key="stage", answer="stg2")
    assert filter_row(row, {("ken", 8)}) == row


def test_input_not_mutated():
    row = _row()
    filter_row(row, {("ken", 8)})
    assert "act08" in row["question"]["criteria"] and row["label"] == 1


def test_answer_missing_raises():
    row = _row(answer="act99")
    with pytest.raises(ValueError):
        filter_row(row, {("ken", 8)})


def test_filter_rows_counts():
    rows = [_row(), _row(code=8, answer="act08"), _row(key="stage", answer="stg1")]
    assert len(list(filter_rows(rows, {("ken", 8)}))) == 2


def test_gate_expects_dropped_actions_absent(tmp_path):
    import json
    from sf2.data.action_gate import dropped_actions
    assert dropped_actions(str(tmp_path)) == frozenset()
    (tmp_path / "filter.json").write_text(json.dumps({"dropped": {"ken act08": 3}}))
    assert dropped_actions(str(tmp_path)) == frozenset({"ken act08"})
