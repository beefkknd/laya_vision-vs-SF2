"""scripts/filter_u_data.py (docs/prereg_u_round2.md): round 1's dataset minus some questions, frames shared, never copied."""
import importlib.util
import json
import os
import sys

import pytest

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, "..", "scripts"))
spec = importlib.util.spec_from_file_location("filter_u", os.path.join(HERE, "..", "scripts", "filter_u_data.py"))
fu = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fu)

QS = ["How far away is he?", "How full is my health bar?", "Is the gap between us changing?"]


def src(tmp_path):
    s = tmp_path / "src"
    for c in ("chunli", "ryu"):
        d = s / c
        (d / "frames").mkdir(parents=True)
        (d / "frames" / "a_now.png").write_bytes(b"png")
        for name in ("train", "val", "test_real"):
            rows = [{"id": "%s-%d" % (q, i), "question": {"instructions": q}, "images": ["frames/a_now.png"]}
                    for i in range(2) for q in QS]
            (d / (name + ".jsonl")).write_text("".join(json.dumps(r) + "\n" for r in rows))
    (s / "build.json").write_text("{}")
    return s


def test_drops_named_questions_and_shares_frames(tmp_path):
    s = src(tmp_path)
    out = tmp_path / "out"
    fu.filter_build(str(s), str(out), QS[1:])
    rows = [json.loads(l) for l in open(out / "chunli" / "train.jsonl")]
    assert {r["question"]["instructions"] for r in rows} == {QS[0]} and len(rows) == 2
    assert os.path.islink(out / "chunli" / "frames")
    assert os.path.exists(out / "chunli" / rows[0]["images"][0])          # resolves through the link
    meta = json.load(open(out / "build.json"))
    assert meta["dropped"] == QS[1:] and meta["counts"]["chunli/train.jsonl"] == [6, 2]


def test_unknown_question_refused(tmp_path):
    with pytest.raises(SystemExit):
        fu.filter_build(str(src(tmp_path)), str(tmp_path / "out"), ["No such question?"])


def test_existing_out_refused(tmp_path):
    s = src(tmp_path)
    (tmp_path / "out").mkdir()
    with pytest.raises(SystemExit):
        fu.filter_build(str(s), str(tmp_path / "out"), QS[1:])
