"""The three movement datasets (sf2.data.action_data) and their gate (sf2.data.action_gate), on a synthetic
collection made by the real collector loop (sf2.data.action_collect_io) from scripted RAM streams."""
import json
import os
from collections import Counter

import numpy as np
import pytest
from ram_rows import row

from sf2.data import action_codes as A
from sf2.data import action_collect_io as IO
from sf2.data import action_data as D
from sf2.data import action_gate as G
from sf2.emu.vs import GROUND_Y

NONE = {"aid": 0, "mclass": 0xFF, "sclass": 0xFF}
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TH = os.path.join(ROOT, "lessons", "perception_thresholds_v2.json")


def r(p1=None, p2=None, **extra):
    return row(dict(NONE, char=5, **(p1 or {})), dict(NONE, **(p2 or {})), **extra)


def image(k):
    im = np.zeros((224, 256, 3), np.uint8)
    im[0, 0] = (k // 256 % 256, k % 256, 7)
    return im


def script(game):
    """Her: stand, a jab (ID 2), crouch; him: walk in, a jump with an attack (ID 25), a cut attack (unknown), stand."""
    rows = []
    for i in range(12):
        rows.append(r({"x": 200}, {"x": 300 - 2 * i}))
    for i in range(9):
        rows.append(r({"x": 200, "state": 0x0A, "aid": 2 if i >= 2 else 0}, {"x": 276, "state": 0x04,
                                                                              "y": GROUND_Y - 10 - i,
                                                                              "aid": 25 if i >= 3 else 0}))
    for i in range(5):
        rows.append(r({"x": 200, "state": 0x02}, {"x": 276, "state": 0x0A}))
    for i in range(10):
        rows.append(r({"x": 200, "state": 0x02}, {"x": 230 + game % 3}))
    return rows


def play(game, sampler):
    for k, x in enumerate(script(game)):
        sampler.feed(x, None if k == 0 else image(k))
    return {}


@pytest.fixture
def built(tmp_path):
    root = str(tmp_path / "mv3")
    IO.collect_opponent(os.path.join(root, "ken"), play, "ken", {1: "chunli", 2: "ken"}, seed=0, cap=10,
                        min_games=100, patience=50, log=lambda *a: None)
    outs = {ds: str(tmp_path / ("data_" + ds)) for ds in D.DATASETS}
    res = D.build(root, outs, thresholds=TH)
    return root, outs, res


def rows_of(out, opp="ken"):
    out_rows = []
    for f in ("train", "val", "test_real"):
        out_rows += [json.loads(x) for x in open(os.path.join(out, opp, f + ".jsonl"))]
    return out_rows


def test_build_three_datasets_no_problems_split_by_game(built):
    root, outs, res = built
    assert res["problems"] == []
    for ds in D.DATASETS:
        rows = rows_of(outs[ds])
        assert rows
        for x in rows:
            assert x["split"] == {2: "test", 5: "test", 8: "test", 0: "val"}.get(x["game"] % 10, "train")
            assert x["state_text"] == "me=chunli" and x["perception"] is True and x["note_version"] == 3
            assert list(x["question"]["criteria"])[x["label"]] == x["answer"]
            assert x["images"] == ["frames/g%04d_k%05d.png" % (x["game"], k) for k in (x["t"] - 3, x["t"] + 1)]
        assert os.path.islink(os.path.join(outs[ds], "ken", "frames"))
        assert len({x["id"] for x in rows}) == len(rows)


def test_act_rows_both_fighters_two_questions_and_label_full(built):
    _, outs, res = built
    rows = rows_of(outs["act"])
    by = Counter((x["decision"], x["player"]) for x in rows)
    assert set(by.values()) == {2}
    assert {x["actor"] for x in rows} == {"chunli", "ken"}
    jump = [x for x in rows if x["actor"] == "ken" and x["code"] == 25]
    assert jump and all(x["label_full"] == "ken act25 stg%d" % x["stg"] for x in jump)
    assert {x["answer"] for x in jump if x["key"] == "act"} == {"act25"}
    q = [x for x in rows if x["key"] == "act" and x["player"] == 2][0]["question"]
    assert "Ken (him)" in q["instructions"] and list(q["criteria"]) == ["act%02d" % c for c in res["options"]["ken"]]
    q = [x for x in rows if x["key"] == "act" and x["player"] == 1][0]["question"]
    assert "Chun-Li (me)" in q["instructions"]
    assert {x["answer"] for x in rows if x["key"] == "stage"} <= {"stg1", "stg2", "stg3"}


def test_unknown_fighter_gets_no_rows(built):
    _, outs, _ = built
    rows = rows_of(outs["act"])
    # his cut attack rows 21-25 are unknown: a pair there has her rows only
    cut = [x for x in rows if 21 <= x["t"] <= 25]
    assert cut and {x["player"] for x in cut} == {1}


def test_where_and_dist_answers(built):
    _, outs, _ = built
    for x in rows_of(outs["where"]):
        assert x["answer"] == ("in the air" if 12 <= x["t"] <= 20 else "on the ground")
    d = {x["t"]: x["answer"] for x in rows_of(outs["dist"])}
    assert all(v in ("throw", "poke", "mid", "far") for v in d.values())
    assert any(v == "throw" for t, v in d.items() if t >= 26) and any(v in ("mid", "far") for t, v in d.items()
                                                                      if t < 12)


def test_gate_passes_on_a_clean_build(built):
    _, outs, _ = built
    for ds in D.DATASETS:
        rep = G.run_gates(outs[ds], TH, check_train=False)
        assert rep["gates"]["labels"]["pass"], rep["gates"]["labels"]
        assert rep["gates"]["labels"]["mismatches"] == 0 and rep["gates"]["disk"]["pass"]
    cov = G.run_gates(outs["act"], TH, check_train=False)["coverage"]
    assert cov["pass"] and "ken act25" in cov["table"]


def _tamper(path, fn):
    rows = [json.loads(x) for x in open(path)]
    rows = fn(rows)
    with open(path, "w") as f:
        f.write("".join(json.dumps(x) + "\n" for x in rows))


@pytest.mark.parametrize("ds,fn", [
    ("act", lambda rs: [dict(x, answer="stg1", label=0) if x["key"] == "stage" and x["answer"] == "stg3" else x
                        for x in rs]),
    ("act", lambda rs: [x for x in rs if not (x["key"] == "stage" and x["player"] == 2)]),
    ("act", lambda rs: [dict(x, code=x["code"] + 1) if x["player"] == 2 else x for x in rs]),
    ("where", lambda rs: [dict(x, answer="in the air", label=1) for x in rs]),
    ("dist", lambda rs: [dict(x, answer="far", label=3) for x in rs]),
    ("dist", lambda rs: [dict(x, images=list(reversed(x["images"]))) for x in rs]),
    ("act", lambda rs: rs + [dict(x, player=2, actor="ken", id=x["id"] + "-x") for x in rs
                             if x["player"] == 1 and 21 <= x["t"] <= 25][:1]),
])
def test_gate_catches_a_wrong_row(built, ds, fn):
    _, outs, _ = built
    _tamper(os.path.join(outs[ds], "ken", "train.jsonl"), fn)
    assert not G.run_gates(outs[ds], TH, check_train=False)["gates"]["labels"]["pass"]


def test_gate_missing_ram_fails(built):
    root, outs, _ = built
    os.remove(os.path.join(root, "ken", "ram", "g0001.json.gz"))
    assert not G.run_gates(outs["where"], TH, check_train=False)["gates"]["labels"]["pass"]


def test_builder_refuses_bad_pairs(tmp_path, built):
    root, _, _ = built
    p = os.path.join(root, "ken", "pairs.jsonl")
    _tamper(p, lambda rs: [dict(x, stg=1 + x["stg"] % 3) if i == 0 else x for i, x in enumerate(rs)])
    outs = {ds: str(tmp_path / ("again_" + ds)) for ds in D.DATASETS}
    assert any("collector label" in x for x in D.build(root, outs, thresholds=TH)["problems"])


def test_builder_refuses_missing_image_and_wrong_split(tmp_path, built):
    root, _, _ = built
    imgs = os.path.join(root, "ken", "images")
    os.remove(os.path.join(imgs, sorted(os.listdir(imgs))[0]))
    p = os.path.join(root, "ken", "pairs.jsonl")
    _tamper(p, lambda rs: [dict(x, split="train") if x["split"] == "test" else x for x in rs])
    outs = {ds: str(tmp_path / ("b_" + ds)) for ds in D.DATASETS}
    probs = D.build(root, outs, thresholds=TH)["problems"]
    assert any("missing image" in x for x in probs) and any("belongs to test" in x for x in probs)


def test_builder_refuses_duplicate_rows_and_existing_out(tmp_path, built):
    root, outs, _ = built
    with pytest.raises(SystemExit):
        D.build(root, outs, thresholds=TH)
    p = os.path.join(root, "ken", "pairs.jsonl")
    _tamper(p, lambda rs: rs + rs[:1])
    outs2 = {ds: str(tmp_path / ("c_" + ds)) for ds in D.DATASETS}
    assert any("pairs at one displayed row" in x for x in D.build(root, outs2, thresholds=TH)["problems"])


def test_head_tokens_counts_question_and_options():
    class Tok:
        def __call__(self, s, add_special_tokens=False):
            return {"input_ids": s.split()}
    q = D.act_question("ken", 2, [1, 2, 70])
    n = D.head_tokens(q, Tok())
    assert n == D.head_tokens(D.act_question("ken", 2, [1, 2]), Tok()) + 3   # "-", "act70", terminator


def test_independent_actions_agree_with_the_collector_labels():
    rows = script(0)
    for p in (1, 2):
        want = {t: (ep.code, stg) for t, (ep, stg) in A.labels_at(rows, p).items()}
        assert G.independent_actions(rows, p) == want


def test_gate_disk_budget(built):
    _, outs, _ = built
    rep = G.run_gates(outs["where"], TH, max_gb=1e-9, check_train=False)
    assert not rep["gates"]["disk"]["pass"] and not rep["pass"]


def test_files_hold_their_own_split(built):
    _, outs, _ = built
    for ds in D.DATASETS:
        for f, split in (("train", "train"), ("val", "val"), ("test_real", "test")):
            rows = [json.loads(x) for x in open(os.path.join(outs[ds], "ken", f + ".jsonl"))]
            assert rows and all(x["split"] == split for x in rows)


@pytest.mark.parametrize("gap,want", [(0, "throw"), (43, "throw"), (44, "poke"), (64, "poke"), (65, "mid"),
                                      (119, "mid"), (120, "far")])
def test_independent_band_edges(gap, want):
    th = json.load(open(TH))
    assert G.independent_band(r({"x": 200}, {"x": 200 + gap}), th) == want
    from sf2.data.perception import range_band
    assert range_band([r({"x": 200}, {"x": 200 + gap})], 0, th) == want
