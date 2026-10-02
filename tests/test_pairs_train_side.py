"""Round 2 of docs/prereg_movement_finetunes.md: the question names the fighter by SCREEN SIDE. Side = which fighter
has the smaller x at the displayed frame t (lag 1: the RAM row t the row's labels come from); equal x -> the row is
dropped and counted; no character name in the question."""
import gzip
import json
import os
import shutil

import pytest

from sf2.data import movement_collect_io as MIO
from sf2.data import pairs_data as D
from sf2.data import pairs_train as T
from ram_rows import row
from test_pairs_collect import BANDS
from test_pairs_down import _knock_collection

WANT = {"movement": "What is the fighter on the %s doing?",
        "facing": "Which way is the fighter on the %s facing?",
        "air": "Is the fighter on the %s on the ground or in the air?",
        "distance": "Is the fighter on the %s close to or far from the other fighter?"}


@pytest.mark.parametrize("q", list(WANT))
@pytest.mark.parametrize("side", ["left", "right"])
def test_the_question_text_names_the_side_exactly_and_no_character(q, side):
    qq = T.question_side(q, side)
    assert qq["instructions"] == WANT[q] % side
    assert not any(n in qq["instructions"] for n in T.NAMES.values())
    assert tuple(qq["criteria"]) == D.QUESTION_ANSWERS[q] and qq["type"] == "choice"


def test_question_side_refuses_another_side():
    with pytest.raises(ValueError):
        T.question_side("movement", "middle")


def test_side_is_the_smaller_x_for_either_slot():
    r = row({"x": 100}, {"x": 300})
    assert T.side_of(r, 1) == "left" and T.side_of(r, 2) == "right"
    r = row({"x": 301}, {"x": 300})
    assert T.side_of(r, 1) == "right" and T.side_of(r, 2) == "left"


def test_equal_x_has_no_side():
    r = row({"x": 250}, {"x": 250})
    assert T.side_of(r, 1) is None and T.side_of(r, 2) is None


def test_side_at_a_crossing_by_one_pixel():
    assert T.side_of(row({"x": 249}, {"x": 250}), 1) == "left"
    assert T.side_of(row({"x": 251}, {"x": 250}), 1) == "right"


def test_the_side_is_read_at_the_displayed_row_t_not_t_minus_4_nor_the_capture_row():
    """A fighter who crossed over between the two frames: left at t - 4, right at t -> right. The capture row t + 1
    (lag 1) shows him back on the left; the displayed row t decides."""
    ram = [row({"x": 100}, {"x": 200}) for _ in range(12)]
    t = 8
    ram[t] = row({"x": 230}, {"x": 200})
    p = {"t": t, "slot": 1}
    assert T.side_at(ram, p) == "right"
    assert T.side_at(ram, dict(p, slot=2)) == "left"
    ram[t] = row({"x": 200}, {"x": 200})
    assert T.side_at(ram, p) is None


# -- the build ------------------------------------------------------------------------------------------------------

@pytest.fixture(scope="module")
def src(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("mv2")
    root, out = str(tmp / "col"), str(tmp / "pairs")
    _knock_collection(root, games=5)
    D.build(root, out, {"train": 40, "test": 20}, bands=BANDS)
    return out


def _ram(src, r):
    root = json.load(open(os.path.join(src, "build.json")))["root"]
    return MIO.read_ram(os.path.join(root, r["pair_name"], "ram", "g%04d.json.gz" % r["game"]))


@pytest.mark.parametrize("dataset", list(T.DATASETS))
def test_side_build_every_row_asks_by_its_side_from_ram(src, tmp_path, dataset):
    out = str(tmp_path / ("mv2_" + dataset))
    meta = T.build_one(src, dataset, out, ask="side")
    q = T.DATASETS[dataset][0]
    rows = [r for fs in T.read_dataset(out).values() for f in T.FILES for r in fs[f]]
    assert rows and meta["problems"] == [] and meta["ask"] == "side"
    for r in rows:
        x = _ram(src, r)[r["t"]]
        me, other = x["p%d_x" % r["slot"]], x["p%d_x" % (3 - r["slot"])]
        assert me != other
        assert r["side"] == ("left" if me < other else "right")
        assert r["question"]["instructions"] == WANT[q] % r["side"]
        assert not any(n in r["question"]["instructions"] for n in T.NAMES.values())
    src_ids = {json.loads(x)["id"] for f in D.FILES for x in open(os.path.join(src, q, f + ".jsonl"))}
    assert len(rows) + meta["dropped_equal_x"]["total"] == len(src_ids)
    assert {r["id"] for r in rows} <= src_ids


def test_by_name_build_is_unchanged(src, tmp_path):
    out = str(tmp_path / "mv_face")
    meta = T.build_one(src, "face", out)
    rows = [r for fs in T.read_dataset(out).values() for f in T.FILES for r in fs[f]]
    assert meta["ask"] == "name" and "side" not in rows[0]
    assert all(r["question"]["instructions"] == T.INSTRUCTIONS["facing"] % T.NAMES[r["char"]] for r in rows)


def _copy_src(src, tmp_path):
    """A private copy of the pairs build and its collection (the RAM is tampered in the copy only)."""
    meta = json.load(open(os.path.join(src, "build.json")))
    root = str(tmp_path / "col")
    shutil.copytree(meta["root"], root, symlinks=True)
    out = str(tmp_path / "pairs")
    shutil.copytree(src, out, symlinks=True)
    with open(os.path.join(out, "build.json"), "w") as f:
        json.dump(dict(meta, root=root), f)
    return out, root


def _set_ram(root, r, t, x1, x2):
    path = os.path.join(root, r["pair_name"], "ram", "g%04d.json.gz" % r["game"])
    with gzip.open(path, "rt") as f:
        rec = json.load(f)
    i1, i2 = rec["names"].index("p1_x"), rec["names"].index("p2_x")
    rec["rows"][t][i1], rec["rows"][t][i2] = x1, x2
    with gzip.open(path, "wt") as f:
        json.dump(rec, f)


def _src_row(src, q, i=0):
    return [json.loads(x) for x in open(os.path.join(src, q, "train.jsonl"))][i]


def test_equal_x_at_t_is_dropped_and_counted(src, tmp_path):
    s2, root = _copy_src(src, tmp_path)
    r = _src_row(s2, "air")
    _set_ram(root, r, r["t"], 222, 222)
    out = str(tmp_path / "mv2_air")
    meta = T.build_one(s2, "air", out, ask="side")
    ids = {x["id"] for fs in T.read_dataset(out).values() for f in T.FILES for x in fs[f]}
    assert r["id"] not in ids
    assert meta["dropped_equal_x"]["total"] >= 1 and meta["problems"] == []
    assert r["id"] in meta["dropped_equal_x"]["ids"]


def test_a_crossover_between_the_frames_takes_the_side_at_t(src, tmp_path):
    s2, root = _copy_src(src, tmp_path)
    r = _src_row(s2, "facing")
    me, other = (100, 300)                               # left at t - 4, right at t
    xs = lambda a, b: (a, b) if r["slot"] == 1 else (b, a)
    _set_ram(root, r, r["t"] - 4, *xs(me, other))
    _set_ram(root, r, r["t"], *xs(other, me))
    _set_ram(root, r, r["t"] + 1, *xs(me, other))        # the capture row disagrees: never used
    out = str(tmp_path / "mv2_face")
    T.build_one(s2, "face", out, ask="side")
    got = next(x for fs in T.read_dataset(out).values() for f in T.FILES for x in fs[f] if x["id"] == r["id"])
    assert got["side"] == "right" and got["question"]["instructions"] == WANT["facing"] % "right"


@pytest.mark.parametrize("tamper", ["side", "question", "name", "dropped_kept"])
def test_problems_catches_a_wrong_side_row(src, tmp_path, tamper):
    out = str(tmp_path / "mv2_dist")
    T.build_one(src, "dist", out, ask="side")
    d = sorted(T.read_dataset(out))[0]
    path = os.path.join(out, d, "train.jsonl")
    rows = [json.loads(x) for x in open(path)]
    r = rows[0]
    flip = "right" if r["side"] == "left" else "left"
    if tamper == "side":
        rows[0] = dict(r, side=flip, question=T.question_side("distance", flip))
    elif tamper == "question":
        rows[0] = dict(r, question=T.question_side("distance", flip))
    elif tamper == "name":
        rows[0] = dict(r, question=T.question(("distance"), r["char"]))
    else:                                                # a row whose x are equal at t kept in the data
        s2, root = _copy_src(src, tmp_path)
        _set_ram(root, r, r["t"], 200, 200)
        assert T.problems(out, s2, "dist", ask="side")
        return
    with open(path, "w") as f:
        f.writelines(json.dumps(x) + "\n" for x in rows)
    assert T.problems(out, src, "dist", ask="side")
