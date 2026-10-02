"""The four fine-tune datasets (sf2.data.pairs_train; docs/prereg_movement_finetunes.md): one question each, cut from
one gated pairs build; val = whole training matches (1 in 6), test untouched; air / dist one dir per answer; laya-vision
loads them; the independent ``problems`` check catches a row in the wrong split or dir."""
import json
import os

import pytest

from sf2.data import pairs_data as D
from sf2.data import pairs_train as T
from test_pairs_collect import BANDS
from test_pairs_down import _knock_collection


@pytest.fixture(scope="module")
def src(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("mv")
    root, out = str(tmp / "col"), str(tmp / "pairs")
    _knock_collection(root, games=5)
    D.build(root, out, {"train": 40, "test": 20}, bands=BANDS)
    return out


def _src_rows(src, q):
    return [json.loads(x) for f in D.FILES for x in open(os.path.join(src, q, f + ".jsonl"))]


def test_val_is_one_in_six_training_matches_and_never_a_test_match():
    matches = [("%s_vs_%s" % (a, b), g) for a in T.NAMES for b in T.NAMES if a != b for g in range(10)]
    val = [m for m in matches if T.is_val_match(*m)]
    train = [m for m in matches if D.split_of_game(*m) == "train"]
    assert set(val) <= set(train)
    assert all(T.split3(*m) == "test" for m in matches if D.split_of_game(*m) == "test")
    assert 0.10 < len(val) / len(train) < 0.24          # 1 in 6 by hash: about 17%


@pytest.mark.parametrize("dataset", list(T.DATASETS))
def test_each_dataset_is_one_question_every_source_row_once_in_its_matchs_split(src, tmp_path, dataset):
    out = str(tmp_path / ("mv_" + dataset))
    meta = T.build_one(src, dataset, out)
    q = T.DATASETS[dataset][0]
    data = T.read_dataset(out)
    rows = {f: [r for fs in data.values() for r in fs[f]] for f in T.FILES}
    assert all(rows[f] for f in T.FILES), {f: len(v) for f, v in rows.items()}
    got = sorted(r["id"] for f in T.FILES for r in rows[f])
    assert got == sorted(r["id"] for r in _src_rows(src, q))
    src_test = {json.loads(x)["id"] for x in open(os.path.join(src, q, "test.jsonl"))}
    assert {r["id"] for r in rows["test"]} == src_test   # test untouched
    tr = {(r["pair_name"], r["game"]) for r in rows["train"]}
    va = {(r["pair_name"], r["game"]) for r in rows["val"]}
    te = {(r["pair_name"], r["game"]) for r in rows["test"]}
    assert not (tr & va) and not (tr & te) and not (va & te)
    assert all(D.split_of_game(*m) == "train" for m in va)
    for f in T.FILES:
        for r in rows[f]:
            assert r["question_key"] == q and r["question"]["instructions"] == T.INSTRUCTIONS[q] % T.NAMES[r["char"]]
            assert list(r["question"]["criteria"])[r["label"]] == r["answer"]
            assert r["answer"] == r[q] if q != "movement" else True
    if T.DATASETS[dataset][1]:
        assert set(data) <= set(D.QUESTION_ANSWERS[q])
        assert all(r["answer"] == d for d, fs in data.items() for f in T.FILES for r in fs[f])
    else:
        assert list(data) == [dataset]
    assert meta["problems"] == []
    with pytest.raises(FileExistsError):
        T.build_one(src, dataset, out)


def test_laya_loads_the_dirs_with_their_images(src, tmp_path):
    import laya.vlm_train as vt
    out = str(tmp_path / "mv_air")
    T.build_one(src, "air", out)
    for d in T.read_dataset(out):
        for f in T.FILES:
            ex = vt.load_jsonl_examples(out, d, f)
            assert ex and all(e["dataset"] == d and e["label"] in (0, 1) for e in ex)
            assert all(os.path.exists(p) for e in ex for p in e["state"]["images"])
            assert all("context" not in e["state"] for e in ex)          # no note: frames + the question


@pytest.mark.parametrize("tamper", ["split", "dir", "label", "name"])
def test_problems_catches_a_tampered_row(src, tmp_path, tamper):
    out = str(tmp_path / "mv_dist")
    T.build_one(src, "dist", out)
    d = sorted(T.read_dataset(out))[0]
    other = [x for x in D.QUESTION_ANSWERS["distance"] if x != d][0]
    path = os.path.join(out, d, "train.jsonl")
    rows = [json.loads(x) for x in open(path)]
    r = rows[0]
    if tamper == "split":                                 # a whole training match moved into val (no duplicate id)
        m = (r["pair_name"], r["game"])
        moved = [x for x in rows if (x["pair_name"], x["game"]) == m]
        with open(os.path.join(out, d, "val.jsonl"), "a") as f:
            f.writelines(json.dumps(dict(x, split="val")) + "\n" for x in moved)
        rows = [x for x in rows if (x["pair_name"], x["game"]) != m]
    elif tamper == "dir":
        rows[0] = dict(r, answer=other, label=list(T.CRITERIA["distance"]).index(other))
    elif tamper == "label":
        rows[0] = dict(r, label=1 - r["label"])
    else:
        rows[0] = dict(r, question=dict(r["question"], instructions="Is he close to or far from the other fighter?"))
    with open(path, "w") as f:
        f.writelines(json.dumps(x) + "\n" for x in rows)
    assert T.problems(out, src, "dist")
