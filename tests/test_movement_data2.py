"""The movement dataset v2 builder (sf2.data.movement_data2; docs/prereg_movement_data.md): one dir per opponent
from the collector's committed games, split by whole games (game % 10 in 2,5,8 test, 0 val, the rest train), the
row format train.py reads (note "me=chunli", note v3, perception, the one movement question), natural counts (no
repetition copies), frames symlinked to the collection's images."""
import json
import os

import pytest
from test_movement_collect_io import FOUR, SMALL, SPEC, fake_play

from sf2.data import movement as M
from sf2.data import movement_collect as C
from sf2.data import movement_collect_io as IO
from sf2.data import movement_data2 as D

OPPS = ("ken", "ryu")


def collection(tmp_path, targets=SMALL, cap=100):
    root = str(tmp_path / "coll")
    for o in OPPS:
        IO.collect_opponent(os.path.join(root, o), fake_play(SPEC), opp=o, cap=cap, seed=0, targets=targets,
                            answers=FOUR, rss_gb=lambda: 1.0, log=lambda *_: None)
    with open(os.path.join(root, "run.json"), "w") as f:
        json.dump({"quota": targets}, f)
    return root


def built(tmp_path, **kw):
    root = collection(tmp_path, **kw)
    out = str(tmp_path / "mv2")
    return D.build(root, out), root, out


def files(out, opp):
    return {n: IO.read_jsonl(os.path.join(out, opp, n + ".jsonl")) for n in D.FILES.values()}


def test_build_is_clean_with_one_dir_per_opponent(tmp_path):
    res, _, out = built(tmp_path)
    assert res["problems"] == []
    assert sorted(res["counts"]) == list(OPPS)
    for o in OPPS:
        assert sorted(os.listdir(os.path.join(out, o))) == ["frames", "stats.json", "test_real.jsonl",
                                                            "train.jsonl", "val.jsonl"]
        assert os.path.islink(os.path.join(out, o, "frames"))


def test_split_by_whole_games(tmp_path):
    _, _, out = built(tmp_path)
    for o in OPPS:
        for name, rows in files(out, o).items():
            assert rows
            for r in rows:
                assert D.FILES[C.split_of_game(r["game"])] == name == D.FILES[r["split"]]


def test_rows_are_in_the_trainers_format(tmp_path):
    _, _, out = built(tmp_path)
    q = M.movement_question()
    for o in OPPS:
        for rows in files(out, o).values():
            for r in rows:
                assert r["state_text"] == "me=chunli" and r["perception"] is True and r["note_version"] == 3
                assert r["key"] == r["task"] == "movement" and r["question"] == q
                assert list(q["criteria"])[r["label"]] == r["answer"] and r["copy"] == 0
                assert r["char"] == "chunli" and r["opp"] == o
                assert r["images"] == ["frames/g%04d_k%05d.png" % (r["game"], k) for k in (r["k_prev"], r["k_now"])]
                assert all(os.path.exists(os.path.join(out, o, p)) for p in r["images"])
                assert r["frame"] == r["k_now"] == r["t"] + 1


def test_no_copies_and_counts_never_exceed_the_quota(tmp_path):
    res, _, out = built(tmp_path)
    for o in OPPS:
        ids = [r["id"] for rows in files(out, o).values() for r in rows]
        assert len(ids) == len(set(ids))
        for split, n in res["counts"][o]["counts"].items():
            for a, c in n.items():
                assert c <= SMALL[split]


def test_trainers_data_gates_pass_on_the_layout(tmp_path):
    from sf2.data.train_data import checkpoint_tags, coverage_problems, load_data, sampling_problems

    _, _, out = built(tmp_path)
    dirs = [os.path.join(out, o) for o in OPPS]
    train, val = load_data(dirs)
    assert train and val
    assert coverage_problems(train, val, dirs, share_check=False) == []
    assert sampling_problems(train, val, dirs, min_train=10, min_val=5) == []
    assert checkpoint_tags(train + val) == {"note_version": 3, "value_questions": False, "perception": True}


def test_pairs_of_an_uncommitted_game_are_dropped(tmp_path):
    root = collection(tmp_path)
    ghost = dict(IO.read_jsonl(os.path.join(root, "ryu", "pairs.jsonl"))[0], game=99)
    with open(os.path.join(root, "ryu", "pairs.jsonl"), "a") as f:
        f.write(json.dumps(ghost) + "\n")
    res = D.build(root, str(tmp_path / "mv2"))
    assert res["problems"] == [] and res["counts"]["ryu"]["dropped_uncommitted"] == 1
    assert all(r["game"] != 99 for rows in files(str(tmp_path / "mv2"), "ryu").values() for r in rows)


def test_a_pair_in_the_wrong_split_is_a_problem(tmp_path):
    root = collection(tmp_path)
    path = os.path.join(root, "ken", "pairs.jsonl")
    pairs = IO.read_jsonl(path)
    pairs[0] = dict(pairs[0], split="test" if pairs[0]["split"] != "test" else "train")
    with open(path, "w") as f:
        f.write("".join(json.dumps(p) + "\n" for p in pairs))
    assert any("split" in p for p in D.build(root, str(tmp_path / "mv2"))["problems"])


def test_a_missing_image_is_a_problem(tmp_path):
    root = collection(tmp_path)
    p = IO.read_jsonl(os.path.join(root, "ken", "pairs.jsonl"))[0]
    os.remove(os.path.join(root, "ken", "images", p["images"][1]))
    assert any("missing" in x for x in D.build(root, str(tmp_path / "mv2"))["problems"])


def test_counts_above_the_quota_are_a_problem(tmp_path):
    root = collection(tmp_path)
    with open(os.path.join(root, "run.json"), "w") as f:
        json.dump({"quota": dict(SMALL, train=2)}, f)
    assert any("quota" in x for x in D.build(root, str(tmp_path / "mv2"))["problems"])


def test_an_unknown_answer_is_a_problem(tmp_path):
    root = collection(tmp_path)
    path = os.path.join(root, "ken", "pairs.jsonl")
    pairs = IO.read_jsonl(path)
    pairs[0] = dict(pairs[0], answer="unknown")
    with open(path, "w") as f:
        f.write("".join(json.dumps(p) + "\n" for p in pairs))
    assert any("answer" in x for x in D.build(root, str(tmp_path / "mv2"))["problems"])


def test_build_refuses_an_existing_out(tmp_path):
    root = collection(tmp_path)
    out = tmp_path / "mv2"
    out.mkdir()
    (out / "x").write_text("x")
    with pytest.raises(SystemExit):
        D.build(root, str(out))


def test_build_json_records_the_collection_and_the_rules(tmp_path):
    res, root, out = built(tmp_path)
    meta = json.load(open(os.path.join(out, "build.json")))
    assert meta["root"] == os.path.abspath(root) and meta["lag"] == 1 and meta["quota"] == SMALL
    assert meta["answers"] == list(M.ANSWERS) and meta["opps"] == list(OPPS)
