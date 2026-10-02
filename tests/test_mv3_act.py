"""Round 3 of docs/prereg_movement_finetunes.md, the "act" question: "What is the fighter on the <side> doing?" ->
moving / attack / special. moving = stand, walk, crouch, jump, block, hit, down; attack = normal, jump attack, throw
(the grid's "attack"); special = the grid's "special" (pressed-move rule, RAM-confirmed). Same frames, sides and
splits as round 2; one data dir per answer (equal draws, no copies); the independent ``problems`` check."""
import json
import os

import pytest

from sf2.data import mv3_act as A
from sf2.data import pairs_collect as PC
from sf2.data import pairs_collect_io as IO
from sf2.data import pairs_data as D
from sf2.data import pairs_labels as L
from sf2.data import pairs_train as T
from test_pairs_collect import BANDS, image
from test_pairs_labels import prow

WANT = {"stand": "moving", "walk toward": "moving", "walk away": "moving", "crouch": "moving", "jump": "moving",
        "block": "moving", "hit": "moving", "down": "moving", "attack": "attack", "special": "special"}


def test_every_grid_movement_maps_to_its_act_answer():
    assert set(WANT) == set(L.MOVEMENTS10)
    for mv, act in WANT.items():
        assert A.act_of(mv) == act
    with pytest.raises(ValueError):
        A.act_of("unknown")


@pytest.mark.parametrize("side", ["left", "right"])
def test_the_question_names_the_side_exactly_and_no_character(side):
    q = A.question_act(side)
    assert q["instructions"] == "What is the fighter on the %s doing?" % side
    assert tuple(q["criteria"]) == ("moving", "attack", "special") == A.ACT_ANSWERS and q["type"] == "choice"
    assert not any(n in q["instructions"] for n in T.NAMES.values())
    with pytest.raises(ValueError):
        A.question_act("middle")


def test_collapse_maps_truths_and_answers_of_the_ten_movements():
    assert A.collapse(["stand", "attack", "special", "hit"]) == ["moving", "attack", "special", "moving"]


# ---- a collection with attacks and specials (the RAM rule: no word pressed) ----------------------------------------

def _act_row(k):
    """Player 1 stands at x 200; player 2 (x 300) cycles every 40 frames: 10 standing, 10 attack (0x0A), 10 special
    (0x0C), 10 crouching."""
    ph = (k % 40) // 10
    p2 = [{}, {"state": 0x0A}, {"state": 0x0C}, {"state": 0x02}][ph]
    return prow({"x": 200}, dict({"x": 300}, **p2))


def _act_collection(root, games=5):
    def play(game, sampler):
        for k in range(480):
            sampler.feed(dict(_act_row(k), timer=k % 256), None if k == 0 else image(k))
        return {"result": "win", "frames": 480, "moves": []}
    for a, b in (("ryu", "ken"), ("ken", "ryu")):
        IO.collect_pair(os.path.join(root, IO.pair_name(a, b)), play, a, b, games, 0, BANDS, per_game=3,
                        log=lambda *x: None, controllers=PC.VS_SLOTS)


@pytest.fixture(scope="module")
def src(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("mv3act")
    root, out = str(tmp / "col"), str(tmp / "pairs")
    _act_collection(root, games=6)
    D.build(root, out, {"train": 40, "test": 20}, bands=BANDS)
    return out


def _rows(out):
    return [r for d, fs in T.read_dataset(out).items() for f in T.FILES for r in fs[f]]


def test_the_build_one_dir_per_answer_every_movement_row_once_by_its_act(src, tmp_path):
    out = str(tmp_path / "act")
    meta = A.build_act(src, out)
    assert meta["problems"] == [] and meta["answer_dirs"] is True and meta["question"] == "act"
    data = T.read_dataset(out)
    assert set(data) == {"moving", "attack", "special"}
    mv = {json.loads(x)["id"]: json.loads(x) for f in D.FILES for x in open(os.path.join(src, "movement", f + ".jsonl"))}
    rows = _rows(out)
    for d, fs in data.items():
        for f, rs in fs.items():
            for r in rs:
                s = mv[r["source_id"]]
                assert r["answer"] == d == WANT[s["answer"]] and r["movement10"] == s["answer"]
                assert r["label"] == list(A.ACT_ANSWERS).index(d) and r["split"] == f == T.split3(s["pair_name"], s["game"])
                assert r["question"]["instructions"] == "What is the fighter on the %s doing?" % r["side"]
                assert r["images"] == s["images"] and r["question_key"] == "act"
                assert os.path.exists(os.path.join(out, d, r["images"][1]))
    assert len(rows) + meta["dropped_equal_x"]["total"] == len(mv)
    assert len({r["id"] for r in rows}) == len(rows)
    assert {r["answer"] for r in rows if r["split"] == "train"} == {"moving", "attack", "special"}


def test_the_sides_and_splits_are_round_2s(src, tmp_path):
    out, out2 = str(tmp_path / "act"), str(tmp_path / "mv2_move")
    A.build_act(src, out)
    T.build_one(src, "move", out2, ask="side")
    r2 = {r["id"]: r for r in _rows(out2)}
    rows = _rows(out)
    assert {r["source_id"] for r in rows} == set(r2)
    for r in rows:
        assert (r["side"], r["split"], r["images"]) == (r2[r["source_id"]]["side"], r2[r["source_id"]]["split"],
                                                        r2[r["source_id"]]["images"])


def test_refuses_an_existing_out(src, tmp_path):
    out = str(tmp_path / "act")
    os.makedirs(out)
    with pytest.raises(FileExistsError):
        A.build_act(src, out)


@pytest.mark.parametrize("tamper", ["wrong_dir", "answer_only", "label", "side", "drop", "split", "name"])
def test_problems_catches_a_tampered_row(src, tmp_path, tamper):
    out = str(tmp_path / "act")
    A.build_act(src, out)
    path = os.path.join(out, "attack", "train.jsonl")
    rows = [json.loads(x) for x in open(path)]
    r = rows[0]
    if tamper == "wrong_dir":
        rows[0] = dict(r, answer="moving", label=0)
    elif tamper == "answer_only":
        rows[0] = dict(r, answer="moving")
    elif tamper == "label":
        rows[0] = dict(r, label=2)
    elif tamper == "side":
        flip = "right" if r["side"] == "left" else "left"
        rows[0] = dict(r, side=flip, question=A.question_act(flip))
    elif tamper == "drop":
        rows = rows[1:]
    elif tamper == "split":
        rows[0] = dict(r, split="test")
    else:
        rows[0] = dict(r, question=dict(r["question"], instructions="What is %s doing?" % T.NAMES[r["char"]]))
    with open(path, "w") as f:
        f.writelines(json.dumps(x) + "\n" for x in rows)
    assert A.problems_act(out, src)
