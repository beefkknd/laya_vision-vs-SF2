"""The movement-pairs datasets and gates (sf2.data.pairs_data, pairs_gate) on a synthetic collection made by the real
collector loop: one file set per question, the split by game, the caps per cell spread over games, the symlinked
frames, the fill report; the gates pass on it and go red on a wrong label, a wrong image, an over-full cell, a lag-0
alignment and a disk overrun."""
import json
import os
import random

import numpy as np
import pytest

from sf2.data import pairs_collect_io as IO
from sf2.data import pairs_data as D
from sf2.data import pairs_gate as G
from sf2.data import pairs_labels as L
from test_pairs_collect import BANDS, image, synth

from sf2.data.movement_gate import CLOCK, HUD_DIGITS


def clock_image(k, timer):
    im = image(k)
    if timer % 2:                                   # draw "digits" when the timer is odd: the clock changes with it
        im[CLOCK][..., :] = HUD_DIGITS[0]
    return im


def play(lag=1, n=500):
    def run(game, sampler):
        rng = random.Random(game)
        off = rng.randrange(13)
        for k in range(n):
            r = synth(k + off)
            r["timer"] = (k // 7) % 200
            shown = max(0, k - lag)
            sampler.feed(r, None if k == 0 else clock_image(k, (shown // 7) % 200))
        return {"result": "loss", "frames": n, "moves": []}
    return run


def collection(root, lag=1, games=4, pairs=(("ryu", "ken"), ("ken", "ryu"))):
    for a, b in pairs:
        IO.collect_pair(os.path.join(root, IO.pair_name(a, b)), play(lag), a, b, games, 0, BANDS, per_game=3,
                        log=lambda *x: None)


@pytest.fixture
def built(tmp_path):
    root, out = str(tmp_path / "col"), str(tmp_path / "data")
    collection(root)
    meta = D.build(root, out, {"train": 4, "test": 2}, bands=BANDS)
    return root, out, meta


def test_one_dataset_per_question_with_rows_about_one_fighter(built):
    root, out, meta = built
    for q, answers in D.QUESTION_ANSWERS.items():
        rows = IO.read_jsonl(os.path.join(out, q, "train.jsonl")) + IO.read_jsonl(os.path.join(out, q, "test.jsonl"))
        assert rows, q
        for r in rows:
            assert answers[r["label"]] == r["answer"] == D.answer(q, r)
            assert r["controller"] == {1: "directed", 2: "cpu"}[r["slot"]]
            assert r["pair"] == r["pair_name"].split("_vs_")
            assert os.path.exists(os.path.join(out, q, r["images"][0]))
    assert os.path.islink(os.path.join(out, "frames", "ryu_vs_ken"))


def test_split_by_game_and_caps_per_cell(built):
    root, out, meta = built
    rows = IO.read_jsonl(os.path.join(out, "movement", "train.jsonl"))
    test = IO.read_jsonl(os.path.join(out, "movement", "test.jsonl"))
    # by whole game (match): each (pair, game) on one side only, as crc32("<pair>:<game>") % 3 says
    assert rows and test
    assert {(r["pair_name"], r["game"]) for r in rows}.isdisjoint({(r["pair_name"], r["game"]) for r in test})
    for r in rows + test:
        assert r["split"] == D.split_of_game(r["pair_name"], r["game"])
    cells = {}
    for r in rows:
        cells[D.cell(r)] = cells.get(D.cell(r), 0) + 1
    assert max(cells.values()) == 4
    full = [c for c, n in cells.items() if n == 4]
    # a full cell takes from both pairs' games when both have it (round-robin over (pair, game))
    spread = {(r["pair_name"], r["game"]) for r in rows if D.cell(r) == full[0]}
    assert len(spread) >= 2


def test_movement_answer_is_the_grid_movement():
    assert D.movement_answer("walk", "toward") == "walk toward"
    assert D.movement_answer("walk", "away") == "walk away"
    assert D.movement_answer("jump", "none") == "jump"            # the jump's direction is not a cell
    assert D.movement_answer("jump", "away") == "jump"
    assert D.movement_answer("attack", "none") == "attack"
    assert D.MOVEMENT_ANSWERS == ("stand", "walk toward", "walk away", "crouch", "jump", "attack", "special",
                                  "block", "hit", "down")


def test_split_is_by_match_and_round_one_games_reach_both_splits():
    names = ["%s_vs_%s" % (a, b) for a in "abcdefgh" for b in "abcdefgh" if a != b]
    splits = [D.split_of_game(n, 0) for n in names]
    assert splits.count("test") > 10 and splits.count("train") > 25          # game 0 of 56 matches: both splits
    assert all(D.split_of_game(n, 0) == D.split_of_game(n, 0) for n in names)   # stable


def test_build_reports_shortfalls_over_every_bucket(built):
    _, _, meta = built
    assert "test|ken|down|left" in meta["short"]                   # never seen at all: short by the whole cap
    assert meta["short"]["test|ken|down|left"] == 2
    assert len(meta["short"]) <= 2 * 2 * 20                        # splits x chars x the 20 grid cells


def test_fill_report(built):
    root, _, _ = built
    pairs, _, names = D.collection_pairs(root)
    rep = D.fill_report(pairs, ["ken", "ryu"], cap=3)
    assert rep["cells"] == 2 * 10 * 2                               # chars x movements x facings
    assert 0 < rep["overall_pct"] < 100
    assert "down left" in rep["zero"]["ryu"]                        # nobody goes down here
    one = [p for p in pairs if D.cell(p) == ("ryu", "stand", "right")]
    assert rep["counts"]["ryu|stand|right"] == len(one)
    assert rep["pct"]["ryu"]["stand|right"] == round(100.0 * min(len(one), 3) / 3, 1)
    # both slots pool into a character's cells: ryu's cells count ryu as player 1 AND as player 2
    assert {p["slot"] for p in pairs if p["char"] == "ryu"} == {1, 2}


def test_the_gates_pass_on_a_clean_build(built):
    _, out, _ = built
    rep = G.run_gates(out, BANDS, sample=50, min_disc=5)
    assert rep["pass"], json.dumps(rep["gates"])[:2000]


def _rewrite(path, change):
    rows = IO.read_jsonl(path)
    rows[0] = change(dict(rows[0]))
    with open(path, "w") as f:
        f.writelines(json.dumps(r) + "\n" for r in rows)


def test_a_wrong_label_or_image_fails_the_label_gate(built):
    _, out, _ = built
    _rewrite(os.path.join(out, "facing", "train.jsonl"),
             lambda r: dict(r, facing="left" if r["facing"] == "right" else "right"))
    assert not G.run_gates(out, BANDS, sample=50, min_disc=5)["gates"]["labels"]["pass"]


def test_a_shifted_image_fails_the_label_gate(built):
    _, out, _ = built
    _rewrite(os.path.join(out, "air", "train.jsonl"),
             lambda r: dict(r, images=[r["images"][0], r["images"][0]]))
    assert not G.run_gates(out, BANDS, sample=50, min_disc=5)["gates"]["labels"]["pass"]


def test_an_over_full_cell_fails_the_cap_gate(built):
    _, out, _ = built
    meta = json.load(open(os.path.join(out, "build.json")))
    meta["caps"] = {"train": 1, "test": 1}
    json.dump(meta, open(os.path.join(out, "build.json"), "w"))
    assert not G.run_gates(out, BANDS, sample=50, min_disc=5)["gates"]["caps"]["pass"]


def test_images_at_lag_zero_fail_the_alignment_gate(tmp_path):
    root, out = str(tmp_path / "col"), str(tmp_path / "data")
    collection(root, lag=0)
    D.build(root, out, {"train": 4, "test": 2}, bands=BANDS)
    assert not G.run_gates(out, BANDS, sample=50, min_disc=5)["gates"]["alignment"]["pass"]


def test_the_disk_gate(built):
    root, out, meta = built
    assert not G.disk_check(root, meta["pairs"], max_gb=1e-9)["pass"]
    assert G.disk_check(root, meta["pairs"])["pass"]


def test_contact_sheet_per_character(built, tmp_path):
    _, out, _ = built
    paths = G.contact_sheets(out, str(tmp_path / "sheets"), per_answer=2)
    assert sorted(os.path.basename(p) for p in paths) == ["contact_ken.png", "contact_ryu.png"]


def test_build_refuses_bad_pairs(tmp_path):
    root, out = str(tmp_path / "col"), str(tmp_path / "data")
    collection(root, pairs=(("ryu", "ken"),), games=1)
    base = os.path.join(root, "ryu_vs_ken")
    _rewrite(os.path.join(base, "pairs.jsonl"), lambda r: dict(r, movement="flying"))
    with pytest.raises(ValueError):
        D.build(root, out)


def test_build_only_the_controlled_player(tmp_path):
    root, out = str(tmp_path / "col"), str(tmp_path / "data")
    collection(root)
    meta = D.build(root, out, {"train": 4, "test": 2}, controllers=("directed",), bands=BANDS)
    rows = [r for q in D.QUESTION_ANSWERS for f in ("train", "test")
            for r in IO.read_jsonl(os.path.join(out, q, f + ".jsonl"))]
    assert rows and {r["controller"] for r in rows} == {"directed"} and {r["slot"] for r in rows} == {1}
    assert meta["controllers"] == ["directed"]



def test_build_refuses_unknown_controller(tmp_path):
    with pytest.raises(ValueError):
        D.build(str(tmp_path), str(tmp_path / "o"), controllers=("human",))
    with pytest.raises(ValueError):
        D.build(str(tmp_path), str(tmp_path / "o"), controllers=())


def _cap_rows(facings):
    return [{"split": "train", "char": "ryu", "movement": "jump", "direction": d, "facing": f, "pair_name": "ryu_vs_ken",
             "game": 0, "slot": 1} for f, d in zip(facings, ["none", "toward", "away", "none", "toward", "away"])]


def test_the_per_game_cap_counts_the_grid_cell_with_its_facing():
    def gate(rows):
        return G.cap_check({"meta": {"caps": {"train": 40, "test": 20}},
                            "files": {"movement": {"train": rows, "test": []}}})["pass"]
    # 3 right + 2 left jumps of one player in one game: two cells, each within PER_GAME (3)
    assert gate(_cap_rows(["right"] * 3 + ["left"] * 2))
    # 4 jumps facing right (any direction: the jump's direction is not a cell): over
    assert not gate(_cap_rows(["right"] * 4))
