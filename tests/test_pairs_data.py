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


def collection(root, lag=1, games=3, pairs=(("ryu", "ken"), ("ken", "ryu"))):
    for a, b in pairs:
        IO.collect_pair(os.path.join(root, IO.pair_name(a, b)), play(lag), a, b, games, 0, BANDS, per_game=3,
                        log=lambda *x: None)


@pytest.fixture
def built(tmp_path):
    root, out = str(tmp_path / "col"), str(tmp_path / "data")
    collection(root)
    meta = D.build(root, out, {"train": 4, "test": 2})
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
    assert {r["game"] % 3 for r in rows} == {0, 1} and {r["game"] % 3 for r in test} == {2}
    cells = {}
    for r in rows:
        cells[D.cell(r)] = cells.get(D.cell(r), 0) + 1
    assert max(cells.values()) == 4
    full = [c for c, n in cells.items() if n == 4]
    # a full cell takes from both pairs' games when both have it (round-robin over (pair, game))
    spread = {(r["pair_name"], r["game"]) for r in rows if D.cell(r) == full[0]}
    assert len(spread) >= 2


def test_movement_answer_combines_direction():
    assert D.movement_answer("walk", "toward") == "walk toward"
    assert D.movement_answer("jump", "none") == "jump up"
    assert D.movement_answer("jump", "away") == "jump away"
    assert D.movement_answer("attack", "none") == "attack"


def test_build_reports_shortfalls_over_every_bucket(built):
    _, _, meta = built
    assert "test|ken|cpu|down|none|left" in meta["short"]          # never seen at all: short by the whole cap
    assert meta["short"]["test|ken|cpu|down|none|left"] == 2


def test_fill_report(built):
    root, _, _ = built
    pairs, _, names = D.collection_pairs(root)
    rep = D.fill_report(pairs, ["ken", "ryu"], cap=3)
    assert rep["buckets"] == 2 * 2 * 12 * 2
    assert 0 < rep["overall_pct"] < 100
    assert "directed down none left" in rep["zero"]["ryu"]          # player 1 never goes down here
    one = [p for p in pairs if D.cell(p) == ("ryu", "directed", "stand", "none", "right")]
    assert rep["counts"]["ryu|directed|stand|none|right"] == len(one)


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
    D.build(root, out, {"train": 4, "test": 2})
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
