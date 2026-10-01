"""The collector's files and its per-opponent loop (sf2.data.movement_collect_io): images written once as the model
sees them (hud_frame), every RAM row of every game (gzip), pairs and games committed per game, quotas and the game
cap stop the loop, a run resumes from what was committed, and the memory check stops it."""
import gzip
import json
import os

import numpy as np
import pytest
from PIL import Image
from test_movement_collect import image, stream

from sf2.data import movement as M
from sf2.data import movement_collect as C
from sf2.data import movement_collect_io as IO


def fake_play(spec):
    """A play function feeding the same synthetic round every game."""
    def play(game, sampler):
        for k, r in enumerate(stream(spec)):
            sampler.feed(r, None if k == 0 else image(k))
        return {"result": "win", "frames": len(stream(spec))}
    return play


SPEC = [("standing", 20), ("crouching", 12), ("attacking", 9), ("jumping", 4), ("standing", 10)]
SMALL = {"train": 6, "val": 3, "test": 3}
FOUR = ("standing", "crouching", "attacking", "jumping")


def collect(tmp_path, cap=100, targets=SMALL, spec=SPEC, **kw):
    base = str(tmp_path / "ryu")
    res = IO.collect_opponent(base, fake_play(spec), opp="ryu", cap=cap, seed=0, targets=targets,
                              rss_gb=lambda: 1.0, log=lambda *_: None, **kw)
    return base, res


def test_image_saver_writes_the_hud_frame_once(tmp_path):
    save = IO.ImageSaver(str(tmp_path), game=3)
    im = image(17)
    name = save(17, im)
    assert name == "g0003_k00017.png"
    got = np.asarray(Image.open(tmp_path / name))
    assert got.shape == (256, 256, 3) and (got[:224] == im).all() and (got[224:] == 0).all()
    mtime = os.path.getmtime(tmp_path / name)
    assert save(17, im) == name and os.path.getmtime(tmp_path / name) == mtime


def test_ram_round_trip(tmp_path):
    rows = stream([("standing", 5), ("crouching", 3)])
    path = IO.write_ram(str(tmp_path), 7, rows)
    assert path.endswith("g0007.json.gz")
    assert IO.read_ram(path) == rows
    with gzip.open(path, "rt") as f:
        rec = json.load(f)
    assert rec["game"] == 7 and rec["names"] == list(rows[0])


def test_the_loop_stops_when_every_quota_is_full(tmp_path):
    base, res = collect(tmp_path, answers=FOUR)
    assert res["reason"] == "quotas full"
    counts = IO.load_progress(base, targets=SMALL).counts
    for a in ("standing", "crouching", "attacking", "jumping"):
        for split, q in SMALL.items():
            assert counts[(a, split)] == q


def test_the_loop_stops_at_the_game_cap_and_says_so(tmp_path):
    base, res = collect(tmp_path, cap=3, targets=dict(SMALL, train=10 ** 6))
    assert res["reason"] == "game cap" and res["games"] == 3
    stop = json.load(open(os.path.join(base, "stop.json")))
    assert stop["reason"] == "game cap" and stop["cap"] == 3


def test_answers_that_never_occur_cannot_fill_and_hit_the_cap(tmp_path):
    base, res = collect(tmp_path, cap=12, targets=SMALL, answers=M.ANSWERS)
    assert res["reason"] == "game cap"
    assert "blocking" in res["short"]


def test_games_of_a_full_split_are_skipped(tmp_path):
    base, res = collect(tmp_path, targets={"train": 10 ** 6, "val": 0, "test": 0}, cap=4)
    games = [g["game"] for g in IO.read_jsonl(os.path.join(base, "games.jsonl"))]
    assert games == [1, 3, 4, 6]
    assert all(C.split_of_game(g) == "train" for g in games)


def test_every_game_keeps_all_its_ram_rows_and_its_pairs_images(tmp_path):
    base, _ = collect(tmp_path, answers=FOUR)
    games = IO.read_jsonl(os.path.join(base, "games.jsonl"))
    pairs = IO.read_jsonl(os.path.join(base, "pairs.jsonl"))
    for g in games:
        assert IO.read_ram(os.path.join(base, "ram", "g%04d.json.gz" % g["game"])) == stream(SPEC)
        assert g["pairs"] == sum(p["game"] == g["game"] for p in pairs)
        assert g["seconds"] >= 0 and g["split"] == C.split_of_game(g["game"])
    for p in pairs:
        assert all(os.path.exists(os.path.join(base, "images", n)) for n in p["images"])
        assert p["split"] == C.split_of_game(p["game"])


def test_resume_continues_from_the_committed_games(tmp_path):
    base, _ = collect(tmp_path, cap=2, targets=dict(SMALL, train=10 ** 6))
    _, res = collect(tmp_path, cap=4, targets=dict(SMALL, train=10 ** 6))
    games = [g["game"] for g in IO.read_jsonl(os.path.join(base, "games.jsonl"))]
    assert games == [0, 1, 2, 3] and res["games"] == 4


def test_resume_ignores_pairs_of_a_game_that_was_never_committed(tmp_path):
    base, _ = collect(tmp_path, cap=1, targets=dict(SMALL, val=10 ** 6))
    with open(os.path.join(base, "pairs.jsonl"), "a") as f:          # a crash after the pairs, before the game line
        f.write(json.dumps({"game": 1, "split": "train", "answer": "standing"}) + "\n")
        f.write('{"game": 1, "spl')                                     # and a torn last line
    prog = IO.load_progress(base, targets=SMALL)
    assert prog.played == 1 and prog.next_game == 2
    assert prog.counts[("standing", "train")] == 0


def test_the_memory_check_stops_the_run(tmp_path):
    base = str(tmp_path / "ryu")
    with pytest.raises(IO.MemoryCapExceeded):
        IO.collect_opponent(base, fake_play(SPEC), opp="ryu", cap=5, seed=0, targets=SMALL, rss_gb=lambda: 99.0,
                            mem_cap_gb=8.0, log=lambda *_: None)
    assert json.load(open(os.path.join(base, "stop.json")))["reason"] == "memory"


def test_a_negative_quota_or_cap_is_refused(tmp_path):
    with pytest.raises(ValueError):
        collect(tmp_path, targets={"train": -1, "val": 1, "test": 1})
    with pytest.raises(ValueError):
        collect(tmp_path, cap=-1)


def test_a_torn_line_is_closed_and_skipped_and_the_run_goes_on(tmp_path):
    base, _ = collect(tmp_path, cap=1, targets=dict(SMALL, val=10 ** 6))
    with open(os.path.join(base, "games.jsonl"), "a") as f:
        f.write('{"game": 1, "spl')
    IO.write_ram(os.path.join(base, "ram"), 1, stream(SPEC))
    _, res = collect(tmp_path, cap=3, targets=dict(SMALL, val=10 ** 6, train=10 ** 6))
    torn = []
    games = IO.read_jsonl(os.path.join(base, "games.jsonl"), torn)
    assert [g["game"] for g in games] == [0, 2, 3] and torn == [1] and res["games"] == 3


def test_a_crashed_games_image_files_block_its_number(tmp_path):
    base, _ = collect(tmp_path, cap=1, targets=dict(SMALL, val=10 ** 6))
    IO.ImageSaver(os.path.join(base, "images"), 5)(9, image(9))
    assert IO.load_progress(base, targets=SMALL).next_game == 6
