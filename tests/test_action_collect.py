"""The action collector (sf2.data.action_collect, sf2.data.action_collect_io): pairs sampled from either fighter's
episodes at start / middle / end, small per-(actor, code, stage, split) caps, one pair per displayed row, and the
stop rule (min games, then patience games without a new (actor, code, split); hard cap). Synthetic streams: every
image carries its capture index in its first pixels."""
import json
import os
import random
from collections import Counter

import numpy as np
import pytest
from ram_rows import row

from sf2.data import action_codes as A
from sf2.data import action_collect as C
from sf2.data import action_collect_io as IO

NONE = {"aid": 0, "mclass": 0xFF, "sclass": 0xFF}
ACTORS = {1: "chunli", 2: "ken"}


def r(p1=None, p2=None, **extra):
    return row(dict(NONE, **(p1 or {})), dict(NONE, **(p2 or {})), **extra)


def image(k):
    im = np.zeros((224, 256, 3), np.uint8)
    im[0, 0] = (k // 256 % 256, k % 256, 7)
    return im


class Saved:
    def __init__(self):
        self.images = {}

    def __call__(self, k, im):
        name = "g0000_k%05d.png" % k
        self.images[name] = im
        return name


def stream(segments):
    """[(p2 overrides, n)] with her standing at x 200, him at x 300."""
    rows = []
    for over, n in segments:
        rows += [r({"x": 200}, dict({"x": 300}, **over)) for _ in range(n)]
    return rows


def run(rows, cap=10 ** 6, split="train", ring=C.RING, counts=None, seed=0):
    saved = Saved()
    s = C.ActionSampler(0, split, ACTORS, Counter(counts or {}), cap, random.Random(seed), saved, ring)
    for k, x in enumerate(rows):
        s.feed(x, None if k == 0 else image(k))
    return s, s.finish(), saved


ATTACK = {"state": 0x0A, "aid": 9}


def test_pairs_come_from_both_fighters_with_stage_thirds_and_lag_one_images():
    rows = stream([({}, 10), (ATTACK, 12), ({}, 10)])
    s, pairs, saved = run(rows)
    mine = [p for p in pairs if p["actor"] == "ken" and p["code"] == 9]
    assert sorted(p["stg"] for p in mine) == [1, 2, 3]
    for p in mine:
        assert p["episode"] == [10, 21] and A.stage_of(p["t"], 10, 21) == p["stg"]
        assert (p["k_prev"], p["k_now"]) == (p["t"] - 3, p["t"] + 1)
        assert [int(saved.images[n][0, 0, 1]) for n in p["images"]] == [p["t"] - 3, p["t"] + 1]
    assert any(p["actor"] == "chunli" and p["code"] == A.STAND for p in pairs)
    assert s.observed[("ken", 9)] == 1


def test_one_pair_per_displayed_row():
    rows = stream([({}, 6), (ATTACK, 3), ({}, 6), (ATTACK, 3), ({}, 30)])
    for seed in range(20):
        _, pairs, _ = run(rows, seed=seed)
        ts = [p["t"] for p in pairs]
        assert len(ts) == len(set(ts))


def test_short_episode_gives_one_pair():
    rows = stream([({}, 10), (ATTACK, 5), ({}, 10)])
    _, pairs, _ = run(rows)
    assert len([p for p in pairs if p["code"] == 9]) == 1


def test_cap_per_actor_code_stage():
    rows = stream([({}, 10)] + [(ATTACK, 9), ({}, 9)] * 5)
    _, pairs, _ = run(rows, cap=2)
    c = Counter((p["actor"], p["code"], p["stg"]) for p in pairs)
    assert c[("ken", 9, 1)] == 2 and c[("ken", 9, 2)] == 2 and c[("ken", 9, 3)] == 2
    assert max(c.values()) <= 2
    _, pairs, _ = run(rows, cap=2, counts={("ken", 9, 1): 2, ("ken", 9, 2): 1})
    c = Counter((p["actor"], p["code"], p["stg"]) for p in pairs)
    assert c[("ken", 9, 1)] == 0 and c[("ken", 9, 2)] == 1


def test_unknown_episodes_counted_never_sampled():
    rows = stream([({}, 10), ({"state": 0x0A}, 5), ({"state": 0x06}, 4), ({}, 10)])
    s, pairs, _ = run(rows)
    assert not [p for p in pairs if p["t"] in range(10, 19) and p["actor"] == "ken"]
    assert s.unknown[("ken", "cut")] == 1 and s.unknown[("ken", "state")] == 1


def test_ring_respected_and_long_flagged():
    rows = stream([({}, 10), ({"state": 0x02}, 60), ({}, 10)])
    _, pairs, _ = run(rows, ring=20)
    mine = [p for p in pairs if p["code"] == A.CROUCH]
    assert mine and all(p["long"] for p in mine)
    assert all(p["t"] >= 69 - 20 + 4 for p in mine)     # its prev capture t - 3 still in the 20-frame ring
    assert [p["stg"] for p in mine] == [3]


def test_round_end_episode_sampled_and_flagged():
    rows = stream([({}, 10), (ATTACK, 9)])
    _, pairs, _ = run(rows)
    mine = [p for p in pairs if p["code"] == 9]
    assert mine and all(p["cut_end"] for p in mine) and all(p["t"] <= len(rows) - 2 for p in mine)


def test_rows_before_first_t_and_row_zero_image():
    rows = stream([(ATTACK, 3), ({}, 20)])
    _, pairs, _ = run(rows)
    assert all(p["t"] >= A.FIRST_T for p in pairs)
    s = C.ActionSampler(0, "train", ACTORS, Counter(), 5, random.Random(0), Saved())
    s.feed(rows[0], None)
    with pytest.raises(ValueError):
        s.feed(rows[1], None)


def test_bad_arguments():
    with pytest.raises(ValueError):
        C.ActionSampler(0, "dev", ACTORS, Counter(), 5, random.Random(0), Saved())
    with pytest.raises(ValueError):
        C.ActionSampler(0, "train", {1: "chunli"}, Counter(), 5, random.Random(0), Saved())
    with pytest.raises(ValueError):
        C.ActionSampler(0, "train", ACTORS, Counter(), 5, random.Random(0), Saved(), ring=4)


def test_positions_whole_episode_thirds_skip_used():
    rng = random.Random(0)
    assert C.positions(10, 18, 0, 100, set(), rng) and len(C.positions(10, 18, 0, 100, set(), rng)) == 3
    got = C.positions(10, 18, 0, 100, {10, 11, 12}, rng)
    assert [A.stage_of(u, 10, 18) for u in got] == [2, 3]
    assert C.positions(10, 12, 0, 100, {10, 11, 12}, rng) == []
    assert C.positions(10, 18, 15, 100, set(), rng) and all(u >= 15 for u in C.positions(10, 18, 15, 100, set(), rng))


# ---- the loop and the stop rule ----------------------------------------------------------------------------------

def fake_play(script):
    """play(game, sampler): feeds the rows script(game) returns."""
    def play(game, sampler):
        for k, x in enumerate(script(game)):
            sampler.feed(x, None if k == 0 else image(k))
        return {"result": "x"}
    return play


def same_game(game):
    return stream([({}, 10), (ATTACK, 9), ({}, 10)])


def test_stop_no_new_action_after_min_and_patience(tmp_path):
    base = str(tmp_path / "ken")
    rec = IO.collect_opponent(base, fake_play(same_game), "ken", ACTORS, seed=0, cap=100, min_games=5, patience=3,
                              log=lambda *a: None)
    # games 0 (val), 1 (train: new), 2 (test: new), 3, 4, 5 -> since_new 3 after game 5 with 6 played >= 5
    assert rec["reason"] == "no new action" and rec["games"] == 6 and rec["last_new_game"] == 2
    assert json.load(open(os.path.join(base, "stop.json")))["reason"] == "no new action"


def test_stop_waits_for_min_games(tmp_path):
    rec = IO.collect_opponent(str(tmp_path / "k"), fake_play(same_game), "ken", ACTORS, seed=0, cap=100,
                              min_games=12, patience=1, log=lambda *a: None)
    assert rec["games"] == 12 and rec["reason"] == "no new action"


def test_new_action_resets_patience_and_hard_cap(tmp_path):
    def script(game):
        return stream([({}, 10), ({"state": 0x0A, "aid": 1 + game % 50}, 9), ({}, 10)])
    rec = IO.collect_opponent(str(tmp_path / "k"), fake_play(script), "ken", ACTORS, seed=0, cap=7, min_games=1,
                              patience=2, log=lambda *a: None)
    assert rec["reason"] == "game cap" and rec["games"] == 7


def test_val_only_novelty_does_not_count(tmp_path):
    def script(game):        # a new code only in val games (game % 10 == 0)
        aid = 40 + game if game % 10 == 0 else 9
        return stream([({}, 10), ({"state": 0x0A, "aid": aid}, 9), ({}, 10)])
    prog = IO.Progress()
    for g in range(4):
        pairs = [{"actor": "ken", "code": 40 + g if g == 0 else 9, "split": C.split_of_game(g), "stg": 1}]
        prog = IO.advance(prog, g, pairs)
    assert prog.last_new_game == 2 and prog.since_new == 1
    assert ("ken", 40, "val") not in prog.seen


def test_resume_replays_progress_and_skips_crashed_game_numbers(tmp_path):
    base = str(tmp_path / "ken")
    IO.collect_opponent(base, fake_play(same_game), "ken", ACTORS, seed=0, cap=4, min_games=100, patience=50,
                        log=lambda *a: None)
    with open(os.path.join(base, "pairs.jsonl"), "a") as f:     # a crashed game 4: pairs, no games.jsonl line
        f.write(json.dumps({"game": 4, "split": "train", "actor": "ken", "code": 33, "stg": 1}) + "\n")
    prog = IO.load_progress(base)
    assert prog.played == 4 and prog.next_game == 5 and ("ken", 33, "train") not in prog.seen
    assert sum(prog.counts["train"].values()) > 0
    rec = IO.collect_opponent(base, fake_play(same_game), "ken", ACTORS, seed=0, cap=6, min_games=100, patience=50,
                              log=lambda *a: None)
    games = [g["game"] for g in IO.read_games(base)]
    assert rec["games"] == 6 and games == [0, 1, 2, 3, 5, 6]


def test_caps_carry_over_between_games(tmp_path):
    base = str(tmp_path / "ken")
    IO.collect_opponent(base, fake_play(same_game), "ken", ACTORS, seed=0, cap=20, min_games=100, patience=50,
                        caps={"train": 2, "val": 1, "test": 1}, log=lambda *a: None)
    pairs, _ = IO.committed_pairs(base)
    c = Counter((p["actor"], p["code"], p["stg"], p["split"]) for p in pairs)
    assert c[("ken", 9, 2, "train")] == 2 and c[("ken", 9, 2, "test")] == 1 and c[("ken", 9, 2, "val")] == 1
    assert max(v for (a, cd, st, sp), v in c.items() if sp == "train") <= 2


def test_memory_cap_stops(tmp_path):
    with pytest.raises(IO.MemoryCapExceeded):
        IO.collect_opponent(str(tmp_path / "k"), fake_play(same_game), "ken", ACTORS, seed=0, cap=5,
                            mem_cap_gb=1.0, rss_gb=lambda: 2.0, log=lambda *a: None)
    assert json.load(open(str(tmp_path / "k" / "stop.json")))["reason"] == "memory"


def test_bad_loop_arguments(tmp_path):
    for kw in ({"cap": -1}, {"patience": 0}, {"caps": {"train": 1}}, {"min_games": -1}):
        with pytest.raises(ValueError):
            IO.collect_opponent(str(tmp_path / "k"), fake_play(same_game), "ken", ACTORS, seed=0,
                                log=lambda *a: None, **kw)


def test_games_jsonl_records_codes_new_and_unknown(tmp_path):
    base = str(tmp_path / "ken")
    IO.collect_opponent(base, fake_play(same_game), "ken", ACTORS, seed=0, cap=3, min_games=100, patience=50,
                        log=lambda *a: None)
    g = IO.read_games(base)
    assert g[1]["new"] and "ken|9" in g[1]["by_code"] and "ken|9" in g[1]["observed"]
    assert g[0]["split"] == "val" and g[1]["split"] == "train" and g[2]["split"] == "test"


def test_resume_never_reuses_a_game_number_left_on_disk(tmp_path):
    base = str(tmp_path / "ken")
    IO.collect_opponent(base, fake_play(same_game), "ken", ACTORS, seed=0, cap=2, min_games=100, patience=50,
                        log=lambda *a: None)
    open(os.path.join(base, "images", "g0007_k00010.png"), "w").close()     # a crashed game 7 left an image only
    assert IO.load_progress(base).next_game == 8
