"""Owner (docs/prereg_movement_finetunes.md): "down" is only once the knocked-down fighter is back ON THE GROUND (lying,
getting up); the airborne part of a knockdown (flying / falling after the hit, or being thrown) is "hit". The labels,
the gate's independent re-derivation and the episodes follow it; the gate's second-fact RAM checks: every "down" pair
on the ground (y == GROUND_Y at rows t - 4 .. t) 100%, and "hit" pairs: health lost within the hit episode (reported).
"""
import gzip
import json
import os

import pytest

from sf2.data import movement_collect_io as MIO
from sf2.data import pairs_collect as PC
from sf2.data import pairs_collect_io as IO
from sf2.data import pairs_data as D
from sf2.data import pairs_gate as G
from sf2.data import pairs_labels as L
from sf2.emu.vs import GROUND_Y
from test_pairs_collect import BANDS, image
from test_pairs_labels import prow

AIR = GROUND_Y - 40
KNOCK = {"state": 0x0E, "sub": 0x04, "react": 0x02}
THROWN = {"state": 0x14}


def _side(p, over):
    return (over, None) if p == 1 else (None, over)


@pytest.mark.parametrize("p", [1, 2])
@pytest.mark.parametrize("over, want", [
    (dict(KNOCK, y=GROUND_Y), "down"), (dict(KNOCK, y=AIR), "hit"),
    (dict(THROWN, y=GROUND_Y), "down"), (dict(THROWN, y=AIR), "hit"),
    (dict(KNOCK, react=0x06, y=AIR), "block"),                # a blocked knock-down sub stays block
    ({"state": 0x0E, "sub": 0x02, "react": 0x02, "y": AIR}, "hit"),
])
def test_down_only_on_the_ground_for_both_fighters_labels_and_gate(p, over, want):
    rows = [prow(*_side(p, over)) for _ in range(5)]
    assert L.movement(rows, 4, p) == want
    assert G.independent_labels(rows, 4, p, BANDS)["movement"] == want
    assert L.labels(rows, 4, p, BANDS) == G.independent_labels(rows, 4, p, BANDS)


def knockdown(n_air=6, n_ground=8, hp_before=176, hp_after=150):
    """Player 2: standing, hit (health lost on the hit row), knocked into the air, lands, lies on the ground."""
    stand = [prow(None, {"hp": hp_before}) for _ in range(6)]
    hit = [prow(None, {"state": 0x0E, "sub": 0x02, "react": 0x02, "hp": hp_after}) for _ in range(3)]
    air = [prow(None, dict(KNOCK, y=AIR + i, hp=hp_after)) for i in range(n_air)]
    ground = [prow(None, dict(KNOCK, y=GROUND_Y, hp=hp_after)) for _ in range(n_ground)]
    return stand + hit + air + ground


def test_a_knockdown_is_hit_in_the_air_and_down_from_the_landing_on():
    rows = knockdown()
    first_ground = 6 + 3 + 6
    grid = [L.grid_movement(rows, t, 2) for t in range(6, len(rows))]
    assert grid == ["hit"] * 9 + ["down"] * 8
    none = [None] * len(rows)
    assert L.same_episode(rows, first_ground + 4, 2, none)            # four rows after landing: one down episode
    assert not L.same_episode(rows, first_ground + 2, 2, none)        # t - 4 still in the air
    assert L.same_episode(rows, first_ground - 1, 2, none)            # the airborne part: one hit episode with the hit
    assert L.episode_key(rows, first_ground + 4, 2) == ("down", "left")


def test_second_fact_hit_health_lost_within_the_episode():
    rows = knockdown()
    t = 6 + 3 + 6 - 1                                    # last airborne row: hit since row 6, health dropped there
    assert G.hit_health_lost(rows, t, 2, BANDS)
    flat = knockdown(hp_after=176)                       # no health lost anywhere
    assert not G.hit_health_lost(flat, t, 2, BANDS)
    assert G.hit_episode(rows, t, 2, BANDS) == (6, 14)


def test_second_fact_down_on_the_ground():
    rows = knockdown()
    assert G.down_on_ground(rows, 6 + 3 + 6 + 4, 2)
    lifted = [dict(r) for r in rows]
    lifted[6 + 3 + 6 + 2]["p2_y"] = AIR                  # one row inside the pair off the ground
    assert not G.down_on_ground(lifted, 6 + 3 + 6 + 4, 2)


# ---- a collection with knockdowns: builder + gate end to end ----------------------------------------------------

def _knock_row(k):
    """Player 1 stands; player 2 cycles every 40 frames: 10 standing, 4 hit (health drops), 10 airborne knockdown,
    16 lying on the ground."""
    ph = k % 40
    hp = 176 - 10 * (k // 40) - (10 if ph >= 10 else 0)
    if ph < 10:
        p2 = {"hp": hp}
    elif ph < 14:
        p2 = {"state": 0x0E, "sub": 0x02, "react": 0x02, "hp": hp}
    elif ph < 24:
        p2 = dict(KNOCK, y=AIR + (ph - 14), hp=hp)
    else:
        p2 = dict(KNOCK, y=GROUND_Y, hp=hp)
    return prow({"x": 200}, dict({"x": 300}, **p2))


def _knock_collection(root, games=3):
    def play(game, sampler):
        for k in range(480):
            sampler.feed(dict(_knock_row(k), timer=k % 256), None if k == 0 else image(k))
        return {"result": "win", "frames": 480, "moves": []}
    for a, b in (("ryu", "ken"), ("ken", "ryu")):
        IO.collect_pair(os.path.join(root, IO.pair_name(a, b)), play, a, b, games, 0, BANDS, per_game=3,
                        log=lambda *x: None, controllers=PC.VS_SLOTS)


def _all_rows(out):
    return [r for f in ("train", "test") for r in MIO.read_jsonl(os.path.join(out, "movement", f + ".jsonl"))]


def test_the_build_has_down_only_on_the_ground_and_the_gate_passes_with_second_facts(tmp_path):
    root, out = str(tmp_path / "col"), str(tmp_path / "data")
    _knock_collection(root)
    D.build(root, out, {"train": 40, "test": 20}, bands=BANDS)
    rows = _all_rows(out)
    downs = [r for r in rows if r["answer"] == "down"]
    hits = [r for r in rows if r["answer"] == "hit"]
    assert downs and hits
    for r in downs:
        ram = MIO.read_ram(os.path.join(root, r["pair_name"], "ram", "g%04d.json.gz" % r["game"]))
        assert all(ram[u]["p%d_y" % r["slot"]] == GROUND_Y for u in range(r["t"] - 4, r["t"] + 1))
    assert any(r["air"] == "air" for r in hits)          # the airborne knockdown is now a hit
    rep = G.run_gates(out, BANDS, sample=50, min_disc=0)
    sf = rep["gates"]["second_fact"]
    assert sf["pass"] and sf["down"]["total"] == len(downs) and sf["down"]["on_ground"] == len(downs), sf
    assert sf["hit"]["total"] == len(hits) and sf["hit"]["health_lost"] == len(hits) and sf["hit"]["pct"] == 100.0
    assert all(rep["gates"][g]["pass"] for g in ("labels", "episode", "caps", "second_fact")), json.dumps(rep)[:1500]


def test_the_gate_fails_a_down_row_whose_ram_is_in_the_air(tmp_path):
    root, out = str(tmp_path / "col"), str(tmp_path / "data")
    _knock_collection(root)
    D.build(root, out, {"train": 40, "test": 20}, bands=BANDS)
    r = next(x for x in _all_rows(out) if x["answer"] == "down")
    path = os.path.join(root, r["pair_name"], "ram", "g%04d.json.gz" % r["game"])
    with gzip.open(path, "rt") as f:
        rec = json.load(f)
    row = dict(zip(rec["names"], rec["rows"][r["t"]]))
    row["p%d_y" % r["slot"]] = AIR
    rec["rows"][r["t"]] = [row[n] for n in rec["names"]]
    with gzip.open(path, "wt") as f:
        json.dump(rec, f)
    rep = G.run_gates(out, BANDS, sample=50, min_disc=0)
    assert not rep["gates"]["labels"]["pass"]            # independently: that row is a hit now
    assert not rep["gates"]["second_fact"]["pass"]
    assert not rep["pass"]


def test_the_second_fact_gate_alone_fails_a_down_row_in_the_air():
    rows = knockdown()
    t = 6 + 3 + 6 + 4
    lifted = [dict(x) for x in rows]
    lifted[t - 1]["p2_y"] = AIR
    rec = {"answer": "down", "movement": "down", "slot": 2, "t": t, "pair_name": "ryu_vs_ken", "game": 0,
           "char": "ken"}
    d = {"meta": {"root": "/nonexistent"}, "files": {"movement": {"train": [rec], "test": []}}}
    ok = G.second_fact_check(d, BANDS, ram_of=lambda pair, game: rows)
    bad = G.second_fact_check(d, BANDS, ram_of=lambda pair, game: lifted)
    assert ok["pass"] and ok["down"]["on_ground"] == 1
    assert not bad["pass"] and bad["down"]["on_ground"] == 0 and bad["down"]["examples"]


def test_a_jump_over_the_other_fighter_keeps_its_pair_direction_unknown_but_a_walk_does_not(tmp_path):
    """Round 7 (2026-10-01): ken_vs_dhalsim g6 t96, a jump crossing the other fighter's x: direction unknown (same x),
    movement "jump" - a grid cell without a direction, so the pair is fine; a walk without a direction is no cell."""
    img = tmp_path / "images"
    img.mkdir()
    for k in (93, 97):
        (img / ("g0006_k%05d.png" % k)).write_bytes(b"x")
    base = {"pair_name": "ken_vs_dhalsim", "game": 6, "slot": 1, "t": 96, "facing": "right", "air": "air",
            "distance": "close", "images": ["g0006_k00093.png", "g0006_k00097.png"]}
    assert D.pair_problems(dict(base, movement="jump", direction="unknown"), str(img)) == []
    assert D.pair_problems(dict(base, movement="walk", direction="unknown"), str(img))
    assert D.pair_problems(dict(base, movement="stand", direction="unknown"), str(img))
