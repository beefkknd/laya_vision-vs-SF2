"""Owner fix after the label quality check (docs/prereg_movement_pairs.md, "Label quality check"): a pair is kept only
if BOTH its displayed frames (t - 4 and t) lie inside the same movement episode of that fighter - every row t-4..t has
the same final movement (grid movement10, after the pressed-word attack / special rule). The builder selects only such
pairs (and counts the dropped ones per movement); the gate re-derives it independently and FAILS on a pair outside."""
import gzip
import json
import os

import pytest

from sf2.data import movement_collect_io as MIO
from sf2.data import pairs_data as D
from sf2.data import pairs_gate as G
from sf2.data import pairs_labels as L
from test_pairs_collect import BANDS
from test_pairs_labels import prow
from test_pairs_pressed import _stale_collection


def _rows(states):
    return [prow({"state": s}) for s in states]


def test_same_episode_needs_rows_t4_to_t_all_one_movement():
    crouch = _rows([2] * 9)
    assert L.same_episode(crouch, 8, 1, [None] * 9)
    blip = _rows([2] * 6 + [0] + [2] * 2)               # a 1-frame stand at t-2: not one episode
    assert not L.same_episode(blip, 8, 1, [None] * 9)
    start = _rows([0] * 5 + [2] * 4)                    # crouch starts at t-3: t-4 is outside
    assert not L.same_episode(start, 8, 1, [None] * 9)
    assert L.same_episode(_rows([0] * 4 + [2] * 5), 8, 1, [None] * 9)     # starts exactly at t-4
    assert not L.same_episode(crouch, 3, 1, [None] * 9)                  # no t-4
    assert not L.same_episode(_rows([6] * 9), 8, 1, [None] * 9)          # unknown is no episode


def test_same_episode_uses_the_final_movement_after_the_pressed_rule():
    att = _rows([0x0A] * 9)
    for r in att:
        r["p1_mclass"] = 0x08                            # RAM rule: special
    # the pressed word switches from a special to a normal at t-1: one RAM state, two final movements
    cls = ["special"] * 7 + ["attack"] * 2
    assert not L.same_episode(att, 8, 1, cls)
    assert L.same_episode(att, 8, 1, ["attack"] * 9)
    assert L.same_episode(att, 8, 1, [None] * 9)        # fallback: RAM special throughout


def test_same_episode_splits_walk_by_direction():
    xs = [200, 200, 200, 200, 204, 208, 212, 216, 220]
    walk = [prow({"x": x}) for x in xs]
    assert L.same_episode(walk, 8, 1, [None] * 9)
    back = [prow({"x": x}) for x in [200, 204, 208, 212, 216, 212, 208, 204, 200]]
    assert not L.same_episode(back, 8, 1, [None] * 9)    # toward at t-4, away at t


def test_the_builder_keeps_only_pairs_inside_one_episode_and_counts_the_dropped(tmp_path):
    root, out = str(tmp_path / "col"), str(tmp_path / "data")
    _stale_collection(root)
    meta = D.build(root, out, {"train": 40, "test": 20}, bands=BANDS)
    rows = [r for f in ("train", "test") for r in MIO.read_jsonl(os.path.join(out, "movement", f + ".jsonl"))]
    assert rows
    logs = {}
    for r in rows:
        ram = MIO.read_ram(os.path.join(root, r["pair_name"], "ram", "g%04d.json.gz" % r["game"]))
        if (r["pair_name"], r["game"]) not in logs:
            g = [x for x in MIO.read_jsonl(os.path.join(root, r["pair_name"], "games.jsonl")) if x["game"] == r["game"]]
            logs[(r["pair_name"], r["game"])] = D.pressed_classes(r["char"], g[0]["moves"], len(ram), r["slot"])
        assert L.same_episode(ram, r["t"], r["slot"], logs[(r["pair_name"], r["game"])])
    assert set(meta["dropped_episode"]) <= set(L.MOVEMENTS10)
    rep = G.run_gates(out, BANDS, sample=50, min_disc=0)
    assert rep["gates"]["episode"]["pass"], json.dumps(rep["gates"]["episode"])[:800]


@pytest.mark.parametrize("back", [4, 2])
def test_the_gate_fails_a_pair_whose_t4_is_outside_its_episode(tmp_path, back):
    root, out = str(tmp_path / "col"), str(tmp_path / "data")
    _stale_collection(root)
    D.build(root, out, {"train": 40, "test": 20}, bands=BANDS)
    r = MIO.read_jsonl(os.path.join(out, "movement", "train.jsonl"))[0]
    _break_episode(root, r, back)                        # row t - back now another movement; t unchanged
    rep = G.run_gates(out, BANDS, sample=50, min_disc=0)
    assert not rep["gates"]["episode"]["pass"]
    assert rep["gates"]["labels"]["pass"]                 # the label at t itself is still right


def test_the_sampler_only_samples_where_t4_is_inside_the_episode():
    import random
    from sf2.data import pairs_collect as PC
    from test_pairs_collect import feed, saver
    s = PC.PairSampler(0, {1: "ryu", 2: "ken"}, random.Random(0), saver({}), BANDS, controllers=PC.VS_SLOTS)
    feed(s, 1500)
    pairs = s.finish()
    assert pairs and all(p["t"] - 4 >= p["episode"][0] for p in pairs)
    assert all(p["length"] >= 5 for p in pairs)


def _break_episode(root, r, back=2):
    """Make row t - ``back`` of pair r's fighter another movement (t itself unchanged)."""
    path = os.path.join(root, r["pair_name"], "ram", "g%04d.json.gz" % r["game"])
    with gzip.open(path, "rt") as f:
        rec = json.load(f)
    names, me = rec["names"], "p%d_" % r["slot"]
    row = dict(zip(names, rec["rows"][r["t"] - back]))
    row[me + "state"] = 0x06 if row[me + "state"] != 0x06 else 0x02
    rec["rows"][r["t"] - back] = [row[n] for n in names]
    with gzip.open(path, "wt") as f:
        json.dump(rec, f)


def test_the_builder_drops_a_pair_outside_its_episode_and_counts_it(tmp_path):
    root, out = str(tmp_path / "col"), str(tmp_path / "data")
    _stale_collection(root)
    from sf2.data import pairs_collect_io as IO
    victim = IO.committed_pairs(os.path.join(root, "zangief_vs_ken"))[0][0]
    victim = dict(victim, pair_name="zangief_vs_ken")
    _break_episode(root, victim)
    meta = D.build(root, out, {"train": 40, "test": 20}, bands=BANDS)
    rows = [r for f in ("train", "test") for r in MIO.read_jsonl(os.path.join(out, "movement", f + ".jsonl"))]
    key = (victim["pair_name"], victim["game"], victim["slot"], victim["t"])
    assert key not in {(r["pair_name"], r["game"], r["slot"], r["t"]) for r in rows}
    assert sum(meta["dropped_episode"].values()) >= 1
