"""Round 4 step 1 of docs/prereg_movement_finetunes.md: the per-round fill line (sf2.data.mv4_fill) must count what
the builds would select - act attack / special rows per split (train / val / test by whole match, as
sf2.data.pairs_train.split3) from the pairs build's selection with the per-movement caps, and fireball left / right rows
per split from the fireball selection with its caps."""
import collections
import json
import os

from sf2.data import mv3_fireball as F
from sf2.data import mv4_fill as M
from sf2.data import pairs_data as D
from sf2.data import pairs_train as T
from test_mv3_fireball import _rows, src  # noqa: F401  (the module fixture: act games + projectile games)
from test_pairs_collect import BANDS

MC = {"attack": {"train": 7, "test": 3}, "special": {"train": 6, "test": 2}}
FC = {"train": 3, "test": 2}


def test_fill_counts_what_the_builds_select(src, tmp_path):  # noqa: F811
    s, root = src
    rep = M.fill(root, caps={"train": 4, "test": 2}, movement_caps=MC, fire_caps=FC, bands=BANDS)
    out = str(tmp_path / "pairs")
    D.build(root, out, {"train": 4, "test": 2}, bands=BANDS, movement_caps=MC)
    mv = [json.loads(x) for f in D.FILES for x in open(os.path.join(out, "movement", f + ".jsonl"))]
    want = collections.Counter((r["answer"], T.split3(r["pair_name"], r["game"])) for r in mv
                               if r["answer"] in ("attack", "special"))
    assert rep["act"] == {a: {f: want[(a, f)] for f in T.FILES} for a in ("attack", "special")}
    assert rep["act"]["attack"]["train"] > 0
    fire = str(tmp_path / "fire")
    F.build_fireball(out, root, fire, caps=FC)
    got = collections.Counter((r["answer"], r["split"]) for r in _rows(fire) if r["answer"] != "none")
    assert rep["fireball"] == {a: {f: got[(a, f)] for f in T.FILES} for a in ("left", "right")}
    # cells not full: act cells (split, char, attack|special, facing) under their cap, fireball cells under theirs
    assert all(k.split("|")[2] in ("attack", "special") for k in rep["act_not_full"])
    assert rep["fire_not_full"] and all(k.split("|")[1] in F.S.THROWERS for k in rep["fire_not_full"])


def test_the_line_names_the_counts_and_the_targets(src):  # noqa: F811
    s, root = src
    rep = M.fill(root, caps={"train": 4, "test": 2}, movement_caps=MC, fire_caps=FC, bands=BANDS)
    line = M.line(rep, target=1000)
    assert "attack %d" % rep["act"]["attack"]["train"] in line and "left %d" % rep["fireball"]["left"]["train"] in line
    assert M.met(rep, target=1) and not M.met(rep, target=10 ** 6)


def test_met_needs_both_act_answers_and_both_fireball_answers():
    full = {"train": 5000, "val": 0, "test": 0}
    rep = {"act": {"attack": full, "special": full}, "fireball": {"left": full, "right": dict(full, train=999)}}
    assert not M.met(rep, 1000)
    rep["fireball"]["right"] = full
    assert M.met(rep, 1000)
    rep["act"]["special"] = dict(full, train=10)
    assert not M.met(rep, 1000)


def test_fill_keeps_the_earlier_build(src, tmp_path):  # noqa: F811
    s, root = src
    old = str(tmp_path / "old")
    D.build(root, old, {"train": 2, "test": 1}, bands=BANDS, seed=7)
    n_old = sum(1 for f in D.FILES for _ in open(os.path.join(old, "movement", f + ".jsonl")))
    rep = M.fill(root, caps={"train": 4, "test": 2}, movement_caps=MC, fire_caps=FC, bands=BANDS, keep_from=old)
    assert n_old > 0 and rep["kept"] == n_old
