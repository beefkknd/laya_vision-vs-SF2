"""Round 4 of docs/prereg_movement_finetunes.md (step 1, data only): the pairs build with per-movement caps (attack
and special cells (character x facing) 100 train / 30 test, the other movements unchanged) and a kept earlier build
(every row of round 3's source build selected again, in the same split, so round 3's test rows stay in the test);
the gate checks both (caps per cell from build.json's movement_caps; "kept": every earlier row present)."""
import collections
import json
import os

import pytest

from sf2.data import pairs_data as D
from sf2.data import pairs_gate as G
from sf2.data import pairs_collect_io as IO
from test_mv3_act import _act_collection
from test_pairs_collect import BANDS

sys_path_scripts = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts")


def _p(pair, game, slot, t, mv="stand", facing="right", char="ryu"):
    return {"pair_name": pair, "game": game, "slot": slot, "t": t, "char": char, "movement": mv, "direction": "none",
            "facing": facing}


def _test_game(pair):
    return next(g for g in range(100) if D.split_of_game(pair, g) == "test")


def _train_games(pair, n):
    return [g for g in range(200) if D.split_of_game(pair, g) == "train"][:n]


def test_cap_of_uses_the_movement_cap_when_given():
    mc = {"attack": {"train": 100, "test": 30}}
    assert D.cap_of({"train": 40, "test": 20}, "train", "attack", mc) == 100
    assert D.cap_of({"train": 40, "test": 20}, "test", "special", mc) == 20
    assert D.cap_of({"train": 40, "test": 20}, "test", "attack", None) == 20


def test_select_caps_attack_and_special_cells_apart_from_the_others():
    games = _train_games("ryu_vs_ken", 10)
    pairs = [_p("ryu_vs_ken", g, 1, t, mv) for g in games for t in range(3) for mv in ("stand", "attack", "special")]
    mc = {"attack": {"train": 7, "test": 3}, "special": {"train": 5, "test": 3}}
    n = collections.Counter(p["movement"] for p in D.select(pairs, {"train": 2, "test": 1}, 0, movement_caps=mc))
    assert n == {"stand": 2, "attack": 7, "special": 5}


def test_select_takes_the_kept_rows_first_then_fills_to_the_cap():
    games = _train_games("ryu_vs_ken", 10)
    pairs = [_p("ryu_vs_ken", g, 1, t, "attack") for g in games for t in range(3)]
    keep = {D.pair_key(p) for p in pairs[-4:]}                       # the last game's rows + one: not round-robin's
    got = D.select(pairs, {"train": 2, "test": 1}, 0, movement_caps={"attack": {"train": 6, "test": 3}}, keep=keep)
    keys = [D.pair_key(p) for p in got]
    assert len(keys) == 6 == len(set(keys)) and keep <= set(keys)
    # with the cap at the old value, only the kept rows (as many as fit: never more kept than the cap)
    with pytest.raises(ValueError):
        D.select(pairs, {"train": 2, "test": 1}, 0, keep=keep)


def test_select_refuses_a_kept_row_missing_from_the_pool():
    pairs = [_p("ryu_vs_ken", g, 1, 0) for g in _train_games("ryu_vs_ken", 3)]
    with pytest.raises(ValueError):
        D.select(pairs, {"train": 2, "test": 1}, 0, keep={("ryu_vs_ken", 999, 1, 0)})


def test_universe_and_shortfalls_use_the_per_movement_caps():
    mc = {"attack": {"train": 100, "test": 30}}
    caps = D.cell_caps(["ryu"], {"train": 40, "test": 20}, mc)
    assert caps[("train", "ryu", "attack", "left")] == 100 and caps[("test", "ryu", "attack", "right")] == 30
    assert caps[("train", "ryu", "stand", "left")] == 40 and len(caps) == 2 * 10 * 2


def test_parse_movement_caps():
    assert D.parse_movement_caps(["attack:100:30", "special:100:30"]) == {
        "attack": {"train": 100, "test": 30}, "special": {"train": 100, "test": 30}}
    for bad in (["flying:1:1"], ["attack:100"], ["attack:-1:3"], ["attack:1:1", "attack:2:2"]):
        with pytest.raises(ValueError):
            D.parse_movement_caps(bad)


# ---- a build that keeps an earlier one, with raised attack / special caps ------------------------------------------

@pytest.fixture(scope="module")
def two_builds(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("mv4caps")
    root = str(tmp / "col")
    _act_collection(root, games=6)
    old = str(tmp / "old")
    D.build(root, old, {"train": 4, "test": 2}, bands=BANDS)
    _act_collection(root, games=12)                                    # more games, resumed
    new = str(tmp / "new")
    mc = {"attack": {"train": 9, "test": 5}, "special": {"train": 9, "test": 5}}
    meta = D.build(root, new, {"train": 4, "test": 2}, bands=BANDS, movement_caps=mc, keep_from=old)
    return root, old, new, meta


def _mv_rows(out):
    return [r for f in D.FILES for r in IO.read_jsonl(os.path.join(out, "movement", f + ".jsonl"))]


def test_the_new_build_holds_every_row_of_the_kept_build_in_its_split(two_builds):
    _, old, new, meta = two_builds
    new_rows = {r["id"]: r for r in _mv_rows(new)}
    olds = _mv_rows(old)
    assert olds and all(r["id"] in new_rows and new_rows[r["id"]]["split"] == r["split"] for r in olds)
    assert meta["kept"] == len(olds) and meta["keep_from"] == os.path.abspath(old)
    assert meta["movement_caps"] == {"attack": {"train": 9, "test": 5}, "special": {"train": 9, "test": 5}}


def test_attack_and_special_cells_grow_to_their_caps_the_others_do_not(two_builds):
    _, old, new, meta = two_builds
    n = collections.Counter((r["split"],) + D.cell(r) for r in _mv_rows(new))
    big = [k for k in n if k[2] in ("attack", "special")]
    assert big and max(n[k] for k in big if k[0] == "train") == 9
    assert max(v for k, v in n.items() if k[2] not in ("attack", "special") and k[0] == "train") <= 4
    assert max(v for k, v in n.items() if k[2] not in ("attack", "special") and k[0] == "test") <= 2
    want = {k: (9 if k[0] == "train" else 5) if k[2] in ("attack", "special") else (4 if k[0] == "train" else 2)
            for k in D.universe(["ken", "ryu"], {"train": 4, "test": 2})}
    assert meta["short"] == {"|".join(k): c - n[k] for k, c in want.items() if n[k] < c}
    assert any(k.split("|")[2] == "attack" and v == 9 for k, v in meta["short"].items())   # an empty attack cell


def test_the_gate_checks_caps_per_movement_and_the_kept_rows(two_builds, tmp_path):
    _, old, new, _ = two_builds
    d = G.load(new)
    assert G.cap_check(d)["pass"] and G.kept_check(d)["pass"]
    # a moving cell over its own cap fails, although the attack cap is larger
    stand = [r for r in d["files"]["movement"]["train"] if r["movement"] == "stand"]
    extra = [dict(stand[0], game=1000 + i) for i in range(5)]          # other games: only the cell cap is over
    bad = {"meta": d["meta"], "files": {"movement": {"train": d["files"]["movement"]["train"] + extra, "test": []}}}
    assert not G.cap_check(bad)["pass"]
    # an attack cell over the raised cap fails
    att = [r for r in d["files"]["movement"]["train"] if r["movement"] == "attack"]
    meta = dict(d["meta"], movement_caps={"attack": {"train": 1, "test": 1}, "special": {"train": 9, "test": 5}})
    assert not G.cap_check({"meta": meta, "files": d["files"]})["pass"] and att
    # a kept row missing (or moved to another split) fails the kept gate
    files = {q: dict(fs) for q, fs in d["files"].items()}
    files["movement"] = {"train": files["movement"]["train"][1:], "test": files["movement"]["test"]}
    assert not G.kept_check({"meta": d["meta"], "files": files})["pass"]


def test_kept_gate_passes_trivially_without_a_kept_build(tmp_path):
    d = {"meta": {"caps": {"train": 1, "test": 1}}, "files": {"movement": {"train": [], "test": []}}}
    assert G.kept_check(d) == {"pass": True, "keep_from": None}


def test_build_pairs_data_cli_passes_movement_caps_and_keep(two_builds, tmp_path):
    import subprocess
    import sys
    root, old, _, _ = two_builds
    out = str(tmp_path / "cli")
    r = subprocess.run([sys.executable, os.path.join(sys_path_scripts, "build_pairs_data.py"), "--root", root,
                        "--out", out, "--cap-train", "4", "--cap-test", "2", "--cap-movement", "attack:9:5",
                        "--cap-movement", "special:9:5", "--keep-from", old], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-2000:]
    meta = json.load(open(os.path.join(out, "build.json")))
    assert meta["movement_caps"]["attack"] == {"train": 9, "test": 5} and meta["kept"] == len(_mv_rows(old))


def test_run_gates_includes_the_kept_gate(two_builds, tmp_path):
    import shutil
    _, old, new, _ = two_builds
    cp = str(tmp_path / "cp")
    shutil.copytree(new, cp, symlinks=True)
    path = os.path.join(cp, "movement", "test.jsonl")
    rows = IO.read_jsonl(path)
    with open(path, "w") as f:
        f.writelines(json.dumps(r) + "\n" for r in rows[1:])
    rep = G.run_gates(cp, BANDS, sample=20, min_disc=1)
    assert "kept" in rep["gates"] and not rep["gates"]["kept"]["pass"] and not rep["pass"]
