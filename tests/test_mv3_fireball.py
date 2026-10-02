"""Round 3 of docs/prereg_movement_finetunes.md, the "fireball" question: "Is there a fireball on the screen?" ->
none / left / right (thrown by the fighter on the left / right). Fireball rows: the projectile-trigger samples
(sf2.data.pairs_shots) whose projectile is DRAWN at the displayed row t, thrown with the thrower's projectile word,
the other slot off at t; the answer = the thrower's side by x at t. "none" rows: existing pairs-build rows with both
slots off over t - 8 .. t, as many per split as the larger fireball answer. Caps per (split, thrower, side, flight
stage); split by whole match (test crc32 % 3 == 2, val 1 in 6 training matches); one dir per answer."""
import collections
import gzip
import json
import os
import shutil

import pytest

from sf2.data import movement_collect_io as MIO
from sf2.data import mv3_fireball as F
from sf2.data import pairs_collect as PC
from sf2.data import pairs_collect_io as IO
from sf2.data import pairs_data as D
from sf2.data import pairs_shots as S
from sf2.data import pairs_train as T
from test_mv3_act import _act_collection
from test_pairs_collect import BANDS, image
from test_pairs_shots import frow

N = 460
WORDS = [(1, "hadoken", 30), (2, "hadoken", 130), (1, "hadoken", 230), (2, "hadoken", 242), (1, "shoryuken", 330)]
FLIGHTS = [(1, 40, 99), (2, 140, 199), (1, 240, 299), (2, 250, 295), (1, 340, 399)]


def drawn_image(k, r):
    """image(k), with a blue blob below the HUD when the row it shows has a projectile drawn (the gate's hadoken
    colour test)."""
    im = image(k)
    if S.visible(r, 1) or S.visible(r, 2):
        im[120:140, 100:130] = (60, 120, 255)
    return im


def shot_play(game, sampler):
    x = (300, 200) if game % 2 else (200, 300)
    moves = []
    for i, (p, w, k0) in enumerate(WORDS):
        nxt = [k for q, _, k in WORDS[i + 1:] if q == p]
        moves.append([w, k0, nxt[0] if nxt else N - 1, p])
    for k in range(N):
        for p, w, k0 in WORDS:
            if k0 == k - 1:
                sampler.press(p, w, k0)
        shown = frow(k - 1, FLIGHTS, 4, *x)               # the capture at row k shows row k - 1 (lag 1)
        sampler.feed(frow(k, FLIGHTS, 4, *x), None if k == 0 else drawn_image(k, shown))
    return {"result": "win", "frames": N, "moves": moves}


def _shot_games(root, games):
    for a, b in (("ryu", "ken"), ("ken", "ryu")):
        IO.collect_pair(os.path.join(root, IO.pair_name(a, b)), shot_play, a, b, games, 0, BANDS, per_game=0,
                        log=lambda *x: None, controllers=PC.VS_SLOTS, sampler_cls=S.ShotSampler)


@pytest.fixture(scope="module")
def src(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("mv3fire")
    root, out = str(tmp / "col"), str(tmp / "pairs")
    _act_collection(root, games=12)
    D.build(root, out, {"train": 40, "test": 20}, bands=BANDS)
    _shot_games(root, 16)                                 # games 12..15: the projectile rounds
    return out, root


def _rows(out):
    return [r for fs in T.read_dataset(out).values() for f in T.FILES for r in fs[f]]


def _ram(root, r):
    return MIO.read_ram(os.path.join(root, r["pair_name"], "ram", "g%04d.json.gz" % r["game"]))


def test_the_question_text_and_answers():
    q = F.question_fireball()
    assert q["instructions"] == "Is there a fireball on the screen?" and q["type"] == "choice"
    assert tuple(q["criteria"]) == ("none", "left", "right") == F.FIRE_ANSWERS


def test_the_build_labels_every_fireball_row_from_ram_and_the_thrower(src, tmp_path):
    s, root = src
    out = str(tmp_path / "fire")
    meta = F.build_fireball(s, root, out)
    assert meta["problems"] == [] and meta["question"] == "fireball" and meta["answer_dirs"] is True
    rows = _rows(out)
    fire = [r for r in rows if r["answer"] != "none"]
    assert {r["answer"] for r in fire} == {"left", "right"} and {r["char"] for r in fire} == {"ryu", "ken"}
    for r in fire:
        ram = _ram(root, r)
        p, t = r["thrower_slot"], r["t"]
        assert ram[t]["shot%d" % p] and not ram[t]["shot%d_hide" % p] & 1 and not ram[t]["shot%d" % (3 - p)]
        me, other = ram[t]["p%d_x" % p], ram[t]["p%d_x" % (3 - p)]
        assert r["answer"] == ("left" if me < other else "right") == r["side"]
        assert r["spawn_word"] == "hadoken" and r["char"] == r["pair_name"].split("_vs_")[p - 1]
        assert r["images"] == ["frames/%s/g%04d_k%05d.png" % (r["pair_name"], r["game"], k) for k in (t - 3, t + 1)]
        assert r["label"] == F.FIRE_ANSWERS.index(r["answer"]) and r["source"] == "shot"
        assert r["split"] == T.split3(r["pair_name"], r["game"])
    assert meta["dropped"]["not_projectile"] > 0 and meta["dropped"]["two_shots"] > 0
    for f in T.FILES:
        n = collections.Counter(r["answer"] for r in rows if r["split"] == f)
        assert n["none"] == max(n["left"], n["right"])


def test_none_rows_come_from_the_pairs_build_with_no_shot_over_t_minus_8_to_t(src, tmp_path):
    s, root = src
    out = str(tmp_path / "fire")
    F.build_fireball(s, root, out)
    src_imgs = {tuple(json.loads(x)["images"]) for f in D.FILES for x in open(os.path.join(s, "movement", f + ".jsonl"))}
    none = [r for r in _rows(out) if r["answer"] == "none"]
    assert none and len({tuple(r["images"]) for r in none}) == len(none)
    for r in none:
        ram = _ram(root, r)
        assert tuple(r["images"]) in src_imgs and r["source"] == "pairs" and r["char"] == "none"
        assert all(ram[u]["shot1"] == 0 == ram[u]["shot2"] for u in range(r["t"] - 8, r["t"] + 1))


def test_caps_per_split_thrower_side_and_stage(src, tmp_path):
    s, root = src
    out = str(tmp_path / "fire")
    meta = F.build_fireball(s, root, out, caps={"train": 2, "test": 1})
    fire = [r for r in _rows(out) if r["answer"] != "none"]
    n = collections.Counter((D.split_of_game(r["pair_name"], r["game"]), r["char"], r["side"], r["flight_stage"])
                            for r in fire)
    assert n and max(v for k, v in n.items() if k[0] == "train") <= 2 and max(
        v for k, v in n.items() if k[0] == "test") <= 1
    assert meta["problems"] == [] and meta["short"]          # guile / dhalsim cells are empty: reported


def test_refuses_an_existing_out_and_another_collection(src, tmp_path):
    s, root = src
    out = str(tmp_path / "fire")
    os.makedirs(out)
    with pytest.raises(FileExistsError):
        F.build_fireball(s, root, out)
    with pytest.raises(ValueError):
        F.build_fireball(s, str(tmp_path), str(tmp_path / "fire2"))


def test_fill_report_counts_the_cells(src):
    s, root = src
    rep = F.fill_report(root)
    assert rep["cells"]["train|ryu|left|start"] > 0 and "train|guile|left|start" in rep["cells"]
    assert 0 < rep["filled_pct"] < 100


def _copy(src, tmp_path):
    s, root = src
    root2 = str(tmp_path / "col")
    shutil.copytree(root, root2, symlinks=True)
    s2 = str(tmp_path / "pairs")
    shutil.copytree(s, s2, symlinks=True)
    meta = json.load(open(os.path.join(s2, "build.json")))
    with open(os.path.join(s2, "build.json"), "w") as f:
        json.dump(dict(meta, root=root2), f)
    for n in os.listdir(os.path.join(s2, "frames")):
        os.remove(os.path.join(s2, "frames", n))
        os.symlink(os.path.join(root2, n, "images"), os.path.join(s2, "frames", n))
    return s2, root2


def _set(r, t, root, **vals):
    path = os.path.join(root, r["pair_name"], "ram", "g%04d.json.gz" % r["game"])
    with gzip.open(path, "rt") as f:
        rec = json.load(f)
    for k, v in vals.items():
        rec["rows"][t][rec["names"].index(k)] = v
    with gzip.open(path, "wt") as f:
        json.dump(rec, f)


@pytest.mark.parametrize("tamper", ["flip", "label", "question", "lag0", "stage", "hidden", "none_shot",
                                    "drop_none", "split", "spawn"])
def test_problems_catches_a_wrong_row(src, tmp_path, tamper):
    s, root = _copy(src, tmp_path)
    out = str(tmp_path / "fire")
    F.build_fireball(s, root, out)
    d = "none" if tamper in ("none_shot", "drop_none") else "left"
    path = os.path.join(out, d, "train.jsonl")
    rows = [json.loads(x) for x in open(path)]
    r = rows[0]
    if tamper == "flip":
        rows[0] = dict(r, answer="right", side="right")
    elif tamper == "label":
        rows[0] = dict(r, label=2)
    elif tamper == "question":
        rows[0] = dict(r, question=dict(r["question"], instructions="Is there a fireball?"))
    elif tamper == "lag0":
        rows[0] = dict(r, images=["frames/%s/g%04d_k%05d.png" % (r["pair_name"], r["game"], k)
                                  for k in (r["t"] - 4, r["t"])])
    elif tamper == "stage":
        rows[0] = dict(r, flight_stage={"start": "end"}.get(r["flight_stage"], "start"))
    elif tamper == "hidden":
        _set(r, r["t"], root, **{"shot%d_hide" % r["thrower_slot"]: 1})
    elif tamper == "none_shot":
        _set(r, r["t"] - 6, root, shot2=1)
    elif tamper == "drop_none":
        rows = rows[1:]
    elif tamper == "split":
        rows[0] = dict(r, split="test")
    else:                                                 # the thrower pressed no projectile at the spawn
        log = os.path.join(root, r["pair_name"], "games.jsonl")
        games = [json.loads(x) for x in open(log)]
        for g in games:
            if g["game"] == r["game"]:
                g["moves"] = [[("shoryuken" if m[0] == "hadoken" else m[0])] + m[1:] for m in g["moves"]]
        with open(log, "w") as f:
            f.writelines(json.dumps(g) + "\n" for g in games)
    with open(path, "w") as f:
        f.writelines(json.dumps(x) + "\n" for x in rows)
    assert F.problems_fireball(out, s, root)


def test_a_sample_whose_projectile_is_hidden_at_t_is_dropped(src, tmp_path):
    s, root = _copy(src, tmp_path)
    x = next(r for r in F.shot_samples(root) if S.is_projectile(r["thrower"], r["spawn_word"]))
    _set(x, x["t"], root, **{"shot%d_hide" % x["slot"]: 1})
    kept, dropped = F.checked(root, F.shot_samples(root))
    assert dropped["not_drawn"] == 1 and not any((k["pair_name"], k["game"], k["t"], k["slot"]) ==
                                                 (x["pair_name"], x["game"], x["t"], x["slot"]) for k in kept)


def test_a_none_row_with_a_shot_within_8_rows_before_t_is_not_in_the_pool(src, tmp_path):
    s, root = _copy(src, tmp_path)
    pool = F.none_rows(s, root)
    r = pool[0]
    _set(r, r["t"] - 6, root=root, shot1=1)
    assert r["images"] not in [x["images"] for x in F.none_rows(s, root)]
