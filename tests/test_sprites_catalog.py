"""The sprite catalog's keys, dedupe and label join (sf2.sprites.catalog / collect / build).

Synthetic snapshots (one 8x8 OBJ on palette 4) check mirror normalisation and dedupe through the real PairSink.
The label join runs sf2.sprites.build on a real collected game (tests/fixtures/sprites/ryu_vs_ken_g0*: game 0 of
ryu vs ken from scripts/collect_sprites.py, 2026-10-02) and recomputes every frame's answer independently from
sf2.data.pairs_labels at the displayed row k - 1.
"""
import gzip
import json
import os
import shutil
import sys
from collections import Counter

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sf2.data import pairs_labels as L  # noqa: E402
from sf2.data.eye_v2 import ACT2_OF  # noqa: E402
from sf2.sprites import build as B  # noqa: E402
from sf2.sprites import report as R  # noqa: E402
from sf2.sprites.catalog import FACE_LEFT, FACE_RIGHT, canonical, key_of  # noqa: E402
from sf2.sprites.collect import ALL_NAMES, PairSink, oam_facing  # noqa: E402
from sf2.sprites.oam import VRAM_LEN, Snapshot, render_groups  # noqa: E402

FIX = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures", "sprites")
ASYM = np.array([[1, 2, 3, 0, 0, 0, 0, 0]] + [[0] * 7 + [r % 15 + 1] for r in range(7)], np.uint8)  # not symmetric


def _tile_bytes(pix: np.ndarray) -> bytes:
    out = bytearray(32)
    for r in range(8):
        for p in range(4):
            byte = sum((((int(pix[r, c]) >> p) & 1) << (7 - c)) for c in range(8))
            out[(p // 2) * 16 + 2 * r + (p % 2)] = byte
    return bytes(out)


def _snap(x: int, y: int, tile: int, hflip: bool) -> Snapshot:
    vram = bytearray(VRAM_LEN)
    vram[0:32] = _tile_bytes(ASYM)
    vram[32:64] = _tile_bytes(ASYM.T)
    oam = bytearray(544)
    for i in range(128):
        oam[4 * i + 1] = 240                                  # parked
    oam[0:4] = bytes([x, y, tile, (4 << 1) | (0x40 if hflip else 0)])
    cg = bytearray(512)
    for c in range(16):
        cg[2 * (128 + 64 + c):2 * (128 + 64 + c) + 2] = (c * 2 + (c << 6)).to_bytes(2, "little")
    return Snapshot(bytes(oam), bytes(cg), bytes(vram), 0, 24576, 4096)


def test_hflip_render_is_the_mirror():
    a = render_groups(_snap(50, 100, 0, False))[4].rgba
    b = render_groups(_snap(50, 100, 0, True))[4].rgba
    assert not np.array_equal(a, b) and np.array_equal(a[:, ::-1], b)


def test_a_sprite_and_its_flip_give_the_same_key():
    img = render_groups(_snap(50, 100, 0, False))[4].rgba
    assert key_of(canonical(img, FACE_LEFT)) == key_of(canonical(img[:, ::-1], FACE_RIGHT))
    assert key_of(canonical(img, FACE_LEFT)) != key_of(canonical(img, FACE_RIGHT))


def test_oam_facing_reads_the_flip():
    assert oam_facing(_snap(50, 100, 0, True), 4) == FACE_RIGHT
    assert oam_facing(_snap(50, 100, 0, False), 4) == FACE_LEFT


def test_dedupe_through_the_sink(tmp_path):
    sink = PairSink(str(tmp_path), {1: "ryu", 2: "ken"})
    row = {n: 0 for n in ALL_NAMES}
    sink.feed(row, None)
    snaps = [_snap(50, 100, 0, False), _snap(120, 60, 0, True), _snap(80, 100, 1, False), _snap(10, 30, 0, False)]
    for s in snaps:
        sink.feed(row, s)
    keys = [f["key"] for f in sink.frames]
    assert keys[0] == keys[1] == keys[3] != keys[2]
    assert len(os.listdir(tmp_path / "sprites")) == 2 and len(os.listdir(tmp_path / "snaps")) == 2
    assert [f["k"] for f in sink.frames] == [1, 2, 3, 4]


def _fixture_out(tmp_path):
    d = tmp_path / "_pairs" / "ryu_vs_ken"
    d.mkdir(parents=True)
    shutil.copyfile(os.path.join(FIX, "ryu_vs_ken_g0.npz"), d / "g0.npz")
    shutil.copyfile(os.path.join(FIX, "ryu_vs_ken_g0_frames.jsonl.gz"), d / "g0_frames.jsonl.gz")
    return str(tmp_path)


def test_label_join_matches_pairs_labels_at_the_displayed_row(tmp_path):
    out = _fixture_out(tmp_path)
    cat = B.build(out)["catalog"]
    z = np.load(os.path.join(out, "_pairs", "ryu_vs_ken", "g0.npz"))
    names = [str(n) for n in z["names"]]
    rows = [dict(zip(names, map(int, r))) for r in z["rows"]]
    pressed = {p: [c or None for c in z["pressed"][p - 1].tolist()] for p in (1, 2)}
    want_act, want_air, want_state = Counter(), Counter(), Counter()
    with gzip.open(os.path.join(out, "_pairs", "ryu_vs_ken", "g0_frames.jsonl.gz"), "rt") as f:
        frames = [json.loads(line) for line in f]
    for fr in frames:
        if fr["group"] not in ("p1", "p2"):
            continue
        p, t = int(fr["group"][1]), fr["k"] - 1
        char = ("ryu", "ken")[p - 1]
        mv = L.movement_pressed(rows, t, p, pressed[p][t])[0]
        want_act[(char, fr["key"], ACT2_OF.get(mv, "unknown"))] += 1
        want_air[(char, fr["key"], L.air(rows, t, p))] += 1
        want_state[(char, fr["key"], "%02x" % rows[t]["p%d_state" % p])] += 1
    got = lambda field: Counter({(e["char"], e["key"], v): n for e in cat.values() if e["char"] != "_projectiles"
                                 for v, n in e["labels"][field].items()})
    assert got("act2") == want_act and got("air") == want_air and got("state") == want_state
    assert sum(e["frames"] for e in cat.values() if e["char"] == "ryu") == len(rows) - 1


def test_lag_one_shares_fewer_sprites_than_lag_zero(tmp_path):
    """The displayed-row convention (k - 1) is also what the pixels say: fewer frames under a sprite's minority
    answer than reading the row the snapshot was taken on."""
    cat = {"%s/%s" % (e["char"], e["key"]): B.to_json(e) for e in B.build(_fixture_out(tmp_path))["catalog"].values()}
    lag1, lag0 = R.shared(cat, "labels")["chars"], R.shared(cat, "lag0")["chars"]
    for c in ("ryu", "ken"):
        assert lag1[c]["minority_frames"] < lag0[c]["minority_frames"]
