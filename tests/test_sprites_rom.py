"""The pose pointer and the pose tables (sf2.sprites.rom_table, sf2.sprites.rom_force.chain_pokes) - no emulator.

Fixtures: tests/fixtures/sprites/ryu_vs_ken_g0* (a real collected game, raw struct bytes per row) and
tests/fixtures/sprites/rom/ryu_wram.bin.gz (WRAM of the 2P versus state vs_ryu_vs_ken, 3 idle frames in;
scripts/probe_pose_table.py, 2026-10-02).
Seeded faults: the wrong lag, the wrong struct byte and the old jump formula must each be caught.
"""
import gzip
import json
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sf2.sprites import rom_table as T  # noqa: E402
from sf2.sprites.rom_force import chain_pokes  # noqa: E402

FIX = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures", "sprites")
PURITY_MIN, REVERSE_MIN = 0.98, 0.99
RYU_STANCE = [0x6A18, 0x6A1C, 0x6A20, 0x6A24]


def _game():
    z = np.load(os.path.join(FIX, "ryu_vs_ken_g0.npz"))
    names = [str(n) for n in z["names"]]
    raw = {p: z["rows"][:, [names.index("p%d_raw%02x" % (p, o)) for o in range(0x80)]].tolist() for p in (1, 2)}
    with gzip.open(os.path.join(FIX, "ryu_vs_ken_g0_frames.jsonl.gz"), "rt") as f:
        frames = [json.loads(line) for line in f]
    return raw, frames, {1: "ryu", 2: "ken"}


@pytest.fixture(scope="module")
def game():
    return _game()


@pytest.fixture(scope="module")
def wram():
    with gzip.open(os.path.join(FIX, "rom", "ryu_wram.bin.gz"), "rb") as f:
        return f.read()


def pointer_holds(raw, frames, chars, lag=1, offset=T.POSE_PTR) -> bool:
    r = T.purity(*T.pose_samples(raw, frames, chars, lag=lag, offset=offset))
    return r["purity"] >= PURITY_MIN and r["reverse"] >= REVERSE_MIN


def test_pose_pointer_purity(game):
    r = T.purity(*T.pose_samples(*game, lag=1))
    assert r["n"] > 5000
    assert r["purity"] >= PURITY_MIN and r["reverse"] >= REVERSE_MIN, r


@pytest.mark.parametrize("fault", [dict(lag=0), dict(lag=2), dict(offset=T.POSE_PTR + 1),
                                   dict(offset=T.ANIM_PTR)])
def test_pose_pointer_seeded_faults(game, fault):
    assert not pointer_holds(*game, **fault)


def test_purity_rejects_bad_input():
    with pytest.raises(ValueError):
        T.purity([], [])
    with pytest.raises(ValueError):
        T.purity([1], [1, 2])


def test_purity_counts():
    r = T.purity([1, 1, 1, 2], ["a", "a", "b", "c"])
    assert (r["purity"], r["reverse"], r["values"], r["keys"]) == (0.75, 1.0, 2, 3)


def test_lorom():
    assert T.lorom_offset(0x10, 0x8000) == 0x80000
    assert T.lorom_address(0x800F6) == (0x10, 0x80F6)
    assert T.lorom_offset(*T.lorom_address(0x12345)) == 0x12345
    with pytest.raises(ValueError):
        T.lorom_offset(0x10, 0x7FFF)


def test_ryu_tables(wram):
    poses = T.image_table(wram)
    assert len(poses) == 123 and None not in poses
    assert poses[0] == 0x71F6 and poses[14] == 0x7490 and poses[16] == 0x7526   # seen as 0x0C1E in play
    assert max(poses) < 0x9500                                                 # inside the copied block
    used = T.poses_used(wram)
    assert all(p < len(poses) for p in used)
    stance = T.walk_anim(wram, 0x6A18, 0x7100)
    assert [r[0] for r in stance] == RYU_STANCE and stance[-1][2] & T.END_FLAG
    assert [r[3] for r in stance] == [0, 1, 0, 2]


def test_offset_table_rejects_garbage():
    buf = bytearray(0x100)
    buf[0:2] = (0x05, 0x00)                    # odd first offset
    with pytest.raises(ValueError):
        T.offset_table(bytes(buf), 0, 0x100)
    buf[0:4] = (0x04, 0x00, 0xF0, 0x0F)        # 2nd offset outside the limit
    with pytest.raises(ValueError):
        T.offset_table(bytes(buf), 0, 0x100)


def _lands(wram, pokes, end):
    w = bytearray(wram)
    for a, v in pokes.items():
        w[a] = v
    rel = w[end + 4] | (w[end + 5] << 8)
    rel -= 0x10000 if rel >= 0x8000 else 0
    return end + 4 + rel


def test_chain_jump_lands_on_script(wram):
    # unpatched, the stance's jump word goes back to the stance start (the formula the pokes rely on)
    assert _lands(wram, {}, 0x6A24) == 0x6A18
    scripts = T.pose_scripts(wram)
    script, target, _ = scripts[15][0]
    pokes = chain_pokes(wram, RYU_STANCE, script, target)
    assert _lands(wram, pokes, 0x6A24) == script and pokes[target] == 0xFF
    # seeded fault: the first (wrong) formula, relative to the end of the 6 bytes
    rel = (script - (0x6A24 + 6)) & 0xFFFF
    assert _lands(wram, {0x6A28: rel & 0xFF, 0x6A29: rel >> 8}, 0x6A24) != script
