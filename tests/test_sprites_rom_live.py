"""Forced poses render exactly as the catalog's sprites (sf2.sprites.rom_force) - needs Mesen + the ROM; skipped
without them.

Expected keys: out/sprite_catalog (2026-10-02) sprites drawn while 0x0C1E held each pose record (Ryu, player 1).
A key is the sha1 of the canonical RGBA, so key equality is pixel equality.
Seeded fault: the table off by one (pose i forced with pose i+1's index and record) must miss every key.
"""
import gzip
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sf2.config import DEFAULT_ROM, REPO  # noqa: E402
from sf2.sprites import rom_table as T  # noqa: E402

FIX = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures", "sprites", "rom")
STATE = os.path.join(REPO, "states", "vs_ryu_vs_ken.state")
ROM = os.environ.get("SF2_ROM") or os.path.join(REPO, DEFAULT_ROM)
PORT = 52221
CATALOG_KEYS = {0: "8098a7b7e032dbb8", 14: "a35d6e1e49324200", 15: "9918074219374d5d", 16: "3232e329f6da7433",
                22: "f1a7777bd8502da7"}   # 15 and 22 reuse their predecessor's tiles: they need the chain method


def _mesen():
    try:
        from sf2.emu.headless import find_mesen
        return find_mesen()
    except Exception:
        return None


pytestmark = pytest.mark.skipif(not (os.path.exists(ROM) and os.path.exists(STATE) and _mesen()),
                                reason="needs Mesen, the ROM and states/vs_ryu_vs_ken.state")


@pytest.fixture(scope="module")
def forced():
    from sf2.sprites import rom_force as F
    from sf2.sprites.emu import open_mesen
    with gzip.open(os.path.join(FIX, "ryu_wram.bin.gz"), "rb") as f:
        wram = f.read()
    with open(STATE, "rb") as f:
        state = f.read()
    poses, scripts = T.image_table(wram), T.pose_scripts(wram)

    def run(b, stance, i):
        script, target, _ = scripts[i][0]
        return F.force_chain(b, state, wram, stance, i, poses[i], script, target)

    with open_mesen(PORT, ROM) as b:
        stance = F.stance_records(b, state)
        good = {i: run(b, stance, i) for i in CATALOG_KEYS}
        shifted = {i: run(b, stance, i + 1) for i in CATALOG_KEYS}          # seeded fault: off by one
        plain = F.force(b, state, stance, 15, poses[15])                     # stance method on a delta pose
    return good, shifted, plain


def test_forced_equals_catalog(forced):
    good, _, _ = forced
    for i, r in good.items():
        assert r.ok, (i, r.why)
        assert r.key == CATALOG_KEYS[i], i


def test_off_by_one_is_caught(forced):
    _, shifted, _ = forced
    assert all(r.key != CATALOG_KEYS[i] for i, r in shifted.items())


def test_stance_method_misses_delta_pose(forced):
    """Why the chain method exists: pose 15 straight from the stance keeps stale tiles."""
    _, _, plain = forced
    assert plain.ok and plain.key != CATALOG_KEYS[15]
