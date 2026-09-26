"""Harness acceptance against the real ROM in a headless Mesen. Skipped unless $SF2_ROM is set.

    SF2_ROM=... pytest -q tests/test_rom_harness.py

Collection and training are only meaningful once this passes.
"""
import argparse
import os
import re

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAM_MAP = os.path.join(ROOT, "ram_maps", "sf2_snes.txt")
SAVESTATE = os.path.join(ROOT, "states", "chunli_vs_dhalsim.state")
PORT = 47960
FULL = 176
GROUND_Y = 192

pytestmark = pytest.mark.skipif(not os.environ.get("SF2_ROM"), reason="needs $SF2_ROM (and Mesen)")


@pytest.fixture(scope="module")
def env():
    from sf2.cli import make_env

    args = argparse.Namespace(port=PORT, launch=None, headless=True, rom=os.environ["SF2_ROM"],
                              mesen=os.environ.get("SF2_MESEN"), capture="auto", seed=0, jitter=0,
                              savestate=SAVESTATE, ram_map=RAM_MAP, me="chunli", opp="dhalsim")
    e = make_env(args)
    yield e
    e.close()


def test_rom_is_the_one_the_ram_map_was_made_for(env):
    want = re.search(r"SHA1 ([0-9A-F]{40})", open(RAM_MAP).read()).group(1)
    assert env.backend.rom_sha1 == want


def test_savestate_starts_with_full_life_on_the_ground(env):
    env.reset()
    f = env.f
    assert (f.my_hp, f.opp_hp) == (FULL, FULL)
    assert (f.my_y, f.opp_y) == (GROUND_Y, GROUND_Y)
    assert env.airborne() == (False, False)


def test_idle_is_one_four_frame_decision(env):
    env.reset()
    res = env.act("idle")
    assert res.frames == 4 and env.frame_no == 4 and env.frame.shape == (224, 256, 3)
