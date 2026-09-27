"""The ROM traces in tests/fixtures replay end to end (recorded with scripts/record_trace.py)."""
import os

import pytest

from sf2.env import FightEnv
from trace_mesen import TraceMesen, fixture_map, load

FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures")
ROM_SHA1 = "7DDCB96E0D9FEA94D9370635262AC7C28DA85214"
MAP = fixture_map(os.path.join(os.path.dirname(FIXTURES), "..", "ram_maps", "sf2_snes.txt"))


@pytest.mark.parametrize("name", ["walk", "facing", "start", "ko_round2", "timeover", "walls", "fireball"])
def test_fixture_replays_its_recorded_inputs(name):
    t = load(os.path.join(FIXTURES, name + ".jsonl.gz"))
    assert t["header"]["rom_sha1"] == ROM_SHA1 and t["header"]["frames"] == len(t["rows"]) - 1
    env = FightEnv(TraceMesen(t), MAP, b"")
    env.reset()
    fs = env.run_frames([r["in"] or [] for r in t["rows"][1:]], capture=False)
    assert len(fs) == t["header"]["frames"]
