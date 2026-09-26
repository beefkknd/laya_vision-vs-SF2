"""Harness behaviour checked against ROM traces (tests/fixtures, recorded by scripts/record_trace.py).

The env runs with the real RAM map in ram_maps/sf2_snes.txt. The facing trace was recorded pressing the
direction that pointed at the opponent's world position, so replaying it through ``env.act("forward")``
fails on the first frame the env's idea of facing is wrong.
"""
import os

from sf2.env import FightEnv
from sf2.ram import load_map
from trace_mesen import TraceMesen, load

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MAP = load_map(os.path.join(ROOT, "ram_maps", "sf2_snes.txt"))


def _env(name):
    t = load(os.path.join(ROOT, "tests", "fixtures", name + ".jsonl.gz"))
    env = FightEnv(TraceMesen(t), MAP, b"", me="chunli", opp="dhalsim")
    env.reset()
    return env, t


def test_walk_across_the_stage_brings_the_fighters_together_and_past_each_other():
    env, t = _env("walk")
    fs = env.run_frames([r["in"] for r in t["rows"][1:]], capture=False)
    assert fs[0].dx > 150                          # the savestate starts about 180 px apart
    assert min(f.dx for f in fs) < 20              # Chun-Li walks through Dhalsim's position
    assert fs[-1].dx < 60                          # and ends next to him
    assert max(abs(b.opp_x - a.opp_x) for a, b in zip(fs, fs[1:])) <= 12   # positions never jump


def test_facing_flips_once_where_the_fighters_cross():
    env, t = _env("walk")
    fs = env.run_frames([r["in"] for r in t["rows"][1:]], capture=False)
    flips = [i for i in range(1, len(fs)) if fs[i].facing_right != fs[i - 1].facing_right]
    assert len(flips) == 1
    assert fs[flips[0]].dx < 20


def test_forward_and_back_press_toward_and_away_from_the_opponent():
    env, t = _env("facing")
    n = len(t["rows"]) - 1
    dx0 = env.f.dx
    for _ in range(8):
        env.act("forward")
    assert env.f.dx < dx0 - 40                     # forward closes the distance (about 9 px a decision)
    dx1 = env.f.dx
    for _ in range(8):
        env.act("back")
    assert env.f.dx > dx1 + 10                     # back opens it again (slower)
    while env.frame_no < n - 64 - 90:              # the recorder idled until the first hit, then 90 frames
        env.act("idle")
    env.run_frames([[]] * 90, capture=False)
    for _ in range(8):                             # still knocked down here; the inputs must still match
        env.act("forward")
    for _ in range(8):
        env.act("back")
    assert env.frame_no == n


def test_ground_level_comes_from_the_savestate_not_the_end_of_the_start_jitter():
    from trace_mesen import make_trace

    rows = [{"my_hp": 176, "opp_hp": 176, "my_x": 200, "opp_x": 384, "my_y": 192,
             "opp_y": 140 if 1 <= i <= 40 else 192} for i in range(101)]      # Dhalsim mid-jump early on
    env = FightEnv(TraceMesen(make_trace(MAP, rows)), MAP, b"", seed=0, jitter=20)
    env.reset()                                    # the jitter idles 1..20 frames, while he is in the air
    for _ in range(12):
        env.act("idle")                            # he has landed by now
    assert env.airborne() == (False, False)
