"""Harness acceptance against the real ROM in a headless Mesen. Skipped unless $SF2_ROM is set.

    SF2_ROM=... pytest -q tests/test_rom_harness.py

Collection and training are only meaningful once this passes. A full passing run writes out/harness_ok.json,
which collect_teacher / play_teacher / play_student require (sf2.cli.check_harness).
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
                              mesen=os.environ.get("SF2_MESEN"), capture="auto", seed=0, jitter=0, jitter_base=0,
                              savestate=SAVESTATE, ram_map=RAM_MAP, me="chunli", opp="dhalsim")
    e = make_env(args, verified=False)
    yield e
    e.close()


@pytest.fixture(scope="module", autouse=True)
def stamp(request, env):
    """Every check in this module ran and none failed: collection and play may use this harness."""
    from sf2.cli import write_harness_stamp

    failed = request.session.testsfailed
    yield
    ran = [i for i in request.session.items if i.module is request.module]
    every = [n for n in dir(request.module) if n.startswith("test_")]
    if request.session.testsfailed == failed and len(ran) == len(every):
        write_harness_stamp(env.backend.rom_sha1, RAM_MAP)


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


def test_forward_closes_and_back_opens_the_distance_on_both_sides(env):
    env.reset()
    dx0 = env.f.dx
    for _ in range(10):
        env.act("forward")
    assert env.f.dx < dx0 - 40
    dx1 = env.f.dx
    for _ in range(10):
        env.act("back")
    assert env.f.dx > dx1 + 10
    side = env.f.facing_right
    for _ in range(150):                # walk through the opponent to the other side
        env.act("forward")
        if env.f.facing_right != side:
            break
    assert env.f.facing_right != side
    env.run_frames([[]] * 30)
    dx2 = env.f.dx
    for _ in range(10):
        env.act("back")
    assert env.f.dx > dx2 + 10


def test_an_idle_match_is_two_lost_rounds_and_every_round_starts_controllable(env):
    env.reset()
    winners = []
    while True:
        x0 = env.f.my_x
        res = env.act("forward")
        assert env.f.my_x != x0, "round %d: the first decision did not move" % env.round
        while not res.round_over:
            res = env.act("idle")
        winners.append(res.winner)
        if not env.next_round():
            break
    assert winners == ["opp", "opp"]


def test_block_guards_the_attack_that_hits_an_idle_fighter(env):
    env.reset()
    while env.f.my_hp == FULL:                      # when does Dhalsim first hit an idle Chun-Li?
        env.run_frames([[]], capture=False)
    first_hit = env.frame_no
    env.reset()
    env.run_frames([[]] * (first_hit - 40), capture=False)
    attacked = False
    while env.frame_no < first_hit + 60:
        env.act("block")
        attacked |= env.f.opp_state == 0x0A
    if not attacked:
        pytest.xfail("Dhalsim did not attack while Chun-Li blocked")
    assert env.f.my_hp == FULL


def _back_until_stopped(env, limit=300):
    """``back`` until world x has not changed for 8 decisions of plain walking (state 00). About 212 px apart the
    screen edge stops her too, so stops there don't count: Dhalsim walks in again and she carries on."""
    xs = []
    for _ in range(limit):
        env.act("back")
        xs = xs + [env.f.my_x] if env.f.my_state == 0 and env.f.dx < 200 else []
        if len(xs) >= 8 and len(set(xs[-8:])) == 1:
            return xs[-1]
    return None


def test_walking_back_stops_at_the_stage_walls(env):
    from sf2.ram import LEFT_WALL, RIGHT_WALL

    env.reset()
    assert env.f.facing_right
    assert _back_until_stopped(env) == LEFT_WALL and env.f.my_cornered
    for _ in range(200):                # through Dhalsim to the other side
        env.act("forward")
        if not env.f.facing_right:
            break
    assert not env.f.facing_right
    assert _back_until_stopped(env) == RIGHT_WALL and env.f.my_cornered


def test_screenshots_are_deterministic_and_show_every_frame(env):
    """Mesen skips rendering frames when it runs fast, which made screenshots stale and different run to run."""
    import numpy as np

    runs = []
    for _ in range(2):
        env.reset()
        obs = env.backend.run([["up"]] * 4 + [[]] * 56, range(61))
        runs.append(np.stack([obs.images[i] for i in range(61)]))
    assert (runs[0] == runs[1]).all()
    assert all((runs[0][i] != runs[0][i - 1]).any() for i in range(10, 50))   # in the air: every frame differs


def test_raw_capture_gives_the_same_image_as_png(env):
    import numpy as np

    env.reset()
    png = env.backend.run([[]] * 30, [30]).images[30]
    env.backend.set_capture("raw")
    try:
        env.reset()
        raw = env.backend.run([[]] * 30, [30]).images[30]
    finally:
        env.backend.set_capture("png")
    assert raw.shape == png.shape == (224, 256, 3) and np.array_equal(raw, png)


def test_prev_is_four_frames_before_cur_after_resets_macros_and_round_starts(env):
    """Replay the same inputs with a screenshot on every frame: each decision's images are frames n-4 and n."""
    import numpy as np

    from sf2 import actions as A

    env.reset()
    assert np.array_equal(env.prev_frame, env.frame)                 # nothing before the savestate
    seen, inputs = [], []
    for a in ["idle", "block", "forward", "hadouken", "lp", "lp", "shoryuken", "jump", "idle"]:
        inputs += [A.to_physical(t, env.f.facing_right) for t in A.expand(a)]
        env.act(a)
        seen.append((env.frame_no, env.prev_frame, env.frame))
    env.reset()
    obs = env.backend.run(inputs, range(len(inputs) + 1))
    for n, prev, cur in seen:
        assert np.array_equal(cur, obs.images[n]) and np.array_equal(prev, obs.images[n - 4]), n

    env.reset()                                                      # an idle round, then round 2's first image
    while not env.act("idle").round_over:
        pass
    assert env.next_round()
    n, prev, cur = env.frame_no, env.prev_frame, env.frame
    env.reset()
    obs = env.backend.run([[]] * n, [n - 4, n])
    assert np.array_equal(cur, obs.images[n]) and np.array_equal(prev, obs.images[n - 4])


def test_yoga_fire_is_read_from_the_projectile_slot(env):
    import random

    rng = random.Random(1)                          # record_trace.py --plan fireball --seed 1: one in round 2
    env.reset()
    for _ in range(600):
        if env.act(rng.choice(["back", "back", "idle", "block", "jump", "forward"])).round_over:
            assert env.next_round()
        if env.f.fireball:
            break
    assert env.f.fireball, "no Yoga Fire within 600 decisions"
    toward = 1 if env.f.my_x > env.f.opp_x else -1
    assert 30 < (env.f.fireball_x - env.f.opp_x) * toward < 80
    fs = env.run_frames([[]] * 12, capture=False)
    assert all(f.fireball for f in fs)
    assert all(2 <= (b.fireball_x - a.fireball_x) * toward <= 4 for a, b in zip(fs, fs[1:]))


def test_the_stick_does_nothing_while_she_is_hit(env):
    """From the first frame of a hit, every input gives the same RAM for as long as she stays in state 0E."""
    env.reset()
    while env.controllable():
        env.act("idle")
    state = env.backend.save_state()
    col = env.names.index("my_state")
    runs = []
    for held in (["right"], ["left"], ["up"], ["down"], ["down", "left"], ["y"], []):
        env.backend.load_state(state)
        runs.append(env.backend.run([held] * 120).rams[1:])
    hit = next(i for i, r in enumerate(runs[-1]) if r[col] != 0x0E)   # leaves the hit state (idle reference)
    assert hit > 10
    assert all(run[:hit] == runs[-1][:hit] for run in runs)
    assert any(run[hit + 8] != runs[-1][hit + 8] for run in runs)      # afterwards the stick works again


def test_headless_mesen_outlives_a_long_collection(env):
    """--timeout is the test runner's total wall-clock limit: Mesen exits mid-run when it passes, keep_alive or not."""
    timeout = [a for a in env.backend.proc.args if a.startswith("--timeout=")]
    assert timeout and int(timeout[0].split("=")[1]) >= 7 * 24 * 3600
