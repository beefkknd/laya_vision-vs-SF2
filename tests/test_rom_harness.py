"""Harness acceptance against the real ROM in a headless Mesen. Skipped unless $SF2_ROM is set.

    SF2_ROM=... pytest -q tests/test_rom_harness.py

Collection and training are only meaningful once this passes. A full passing run writes out/harness_ok.json,
which collect_teacher / play_teacher / play_student require (sf2.cli.check_harness).
"""
import argparse
import os
import re

import pytest

from sf2 import actions as A

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
    for a in ["idle", "block", "forward", "hk", "lp", "lp", "block", "jump", "idle"]:
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


def test_every_action_does_what_it_says(env):
    """From the savestate Dhalsim is ~96 px away and nothing connects for 45 frames. Chun-Li's action state per
    frame (0x0C03): 00 stand, 02 crouch, 04 jump, 0A attack. Attack lengths tell jab / fierce / short / roundhouse
    apart (the PAD mapping)."""
    from sf2.actions import ACTIONS

    got = {}
    for a in ACTIONS:
        env.reset()
        env.run_frames([[]] * 4, capture=False)
        f0 = env.f
        fs = []
        env.act(a, on_frame=fs.append)
        fs += env.run_frames([[]] * 41, capture=False)
        got[a] = (f0, fs, [f.my_state for f in fs])
    toward = lambda a: (got[a][1][-1].my_x - got[a][0].my_x) * (1 if got[a][0].facing_right else -1)  # noqa: E731
    assert set(got["idle"][2]) == {0} and toward("idle") == 0
    assert toward("forward") > 0 and toward("back") < 0 and set(got["forward"][2]) == set(got["back"][2]) == {0}
    f0, fs, st = got["jump"]
    assert 0x04 in st and min(f.my_y for f in fs) < f0.my_y - 80 and toward("jump") == 0     # straight up
    assert got["crouch"][2][:4] == [0x02] * 4 and got["block"][2][:6] == [0x02] * 6        # nothing to guard yet
    frames = {a: got[a][2].count(0x0A) for a in ("lp", "hp", "lk", "hk", "throw")}
    assert (frames["lp"], frames["hp"], frames["lk"], frames["hk"]) == (13, 30, 17, 33)
    assert frames["throw"] == 30 and 0x14 not in [f.opp_state for f in got["throw"][1]]  # out of reach: a fierce


def test_the_same_button_on_consecutive_decisions_presses_again(env):
    """Each tap is 2 frames down, 2 up, so back-to-back jabs are separate presses: a new jab starts on the first
    decision after the last one ends (13 frames). Held down, the button would give one jab."""
    env.reset()
    env.run_frames([[]] * 4, capture=False)
    st = []
    for _ in range(8):
        env.act("lp", on_frame=lambda f: st.append(f.my_state))
    starts = [i for i in range(len(st)) if st[i] == 0x0A and (i == 0 or st[i - 1] != 0x0A)]
    assert starts == [0, 16]


def test_block_guards_on_the_right_side_too(env):
    """After the fighters swap sides, block is down + right: Dhalsim's attacks are guarded (08, then 0E block
    stun) and cost no life."""
    env.reset()
    for _ in range(200):
        env.act("forward")
        if not env.f.facing_right:
            break
    assert not env.f.facing_right
    hp, st = env.f.my_hp, []
    for _ in range(200):
        env.act("block", on_frame=lambda f: st.append(f.my_state))
    guarded = [i for i in range(1, len(st)) if st[i] == 0x0E and st[i - 1] == 0x08]
    assert len(guarded) >= 3 and env.f.my_hp == hp


def _hold_each(env, runs, extra=()):
    """RAM rows (the map's variables, then ``extra``) of each named input sequence from the current state; the
    state is restored afterwards."""
    from sf2.ram import load_map

    state = env.backend.save_state()
    ram_map = load_map(RAM_MAP)
    env.backend.set_vars(ram_map + list(extra))
    try:
        out = {}
        for name, frames in runs.items():
            env.backend.load_state(state)
            out[name] = env.backend.run(frames).rams[1:]
    finally:
        env.backend.set_vars(ram_map)
        env.backend.load_state(state)
    return out


def test_in_block_stun_only_down_does_anything(env):
    """Block stun is state 0E like hit stun, with reaction 06 / 08 at 0x0C4A. Holding down in it switches her to a
    crouching guard (+0x43: 1 standing, 2 crouching) before it ends, so it is controllable; nothing else changes
    her until it ends."""
    from sf2.ram import Var

    env.reset()
    for _ in range(100):
        env.act("block")
        if env.f.my_state == 0x0E:
            break
    assert env.f.my_state == 0x0E and env.f.my_react in (6, 8) and env.f.my_hp == FULL
    assert env.controllable() and env.text().split()[1] == "block"
    inputs = ([], ["right"], ["left"], ["up"], ["y"], ["r"], ["down"], ["down", "left"], ["down", "right"])
    runs = _hold_each(env, {tuple(h): [h] * 150 for h in inputs}, [Var("guard", 0x0C43, 1, False)])
    ref, col = runs[()], env.names.index("my_state")
    end = next(i for i, r in enumerate(ref) if r[col] != 0x0E)
    assert end > 10
    assert all(r[:end] == ref[:end] for held, r in runs.items() if "down" not in held)
    assert all(any(a[-1] == 2 != b[-1] for a, b in zip(r[:end], ref[:end])) for held, r in runs.items() if "down" in held)


def _close_plan(env, seed, until):
    """record_trace.py's close plan (fixtures/close is seed 1, fixtures/win seed 13) until ``until(env)``."""
    import random

    rng = random.Random(seed)
    env.reset()
    for _ in range(3000):
        res = env.act("forward" if env.f.dx > 30 else rng.choice(["idle", "idle", "crouch", "lp", "forward"]))
        if until(env):
            return True
        if res.round_over and not env.next_round():
            return False
    return False


HELD = ([], ["right"], ["left"], ["up"], ["down"], ["down", "left"], ["y"], ["l"], ["r"], ["b"])
MASH = [[["y"], []][k % 2] if k % 4 < 2 else [["left"], ["right"]][k % 2] for k in range(400)]  # jab, left, right


def test_the_stick_does_nothing_while_she_is_held_up_or_thrown(env):
    """Dhalsim's throw lifts her to y 136 in state 00, then throws her in state 14 (seed 1: at frame ~323)."""
    assert _close_plan(env, 1, lambda e: e.f.my_state == 0 and e.f.my_y < 185)
    assert not env.controllable()
    runs = _hold_each(env, dict({tuple(h): [h] * 250 for h in HELD}, mash=MASH[:250]))
    ref, st, y = runs[()], env.names.index("my_state"), env.names.index("my_y")
    end = next(i for i, r in enumerate(ref) if not (r[st] == 0x14 or (r[st] == 0 and r[y] < 185)))
    assert end > 80 and any(r[st] == 0x14 for r in ref[:end])
    assert all(r[:end] == ref[:end] for r in runs.values())
    assert any(r[end + 8] != ref[end + 8] for r in runs.values())       # afterwards the stick works again


def test_mashing_shortens_a_dizzy(env):
    """Dizzy is state 0E, sub-state 08 with the flag at 0x0C89 (seed 13: at frame ~2095). Holding any one input
    changes nothing; mashing gets her out far sooner, so a dizzy is controllable."""
    assert _close_plan(env, 13, lambda e: e.text().split()[1] == "dizzy")
    assert env.controllable()
    runs = _hold_each(env, dict({tuple(h): [h] * 400 for h in HELD}, mash=MASH))
    st = env.names.index("my_state")
    end = {k: next(i for i, r in enumerate(r) if r[st] != 0x0E) for k, r in runs.items()}
    assert end[()] > 60 and all(e == end[()] for k, e in end.items() if k != "mash")
    assert end["mash"] < end[()] / 2


def _to_other_side(env):
    for _ in range(200):                # walk through Dhalsim
        env.act("forward")
        if not env.f.facing_right:
            return
    raise AssertionError("never got past Dhalsim")


def test_jump_forward_leaves_the_ground_toward_dhalsim_on_both_sides_and_lands(env):
    """Up + toward him: in the air for about 45 frames, rising 90+ px and travelling ~90 px toward him; after the
    fighters swap sides it still goes toward him. A kick tapped in it is an air attack: the ROM keeps state 04 and
    moves the sub-state from 02 to 06 (not 0A, which is Dhalsim's air attack); the note says jumpattack."""
    for side in ("left", "right"):
        env.reset()
        env.run_frames([[]] * 4, capture=False)
        if side == "right":
            _to_other_side(env)
        for _ in range(100):            # back off until she stands free at mid range
            env.act("back")
            if env.f.dx >= 110 and env.f.my_state == 0 and env.f.opp_state == 0 and env.controllable():
                break
        f0, fs = env.f, []
        env.act("jump_forward", on_frame=fs.append)
        for _ in range(3):
            env.act("idle", on_frame=fs.append)
        env.act("hk", on_frame=fs.append)
        words = [env.text().split()[1]]
        for _ in range(12):
            env.act("idle", on_frame=fs.append)
            words.append(env.text().split()[1])
        assert "jumpattack" in words, (side, words)
        toward = 1 if f0.facing_right else -1
        assert (fs[24].my_x - f0.my_x) * toward > 25, side                  # rising, before anything connects
        assert min(f.my_y for f in fs) < f0.my_y - 80, side
        assert any(f.my_state == 0x04 and f.my_sub == 0x06 and f.my_y < f0.my_y - 6 for f in fs), side
        assert not any(f.my_state == 0x0A for f in fs), side
        assert fs[-1].my_y == f0.my_y, side                                # landed (sub-state 04), maybe across him


def test_crouch_decisions_hold_the_crouch(env):
    env.reset()
    env.run_frames([[]] * 4, capture=False)
    st = []
    for _ in range(10):
        env.act("crouch", on_frame=lambda f: st.append(f.my_state))
    assert set(st) == {0x02}


def test_block_is_a_crouching_guard_and_back_a_standing_one(env):
    """block (down + back) guards Dhalsim's standing and low attacks with the crouching guard (+0x43 = 2) and loses
    at most a Yoga Fire's chip over 20 s. back guards with the standing guard (+0x43 = 1); lows get through it."""
    from sf2.ram import Var

    env.reset()
    runs = _hold_each(env, {"block": [["down", "left"]] * 1200, "back": [["left"]] * 1200},
                      [Var("guard", 0x0C43, 1, False)])
    st, life = env.names.index("my_state"), env.names.index("my_life")
    for held, guard in (("block", 2), ("back", 1)):
        r = runs[held]
        entries = [i for i in range(1, len(r)) if r[i][st] == 0x08 and r[i - 1][st] != 0x08]
        assert len(entries) >= 3 and all(r[i][-1] == guard for i in entries), held
    assert FULL - runs["block"][-1][life] <= 8
    assert FULL - runs["back"][-1][life] > FULL - runs["block"][-1][life]


def test_every_action_near_both_walls(env):
    """Cornered at either wall, every action keeps her inside the walls and facing Dhalsim; forward and
    jump_forward go toward him, back goes nowhere."""
    from sf2.actions import ACTIONS
    from sf2.ram import LEFT_WALL, RIGHT_WALL

    for side in ("left", "right"):
        env.reset()
        if side == "right":
            _to_other_side(env)
        assert _back_until_stopped(env) in (LEFT_WALL, RIGHT_WALL) and env.f.my_cornered, side
        state, f0 = env.backend.save_state(), env.f
        toward = 1 if f0.facing_right else -1
        for a in ACTIONS:
            env.backend.load_state(state)
            env.f = f0
            fs = []
            env.act(a, on_frame=fs.append)
            fs += env.run_frames([[]] * 12, capture=False)
            assert all(LEFT_WALL <= f.my_x <= RIGHT_WALL for f in fs), (side, a)
            assert all(f.facing_right == f0.facing_right for f in fs), (side, a)
            moved = (fs[11].my_x - f0.my_x) * toward
            if a in ("forward", "jump_forward"):
                assert moved > 0, (side, a, moved)
            elif a == "back":
                assert moved == 0, (side, a, moved)


def test_lightning_legs_need_ten_kick_decisions_in_a_row(env):
    """Kick taps build Chun-Li's Lightning Legs (state 0C): nine lk or hk decisions in a row do not start them,
    eleven do. Mixed kick patterns start them sooner (lk, lk, idle repeated; lk and hk alternating)."""
    def legs(seq):
        env.reset()
        env.run_frames([[]] * 4, capture=False)
        st = []
        for a in seq:
            env.act(a, on_frame=lambda f: st.append(f.my_state))
        st += [f.my_state for f in env.run_frames([[]] * 30, capture=False)]
        return st.count(0x0C)

    for b in ("lk", "hk"):
        assert legs([b] * 9) == 0 and legs([b] * 11) > 0, b
    assert legs(["lk", "hk"] * 10) > 0


def _up_close(env, reach=38):
    """Walk in until Dhalsim is within ``reach`` px, both on the ground, him standing or crouching."""
    for _ in range(300):
        f = env.f
        if (f.dx <= reach and not any(env.airborne()) and env.controllable() and f.my_state in (0, 2)
                and f.opp_state in (0, 2) and f.opp_y == f.my_y):
            return
        env.act("forward" if f.dx > reach else "idle")
    raise AssertionError("never got up close")


def test_throw_throws_him_up_close_on_both_sides(env):
    """Toward + fierce within 42 px: she holds him (0A, 61 frames), he is thrown (14) ~31 frames after the press and
    loses 46 life, landing further away on the same side. Back + fierce throws him behind her: the sides swap."""
    for side in ("left", "right"):
        env.reset()
        env.run_frames([[]] * 4, capture=False)
        if side == "right":
            _to_other_side(env)
        _up_close(env)
        state, f0 = env.backend.save_state(), env.f
        for arm in ("throw", "back throw"):
            env.backend.load_state(state)
            env.f = f0
            fs = []
            if arm == "throw":
                env.act("throw", on_frame=fs.append)
            else:
                fs += env.run_frames([A.to_physical(t, f0.facing_right) for t in [("B", "hp")] * 2 + [("B",)] * 2],
                                     capture=False)
            fs += env.run_frames([[]] * 70, capture=False)
            assert any(f.opp_state == 0x14 for f in fs), (side, arm)
            assert f0.life[1] - fs[-1].life[1] == 46, (side, arm)
            assert fs[-1].facing_right == (f0.facing_right if arm == "throw" else not f0.facing_right), (side, arm)
