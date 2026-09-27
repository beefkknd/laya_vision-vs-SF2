"""Per-character moves on the real ROM, from both facings (docs/TWO_SYSTEM_PLAN.md P2 / P3). Skipped unless $SF2_ROM.

    SF2_ROM=... pytest -q tests/test_rom_moves.py

Each process starts its own headless Mesen on a free port and writes one record file per verified move
(sf2.verified), so the suite can run as several processes at once, e.g. one per facing:

    SF2_ROM=... pytest -q tests/test_rom_moves.py -k right & SF2_ROM=... pytest -q tests/test_rom_moves.py -k left

Player 1 is the character under test, the CPU is player 2. The CPU attacks, walks in and throws, so every check
first finds a moment where the character stands free at the right distance, saves the state there, and runs the
inputs from that state; a run in which the CPU hit or threw the character first proves nothing and is tried again
at a later moment. The other facing: walk forward through the CPU (sf2.ram.Fighters.facing_right flips), then only
use moments where the ROM's own facing byte (my_facing) agrees.
"""
import os

import pytest

from sf2 import actions as A
from sf2 import ram

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAM_MAP = os.path.join(ROOT, "ram_maps", "sf2_snes.txt")
# character -> (fight-start savestate, CPU opponent). Add a character here with its specials' checks below.
FIGHTS = {"ryu": ("p1_ryu_vs_ken.state", "ken")}
FACINGS = ("right", "left")

pytestmark = pytest.mark.skipif(not os.environ.get("SF2_ROM"), reason="needs $SF2_ROM (and Mesen)")


@pytest.fixture(scope="module", params=sorted(FIGHTS))
def env(request):
    import argparse

    from sf2.cli import make_env

    state, opp = FIGHTS[request.param]
    args = argparse.Namespace(port=_free_port(), launch=None, headless=True, rom=os.environ["SF2_ROM"],
                              mesen=os.environ.get("SF2_MESEN"), capture="auto", seed=0, jitter=0, jitter_base=0,
                              savestate=os.path.join(ROOT, "states", state), ram_map=RAM_MAP, me=request.param, opp=opp)
    e = make_env(args, verified=False)
    yield e
    e.close()


def _free_port():
    import socket

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _to_other_side(env):
    for _ in range(200):
        env.act("forward")
        if not env.f.facing_right:
            return
    raise AssertionError("never got past the opponent")


def _setup(env, facing, k=0):
    """From the savestate, idle a different number of frames each time (``k``: the CPU plays differently), then
    for ``left`` walk through the CPU."""
    env.reset()
    env.run_frames([[]] * (4 + 9 * k), capture=False)
    if facing == "left":
        _to_other_side(env)


def _free(env, facing):
    """Standing or crouching on the ground, in control, facing ``facing`` by x and by the ROM, no attack or
    projectile on the way."""
    f = env.f
    return (env.controllable() and not any(env.airborne()) and f.my_state in (0, 2) and f.opp_state in (0, 2)
            and f.facing_right == (facing == "right") and f.game_facing_right == f.facing_right
            and not f.fireball and not f.my_fireball)


def _run(env, frames):
    """Fighters after each of ``frames`` (physical buttons) from the current state; the state is restored."""
    state = env.backend.save_state()
    try:
        return [env._f(r) for r in env.backend.run(frames).rams[1:]]
    finally:
        env.backend.load_state(state)


def _physical(tokens, facing_right):
    return [A.to_physical(t, facing_right) for t in tokens]


def _clean(f0, fs):
    """The CPU did not hit, throw or hold the character during the run."""
    return all(f.my_state not in (ram.HIT_STATE, ram.THROWN_STATE) and f.my_life == f0.my_life for f in fs)


def _find(env, facing, lo, hi, attempt, limit=400):
    """At each moment the character stands free facing ``facing`` with the opponent ``lo``..``hi`` px away,
    ``attempt(f0)`` runs inputs from there and returns a verdict, or None when the CPU interfered. Returns the first
    verdict."""
    k = 0
    _setup(env, facing)
    for _ in range(limit):
        f = env.f
        if f.facing_right != (facing == "right") or not env.in_round:
            k += 1
            _setup(env, facing, k)
            continue
        if _free(env, facing) and lo <= f.dx <= hi:
            verdict = attempt(f)
            if verdict is not None:
                return verdict
            res = env.act("idle")
        else:
            res = env.act("back" if f.dx < lo else "forward" if f.dx > hi else "idle")
        if res.round_over:
            k += 1
            _setup(env, facing, k)
    raise AssertionError("no free moment facing %s at %d-%d px" % (facing, lo, hi))


def _toward(f0, f):
    return (f.my_x - f0.my_x) * (1 if f0.facing_right else -1)


# ---------------------------------------------------------------------------------------------------------- P2: RAM
# Raw motions (relative tokens, 2 frames per direction, the button on the last one), independent of sf2.actions.
def _motion(dirs, button):
    return [d for d in dirs[:-1] for _ in range(2)] + [dirs[-1] + (button,)] * 2 + [()] * 2


RYU_MOTIONS = {  # docs/MOVES.md, fierce / roundhouse
    "hadoken": _motion([("D",), ("D", "F"), ("F",)], "hp"),
    "shoryuken": _motion([("F",), ("D",), ("D", "F")], "hp"),
    "tatsumaki": _motion([("D",), ("D", "B"), ("B",)], "hk"),
}
RYU_SPECIAL_IDS = {"hadoken": 0x00, "tatsumaki": 0x02, "shoryuken": 0x04}


@pytest.mark.parametrize("facing", FACINGS)
def test_each_special_has_its_own_id_and_the_hadoken_uses_player_1s_projectile_slot(env, facing):
    """Every Ryu special is state 0C, with 0x0D80 telling which (00 Hadoken, 02 Hurricane Kick, 04 Shoryuken). The
    Hadoken puts a projectile in player 1's slot (0x1000, x at 0x1007) in front of him within 14 frames of the
    press, travelling away from him ~3 px a frame (it stops on the frame it hits). (Player 2's slot 0x1050 holds the
    CPU's own Hadoken, which can be on screen at the same time: checked by hand, 2026-09-27.)"""
    if env.me != "ryu":
        pytest.skip("Ryu's ids")

    def attempt(f0):
        runs = {m: _run(env, _physical(t, f0.facing_right) + [[]] * 40) for m, t in RYU_MOTIONS.items()}
        if not all(_clean(f0, fs[:30]) for fs in runs.values()):
            return None
        return f0, runs

    f0, runs = _find(env, facing, 90, 200, attempt)
    for m, fs in runs.items():
        sp = [f.my_special for f in fs if f.my_state == ram.SPECIAL_STATE]
        assert len(sp) > 10 and set(sp) == {RYU_SPECIAL_IDS[m]}, (facing, m, sp)
    fb = [f for f in runs["hadoken"] if f.my_fireball]
    first = runs["hadoken"].index(fb[0])
    toward = 1 if f0.facing_right else -1
    xs = [(f.my_fireball_x - f0.my_x) * toward for f in fb]
    assert first <= 4 + 14 and len(fb) >= 2, (facing, first, xs)
    steps = [b - a for a, b in zip(xs, xs[1:])]
    assert all(x > 0 for x in xs) and all(0 <= d <= 4 for d in steps) and any(d >= 2 for d in steps), (facing, xs)
    assert not any(f.my_fireball for m in ("shoryuken", "tatsumaki") for f in runs[m]), facing


@pytest.mark.parametrize("facing", FACINGS)
def test_attack_result_reads_hit_blocked_and_whiffed(env, facing):
    """ram.attack_result against what the ROM shows independently: a hit costs him life and he was not guarding
    (08) the frame before; blocked means he was guarding, then block stun, losing at most a special's chip (12); a
    whiff costs him nothing and he never enters 0E. Ryu's three specials and a fierce, at many moments."""
    if env.me != "ryu":
        pytest.skip("Ryu's moves")
    moves = dict(RYU_MOTIONS, hp=[("hp",)] * 2 + [()] * 2)
    seen = {"hit": 0, "blocked": 0, "whiffed": 0}
    k = 0
    _setup(env, facing)
    for step in range(600):
        f0 = env.f
        if f0.facing_right != (facing == "right") or not env.in_round:
            k += 1
            _setup(env, facing, k)
            continue
        if _free(env, facing):
            for m, tokens in moves.items():
                fs = _run(env, _physical(tokens, f0.facing_right) + [[]] * 60)
                end = next((i for i in range(len(tokens), len(fs)) if fs[i].my_state not in (0x0A, 0x0C)
                            and not fs[i].my_fireball), None)
                if end is None or not _clean(f0, fs[:end]):
                    continue
                window = [f0] + fs[:end]
                result = ram.attack_result(window)
                lost = f0.opp_life - window[-1].opp_life
                entry = next((i for i in range(1, len(window)) if window[i].opp_state == ram.HIT_STATE
                              and window[i - 1].opp_state != ram.HIT_STATE), None)
                if result == "hit":
                    assert lost > 0 and window[entry - 1].opp_state != 0x08, (facing, m, step, lost)
                elif result == "blocked":
                    assert window[entry - 1].opp_state == 0x08 and 0 <= lost <= 12, (facing, m, step, lost)
                else:
                    assert lost == 0 and entry is None, (facing, m, step, lost)
                seen[result] += 1
        res = env.act(["idle", "back", "forward", "idle", "forward"][step % 5])
        if res.round_over:
            k += 1
            _setup(env, facing, k)
        if min(seen.values()) >= 3:
            break
    assert min(seen.values()) >= 3, (facing, seen)


def test_the_rom_facing_byte_agrees_with_x_when_standing_apart(env):
    """0x0CF4 is 40 while he faces right and 00 while he faces left. It lags the x positions (in the air, guarding,
    in hit stun, turning round), but standing or crouching at least 30 px apart it always agrees. Seeded play."""
    import random

    rng = random.Random(3)
    env.reset()
    seen = {True: 0, False: 0}
    for _ in range(1500):
        f = env.f
        if env.in_round and f.my_state in (0, 2) and not any(env.airborne()) and f.dx >= 30:
            assert f.my_facing in (0x00, 0x40) and f.game_facing_right == f.facing_right, (f.my_x, f.opp_x, f.my_facing)
            seen[f.facing_right] += 1
        if env.act(rng.choice(["forward", "forward", "back", "idle", "jump_forward", "crouch"])).round_over:
            if not env.next_round():
                env.reset()
    assert min(seen.values()) > 20, seen


# ------------------------------------------------------------------------------------------------ P3: the move macros
# Far standing normals: frames in the attack state 0A (measured on the ROM, the opponent 85+ px away). The button
# mapping shows in them: jab / fierce / short / roundhouse all differ.
ATTACK_FRAMES = {"ryu": {"lp": 13, "hp": 36, "lk": 21, "hk": 33}}


def _run_length(fs, state):
    """Frames in the first run of ``state``."""
    i = next((i for i, f in enumerate(fs) if f.my_state == state), None)
    if i is None:
        return 0
    return next((j for j, f in enumerate(fs[i:]) if f.my_state != state), len(fs) - i)


def _rise(f0, fs):
    return f0.my_y - min(f.my_y for f in fs)


def _special(fs, sid):
    return [f for f in fs if f.my_state == ram.SPECIAL_STATE and f.my_special == sid]


def _attack(button):
    return lambda c, f0, fs: _run_length(fs, 0x0A) == ATTACK_FRAMES[c][button]


# move -> (opponent lo..hi px away, frames watched after the macro, effect(character, f0, frames) -> bool, a wrong
# input). The effect is what docs/MOVES.md defines, read from RAM; the wrong input must not produce it.
CHECKS = {
    "idle": (80, 200, 0, lambda c, f0, fs: all(f.my_state == 0 and f.my_x == f0.my_x for f in fs), [("F",)] * 4),
    "forward": (80, 200, 0, lambda c, f0, fs: _toward(f0, fs[-1]) > 0, [("B",)] * 4),
    "back": (80, 200, 0, lambda c, f0, fs: _toward(f0, fs[-1]) < 0, [("F",)] * 4),
    "jump": (80, 200, 40, lambda c, f0, fs: any(f.my_state == ram.JUMP_STATE for f in fs) and _rise(f0, fs) > 60
             and _toward(f0, fs[30]) == 0, [("U", "F")] * 4),
    "jump_forward": (80, 200, 40, lambda c, f0, fs: _rise(f0, fs) > 60 and _toward(f0, fs[20]) > 8, [("U",)] * 4),
    "crouch": (80, 200, 0, lambda c, f0, fs: all(f.my_state == 0x02 for f in fs), [()] * 4),
    "lp": (85, 200, 40, _attack("lp"), [("hp",)] * 2 + [()] * 2),
    "hp": (85, 200, 40, _attack("hp"), [("lp",)] * 2 + [()] * 2),
    "lk": (85, 200, 40, _attack("lk"), [("hk",)] * 2 + [()] * 2),
    "hk": (85, 200, 40, _attack("hk"), [("lk",)] * 2 + [()] * 2),
    # toward + fierce up close throws him (state 14); fierce alone does not
    "throw": (0, 36, 70, lambda c, f0, fs: any(f.opp_state == ram.THROWN_STATE for f in fs), [("hp",)] * 2 + [()] * 2),
    # down + roundhouse knocks him down (0E, sub-state 04) when it connects; the standing roundhouse does not
    "sweep": (50, 68, 60, lambda c, f0, fs: any(f.opp_state == ram.HIT_STATE and f.opp_sub == 0x04 for f in fs),
              [("hk",)] * 2 + [()] * 2),
    # a projectile in his own slot, in front of him (the P2 test above has the details)
    "hadoken": (90, 200, 40, lambda c, f0, fs: bool(_special(fs, 0x00)) and any(
        f.my_fireball and (f.my_fireball_x - f0.my_x) * (1 if f0.facing_right else -1) > 0 for f in fs[:20]),
        [("D",)] * 2 + [("D", "B")] * 2 + [("B", "hp")] * 2 + [()] * 2),                     # motion mirrored
    # rises 60+ px
    "shoryuken": (60, 200, 40, lambda c, f0, fs: bool(_special(fs, 0x04)) and _rise(f0, fs) > 60,
                  [("F",)] * 2 + [("D",)] * 2 + [("D", "hp")] * 2 + [()] * 2),                 # without the DF
    # lifts off a little and travels toward him
    "tatsumaki": (70, 200, 40, lambda c, f0, fs: bool(_special(fs, 0x02)) and _rise(f0, fs) >= 10
                  and _toward(f0, fs[30]) > 20,
                  [("D",)] * 2 + [("D", "F")] * 2 + [("F", "hk")] * 2 + [()] * 2),             # motion mirrored
}


def _record(env, move, facing, request):
    from sf2 import verified

    verified.record(env.me, move, facing, verified.stamp_key(env.backend.rom_sha1, RAM_MAP), request.node.nodeid)


@pytest.mark.parametrize("facing", FACINGS)
@pytest.mark.parametrize("move", list(CHECKS))
def test_move_does_what_it_is_defined_to_do(env, move, facing, request):
    """The macro (sf2.actions, directions resolved by sf2.actions.to_physical) produces the move's effect from a
    free moment facing ``facing``; the wrong input from the same moment does not. Passing records it for the ALL-8
    gate (sf2.verified)."""
    if move not in A.moves(env.me):
        pytest.skip("not on %s's move list" % env.me)
    lo, hi, after, effect, wrong = CHECKS[move]

    def attempt(f0):
        right = _run(env, _physical(A.expand(move), f0.facing_right) + [[]] * after)
        bad = _run(env, _physical(wrong, f0.facing_right) + [[]] * after)
        if not (_clean(f0, right) and _clean(f0, bad)):
            return None
        if move == "sweep" and right[-1].opp_life >= f0.opp_life:      # it must connect to show the knockdown
            return None
        return f0, right, bad

    f0, right, bad = _find(env, facing, lo, hi, attempt)
    assert effect(env.me, f0, right), (move, facing, f0.dx, [(f.my_state, f.my_x, f.my_y) for f in right[:12]])
    assert not effect(env.me, f0, bad), (move, facing, "the wrong input did it too")
    _record(env, move, facing, request)


@pytest.mark.parametrize("facing", FACINGS)
def test_block_is_the_crouching_guard_facing_either_way(env, facing, request):
    """block (down + back) held 20 s from a free moment: each time the CPU attacks he guards (08) crouching
    (+0x43 = 2), and a guard leads to block stun (0E, reaction 06 / 08). Down + toward never guards. (Jump-ins and
    throws still get through a crouching guard: SF2 rules, docs/MOVES.md.)"""
    from sf2.ram import Var, load_map

    def attempt(f0):
        ram_map = load_map(RAM_MAP)
        env.backend.set_vars(ram_map + [Var("guard", 0x0C43, 1, False)])
        state = env.backend.save_state()
        try:
            runs = {}
            for name, tokens in (("block", A.expand("block")), ("wrong", [("D", "F")] * 6)):
                env.backend.load_state(state)
                runs[name] = env.backend.run(_physical(tokens, f0.facing_right) * 200).rams[1:]
        finally:
            env.backend.set_vars(ram_map)
            env.backend.load_state(state)
        return f0, runs

    f0, runs = _find(env, facing, 80, 200, attempt)
    st, react, guard = env.names.index("my_state"), env.names.index("my_react"), len(env.names)
    mx, ox = env.names.index("my_x"), env.names.index("opp_x")
    for name, rows in runs.items():
        # until the CPU jumps over him: after that the same buttons are down + toward / down + back
        rows = rows[:next((i for i, r in enumerate(rows) if (r[mx] <= r[ox]) != f0.facing_right), len(rows))]
        entries = [i for i in range(1, len(rows)) if rows[i][st] == 0x08 and rows[i - 1][st] != 0x08]
        stun = [i for i in range(1, len(rows)) if rows[i][st] == ram.HIT_STATE and rows[i - 1][st] == 0x08
                and rows[i][react] in ram.BLOCK_REACTS]
        if name == "block":
            assert len(entries) >= 2 and all(rows[i][guard] == 2 for i in entries), (facing, entries)
            assert stun, (facing, entries, stun)
        else:
            assert len(rows) > 300 and not entries and not stun, (facing, len(rows), entries)
    _record(env, "block", facing, request)


def test_hadoken_comes_out_while_the_rom_facing_byte_lags_x(env):
    """Right after the fighters cross, x says he faces one way while the ROM (0x0CF4) still has him facing the
    other, and the ROM mirrors the stick by its own byte. At such a moment on the ground, where the Hadoken motion
    resolved from the byte comes out (a control run), sf2.env's ``act("hadoken")`` must come out too (state 0C,
    special 00). Seeded play reaches such moments on the ROM."""
    import random

    if env.me != "ryu":
        pytest.skip("Ryu's Hadoken")
    rng = random.Random(3)
    env.reset()
    for _ in range(3000):
        f0 = env.f
        if (env.in_round and env.controllable() and not env.airborne()[0] and f0.my_state in (0, 2)
                and f0.game_facing_right is not None and f0.game_facing_right != f0.facing_right
                and not f0.fireball and not f0.my_fireball):
            control = _run(env, _physical(RYU_MOTIONS["hadoken"], f0.game_facing_right) + [[]] * 20)
            if _clean(f0, control) and _special(control, 0x00):
                seen = []
                env.act("hadoken", on_frame=seen.append)
                seen += env.run_frames([[]] * 20, capture=False)
                if _clean(f0, seen):
                    assert _special(seen, 0x00), (f0.my_x, f0.opp_x, f0.my_facing,
                                                  [(f.my_state, f.my_special, f.my_x) for f in seen[:16]])
                    return
        if env.act(rng.choice(["forward", "forward", "jump_forward", "idle", "back", "crouch"])).round_over:
            if not env.next_round():
                env.reset()
    raise AssertionError("no clean moment where the two facings disagree")
