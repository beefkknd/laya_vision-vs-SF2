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
FIGHTS = {"ryu": ("p1_ryu_vs_ken.state", "ken"), "ken": ("p1_ken_vs_ryu.state", "ryu"),
          "guile": ("p1_guile_vs_ryu.state", "ryu"), "chunli": ("p1_chunli_vs_ryu.state", "ryu"),
          "zangief": ("p1_zangief_vs_ryu.state", "ryu")}
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


# Characters that jump over the CPU instead of walking through it: CPU Ryu backs into the corner, and when Ken walks
# past him there he throws Ken straight back.
JUMP_OVER = {"ken"}


def _to_other_side(env):
    """Walk forward until past the CPU. Some openings never get past it (it keeps blocking or throwing): _find then
    sees the wrong facing and tries the next opening."""
    for _ in range(200):
        env.act("jump_forward" if env.me in JUMP_OVER and env.f.dx <= 70 else "forward")
        if not env.f.facing_right:
            return


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


def _until_recovered(fs, n):
    """The frames of a macro ``n`` frames long up to the first one after it where the character stands or crouches
    again: what the CPU does to him after that does not change what the move did (Guile vs Ryu: Ryu walks in and
    throws him right after a far kick)."""
    end = next((i for i in range(n, len(fs)) if fs[i].my_state in (0, 2)), len(fs) - 1)
    return fs[:end + 1]


# Zangief facing left up close: the first clean moment came after ~500 steps (CPU Ryu keeps attacking him there).
def _find(env, facing, lo, hi, attempt, limit=5000):
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
# character -> {special: my_special id}. Ken has the same inputs, ids and projectile slot as Ryu (checked on the ROM).
SPECIAL_IDS = {"ryu": {"hadoken": 0x00, "tatsumaki": 0x02, "shoryuken": 0x04},
               "ken": {"hadoken": 0x00, "tatsumaki": 0x02, "shoryuken": 0x04}}


@pytest.mark.parametrize("facing", FACINGS)
def test_each_special_has_its_own_id_and_the_hadoken_uses_player_1s_projectile_slot(env, facing):
    """Every Ryu special is state 0C, with 0x0D80 telling which (00 Hadoken, 02 Hurricane Kick, 04 Shoryuken). The
    Hadoken puts a projectile in player 1's slot (0x1000, x at 0x1007) in front of him within 14 frames of the
    press, travelling away from him ~3 px a frame (it stops on the frame it hits). (Player 2's slot 0x1050 holds the
    CPU's own Hadoken, which can be on screen at the same time: checked by hand, 2026-09-27.) Ken: the same."""
    if env.me not in SPECIAL_IDS:
        pytest.skip("Ryu's and Ken's ids")

    def attempt(f0):
        runs = {m: _run(env, _physical(t, f0.facing_right) + [[]] * 40) for m, t in RYU_MOTIONS.items()}
        # the CPU can walk into the fireball as it appears (seen: hit on its 2nd frame, no travel): another moment
        if not all(_clean(f0, fs[:30]) for fs in runs.values()) or any(
                f.opp_state == ram.HIT_STATE for f in runs["hadoken"][:30]):
            return None
        return f0, runs

    f0, runs = _find(env, facing, 90, 200, attempt)
    for m, fs in runs.items():
        sp = [f.my_special for f in fs if f.my_state == ram.SPECIAL_STATE]
        assert len(sp) > 10 and set(sp) == {SPECIAL_IDS[env.me][m]}, (facing, m, sp)
    fb = [f for f in runs["hadoken"] if f.my_fireball]
    first = runs["hadoken"].index(fb[0])
    toward = 1 if f0.facing_right else -1
    xs = [(f.my_fireball_x - f0.my_x) * toward for f in fb]
    assert first <= 4 + 14 and len(fb) >= 2, (facing, first, xs)
    steps = [b - a for a, b in zip(xs, xs[1:])]
    assert all(x > 0 for x in xs) and all(0 <= d <= 4 for d in steps) and any(d >= 2 for d in steps), (facing, xs)
    assert not any(f.my_fireball for m in ("shoryuken", "tatsumaki") for f in runs[m]), facing


# Guile's charges (docs/MOVES.md, fierce / roundhouse): hold the charge direction ``n`` frames, then the release
# direction + the button 2 frames. On the ROM 61 frames of charge works and 60 never does (both facings, both moves).
def _charge(hold, n, release, button):
    return [hold] * n + [release + (button,)] * 2 + [()] * 2


GUILE_CHARGES = {"sonic_boom": _charge(("B",), 64, ("F",), "hp"), "flash_kick": _charge(("D",), 64, ("U",), "hk")}
GUILE_SPECIAL_IDS = {"sonic_boom": 0x00, "flash_kick": 0x02}


def _same_sides(f0, fs):
    """The fighters did not swap sides, by x or by the ROM's facing byte (a CPU jump or Hurricane Kick over him): the
    held direction kept its meaning."""
    return all(f.facing_right == f.game_facing_right == f0.facing_right for f in fs)


@pytest.mark.parametrize("facing", FACINGS)
def test_guiles_specials_have_their_own_ids_and_the_sonic_boom_uses_player_1s_projectile_slot(env, facing):
    """Both Guile specials are state 0C, with 0x0D80 telling which (00 Sonic Boom, 02 Flash Kick). The Sonic Boom puts
    a projectile in player 1's slot (0x1000, x at 0x1007) in front of him ~13 frames after the press, moving away
    from him until it hits; the Flash Kick rises 60+ px and throws nothing. 60 frames of charge give neither."""
    if env.me != "guile":
        pytest.skip("Guile's ids")
    runs_of = dict(GUILE_CHARGES, short_boom=_charge(("B",), 60, ("F",), "hp"),
                   short_flash=_charge(("D",), 60, ("U",), "hk"))

    def attempt(f0):
        runs = {m: _run(env, _physical(t, f0.facing_right) + [[]] * 40) for m, t in runs_of.items()}
        if not all(_clean(f0, runs[m]) and _same_sides(f0, runs[m][:len(runs_of[m])]) for m in runs):
            return None
        return f0, runs

    f0, runs = _find(env, facing, 120, 220, attempt)
    press = len(GUILE_CHARGES["sonic_boom"]) - 4
    for m in GUILE_CHARGES:
        sp = [f.my_special for f in runs[m] if f.my_state == ram.SPECIAL_STATE]
        assert len(sp) > 10 and set(sp) == {GUILE_SPECIAL_IDS[m]}, (facing, m, sp)
    for m in ("short_boom", "short_flash"):
        assert not any(f.my_state == ram.SPECIAL_STATE for f in runs[m]), (facing, m)
    fb = [f for f in runs["sonic_boom"] if f.my_fireball]
    first = runs["sonic_boom"].index(fb[0]) - press
    toward = 1 if f0.facing_right else -1
    xs = [(f.my_fireball_x - f.my_x) * toward for f in fb]
    assert 0 <= first <= 16 and all(x > 0 for x in xs), (facing, first, xs)
    steps = [b - a for a, b in zip(xs, xs[1:])]
    assert all(d >= 0 for d in steps), (facing, xs)
    assert _rise(f0, runs["flash_kick"]) > 60 and not any(f.my_fireball for f in runs["flash_kick"]), facing


ZANGIEF_MOTIONS = {  # docs/MOVES.md, as the ROM takes them (sf2/actions.py has why)
    "spinning_piledriver": _motion([("F",), ("D", "F"), ("D",), ("D", "B"), ("B",), ("U",)], "lp"),
    "clothesline": [("lp", "mp", "hp")] * 2 + [()] * 2,
}


@pytest.mark.parametrize("facing", FACINGS)
def test_the_pile_driver_is_special_00_and_the_clothesline_is_an_attack(env, facing):
    """Zangief. The Spinning Pile Driver is state 0C with 0x0D80 = 00 (his only 0C move): the grab pulls Ryu in, Ryu
    keeps his own state (never hit stun 0E) until the slam throws him (14) and costs him life. The Clothesline is
    not a special on the ROM: action state 0A like a normal, 61+ frames, and it never enters 0C. The Pile Driver up
    close, so it can grab; the Clothesline far, so the CPU's attacks do not cut it short."""
    if env.me != "zangief":
        pytest.skip("Zangief's ids")

    def attempt(move):
        def run(f0):
            fs = _run(env, _physical(ZANGIEF_MOTIONS[move], f0.facing_right) + [[]] * 130)
            return (f0, fs) if _clean(f0, _until_recovered(fs, len(ZANGIEF_MOTIONS[move]))) else None
        return run

    f0, spd = _find(env, facing, 0, 50, attempt("spinning_piledriver"))
    sp = [f.my_special for f in spd if f.my_state == ram.SPECIAL_STATE]
    start = next(i for i, f in enumerate(spd) if f.my_state == ram.SPECIAL_STATE)
    thrown = next(i for i, f in enumerate(spd) if f.opp_state == ram.THROWN_STATE)
    assert len(sp) > 60 and set(sp) == {0x00}, (facing, sp)
    assert all(f.opp_state != ram.HIT_STATE for f in spd[start:thrown]) and spd[-1].opp_life < f0.opp_life, facing
    _, lar = _find(env, facing, 85, 200, attempt("clothesline"))
    assert not any(f.my_state == ram.SPECIAL_STATE for f in lar) and _run_length(lar, 0x0A) >= 60, facing


# The specials each character's attack_result check runs (raw inputs, independent of sf2.actions). Zangief's Pile
# Driver is a throw: neither hit nor block.
RESULT_MOVES = {"ryu": RYU_MOTIONS, "ken": RYU_MOTIONS, "guile": GUILE_CHARGES,
                "zangief": {"clothesline": ZANGIEF_MOTIONS["clothesline"]}}


@pytest.mark.parametrize("facing", FACINGS)
def _taps(n):
    """``n`` short taps, 1 frame down and 1 up (the lightning_legs macro is 12)."""
    return [("lk",), ()] * n


def _down_charge(hold):
    """Down held ``hold`` frames, then up + roundhouse (the spinning_bird_kick macro holds 64)."""
    return [("D",)] * hold + [("U", "hk")] * 2 + [()] * 2


@pytest.mark.parametrize("facing", FACINGS)
def test_chunli_specials_have_their_own_id_and_thresholds(env, facing):
    """Chun-Li's specials are state 0C, 0x0D80 telling which: 02 Lightning Legs, 00 Spinning Bird Kick; neither uses
    her projectile slot. From one free moment: 10 short taps (1 frame down, 1 up) start the Legs, 9 do not; down
    held 61 frames, then up + roundhouse, is a Spinning Bird Kick, 60 frames is not (a binary search found 61 at 20
    of 20 free moments, 10 per facing, 2026-09-27)."""
    if env.me != "chunli":
        pytest.skip("Chun-Li's ids")
    tries = {"legs": _taps(12), "taps10": _taps(10), "taps9": _taps(9),
             "sbk": _charge(64), "hold61": _charge(61), "hold60": _charge(60)}

    def attempt(f0):
        runs = {m: _run(env, _physical(t, f0.facing_right) + [[]] * 40) for m, t in tries.items()}
        if not all(_clean(f0, runs[m][:len(t) + 20]) for m, t in tries.items()):   # until well after the input
            return None
        return runs

    runs = _find(env, facing, 90, 200, attempt)
    ids = {m: {f.my_special for f in fs if f.my_state == ram.SPECIAL_STATE} for m, fs in runs.items()}
    assert ids == {"legs": {0x02}, "taps10": {0x02}, "taps9": set(), "sbk": {0x00}, "hold61": {0x00},
                   "hold60": set()}, (facing, ids)
    assert not any(f.my_fireball for fs in runs.values() for f in fs), facing


@pytest.mark.parametrize("facing", FACINGS)
def test_attack_result_reads_hit_blocked_and_whiffed(env, facing):
    """ram.attack_result against what the ROM shows independently: a hit costs him life and he was not guarding
    (08) the frame before; blocked means he was guarding, then block stun, losing at most a special's chip (12); a
    whiff costs him nothing and he never enters 0E. The character's specials and a fierce, at many moments."""
    if env.me not in RESULT_MOVES:
        pytest.skip("no specials to check for %s" % env.me)
    moves = dict(RESULT_MOVES[env.me], hp=[("hp",)] * 2 + [()] * 2)
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
                if end is None or not _clean(f0, fs[:end]) or not _same_sides(f0, fs[:len(tokens) - 4]):
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
# Far standing normals: frames in the attack state 0A when they whiff (measured on the ROM, the opponent 85+ px away;
# a run where one connects is retried at a later moment). The button mapping shows in them: jab / fierce / short /
# roundhouse all differ.
ATTACK_FRAMES = {"ryu": {"lp": 13, "hp": 36, "lk": 21, "hk": 33}, "ken": {"lp": 13, "hp": 36, "lk": 21, "hk": 33},
                 "guile": {"lp": 13, "hp": 33, "lk": 15, "hk": 35}, "chunli": {"lp": 13, "hp": 30, "lk": 17, "hk": 33},
                 "zangief": {"lp": 12, "hp": 42, "lk": 10, "hk": 24}}


def _run_length(fs, state):
    """Frames in the first run of ``state``."""
    i = next((i for i, f in enumerate(fs) if f.my_state == state), None)
    if i is None:
        return 0
    return next((j for j, f in enumerate(fs[i:]) if f.my_state != state), len(fs) - i)


def _rise(f0, fs):
    return f0.my_y - min(f.my_y for f in fs)


def _hit_on_ground(f0, fs):
    """When he is first hit he is on the ground and was not starting a jump the frame before (a hit in the jump's
    first frames, still at ground height, knocks down too)."""
    i = next((i for i, f in enumerate(fs) if f.opp_state == ram.HIT_STATE), None)
    before = f0 if not i else fs[i - 1]
    return i is not None and fs[i].opp_y == f0.opp_y and before.opp_state != ram.JUMP_STATE


def _special(fs, sid):
    return [f for f in fs if f.my_state == ram.SPECIAL_STATE and f.my_special == sid]


def _attack(button):
    return lambda c, f0, fs: _run_length(fs, 0x0A) == ATTACK_FRAMES[c][button]


# move -> (opponent lo..hi px away, frames watched after the macro, effect(character, f0, frames) -> bool, a wrong
# input). The effect is what docs/MOVES.md defines, read from RAM; the wrong input must not produce it.
CHECKS = {
    # (right after walking back, x still moves 1 px on the first frame: 5 of 51 free moments with Ken; so the
    # reference is the first frame)
    "idle": (80, 200, 0, lambda c, f0, fs: all(f.my_state == 0 and f.my_x == fs[0].my_x for f in fs), [("F",)] * 4),
    "forward": (80, 200, 0, lambda c, f0, fs: _toward(f0, fs[-1]) > 0, [("B",)] * 4),
    "back": (80, 200, 0, lambda c, f0, fs: _toward(f0, fs[-1]) < 0, [("F",)] * 4),
    "jump": (80, 200, 40, lambda c, f0, fs: any(f.my_state == ram.JUMP_STATE for f in fs) and _rise(f0, fs) > 60
             and _toward(f0, fs[30]) == 0, [("U", "F")] * 4),
    # (Zangief's short forward jump rises exactly 60 px, and right after walking through the CPU it leaves the ground
    # only ~15 frames after the press)
    "jump_forward": (80, 200, 40, lambda c, f0, fs: _rise(f0, fs) > 50 and _toward(f0, fs[30]) > 8, [("U",)] * 4),
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
    "hadoken": (90, 200, 40, lambda c, f0, fs: bool(_special(fs, SPECIAL_IDS[c]["hadoken"])) and any(
        f.my_fireball and (f.my_fireball_x - f0.my_x) * (1 if f0.facing_right else -1) > 0 for f in fs[:20]),
        [("D",)] * 2 + [("D", "B")] * 2 + [("B", "hp")] * 2 + [()] * 2),                     # motion mirrored
    # rises 60+ px
    "shoryuken": (60, 200, 40, lambda c, f0, fs: bool(_special(fs, SPECIAL_IDS[c]["shoryuken"])) and _rise(f0, fs) > 60,
                  [("F",)] * 2 + [("D",)] * 2 + [("D", "hp")] * 2 + [()] * 2),                 # without the DF
    # lifts off a little and travels toward him
    "tatsumaki": (70, 200, 40, lambda c, f0, fs: bool(_special(fs, SPECIAL_IDS[c]["tatsumaki"])) and _rise(f0, fs) >= 10
                  and _toward(f0, fs[30]) > 20,
                  [("D",)] * 2 + [("D", "F")] * 2 + [("F", "hk")] * 2 + [()] * 2),             # motion mirrored
    # Guile. The CPU walks in during the 64-frame charge, so the boom can hit him at once: it is enough that it
    # appears in his own slot, in front of him. The wrong input is the same release after 60 frames of charge.
    "sonic_boom": (90, 220, 40, lambda c, f0, fs: bool(_special(fs, 0x00)) and any(
        f.my_fireball and (f.my_fireball_x - f.my_x) * (1 if f0.facing_right else -1) > 0 for f in fs),
        _charge(("B",), 60, ("F",), "hp")),
    # rises 60+ px
    "flash_kick": (60, 220, 40, lambda c, f0, fs: bool(_special(fs, 0x02)) and _rise(f0, fs) > 60,
                   _charge(("D",), 60, ("U",), "hk")),
    # Chun-Li: kicks on the spot (the P2 test above has the ids and thresholds)
    "lightning_legs": (80, 200, 30, lambda c, f0, fs: bool(_special(fs, 0x02)) and _toward(f0, fs[-1]) == 0,
                       _taps(8)),                                                               # too few taps
    # upside down, up off the ground and across toward him
    "spinning_bird_kick": (90, 200, 75, lambda c, f0, fs: bool(_special(fs, 0x00)) and _rise(f0, fs) >= 10
                           and max(_toward(f0, f) for f in fs) > 40, _down_charge(40)),             # charge too short
    # Zangief. The Pile Driver is state 0C, 0x0D80 = 00; the grab reaches ~75 px (it pulls him in), he stays in his
    # own state until the slam throws him (14) and it costs him life. The wrong input is the circle without the U.
    "spinning_piledriver": (0, 50, 130, lambda c, f0, fs: bool(_special(fs, 0x00)) and any(
        f.opp_state == ram.THROWN_STATE for f in fs) and fs[-1].opp_life < f0.opp_life,
        [("F",)] * 2 + [("D", "F")] * 2 + [("D",)] * 2 + [("D", "B")] * 2 + [("B", "lp")] * 2 + [()] * 2),
    # not a special on the ROM: an attack (0A) of 60+ frames (his longest far normal is 42 whiffing, 56 hitting), in
    # place; a jab is the wrong input
    "clothesline": (85, 200, 80, lambda c, f0, fs: _run_length(fs, 0x0A) >= 60
                    and all(abs(f.my_x - f0.my_x) <= 2 for f in fs[:60]), [("lp",)] * 2 + [()] * 2),
}


def _grounded(fs):
    """The opponent stood or crouched until the first hit or block on him (the CPU did not jump or attack)."""
    end = next((i for i, f in enumerate(fs) if f.opp_state == ram.HIT_STATE), len(fs))
    return all(f.opp_state in (0, 2) for f in fs[:end])


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
        n = len(A.expand(move))
        if not (_clean(f0, _until_recovered(right, n)) and _clean(f0, _until_recovered(bad, len(wrong)))):
            return None
        if not (_same_sides(f0, right[:n - 4]) and _same_sides(f0, bad[:len(wrong) - 4])):
            return None                                                     # a charge held the wrong way

        if move == "sweep" and (right[-1].opp_life >= f0.opp_life     # it must connect to show the knockdown,
                                or not (_grounded(right) and _grounded(bad))):
            return None       # on him standing: any hit knocks him down out of a jump or a Shoryuken
        # any hit on a CPU starting a jump knocks down (Zangief's standing roundhouse catches Ryu's): only ground hits
        if move == "sweep" and not all(_hit_on_ground(f0, fs) for fs in (right, bad)
                                       if any(f.opp_state == ram.HIT_STATE for f in fs)):
            return None
        if move in ("lp", "hp", "lk", "hk") and (right[-1].opp_life < f0.opp_life
                                                 or any(f.opp_state == ram.HIT_STATE for f in right + bad)):
            return None                                                 # ATTACK_FRAMES are whiffs

        return f0, right, bad

    f0, right, bad = _find(env, facing, lo, hi, attempt)
    assert effect(env.me, f0, right), (move, facing, f0.dx, [(f.my_state, f.my_x, f.my_y) for f in right[:12]])
    assert not effect(env.me, f0, bad), (move, facing, "the wrong input did it too")
    _record(env, move, facing, request)


@pytest.mark.parametrize("facing", FACINGS)
def test_block_is_the_crouching_guard_facing_either_way(env, facing, request):
    """block (down + back) for 200 decisions (20 s) from a free moment, each resolved from the ROM's facing byte at
    that decision as sf2.env does, so the CPU crossing over (Ryu vs Guile, often within a second) does not end the
    check: each time the CPU attacks he guards (08) crouching (+0x43 = 2), and a guard leads to block stun (0E,
    reaction 06 / 08); he never guards standing (+0x43 = 1; it read 0 on one guard against Ryu). Down + toward never
    guards while the ROM still has him facing the way the decision was resolved for (the CPU crossing over in the
    middle of a 6-frame decision turns the held toward into back). (Jump-ins and throws still get through a crouching
    guard: SF2 rules, docs/MOVES.md.)"""
    from sf2.ram import Var, load_map

    def play(tokens, facing_right):
        """RAM rows, each with the facing its decision was resolved for appended."""
        rows = []
        for _ in range(200):
            rows += [r + [facing_right] for r in env.backend.run(_physical(tokens, facing_right)).rams[1:]]
            facing_right = rows[-1][mf] == ram.FACING_RIGHT
        return rows

    def attempt(f0):
        ram_map = load_map(RAM_MAP)
        env.backend.set_vars(ram_map + [Var("guard", 0x0C43, 1, False)])
        state = env.backend.save_state()
        try:
            runs = {}
            for name, tokens in (("block", A.expand("block")), ("wrong", [("D", "F")] * 6)):
                env.backend.load_state(state)
                runs[name] = play(tokens, f0.facing_right)
        finally:
            env.backend.set_vars(ram_map)
            env.backend.load_state(state)
        # the CPU never reached him (hit or block stun) while he held block: nothing to judge (seen: CPU Ryu's
        # attacks all falling short of a crouching Zangief for 20 s): another moment
        if not any(r[st] == ram.HIT_STATE for r in runs["block"]):
            return None
        return f0, runs

    st, react, guard = env.names.index("my_state"), env.names.index("my_react"), len(env.names)
    mf = env.names.index("my_facing")
    f0, runs = _find(env, facing, 80, 200, attempt)
    for name, rows in runs.items():
        entries = [i for i in range(1, len(rows)) if rows[i][st] == 0x08 and rows[i - 1][st] != 0x08]
        stun = [i for i in range(1, len(rows)) if rows[i][st] == ram.HIT_STATE and rows[i - 1][st] == 0x08
                and rows[i][react] in ram.BLOCK_REACTS]
        if name == "block":
            crouching = [i for i in entries if rows[i][guard] == 2]
            assert len(crouching) >= 2 and not any(rows[i][guard] == 1 for i in entries), (facing, entries)
            assert stun, (facing, entries, stun)
        else:
            meant = [i for i in entries + stun if (rows[i][mf] == ram.FACING_RIGHT) == rows[i][-1]]
            assert len(rows) == 1200 and not meant, (facing, entries, stun)
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
