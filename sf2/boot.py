"""Power-on to round 1 of Chun-Li vs Dhalsim in arcade mode, by a fixed input script (no savestate needed), and
on from one arcade fight to the next without reloading anything.

The ROM is deterministic from reset: the same inputs on the same frames always give the same fight. The CPU's
first opponent depends on the frame the pick is confirmed; the idle count before the jab below gives Dhalsim.
The script ends on the frame ``states/chunli_vs_dhalsim.state`` was saved on (clock 99, x 208 vs 304; holding
right from here moves her as from that savestate), checked by tests/test_rom_harness.py.

Frame counts are bridge RUN frames (input polls). The game state they reach does not depend on what the emulator
did before the reset, but the reset's sub-frame timing does, and with it which frame an input lands on: after some
histories her first step comes one frame sooner than from the savestate (a fresh Mesen does this). Nothing in RAM
shows it, so the script cannot correct it.
"""
import os
from typing import Callable, List, Optional, Sequence, Tuple

from .ram import Var, load_map

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHUNK = 60  # frames per RUN: a windowed Mesen shows the launch smoothly and the caller can watch each chunk

BOOT: List[Tuple[Tuple[str, ...], int]] = [
    ((), 900),            # Capcom logo, the STREET FIGHTER II logo animation, then the title screen settles
    (("start",), 2),
    ((), 60),             # title -> menu, cursor on GAME START
    (("start",), 2),
    ((), 120),            # GAME START -> player select, cursor on Ryu
    (("down",), 2),
    ((), 10),
    (("right",), 2),      # cursor on Chun-Li
    ((), 10),             # this idle count picks the first opponent: 10 gives Dhalsim (0-2 too; 3 Ken, 5 Blanka)
    (("y",), 2),          # jab confirms
    ((), 679),            # VS screen, stage, "ROUND 1 / FIGHT": ends on the first frame she answers the stick
]


def frames_of(steps: Sequence[Tuple[Tuple[str, ...], int]]) -> List[List[str]]:
    return [list(buttons) for buttons, n in steps for _ in range(n)]


def boot(bridge, on_frames: Optional[Callable[[List[List[str]]], None]] = None) -> bytes:
    """Reset the console, play BOOT, return the savestate at the first controllable frame of round 1."""
    bridge.reset()
    frames = frames_of(BOOT)
    for i in range(0, len(frames), CHUNK):
        chunk = frames[i:i + CHUNK]
        if on_frames:
            on_frames(chunk)
        bridge.run(chunk)
    return bridge.save_state()


# Player 2's character (0x0CD1 is player 1's; ids from the VS screens and the savestates in states/). The select
# screen's grid holds the same ids: top row 0-3, bottom row 4-7.
P2_CHAR = 0x0ED1
CHARACTERS = {0: "ryu", 1: "honda", 2: "blanka", 3: "guile", 4: "ken", 5: "chunli", 6: "zangief", 7: "dhalsim",
              10: "balrog", 11: "vega"}
CHUNLI = 5
# Screen bytes, observed on the ROM. 0x0B: 63 while the player select screen waits for a pick (it blinks to 49 once
# one is made). 0x0C: 255 while the continue countdown runs. 0x0D: the select cursor. 0x1A7B: 0 outside a game
# (Capcom logo, title, menu, attract demo, and the select screen of a new game), 255 or 1 in one, 2 from the
# ending on (story, high-score entry, "THANKS FOR PLAYING!!", which waits for START, and the title after it).
WATCH = [Var("mode", 0x0B, 1, False), Var("sub", 0x0C, 1, False), Var("cursor", 0x0D, 1, False),
         Var("game", 0x1A7B, 1, False)]
# Round 1 of a fight laid out (clock 99, full bars, both on the ground at the start x); it holds until someone moves.
SETUP = {"timer": 0x99, "my_hp": 176, "opp_hp": 176, "my_x": 208, "opp_x": 304, "my_y": 192, "opp_y": 192,
         "result": 0, "game": 255}
READY = 168         # frames from the first SETUP frame to the first frame she answers the stick ("FIGHT!")
MAX_FRAMES = 60000  # measured: win ~1600-1900 (with a bonus stage ~3600-3800), loss ~1700-1900, ending ~9200
ENDING = 7500       # frames of Chun-Li's ending before "THANKS FOR PLAYING!!" (it shows at ~6900)
TAP = 12            # a press (2 frames) and the idle after it
COOL = {"start": 150}  # frames to wait after START before the next press (the screens change slowly)
PICKED = 600          # after the jab the select screen stays up (and 0x0D is reused) for a while: leave it alone


def _press(r, ending: int) -> List[str]:
    """The button the screen in RAM row ``r`` waits for, if any; ``ending``: frames since the ending began."""
    if r["sub"] == 255:                                  # continue? - yes
        return ["start"]
    if r["mode"] == 63:                                  # player select: walk the cursor to Chun-Li, jab
        c = r["cursor"]
        if c == CHUNLI:
            return ["y"]
        return ["down"] if c < 4 else ["right"] if c % 4 < CHUNLI % 4 else ["left"]
    if r["game"] == 0 or ending >= ENDING:               # attract / title / menu: START until GAME START
        return ["start"]
    return []


def next_fight(bridge, on_frames: Optional[Callable[[List[List[str]]], None]] = None,
               ram_map: Optional[List[Var]] = None) -> bytes:
    """From anywhere after a match (usually right after it ends) to the first controllable frame of round 1 of the
    next fight, returned as a savestate. It plays on through the win quote / map / bonus stages / VS screen; after
    a loss it continues (START on the countdown, jab on the select screen); after the last boss it lets her ending
    run, then starts a new game (START past "THANKS FOR PLAYING!!" and the title, GAME START, Chun-Li; the first
    opponent is whoever the CPU picks). The bridge's VARS are replaced; pass the caller's ``ram_map`` to have them
    set back."""
    watch = load_map(os.path.join(ROOT, "ram_maps", "sf2_snes.txt")) + WATCH
    names = [v.name for v in watch]
    bridge.set_vars(watch)
    chunk, done, since, ending, last = [[]] * CHUNK, 0, 10 ** 9, 0, None
    while True:
        if done > MAX_FRAMES:
            raise RuntimeError("next_fight: no fight after %d frames" % done)
        if on_frames:
            on_frames(chunk)
        rows = [dict(zip(names, r)) for r in bridge.run(chunk).rams]
        done, since = done + len(chunk), since + len(chunk)
        setup = [i for i, r in enumerate(rows) if all(r[k] == v for k, v in SETUP.items())]
        if setup:
            left = READY - (len(rows) - 1 - setup[0])
            break
        ending = ending + len(chunk) if rows[-1]["game"] == 2 else 0
        press = _press(rows[-1], ending)
        if last == "y" and since < PICKED and press != ["start"]:
            press = []
        if press and since >= COOL.get(last, 0):
            chunk, since, last = [press] * 2 + [[]] * (TAP - 2), 0, press[0]
        else:
            chunk = [[]] * (TAP if press else CHUNK)
    for i in range(0, left, CHUNK):
        chunk = [[]] * min(CHUNK, left - i)
        if on_frames:
            on_frames(chunk)
        bridge.run(chunk)
    if ram_map is not None:
        bridge.set_vars(ram_map)
    return bridge.save_state()
