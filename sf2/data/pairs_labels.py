"""The movement-pairs labels (docs/prereg_movement_pairs.md): five questions about ONE fighter (player ``p``, 1 or 2),
each read from RAM at the DISPLAYED row t = n - LAG (perception.LAG = 1). Pure functions; no emulator, no files.

movement (from the state byte, 9 answers; precedence top to bottom):
  0x14 (thrown), or 0x0E with sub 0x04 (the knock-down landing, sf2.data.perception)  -> down   (thrown / knocked down)
  0x0E with a block react (06 / 08), or 0x08 (guard)                                    -> block
  0x0E otherwise                                                                        -> hit    (being hit)
  0x0C, or 0x0A with MOVE_CLASS 0x08 (the CPU's specials run in 0x0A, sf2.data.action_probe) -> special (incl. projectile)
  0x0A                                                                                  -> attack
  0x04, or y off the ground in any other state, with the attack box out (0x?C3E != 0)  -> attack (a jump attack;
                Plan B: "a jump attack counts as attack", the air question says "in the air". The state stays 0x04
                and the sub byte that marks it differs by character, so the attack-ID byte is read: the rows while
                the box is out are "attack", the rest of that jump "jump")
  0x04, or y off the ground in any other state                                           -> jump
  0x02                                                                                  -> crouch
  0x00: own x moved >= WALK_PX over t - 4 -> t                                           -> walk, else stand
  anything else (0x06 turning, ...)                                                     -> unknown
direction: for walk and jump, the x travel over t - 4 -> t toward / away from the other fighter (|dx| < WALK_PX: none,
  a straight jump); every other movement: none. The other fighter on the same x: unknown.
facing: the facing byte (0x?CF4): 0x40 right, 0x00 left (sf2.system1.system1._act reads 0x40 as facing right), else
  unknown.
air: y == GROUND_Y ground, else air.
distance: |x1 - x2| <= the fighter's own calibrated poke band (lessons/perception_thresholds_v2.json poke_max, which
  contains the throw band) -> close, else far.
movement10 (the owner's grid, Plan B): the 10 movement cells - walk split by direction, jump not:
  stand, walk toward, walk away, crouch, jump, attack, special, block, hit (being hit), down (knocked down).
Every label that reads an x is unknown when an x is impossible (perception.STAGE_X) or the t - 4 row is missing.
"""
import json
import os
from typing import Dict, Optional, Tuple

from ..config import REPO
from ..emu.vs import GROUND_Y
from ..vocab import CHARACTERS
from .perception import BLOCK_REACTS, LAG, STAGE_X, UNKNOWN

Rows = list
MOVEMENTS = ("stand", "walk", "crouch", "jump", "attack", "special", "block", "hit", "down")
DIRECTIONS = ("toward", "away", "none")
FACINGS = ("left", "right")
AIRS = ("ground", "air")
DISTANCES = ("close", "far")
MOVEMENTS10 = ("stand", "walk toward", "walk away", "crouch", "jump", "attack", "special", "block", "hit", "down")
QUESTIONS: Dict[str, Tuple[str, ...]] = {"movement": MOVEMENTS, "direction": DIRECTIONS, "facing": FACINGS,
                                         "air": AIRS, "distance": DISTANCES}
STAND, CROUCH, JUMP, GUARD, ATTACK, SPECIAL, HIT, THROWN = 0x00, 0x02, 0x04, 0x08, 0x0A, 0x0C, 0x0E, 0x14
KNOCKDOWN_SUB, CPU_SPECIAL_CLASS = 0x04, 0x08
FACING = {0x40: "right", 0x00: "left"}
WALK_PX = 2
GAP = 4                                        # the two displayed frames are t - 4 and t
THRESHOLDS = os.path.join(REPO, "lessons", "perception_thresholds_v2.json")


def poke_bands(path: str = THRESHOLDS) -> Dict[str, int]:
    with open(path) as f:
        return dict(json.load(f)["poke_max"])


def _f(r: Dict[str, int], p: int, name: str) -> int:
    return r["p%d_%s" % (p, name)]


def _x_ok(r: Optional[Dict[str, int]], *keys: str) -> bool:
    return r is not None and all(0 <= r[k] <= STAGE_X for k in keys)


def _dx(rows: Rows, t: int, p: int) -> Optional[int]:
    """Own x travel over t - 4 -> t; None when a row is missing or an x impossible."""
    if t - GAP < 0 or t >= len(rows):
        return None
    a, b = rows[t - GAP], rows[t]
    if not (_x_ok(a, "p%d_x" % p) and _x_ok(b, "p1_x", "p2_x")):
        return None
    return _f(b, p, "x") - _f(a, p, "x")


def movement(rows: Rows, t: int, p: int) -> str:
    if not 0 <= t < len(rows):
        return UNKNOWN
    r = rows[t]
    st, sub = _f(r, p, "state"), _f(r, p, "sub")
    if st == THROWN or (st == HIT and sub == KNOCKDOWN_SUB and _f(r, p, "react") not in BLOCK_REACTS):
        return "down"
    if st == GUARD or (st == HIT and _f(r, p, "react") in BLOCK_REACTS):
        return "block"
    if st == HIT:
        return "hit"
    if st == SPECIAL or (st == ATTACK and _f(r, p, "mclass") == CPU_SPECIAL_CLASS):
        return "special"
    if st == ATTACK:
        return "attack"
    if st == JUMP or _f(r, p, "y") != GROUND_Y:
        return "attack" if _f(r, p, "aid") else "jump"
    if st == CROUCH:
        return "crouch"
    if st == STAND:
        dx = _dx(rows, t, p)
        if dx is None:
            return UNKNOWN
        return "walk" if abs(dx) >= WALK_PX else "stand"
    return UNKNOWN


ATTACK_MOVES = ("attack", "special")


def movement_pressed(rows: Rows, t: int, p: int, pressed: Optional[str]) -> Tuple[str, str]:
    """(movement, source) with attack vs special from the move WE pressed (owner after round 1): when RAM shows an
    attack state (the RAM rule says attack or special), ``pressed`` ('attack' / 'special', the class of our pressed
    word, sf2.data.pairs_moves.pressed_class) decides -> source 'pressed'; with no attack word pressed the RAM rule
    stays -> 'fallback'; any other movement is RAM's -> 'ram'."""
    mv = movement(rows, t, p)
    if mv not in ATTACK_MOVES:
        return mv, "ram"
    if pressed in ATTACK_MOVES:
        return pressed, "pressed"
    return mv, "fallback"


def direction(rows: Rows, t: int, p: int, mv: Optional[str] = None) -> str:
    mv = movement(rows, t, p) if mv is None else mv
    if mv == UNKNOWN:
        return UNKNOWN
    if mv not in ("walk", "jump"):
        return "none"
    dx = _dx(rows, t, p)
    if dx is None:
        return UNKNOWN
    if abs(dx) < WALK_PX:
        return "none"
    side = _f(rows[t], 3 - p, "x") - _f(rows[t], p, "x")
    if side == 0:
        return UNKNOWN
    return "toward" if dx * side > 0 else "away"


def facing(rows: Rows, t: int, p: int) -> str:
    if not 0 <= t < len(rows):
        return UNKNOWN
    return FACING.get(_f(rows[t], p, "facing"), UNKNOWN)


def air(rows: Rows, t: int, p: int) -> str:
    if not 0 <= t < len(rows):
        return UNKNOWN
    return "ground" if _f(rows[t], p, "y") == GROUND_Y else "air"


def distance(rows: Rows, t: int, p: int, bands: Dict[str, int]) -> str:
    if not 0 <= t < len(rows) or not _x_ok(rows[t], "p1_x", "p2_x"):
        return UNKNOWN
    r = rows[t]
    char = CHARACTERS.get(_f(r, p, "char"))
    band = bands.get(char, bands["all"])
    return "close" if abs(r["p1_x"] - r["p2_x"]) <= band else "far"


def labels(rows: Rows, t: int, p: int, bands: Dict[str, int], pressed: Optional[str] = None) -> Dict[str, str]:
    """All five answers for fighter ``p`` at displayed row ``t``; ``pressed``: the class of our pressed word there
    (movement_pressed), None = the RAM rule."""
    mv = movement_pressed(rows, t, p, pressed)[0]
    return {"movement": mv, "direction": direction(rows, t, p, mv), "facing": facing(rows, t, p),
            "air": air(rows, t, p), "distance": distance(rows, t, p, bands)}


def movement10(mv: str, direction_: str) -> str:
    """The grid's movement cell from (movement, direction): walk -> walk toward / away, everything else as is (a jump's
    direction is not a cell). UNKNOWN for a walk without a direction."""
    if mv == "walk":
        return "walk " + direction_ if direction_ in ("toward", "away") else UNKNOWN
    return mv


def episode_key(rows: Rows, t: int, p: int, pressed: Optional[str] = None) -> Optional[Tuple[str, str]]:
    """(movement10, facing): what an episode is a run of, and (with the character) the grid cell a cap counts; None if
    either is unknown."""
    mv = movement_pressed(rows, t, p, pressed)[0]
    key = (movement10(mv, direction(rows, t, p, mv)), facing(rows, t, p))
    return None if UNKNOWN in key else key


def labels_at(rows: Rows, n: int, p: int, bands: Dict[str, int], lag: int = LAG) -> Dict[str, str]:
    """The answers for the image captured at row ``n``: it shows row n - ``lag``."""
    return labels(rows, n - lag, p, bands)
