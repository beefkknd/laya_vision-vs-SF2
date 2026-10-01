"""The movement question (docs/prereg_movement.md): "What is he doing right now?", asked of the screen only (frames
n-4 and n, HUD visible, note "me=<char>"), labelled from the RAM rows the U collection stores around each decision.
"Me" is player 1, "him" player 2 (sf2.data.perception).

The label is his state at the DISPLAYED frame t = n - LAG (perception.LAG = 1, as measured):
  0x0E with a non-block react (not 06 / 08), or 0x14 (thrown)    -> being hit
  0x08 (guard), or 0x0E with a block react (06 / 08)             -> blocking
  0x0A / 0x0C (attack or special), on the ground or in the air   -> attacking
  0x04 (jump), or y off the ground (y != GROUND_Y) in any other state -> jumping
  0x02                                                            -> crouching
  0x00: x moved >= WALK_PX toward / away from me between the two displayed frames (t - 4 -> t)
                                                                  -> walking toward me / walking away, else standing
  any other state (0x06, turning round, on the ground)            -> unknown
Precedence (the prereg lists "0x04 jump or y above ground" next to the other states without an order): a state the
prereg names with its own answer (hit / thrown, guard / block react, attack) keeps it in the air too - a jump attack
is attacking, a fighter knocked into the air is being hit; "y above ground" makes jumping of every other state.

Unknown (no row): the displayed row missing; for standing (the only answer that reads x), the row t - 4 missing, an
impossible x (perception.STAGE_X guard) at t or t - 4, or him on my x (no direction).
"""
from typing import Dict, Optional

from ..emu.vs import GROUND_Y
from .perception import (
    ATTACK,
    BLOCK_REACTS,
    GUARD,
    HIT,
    LAG,
    THROWN,
    UNKNOWN,
    Rows,
    _row,
    _x_ok,
)

KEY = "movement"
INSTRUCTIONS = "What is he doing right now?"
ANSWERS = ("standing", "walking toward me", "walking away", "crouching", "jumping", "attacking", "being hit",
           "blocking")
CRITERIA: Dict[str, str] = {
    "standing": "standing still", "walking toward me": "walking toward me", "walking away": "walking away from me",
    "crouching": "crouching", "jumping": "in the air or jumping", "attacking": "attacking",
    "being hit": "being hit or thrown", "blocking": "blocking"}
STAND, CROUCH, JUMP = 0x00, 0x02, 0x04
WALK_PX = 2
FRAME_GAP = 4               # the two displayed frames are n - 4 and n


def movement_question() -> Dict:
    if tuple(CRITERIA) != ANSWERS:
        raise AssertionError("criteria %s != answers %s" % (tuple(CRITERIA), ANSWERS))
    return {"type": "choice", "instructions": INSTRUCTIONS, "criteria": dict(CRITERIA)}


def _walk(rows: Rows, t: int, walk_px: int) -> str:
    r, p = rows[t], _row(rows, t - FRAME_GAP)
    if not (_x_ok(p, "p2_x") and _x_ok(r, "p1_x", "p2_x")):
        return UNKNOWN
    dx = r["p2_x"] - p["p2_x"]
    if abs(dx) < walk_px:
        return "standing"
    side = r["p1_x"] - r["p2_x"]
    if side == 0:
        return UNKNOWN
    return "walking toward me" if dx * side > 0 else "walking away"


def movement(rows: Rows, t: int, walk_px: int = WALK_PX) -> str:
    """The answer for the displayed row ``t`` (already n - LAG)."""
    r: Optional[Dict[str, int]] = _row(rows, t)
    if r is None:
        return UNKNOWN
    st = r["p2_state"]
    if st == THROWN or (st == HIT and r["p2_react"] not in BLOCK_REACTS):
        return "being hit"
    if st in (GUARD, HIT):
        return "blocking"
    if st in ATTACK:
        return "attacking"
    if st == JUMP or r["p2_y"] != GROUND_Y:
        return "jumping"
    if st == CROUCH:
        return "crouching"
    if st == STAND:
        return _walk(rows, t, walk_px)
    return UNKNOWN


def movement_at(rows: Rows, n: int, lag: int = LAG, walk_px: int = WALK_PX) -> str:
    """The answer for the image captured at decision row ``n``: the displayed frame is n - ``lag``."""
    return movement(rows, n - lag, walk_px)
