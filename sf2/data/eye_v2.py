"""The eye's questions v2 labels (docs/eye_questions_v1.md, "Questions v2 - relabel"; owner 2026-10-02). Pure functions
over the stored RAM rows, read at the DISPLAYED row t (lag 1, as v1); no files.

q3 v2 "What is the fighter on the <side> doing?" -> attack / special attack / block / walk / jump / stand / hit, from
the movement rule with the pressed-word rule (sf2.data.pairs_labels.movement_pressed; attack vs special decided by the
move we pressed when RAM shows an attack state, RAM's rule otherwise):
  attack          movement attack: a standing or crouching normal (0x0A), a jump attack (box out, 0x?C3E != 0), a throw
  special attack  movement special: the character's special (0x0C, or 0x0A with move class 0x08; the pressed word decides)
  block           movement block: guard 0x08 (standing or crouching) or a block react (0x0E with react 06 / 08)
  walk            movement walk (toward or away)
  jump            movement jump: in the air, no attack box out (incl. take-off and landing rows of state 0x04)
  stand           movement stand or crouch (crouching is a position, not an action)
  hit             movement hit or down: being hit, thrown (0x14), knocked down (incl. on the ground after the knockdown)
  None            movement unknown (e.g. 0x06 turning, a missing t - 4 row)
q3 v2 rows keep the episode rule: rows t - 4 .. t of the fighter all have the same v2 answer (``act2_in_episode``).

q4 v2 "Is the fighter on the <side> high, normal or low?" -> high / normal / low:
  high    y != GROUND_Y (in the air)
  low     on the ground AND crouching (``low_kind``), decided by these RAM fields (verified on games 0-31, see the doc):
            crouch          state 0x02 (sub 0 crouched, sub 2 the rows rising out of it)
            crouch_attack   state 0x0A with move class 0x?CBA == 0x02 (a crouching normal; standing normals are 0x00)
            crouch_block    state 0x08 (guard) with sub 0x04 / 0x06 (standing guard: sub 0x00 / 0x02), or state 0x0E
                            with react 0x08 (crouching block stun; standing block stun is react 0x06)
  normal  on the ground otherwise (standing, walking, standing attacks / blocks, specials, hit on the ground, lying
          after a knockdown, the jump state's take-off / landing rows on the ground)
"""
from typing import Optional, Sequence

from ..emu.vs import GROUND_Y
from . import pairs_labels as L
from .perception import UNKNOWN

ACT2 = ("attack", "special attack", "block", "walk", "jump", "stand", "hit")
POS = ("high", "normal", "low")
ACT2_OF = {"attack": "attack", "special": "special attack", "block": "block", "walk": "walk", "jump": "jump",
           "stand": "stand", "crouch": "stand", "hit": "hit", "down": "hit"}
CROUCH_ATTACK_CLASS = 0x02
CROUCH_GUARD_SUBS = (0x04, 0x06)
CROUCH_BLOCK_REACT = 0x08
GAP = 4


def dir_of(answer: str) -> str:
    """The data dir of an answer (no spaces in paths: "special attack" -> special_attack)."""
    return answer.replace(" ", "_")


def act2(rows, t: int, p: int, pressed: Optional[str] = None) -> Optional[str]:
    """The q3 v2 answer of fighter ``p`` at row ``t`` (``pressed``: the class of our pressed word there); None when the
    movement is unknown."""
    if not 0 <= t < len(rows):
        return None
    mv = L.movement_pressed(rows, t, p, pressed)[0]
    return None if mv == UNKNOWN else ACT2_OF[mv]


def act2_in_episode(rows, t: int, p: int, pressed: Sequence[Optional[str]], gap: int = GAP) -> bool:
    """Both shown frames (rows t - gap and t) inside one q3 v2 episode: every row t - gap .. t has the same answer."""
    if t - gap < 0 or t >= len(rows):
        return False
    a = act2(rows, t, p, pressed[t])
    return a is not None and all(act2(rows, u, p, pressed[u]) == a for u in range(t - gap, t))


def low_kind(row, p: int) -> Optional[str]:
    """crouch / crouch_attack / crouch_block from the state, move-class, sub and react bytes (the y is not read)."""
    st = row["p%d_state" % p]
    if st == L.CROUCH:
        return "crouch"
    if st == L.ATTACK and row["p%d_mclass" % p] == CROUCH_ATTACK_CLASS:
        return "crouch_attack"
    if (st == L.GUARD and row["p%d_sub" % p] in CROUCH_GUARD_SUBS) or \
            (st == L.HIT and row["p%d_react" % p] == CROUCH_BLOCK_REACT):
        return "crouch_block"
    return None


def position(rows, t: int, p: int) -> Optional[str]:
    """The q4 v2 answer of fighter ``p`` at row ``t``; None when the row is missing."""
    if not 0 <= t < len(rows):
        return None
    r = rows[t]
    if r["p%d_y" % p] != GROUND_Y:
        return "high"
    return "low" if low_kind(r, p) else "normal"
