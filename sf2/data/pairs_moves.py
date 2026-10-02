"""The directed player 1 of the movement pairs (docs/prereg_movement_pairs.md, "Fallback"): the character's FULL set of
attacks and defence as move words, each a relative-token script (F / B / U / D + lp mp hp lk mk hk; resolved per side
by sf2.emu.vs.physical), and the cycle that tells player 1 which move to do next.

The move words (per character):
  standing normals  lp mp hp lk mk hk          (close / far: the game decides by the gap)
  crouching normals c.lp c.mp c.hp c.lk c.mk sweep (= c.hk)
  jumps             jump, jump_forward, jump_back, and each of them with each button (jump_lp, jump_forward_hk, ...):
                    the stick held JUMP_HOLD frames, JUMP_WAIT idle frames, then the button
  throw             F + hp (up close a throw, else a forward hp)
  specials          the character's own, ROM-verified inputs (sf2.data.vs_sweep.SPECIALS)
  defence           block_high (hold B), block_low (hold D + B), HOLD frames each (sf2.data.vs_defense)
  movement          forward / back (walk toward / away), crouch, idle (stand), HOLD_MOVE frames each
The normals, specials and blocks are sf2.data.vs_sweep.actions' own scripts (what System 1 presses); the jump attacks,
c.lp / c.mp (and c.hp for Ryu / Ken) are added here from the same tokens. Movement is held longer than System 1's
4 frames so a walk or a crouch lasts long enough to be seen.

The cycle: every move word once in a seeded shuffled order, then a new shuffle, for as long as the game lasts.
"""
import random
from typing import Dict, List, Sequence, Tuple

from ..emu.vs import GROUND_Y, Step
from .vs_defense import BLOCKS, block_steps
from .vs_sweep import NORMALS, SPECIALS, _crouch

BUTTONS = ("lp", "mp", "hp", "lk", "mk", "hk")
CROUCH_NAMES = {"lp": "c.lp", "mp": "c.mp", "hp": "c.hp", "lk": "c.lk", "mk": "c.mk", "hk": "sweep"}
JUMPS = {"jump": ("U",), "jump_forward": ("U", "F"), "jump_back": ("U", "B")}
JUMP_HOLD, JUMP_WAIT, PRESS = 4, 10, 2
HOLD_MOVE = 16
WALKS = {"forward": ("F",), "back": ("B",), "crouch": ("D",), "idle": ()}
THROW = "throw"
BASE_KEYS = ("c.lk", "c.mk", "sweep", "c.hp", "throw")
# which words resolve F / B from x (where the other fighter is) rather than the facing byte: the walks and the jumps,
# as sf2.system1.system1._act does for its MOVEMENT words
JUMP_ATTACKS = {"%s_%s" % (j, b) for j in JUMPS for b in BUTTONS}
BY_X = set(WALKS) | set(JUMPS) | JUMP_ATTACKS

KIND_NORMAL, KIND_CROUCH, KIND_JUMP, KIND_JUMP_ATTACK, KIND_THROW, KIND_SPECIAL, KIND_BLOCK, KIND_WALK = (
    "normal", "crouch_normal", "jump", "jump_attack", "throw", "special", "block", "movement")


def _jump(tokens: Tuple[str, ...], button: str = "") -> Tuple[Step, ...]:
    steps: Tuple[Step, ...] = ((tokens, JUMP_HOLD),)
    if button:
        steps += (((), JUMP_WAIT), ((button,), PRESS), ((), PRESS))
    return steps


def jump_of(word: str) -> str:
    """The jump a jump word starts with: jump_forward_hk -> jump_forward."""
    for j in JUMPS:
        if word == j or (word.startswith(j + "_") and word[len(j) + 1:] in BUTTONS):
            return j
    raise ValueError("%r is not a jump word" % word)


def specials_of(char: str) -> Dict[str, Tuple[Step, ...]]:
    return {k: v for k, v in SPECIALS[char].items() if k not in BASE_KEYS}


def moves(char: str) -> Dict[str, Tuple[Step, ...]]:
    """Every move word of ``char`` -> its input script."""
    if char not in SPECIALS:
        raise ValueError("unknown character %r" % char)
    out: Dict[str, Tuple[Step, ...]] = dict(NORMALS)
    out.update({CROUCH_NAMES[b]: _crouch(b) for b in BUTTONS})
    for j, toks in JUMPS.items():
        out[j] = _jump(toks)
        out.update({"%s_%s" % (j, b): _jump(toks, b) for b in BUTTONS})
    out[THROW] = SPECIALS[char][THROW]
    out.update(specials_of(char))
    out.update({b: block_steps(b) for b in BLOCKS})
    out.update({w: ((toks, HOLD_MOVE),) for w, toks in WALKS.items()})
    return out


def kind(char: str, word: str) -> str:
    if word in NORMALS:
        return KIND_NORMAL
    if word in CROUCH_NAMES.values():
        return KIND_CROUCH
    if word in WALKS:
        return KIND_WALK
    if word in JUMPS:
        return KIND_JUMP
    if word in JUMP_ATTACKS:
        return KIND_JUMP_ATTACK
    if word == THROW:
        return KIND_THROW
    if word in BLOCKS:
        return KIND_BLOCK
    if word in specials_of(char):
        return KIND_SPECIAL
    raise ValueError("%r is not a move word of %s" % (word, char))


class Cycle:
    """Which move to do next: every word once per round of the cycle, in a fresh seeded shuffle each round."""

    def __init__(self, words: Sequence[str], rng: random.Random):
        if not words:
            raise ValueError("an empty move list")
        self.words, self.rng = list(words), rng
        self.queue: List[str] = []
        self.rounds = 0

    def next(self) -> str:
        if not self.queue:
            self.queue = list(self.words)
            self.rng.shuffle(self.queue)
            self.rounds += 1
        return self.queue.pop(0)


# ---- did player 1 do the directed move? (a check over the RAM rows of the move's window; p1_ / p2_ names) ---------
MASH_OR_ATTACK = {"lightning_legs", "hundred_hand_slap", "electricity", "clothesline", "spinning_piledriver"}
STRAIGHT_PX = 8          # a straight jump's x travel (pushes included) stays under this


def _travel(rows: Sequence[Dict[str, int]]) -> int:
    """Player 1's x travel toward where player 2 was at the start (negative: away)."""
    sign = 1 if rows[0]["p1_x"] < rows[0]["p2_x"] else -1
    return sign * (rows[-1]["p1_x"] - rows[0]["p1_x"])


def _evidence(char: str, word: str, rows: Sequence[Dict[str, int]]) -> str:
    """'done' when the RAM shows the move, '' when it does not; blocks: 'held' when no attack came to block."""
    st = [r["p1_state"] for r in rows]
    air = [r["p1_y"] != GROUND_Y for r in rows]
    guarded = any(r["p1_state"] == 0x08 or (r["p1_state"] == 0x0E and r["p1_react"] in (0x06, 0x08)) for r in rows)
    k = kind(char, word)
    if k in (KIND_NORMAL, KIND_CROUCH):
        ok = any(r["p1_state"] == 0x0A and r["p1_aid"] for r in rows)
    elif k == KIND_THROW:
        ok = 0x0A in st
    elif k == KIND_SPECIAL:
        ok = 0x0C in st or (word in MASH_OR_ATTACK and 0x0A in st)
    elif k in (KIND_JUMP, KIND_JUMP_ATTACK):
        tr = _travel(rows)
        # back: no x travel either (backed up against the wall - which is wider for a wide character - a back
        # jump goes straight up)
        way = {"jump": abs(tr) < STRAIGHT_PX, "jump_forward": tr > 0, "jump_back": tr <= 0}[jump_of(word)]
        ok = any(air) and way
        if k == KIND_JUMP_ATTACK:                       # an attack box (0x0C3E) out while in the air
            ok = ok and any(a and r["p1_aid"] for a, r in zip(air, rows))
    elif k == KIND_BLOCK:
        return "done" if guarded else "held"
    elif word == "forward":
        ok = 0x00 in st and _travel(rows) >= 2
    elif word == "back":                                    # holding back while he attacks is a block
        ok = (0x00 in st and _travel(rows) <= -2) or guarded
    elif word == "crouch":
        ok = 0x02 in st
    else:                                                   # idle: stays standing
        ok = all(s == 0x00 for s in st)
    return "done" if ok else ""


def as_p1(row: Dict[str, int], p: int) -> Dict[str, int]:
    """The row seen from player ``p``: its p<p>_ fields under p1_, the other's under p2_ (shared fields as they are)."""
    if p == 1:
        return row
    if p != 2:
        raise ValueError("player %r" % p)
    out = {}
    for k, v in row.items():
        if k.startswith("p1_"):
            out["p2_" + k[3:]] = v
        elif k.startswith("p2_"):
            out["p1_" + k[3:]] = v
        else:
            out[k] = v
    return out


def executed(char: str, word: str, rows: Sequence[Dict[str, int]], p: int = 1) -> Dict:
    """status: done (RAM shows the move) / held (a block with nothing to block) / interrupted (player ``p`` was hit or
    thrown before showing it) / cut (the round ended first) / missed; aids: the non-zero attack IDs (0x?C3E) seen."""
    if not rows:
        raise ValueError("no rows for %s" % word)
    rows = [as_p1(r, p) for r in rows]
    ev = _evidence(char, word, rows)
    if ev:
        status = ev
    elif any(r["p1_state"] in (0x0E, 0x14) for r in rows):
        status = "interrupted"
    elif rows[-1]["result"]:
        status = "cut"
    else:
        status = "missed"
    return {"status": status, "aids": sorted({r["p1_aid"] for r in rows if r["p1_aid"]})}
