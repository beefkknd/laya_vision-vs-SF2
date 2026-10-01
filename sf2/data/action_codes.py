"""The action label (docs/prereg_movement_data.md, "Owner decisions on the label"): for each fighter on a pair,
"<actor> act<NN> stg<1-3>". Pure logic over the RAM rows the movement collector stores; no emulator, no files.

Per fighter (player 1 = me = Chun-Li, player 2 = him), per displayed row t (t >= FIRST_T; earlier rows are unknown):

  state 0x0A / 0x0C (attack, special) and 0x04 (jump) are RUN states: the whole run of that state byte is ONE episode
  and gets one code, known when the run ends (sf2.data.action_probe):
    the run's first non-zero attack ID (0x?C3E)                           -> that ID (1..ATTACK_ID_MAX)
      (a jump with an attack in it: the whole jump run gets the attack's ID - not split; the owner's rule)
    no ID, state 0x04                                                      -> JUMP
    no ID, attack state: MOVE_CLASS (0x?CBA) mode over the run after its 1st row in (0x06, 0x0C) -> THROW
                         its own projectile slot (shot1 / shot2) active on any row of the run   -> PROJECTILE
                         state 0x0C and SPECIAL_CLASS (0x?C49) mode after the 1st row != 0xFF   -> SPECIAL_BASE + it
                         none of these (an attack cut in its startup)                           -> unknown, dropped
    an ID above ATTACK_ID_MAX (it would collide with a reserved code)     -> unknown ("id_range"), dropped
  every other row has its own code, from sf2.data.movement's rules (precedence as there):
    0x14 THROWN; 0x0E with a block react (06 / 08) or 0x08 BLOCK; 0x0E otherwise HIT; y != GROUND_Y JUMP;
    0x02 CROUCH; 0x00 by x over t - 4 -> t: |dx| < WALK_PX STAND, else WALK_TOWARD / WALK_AWAY (toward = the other
    fighter's side); any other state (0x06 turning, ...) or an impossible x: unknown.
  Episodes: a run state's run; otherwise a run of rows with one code (so guard 0x08 -> block react 0x0E is one BLOCK
  episode). Unknown rows break episodes and are never labelled. The stage of row t in its episode [s, e]:
  stg = 1 + 3 * (t - s) // (e - s + 1).
"""
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from ..emu.ram import Var
from ..emu.vs import GROUND_Y
from .perception import BLOCK_REACTS, STAGE_X

FIRST_T = 4
WALK_PX = 2
ATTACK_STATES = (0x0A, 0x0C)
JUMP_STATE = 0x04
RUN_STATES = ATTACK_STATES + (JUMP_STATE,)
STAND_STATE, CROUCH_STATE, GUARD_STATE, HIT_STATE, THROWN_STATE = 0x00, 0x02, 0x08, 0x0E, 0x14
THROW_CLASSES = (0x06, 0x0C)
NO_SPECIAL = 0xFF

ATTACK_ID_MAX = 59
THROW, PROJECTILE = 60, 61
STAND, WALK_TOWARD, WALK_AWAY, CROUCH, JUMP, BLOCK, HIT, THROWN = range(70, 78)
SPECIAL_BASE, SPECIAL_MAX = 80, 99          # a special without a box: 80 + its SPECIAL_CLASS (0x00..0x13)
RESERVED: Dict[int, str] = {
    THROW: "throw (no attack box; MOVE_CLASS 0x06 / 0x0C)", PROJECTILE: "fireball (own projectile slot active)",
    STAND: "standing", WALK_TOWARD: "walking toward the other fighter", WALK_AWAY: "walking away from the other fighter",
    CROUCH: "crouching", JUMP: "jumping (no attack in the jump)", BLOCK: "blocking", HIT: "being hit",
    THROWN: "being thrown"}
STAGES = (1, 2, 3)

# the extra RAM the collector logs per fighter (player 2 = +0x200), after sf2.emu.vs.VARS
EXTRA_FIELDS = (("aid", 0x0C3E), ("mclass", 0x0CBA), ("sclass", 0x0C49))
EXTRA_VARS: List[Var] = [Var("p%d_%s" % (p, n), a + 0x200 * (p - 1), 1, False) for p in (1, 2) for n, a in EXTRA_FIELDS]
EXTRA_NAMES = [v.name for v in EXTRA_VARS]

Rows = List[Dict[str, int]]


def code_name(code: int) -> str:
    return "act%02d" % code


def label(actor: str, code: int, stg: int) -> str:
    return "%s %s stg%d" % (actor, code_name(code), stg)


def stage_of(t: int, s: int, e: int) -> int:
    if not s <= t <= e:
        raise ValueError("row %d outside the episode [%d, %d]" % (t, s, e))
    return 1 + 3 * (t - s) // (e - s + 1)


def other(p: int) -> int:
    if p not in (1, 2):
        raise ValueError("player %r" % p)
    return 3 - p


def _f(r: Dict[str, int], p: int, name: str) -> int:
    return r["p%d_%s" % (p, name)]


def frame_code(rows: Rows, t: int, p: int) -> Optional[int]:
    """The code of a non-run row (state not in RUN_STATES); None = unknown."""
    r = rows[t]
    st = _f(r, p, "state")
    if st == THROWN_STATE:
        return THROWN
    if st == HIT_STATE:
        return BLOCK if _f(r, p, "react") in BLOCK_REACTS else HIT
    if st == GUARD_STATE:
        return BLOCK
    if _f(r, p, "y") != GROUND_Y:
        return JUMP
    if st == CROUCH_STATE:
        return CROUCH
    if st != STAND_STATE or t < FIRST_T:
        return None
    q = other(p)
    prev = rows[t - FIRST_T]
    xs = (_f(prev, p, "x"), _f(r, p, "x"), _f(r, q, "x"))
    if not all(0 <= x <= STAGE_X for x in xs):
        return None
    dx = xs[1] - xs[0]
    if abs(dx) < WALK_PX:
        return STAND
    side = xs[2] - xs[1]
    if side == 0:
        return None
    return WALK_TOWARD if dx * side > 0 else WALK_AWAY


def _mode(values: Sequence[int]) -> Optional[int]:
    if not values:
        return None
    counts: Dict[int, int] = {}
    for v in values:
        counts[v] = counts.get(v, 0) + 1
    return max(counts, key=lambda v: (counts[v], -values.index(v)))


def run_code(rows: Rows, s: int, e: int, p: int) -> Tuple[Optional[int], str]:
    """(code, reason) of the run-state run [s, e] of player ``p``; code None = unknown (reason says why)."""
    st = _f(rows[s], p, "state")
    first = next((_f(rows[i], p, "aid") for i in range(s, e + 1) if _f(rows[i], p, "aid")), 0)
    if first:
        return (first, "id") if first <= ATTACK_ID_MAX else (None, "id_range")
    if st == JUMP_STATE:
        return JUMP, "jump"
    tail = rows[s + 1:e + 1]
    if _mode([_f(r, p, "mclass") for r in tail]) in THROW_CLASSES:
        return THROW, "throw"
    if any(rows[i]["shot%d" % p] for i in range(s, e + 1)):
        return PROJECTILE, "projectile"
    sc = _mode([_f(r, p, "sclass") for r in tail])
    if st == 0x0C and sc is not None and sc != NO_SPECIAL:
        return (SPECIAL_BASE + sc, "special") if SPECIAL_BASE + sc <= SPECIAL_MAX else (None, "special_range")
    return None, "cut"


@dataclass(frozen=True)
class Episode:
    player: int
    start: int
    end: int                  # inclusive
    code: Optional[int]       # None: unknown, never labelled
    reason: str               # "id", "jump", "throw", "projectile", "special", "frame" / why unknown
    cut_end: bool = False     # closed by the round end (its last row has no now-image)

    @property
    def length(self) -> int:
        return self.end - self.start + 1


class ActorTrack:
    """One fighter's episodes, online: ``step(rows, k)`` after row k arrives returns the episode that row k closed
    (or None); ``finish(rows)`` closes the last one."""

    def __init__(self, player: int):
        other(player)
        self.p = player
        self.key: Optional[Tuple] = None
        self.start = 0

    def _key(self, rows: Rows, k: int) -> Tuple:
        if k < FIRST_T:
            return ("pre",)
        st = _f(rows[k], self.p, "state")
        if st in RUN_STATES:
            run_start = self.start if self.key is not None and self.key[0] == "run" and self.key[1] == st else k
            return ("run", st, run_start)
        return ("frame", frame_code(rows, k, self.p))

    def _episode(self, rows: Rows, e: int, cut_end: bool) -> Optional[Episode]:
        if self.key is None or self.key[0] == "pre" or e < self.start:
            return None
        if self.key[0] == "run":
            code, why = run_code(rows, self.start, e, self.p)
        else:
            code, why = self.key[1], ("frame" if self.key[1] is not None else "state")
        return Episode(self.p, self.start, e, code, why, cut_end)

    def step(self, rows: Rows, k: int) -> Optional[Episode]:
        key = self._key(rows, k)
        if key == self.key:
            return None
        closed = self._episode(rows, k - 1, False)
        self.key, self.start = key, k
        return closed

    def finish(self, rows: Rows) -> Optional[Episode]:
        ep = self._episode(rows, len(rows) - 1, True)
        self.key = None
        return ep


def episodes_of(rows: Rows, p: int) -> List[Episode]:
    """Every episode of player ``p`` in a whole game's rows (the online tracker run over them)."""
    tr, out = ActorTrack(p), []
    for k in range(len(rows)):
        ep = tr.step(rows, k)
        if ep is not None:
            out.append(ep)
    last = tr.finish(rows)
    if last is not None:
        out.append(last)
    return out


def labels_at(rows: Rows, p: int) -> Dict[int, Tuple[Episode, int]]:
    """Row t -> (its episode, stage) for every labelled row (code known) of player ``p``."""
    out: Dict[int, Tuple[Episode, int]] = {}
    for ep in episodes_of(rows, p):
        if ep.code is not None:
            for t in range(ep.start, ep.end + 1):
                out[t] = (ep, stage_of(t, ep.start, ep.end))
    return out


def table() -> Dict:
    """The fixed code table (lessons/action_codes.json)."""
    return {
        "label": "<actor> act<NN> stg<1-3>",
        "attack_ids": "1..%d: the fighter's own ROM attack ID (0x0C3E me / 0x0E3E him), per character" % ATTACK_ID_MAX,
        "reserved": {code_name(c): d for c, d in sorted(RESERVED.items())},
        "specials_without_box": "%s..%s: %s + SPECIAL_CLASS (0x0C49 / 0x0E49), a special put out no box" % (
            code_name(SPECIAL_BASE), code_name(SPECIAL_MAX), code_name(SPECIAL_BASE)),
        "unknown_dropped": ["an attack cut before any ID or class (cut)", "an ID above %d (id_range)" % ATTACK_ID_MAX,
                            "state 0x06 and other unnamed states", "an impossible x on a standing row",
                            "rows before t = %d" % FIRST_T],
        "stage": "stg = 1 + 3 * (t - start) // length of the fighter's episode (thirds: start / middle / end)",
        "episode": "a run of state 0x0A / 0x0C / 0x04 (one code, the run's first non-zero attack ID; a jump with an "
                   "attack is labelled with the attack, not split), else a run of rows with one code",
        "ram": {v.name: "0x%04X" % v.addr for v in EXTRA_VARS},
    }
