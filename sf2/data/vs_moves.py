"""The move list for a VS BATTLE move test: each fighter's movement, 18 normals, throws, blocks, specials and combos,
as relative-token scripts (F / B / U / D + lp mp hp lk mk hk; directions resolved per side by sf2.emu.vs.physical), the
range each is tried at, and the RAM fact that proves it came out (docs: laya_two_system/docs/MOVES.md; special
input timings from its ROM-verified macros).

A Move's ``check`` is a mechanical yes/no over the recorded rows (a_ = the fighter doing the move, d_ = the other);
the measurements (sf2/data/vs_metrics.py) are data, not a gate.

The RAM-FREE half of this module -- the move names, kinds, setup gaps and button-step scripts -- lives in
sf2.moves_free (which imports no RAM), so the screen-only play runner can import the move menu without pulling RAM in.
Here each RAM-free ``MoveSteps`` descriptor (``_free_*``) gets its ``check`` / ``expect`` attached (``_attach``); the
fact predicates (seen, life_drops, connected, gap, toward, combo, _special, _fireball) stay here: they read RAM rows.
"""
from dataclasses import dataclass
from typing import Callable, Dict, List, Sequence, Tuple

from ..emu.vs import GROUND_Y
from ..moves_free import (BIRD, BUTTONS, GAPS, HADOKEN, JUMP_IN, LEGS, MoveSteps, REACH_GAPS,  # noqa: F401
                          SHORYUKEN, Step, TATSUMAKI)
from ..moves_free import blocks as _free_blocks
from ..moves_free import chunli as _free_chunli
from ..moves_free import movement as _free_movement
from ..moves_free import normals as _free_normals
from ..moves_free import ryu as _free_ryu
from ..moves_free import throws as _free_throws

ATTACK, SPECIAL, HIT, GUARD, THROWN, JUMP = 0x0A, 0x0C, 0x0E, 0x08, 0x14, 0x04
BLOCK_REACTS = (0x06, 0x08)
Rows = List[Dict[str, int]]


@dataclass(frozen=True)
class Move:
    name: str
    kind: str                      # movement | normal | throw | block | special | combo
    gap: str                       # key of GAPS
    steps: Tuple[Step, ...]        # the measured fighter's inputs
    check: Callable[[Rows], bool]
    expect: str                    # the check in words (goes into the ledger)
    other: Tuple[Step, ...] = ()   # the other fighter's inputs (blocks: the attack being blocked)
    defends: bool = False          # the measured fighter is the defender (blocks)
    sweep: bool = False            # measure reach: try it at every gap in REACH_GAPS
    hits: int = 0                  # combos: connected hits the check requires


# ---------------------------------------------------------------------------------------------------- facts
def seen(rows: Rows, **want) -> bool:
    return any(all(r[k] == v for k, v in want.items()) for r in rows)


def life_drops(rows: Rows, who: str = "d") -> List[int]:
    """Frame indices where ``who``'s true life went down (one per hit; a KO wrap to 255 is not a drop)."""
    key = who + "_life"
    return [i for i in range(1, len(rows)) if rows[i][key] < rows[i - 1][key]]


def connected(rows: Rows) -> bool:
    """The attack touched the other fighter: hit (life drop / hit stun), block stun, or thrown."""
    return bool(life_drops(rows)) or any(r["d_state"] in (HIT, THROWN) for r in rows)


def gap(r: Dict[str, int]) -> int:
    return abs(r["a_x"] - r["d_x"])


def toward(rows: Rows) -> int:
    """Attacker's x travel toward where the defender was at the start (negative: away)."""
    sign = 1 if rows[0]["a_x"] < rows[0]["d_x"] else -1
    return sign * (rows[-1]["a_x"] - rows[0]["a_x"])


def combo(rows: Rows, n: int) -> bool:
    """n or more hits and the defender never left hit stun between the first and the last (no block, no reset)."""
    drops = life_drops(rows)
    if len(drops) < n:
        return False
    return all(rows[i]["d_state"] in (HIT, THROWN) for i in range(drops[0], drops[-1] + 1))


def _attack_seen(rows: Rows) -> bool:
    return any(r["a_state"] in (ATTACK, SPECIAL) or (r["a_state"] == JUMP and r["a_sub"] == 0x06) for r in rows)


def _blocked(rows: Rows) -> bool:
    """The measured fighter (the DEFENDER of the exchange) guarded: guard state or block stun, no life lost."""
    guard = any(r["d_state"] == GUARD or (r["d_state"] == HIT and r["d_react"] in BLOCK_REACTS) for r in rows)
    return guard and not life_drops(rows)


def _special(sid):
    """State 0C seen; for player 1 also the special id (0x0D80) -- player 2's id byte is not verified."""
    def check(rows: Rows) -> bool:
        hits = [r for r in rows if r["a_state"] == SPECIAL]
        return bool(hits) and (sid is None or rows[0]["attacker"] != 1 or hits[0]["a_special"] == sid)
    return check


def _fireball(rows: Rows) -> bool:
    return _special(0x00)(rows) and any(r["shot1"] or r["shot2"] for r in rows)


# ---------------------------------------------------------------------------------------------------- builders
# Each RAM-free descriptor (sf2.moves_free) gets its RAM check / expect words attached here.
def _attach(ms: MoveSteps, check: Callable[[Rows], bool], expect: str) -> Move:
    return Move(ms.name, ms.kind, ms.gap, ms.steps, check, expect,
                other=ms.other, defends=ms.defends, sweep=ms.sweep, hits=ms.hits)


def movement() -> List[Move]:
    checks: Dict[str, Tuple[Callable[[Rows], bool], str]] = {
        "walk_forward": (lambda r: toward(r) >= 20, "moves >=20 px toward"),
        "walk_back": (lambda r: toward(r) <= -20, "moves >=20 px away"),
        "crouch": (lambda r: seen(r, a_state=2), "state 02 crouch"),
        "jump_up": (lambda r: min(x["a_y"] for x in r) < GROUND_Y - 30 and abs(toward(r)) < 4,
                    "rises >30 px, lands in place"),
        "jump_forward": (lambda r: min(x["a_y"] for x in r) < GROUND_Y - 30 and toward(r) > 10,
                         "rises, lands >10 px toward"),
        "jump_back": (lambda r: min(x["a_y"] for x in r) < GROUND_Y - 30 and toward(r) < -10,
                      "rises, lands >10 px away"),
    }
    return [_attach(ms, *checks[ms.name]) for ms in _free_movement()]


def normals() -> List[Move]:
    out: List[Move] = []
    for ms in _free_normals():
        prefix = ms.name.split(".", 1)[0]
        if prefix == "s":
            chk, exp = _attack_seen, "attack state 0A"
        elif prefix == "cl":
            chk, exp = (lambda r: _attack_seen(r) and connected(r)), "attack state 0A and it connects"
        elif prefix == "c":
            chk = lambda r: _attack_seen(r) and connected(r) and seen(r, a_state=ATTACK)  # noqa: E731
            exp = "crouching attack connects"
        elif prefix == "j":
            chk = lambda r: _attack_seen(r) and min(x["a_y"] for x in r) < GROUND_Y - 30  # noqa: E731
            exp = "air attack (0A or 04/06) in a neutral jump"
        else:  # jf
            chk, exp = _attack_seen, "air attack in a forward jump (jump-in)"
        out.append(_attach(ms, chk, exp))
    return out


def blocks() -> List[Move]:
    exp = {"block_high": "guards the standing fierce, no life lost",
           "block_low": "guards the crouching forward, no damage"}
    return [_attach(ms, _blocked, exp[ms.name]) for ms in _free_blocks()]


def throws(buttons: Sequence[str]) -> List[Move]:
    return [_attach(ms, lambda r: seen(r, d_state=THROWN), "defender state 14 (thrown)")
            for ms in _free_throws(buttons)]


def ryu() -> List[Move]:
    specials: Dict[str, Tuple[Callable[[Rows], bool], str]] = {
        "hadoken_lp": (_fireball, "state 0C, special 00, projectile slot active"),
        "hadoken_hp": (_fireball, "state 0C, special 00, projectile slot active"),
        "shoryuken_hp": (lambda r: _special(0x04)(r) and min(x["a_y"] for x in r) < GROUND_Y - 30,
                         "state 0C, special 04, rises"),
        "tatsumaki_hk": (_special(0x02), "state 0C, special 02"),
        "c.mk_xx_hadoken": (lambda r: combo(r, 2), "2 hits, no gap"),
        "c.mk_xx_shoryuken": (lambda r: combo(r, 2), "2+ hits, no gap"),
        "jf.hk_cl.hp_xx_hadoken": (lambda r: combo(r, 3), "3 hits, no gap"),
    }
    base = movement() + normals() + blocks() + throws(["hp", "hk"])
    return base + [_attach(ms, *specials[ms.name]) for ms in _free_ryu() if ms.name in specials]


def chunli() -> List[Move]:
    specials: Dict[str, Tuple[Callable[[Rows], bool], str]] = {
        "lightning_legs": (_special(0x02), "state 0C, special 02"),
        "spinning_bird_kick": (_special(0x00), "state 0C, special 00"),
        "jf.hk_s.mp_s.hp": (lambda r: combo(r, 3), "3 hits, no gap"),
        "jf.mk_legs": (lambda r: combo(r, 2), "jump-in then Lightning Legs, no gap"),
    }
    base = movement() + normals() + blocks() + throws(["hp", "mp"])
    return base + [_attach(ms, *specials[ms.name]) for ms in _free_chunli() if ms.name in specials]


MOVESETS: Dict[str, Callable[[], List[Move]]] = {"ryu": ryu, "ken": ryu, "chunli": chunli}
CONDS = {"landed": lambda now, start: now["a_y"] == GROUND_Y and now["a_state"] != JUMP}
