"""The move list for a VS BATTLE move test: each fighter's movement, 18 normals, throws, blocks, specials and combos,
as relative-token scripts (F / B / U / D + lp mp hp lk mk hk; directions resolved per side by sf2.emu.vs.physical), the
range each is tried at, and the RAM fact that proves it came out (docs: laya_two_system/docs/MOVES.md; special
input timings from its ROM-verified macros).

A Move's ``check`` is a mechanical yes/no over the recorded rows (a_ = the fighter doing the move, d_ = the other);
the measurements (sf2/data/vs_metrics.py) are data, not a gate.
"""
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from ..emu.vs import GROUND_Y, Step

ATTACK, SPECIAL, HIT, GUARD, THROWN, JUMP = 0x0A, 0x0C, 0x0E, 0x08, 0x14, 0x04
BLOCK_REACTS = (0x06, 0x08)
BUTTONS = ("lp", "mp", "hp", "lk", "mk", "hk")
# Gaps (|x1 - x2|, world px) the setups use. Fight start is 96.
GAPS = {"close": 36, "mid": 70, "far": 96, "wide": 150}
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


# ---------------------------------------------------------------------------------------------------- builders
def _btn(b: str, n: int = 2) -> Tuple[Step, ...]:
    return (((b,), n),)


def movement() -> List[Move]:
    j = lambda d: (((("U",) + d), 4),)  # noqa: E731
    return [
        Move("walk_forward", "movement", "wide", ((("F",), 40),), lambda r: toward(r) >= 20, "moves >=20 px toward"),
        Move("walk_back", "movement", "far", ((("B",), 40),), lambda r: toward(r) <= -20, "moves >=20 px away"),
        Move("crouch", "movement", "far", ((("D",), 30),), lambda r: seen(r, a_state=2), "state 02 crouch"),
        Move("jump_up", "movement", "wide", j(()),
             lambda r: min(x["a_y"] for x in r) < GROUND_Y - 30 and abs(toward(r)) < 4, "rises >30 px, lands in place"),
        Move("jump_forward", "movement", "wide", j(("F",)),
             lambda r: min(x["a_y"] for x in r) < GROUND_Y - 30 and toward(r) > 10, "rises, lands >10 px toward"),
        Move("jump_back", "movement", "far", j(("B",)),
             lambda r: min(x["a_y"] for x in r) < GROUND_Y - 30 and toward(r) < -10, "rises, lands >10 px away"),
    ]


def normals() -> List[Move]:
    out: List[Move] = []
    for b in BUTTONS:
        out.append(Move("s." + b, "normal", "far", _btn(b), _attack_seen, "attack state 0A", sweep=True))
        out.append(Move("cl." + b, "normal", "close", _btn(b), lambda r: _attack_seen(r) and connected(r),
                        "attack state 0A and it connects"))
        out.append(Move("c." + b, "normal", "close", ((("D", b), 2), (("D",), 14)),
                        lambda r: _attack_seen(r) and connected(r) and seen(r, a_state=ATTACK),
                        "crouching attack connects", sweep=True))
        out.append(Move("j." + b, "normal", "close", ((("U",), 4), ((), 14), ((b,), 2)),
                        lambda r: _attack_seen(r) and min(x["a_y"] for x in r) < GROUND_Y - 30,
                        "air attack (0A or 04/06) in a neutral jump"))
        out.append(Move("jf." + b, "normal", "far", ((("U", "F"), 4), ((), 16), ((b,), 2)),
                        _attack_seen, "air attack in a forward jump (jump-in)"))
    return out


def blocks() -> List[Move]:
    hp = (((), 2), (("hp",), 2))
    low = (((), 2), (("D", "mk"), 2), (("D",), 12))
    return [
        Move("block_high", "block", "close", ((("B",), 40),), _blocked, "guards the standing fierce, no life lost",
             other=hp, defends=True),
        Move("block_low", "block", "close", ((("D", "B"), 40),), _blocked, "guards the crouching forward, no damage",
             other=low, defends=True),
    ]


def throws(buttons: Sequence[str]) -> List[Move]:
    return [Move("throw_F+" + b, "throw", "close", ((("F", b), 2), (("F",), 2)), lambda r: seen(r, d_state=THROWN),
                 "defender state 14 (thrown)", sweep=True) for b in buttons]


def _special(sid: Optional[int]) -> Callable[[Rows], bool]:
    """State 0C seen; for player 1 also the special id (0x0D80) — player 2's id byte is not verified."""
    def check(rows: Rows) -> bool:
        hits = [r for r in rows if r["a_state"] == SPECIAL]
        return bool(hits) and (sid is None or rows[0]["attacker"] != 1 or hits[0]["a_special"] == sid)
    return check


def _fireball(rows: Rows) -> bool:
    return _special(0x00)(rows) and any(r["shot1"] or r["shot2"] for r in rows)


HADOKEN = ((("D",), 2), (("D", "F"), 2), (("F", "hp"), 2), ((), 2))
SHORYUKEN = ((("F",), 2), (("D",), 2), (("D", "F", "hp"), 2), ((), 2))
TATSUMAKI = ((("D",), 2), (("D", "B"), 2), (("B", "hk"), 2), ((), 2))
LEGS = ((("lk",), 1), ((), 1)) * 12
BIRD = ((("D",), 64), (("U", "hk"), 2), ((), 2))
JUMP_IN = ((("U", "F"), 4), ((), 18), (("hk",), 2), ("until", "landed", (), 60))


def ryu() -> List[Move]:
    lp_fb = ((("D",), 2), (("D", "F"), 2), (("F", "lp"), 2), ((), 2))
    return movement() + normals() + blocks() + throws(["hp", "hk"]) + [
        Move("hadoken_lp", "special", "far", lp_fb, _fireball, "state 0C, special 00, projectile slot active"),
        Move("hadoken_hp", "special", "far", HADOKEN, _fireball, "state 0C, special 00, projectile slot active",
             sweep=True),
        Move("shoryuken_hp", "special", "close", SHORYUKEN,
             lambda r: _special(0x04)(r) and min(x["a_y"] for x in r) < GROUND_Y - 30, "state 0C, special 04, rises",
             sweep=True),
        Move("tatsumaki_hk", "special", "mid", TATSUMAKI, _special(0x02), "state 0C, special 02", sweep=True),
        Move("c.mk_xx_hadoken", "combo", "close", ((("D", "mk"), 2), (("D",), 2), (("D", "F"), 2), (("F", "hp"), 2),
                                                   ((), 2)), lambda r: combo(r, 2), "2 hits, no gap", hits=2),
        Move("c.mk_xx_shoryuken", "combo", "close", ((("D", "mk"), 2), (("F",), 2), (("D",), 2),
                                                     (("D", "F", "hp"), 2), ((), 2)),
             lambda r: combo(r, 2), "2+ hits, no gap", hits=2),
        Move("jf.hk_cl.hp_xx_hadoken", "combo", "far", JUMP_IN + ((("hp",), 2),) + HADOKEN,
             lambda r: combo(r, 3), "3 hits, no gap", hits=3),
    ]


def chunli() -> List[Move]:
    return movement() + normals() + blocks() + throws(["hp", "mp"]) + [
        Move("lightning_legs", "special", "close", LEGS, _special(0x02), "state 0C, special 02", sweep=True),
        Move("spinning_bird_kick", "special", "mid", BIRD, _special(0x00), "state 0C, special 00", sweep=True),
        Move("jf.hk_s.mp_s.hp", "combo", "far", JUMP_IN + ((("mp",), 2), ((), 8), (("hp",), 2)),
             lambda r: combo(r, 3), "3 hits, no gap", hits=3),
        Move("jf.mk_legs", "combo", "far", ((("U", "F"), 4), ((), 18), (("mk",), 2), ("until", "landed", (), 60))
             + LEGS, lambda r: combo(r, 2), "jump-in then Lightning Legs, no gap", hits=2),
    ]


MOVESETS: Dict[str, Callable[[], List[Move]]] = {"ryu": ryu, "ken": ryu, "chunli": chunli}
CONDS = {"landed": lambda now, start: now["a_y"] == GROUND_Y and now["a_state"] != JUMP}
REACH_GAPS = list(range(16, 204, 6))
