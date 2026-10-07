"""RAM-free move mechanics: every move's NAME, kind, setup gap and button-step script, with NO RAM (nothing here
imports sf2.emu, sf2.data.vs_defense or any RAM-row builder), so the screen-only play runner can import the move
menu and press the buttons without pulling the emulator / RAM map in.

This is the mechanics half of sf2.data.vs_moves: the names match sf2.system1.action_menu.CATEGORIES and
sf2.data.vs_moves.chunli() exactly. The RAM half (each move's ``check`` over recorded RAM rows, the ``expect``
words, the fact predicates) stays in vs_moves, which re-imports these descriptors and attaches a check to each.

A ``MoveSteps`` carries everything about a move except the RAM check: the relative-token steps (F / B / U / D +
lp mp hp lk mk hk; the side is resolved per fighter at press time, by the play runner's own physical map), the kind,
the setup gap key, and the measurement flags (other inputs, defends, sweep, combo hits) that are plain data.
"""
from typing import Dict, List, NamedTuple, Sequence, Tuple

Step = Tuple  # (tokens, n) fixed frames | ("until", cond_name, tokens, max_frames) -- same shape as sf2.emu.vs.Step

BUTTONS: Tuple[str, ...] = ("lp", "mp", "hp", "lk", "mk", "hk")
# Setup gaps (|x1 - x2|, world px) a move is tried at; the reach sweep tries every REACH_GAPS gap. Pure numbers.
GAPS: Dict[str, int] = {"close": 36, "mid": 70, "far": 96, "wide": 150}
REACH_GAPS: List[int] = list(range(16, 204, 6))


class MoveSteps(NamedTuple):
    """A move's RAM-free description: what to press and the plain metadata, never how RAM judges it."""
    name: str
    kind: str                       # movement | normal | throw | block | special | combo
    gap: str                        # key of GAPS
    steps: Tuple[Step, ...]         # the acting fighter's inputs
    other: Tuple[Step, ...] = ()    # the other fighter's inputs (blocks: the attack being blocked)
    defends: bool = False           # the acting fighter is the defender (blocks)
    sweep: bool = False             # measure reach: try at every gap in REACH_GAPS
    hits: int = 0                   # combos: connected hits the move aims for


# ---------------------------------------------------------------------------------------------------- step scripts
# Special / combo input scripts, relative tokens. ROM-verified timings (laya_two_system sf2/actions.py).
HADOKEN: Tuple[Step, ...] = ((("D",), 2), (("D", "F"), 2), (("F", "hp"), 2), ((), 2))
SHORYUKEN: Tuple[Step, ...] = ((("F",), 2), (("D",), 2), (("D", "F", "hp"), 2), ((), 2))
TATSUMAKI: Tuple[Step, ...] = ((("D",), 2), (("D", "B"), 2), (("B", "hk"), 2), ((), 2))
LEGS: Tuple[Step, ...] = ((("lk",), 1), ((), 1)) * 12
BIRD: Tuple[Step, ...] = ((("D",), 64), (("U", "hk"), 2), ((), 2))
JUMP_IN: Tuple[Step, ...] = ((("U", "F"), 4), ((), 18), (("hk",), 2), ("until", "landed", (), 60))
# E.Honda: Hundred Hand Slap = mash punch (no motion); Sumo Headbutt / Sumo Smash = charge moves (hold the direction
# inside the macro, like BIRD, so no charge STATE is needed in the key). Which fire on this World-Warrior ROM is
# verified on screen; non-firing candidates are dropped from honda().
HUNDRED_HAND: Tuple[Step, ...] = ((("lp",), 1), ((), 1)) * 14
HEADBUTT: Tuple[Step, ...] = ((("B",), 56), (("F", "hp"), 2), ((), 2))
SUMO_SMASH: Tuple[Step, ...] = ((("D",), 56), (("U", "hk"), 2), ((), 2))
# Zangief: Spinning Piledriver = a 360 joystick rotation + P (a CLOSE command throw). On-screen probe
# (scratchpad zangief_probe*) verified the ROM only registers the special (p1_state 0x0C) when the
# rotation is a FULL semicircle-plus with the punch on a dedicated final up-frame; a shorter rotation
# with punch folded into the last diagonal came out as a plain normal. This script fired 0x0C with
# only ~2 stray jump frames (the mirror B->..->UF variant jumped 47 frames -- rejected). The grab is
# range-gated like every throw (learned in play). Double/Quick Lariat = all-punch / all-kick, no motion.
SPD: Tuple[Step, ...] = ((("F",), 2), (("D", "F"), 2), (("D",), 2), (("D", "B"), 2),
                         (("B",), 2), (("U", "B"), 2), (("U", "hp"), 1), ((), 2))
LARIAT: Tuple[Step, ...] = ((("lp", "mp", "hp"), 2), ((), 6))
QLARIAT: Tuple[Step, ...] = ((("lk", "mk", "hk"), 2), ((), 6))


def _btn(b: str, n: int = 2) -> Tuple[Step, ...]:
    return (((b,), n),)


# ---------------------------------------------------------------------------------------------------- builders
def movement() -> List[MoveSteps]:
    j = lambda d: (((("U",) + d), 4),)  # noqa: E731
    return [
        MoveSteps("walk_forward", "movement", "wide", ((("F",), 40),)),
        MoveSteps("walk_back", "movement", "far", ((("B",), 40),)),
        MoveSteps("crouch", "movement", "far", ((("D",), 30),)),
        MoveSteps("jump_up", "movement", "wide", j(())),
        MoveSteps("jump_forward", "movement", "wide", j(("F",))),
        MoveSteps("jump_back", "movement", "far", j(("B",))),
    ]


def normals() -> List[MoveSteps]:
    out: List[MoveSteps] = []
    for b in BUTTONS:
        out.append(MoveSteps("s." + b, "normal", "far", _btn(b), sweep=True))
        out.append(MoveSteps("cl." + b, "normal", "close", _btn(b)))
        out.append(MoveSteps("c." + b, "normal", "close", ((("D", b), 2), (("D",), 14)), sweep=True))
        out.append(MoveSteps("j." + b, "normal", "close", ((("U",), 4), ((), 14), ((b,), 2))))
        out.append(MoveSteps("jf." + b, "normal", "far", ((("U", "F"), 4), ((), 16), ((b,), 2))))
    return out


def blocks() -> List[MoveSteps]:
    hp = (((), 2), (("hp",), 2))
    low = (((), 2), (("D", "mk"), 2), (("D",), 12))
    return [
        MoveSteps("block_high", "block", "close", ((("B",), 40),), other=hp, defends=True),
        MoveSteps("block_low", "block", "close", ((("D", "B"), 40),), other=low, defends=True),
    ]


def throws(buttons: Sequence[str]) -> List[MoveSteps]:
    return [MoveSteps("throw_F+" + b, "throw", "close", ((("F", b), 2), (("F",), 2)), sweep=True) for b in buttons]


def ryu() -> List[MoveSteps]:
    lp_fb = ((("D",), 2), (("D", "F"), 2), (("F", "lp"), 2), ((), 2))
    return movement() + normals() + blocks() + throws(["hp", "hk"]) + [
        MoveSteps("hadoken_lp", "special", "far", lp_fb),
        MoveSteps("hadoken_hp", "special", "far", HADOKEN, sweep=True),
        MoveSteps("shoryuken_hp", "special", "close", SHORYUKEN, sweep=True),
        MoveSteps("tatsumaki_hk", "special", "mid", TATSUMAKI, sweep=True),
        MoveSteps("c.mk_xx_hadoken", "combo", "close",
                  ((("D", "mk"), 2), (("D",), 2), (("D", "F"), 2), (("F", "hp"), 2), ((), 2)), hits=2),
        MoveSteps("c.mk_xx_shoryuken", "combo", "close",
                  ((("D", "mk"), 2), (("F",), 2), (("D",), 2), (("D", "F", "hp"), 2), ((), 2)), hits=2),
        MoveSteps("jf.hk_cl.hp_xx_hadoken", "combo", "far", JUMP_IN + ((("hp",), 2),) + HADOKEN, hits=3),
    ]


def chunli() -> List[MoveSteps]:
    return movement() + normals() + blocks() + throws(["hp", "mp"]) + [
        MoveSteps("lightning_legs", "special", "close", LEGS, sweep=True),
        MoveSteps("spinning_bird_kick", "special", "mid", BIRD, sweep=True),
        MoveSteps("jf.hk_s.mp_s.hp", "combo", "far", JUMP_IN + ((("mp",), 2), ((), 8), (("hp",), 2)), hits=3),
        MoveSteps("jf.mk_legs", "combo", "far",
                  ((("U", "F"), 4), ((), 18), (("mk",), 2), ("until", "landed", (), 60)) + LEGS, hits=2),
    ]


def honda() -> List[MoveSteps]:
    return movement() + normals() + blocks() + throws(["hp", "mp"]) + [
        MoveSteps("hundred_hand_slap", "special", "close", HUNDRED_HAND, sweep=True),   # mash punch (signature)
        MoveSteps("sumo_headbutt", "special", "mid", HEADBUTT, sweep=True),             # charge B->F+hp (verify fires)
        MoveSteps("sumo_smash", "special", "mid", SUMO_SMASH, sweep=True),              # charge D->U+hk (verify fires)
    ]


def zangief() -> List[MoveSteps]:
    return movement() + normals() + blocks() + throws(["hp", "hk"]) + [
        MoveSteps("spinning_piledriver", "special", "close", SPD, sweep=True),   # 360+P command throw (verify fires)
        MoveSteps("double_lariat", "special", "mid", LARIAT, sweep=True),         # PPP spin (anti-air / anti-fireball)
        MoveSteps("quick_lariat", "special", "mid", QLARIAT, sweep=True),         # KKK spin (faster, verify fires)
    ]


MENUS: Dict[str, "callable"] = {"ryu": ryu, "ken": ryu, "chunli": chunli, "honda": honda, "zangief": zangief}


def menu(char: str) -> List[MoveSteps]:
    """Every move the character has, as RAM-free descriptors (the mechanics for sf2.system1.action_menu's names)."""
    if char not in MENUS:
        raise ValueError("no move menu for %r" % char)
    return MENUS[char]()


def steps_of(char: str) -> Dict[str, Tuple[Step, ...]]:
    """name -> button-step script, for the play runner to press."""
    return {m.name: m.steps for m in menu(char)}


def kind_of(char: str) -> Dict[str, str]:
    """name -> kind (movement / normal / throw / block / special / combo)."""
    return {m.name: m.kind for m in menu(char)}
