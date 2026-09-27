"""The options laya-vision chooses from, and the frame-by-frame inputs the glue code stuffs for each.

Directions are *relative*: F = toward the opponent, B = away. The glue resolves them to LEFT/RIGHT at decision
time, so the model never has to know which side it is on: WALKS from the fighters' x positions, every other move from
the ROM's own facing byte, which it mirrors the stick by (sf2.env.FightEnv.act). (The plan's
"left/right" become "forward/back"; "block" is down-back, the crouching guard, so it differs from "back".)
"""
from typing import Dict, List, Sequence, Tuple

from .config import PAD

CHARACTERS: List[str] = ["ryu", "ken", "honda", "blanka", "guile", "chunli", "zangief", "dhalsim"]
# Shared by all 8 characters (docs/MOVES.md), verified on the ROM with Chun-Li (tests/test_rom_harness.py).
BASICS: List[str] = [
    "idle", "forward", "back", "jump", "jump_forward", "crouch",
    "lp", "hp", "lk", "hk",
    "block", "throw", "sweep",
]
# Each character's specials, in the order the model is shown them. A character missing here has no specials in the
# code yet (the ALL-8 gate counts that as missing moves).
SPECIALS: Dict[str, List[str]] = {
    "ryu": ["hadoken", "shoryuken", "tatsumaki"],
    "ken": ["hadoken", "shoryuken", "tatsumaki"],
    "chunli": ["lightning_legs"],
}


def moves(character: str) -> List[str]:
    """The move list of one character: the shared basics, then its specials."""
    if character not in CHARACTERS:
        raise KeyError("unknown character %r (one of %s)" % (character, ", ".join(CHARACTERS)))
    return BASICS + SPECIALS.get(character, [])


# The 14-action set of the Chun-Li datasets and checkpoints (datasets, metrics, the playbook contract use it).
ACTIONS: List[str] = moves("chunli")
INDEX = {a: i for i, a in enumerate(ACTIONS)}

CRITERIA: Dict[str, str] = {
    "idle": "do nothing this moment",
    "forward": "walk toward the opponent",
    "back": "walk away from the opponent (standing guard)",
    "jump": "jump straight up",
    "jump_forward": "jump toward the opponent",
    "crouch": "crouch down",
    "lp": "light punch, fast and short",
    "hp": "fierce punch, slow and strong",
    "lk": "light kick, fast low-risk poke",
    "hk": "roundhouse kick, long reach",
    "block": "crouching block against an incoming attack",
    "throw": "throw the opponent when right next to him (a fierce punch otherwise)",
    "sweep": "crouching roundhouse, knocks him down",
    "lightning_legs": "Lightning Legs: a flurry of kicks, strong up close",
    "hadoken": "Hadoken: a fireball that travels along the ground toward him",
    "shoryuken": "Shoryuken: a rising uppercut, beats jump-ins",
    "tatsumaki": "Hurricane Kick: spins forward through the air with one leg out",
}

INSTRUCTIONS = ("You are the fighter on the left health bar in Street Fighter II. The images are the screen a "
                "moment ago and now; the note gives both fighters' state. Which move should you make now?")

# One step = (tokens held, number of frames). Tokens: U D F B + attack names from config.PAD.
Step = Tuple[Tuple[str, ...], int]
_TAP = 2  # frames an attack button is held, then released for _TAP frames so the next press registers
MACROS: Dict[str, List[Step]] = {
    "idle": [((), 4)],
    "forward": [(("F",), 4)],
    "back": [(("B",), 4)],
    "jump": [(("U",), 4)],
    "jump_forward": [(("U", "F"), 4)],
    "crouch": [(("D",), 4)],
    "lp": [(("lp",), _TAP), ((), _TAP)],
    "hp": [(("hp",), _TAP), ((), _TAP)],
    "lk": [(("lk",), _TAP), ((), _TAP)],
    "hk": [(("hk",), _TAP), ((), _TAP)],
    "block": [(("D", "B"), 6)],
    # toward + fierce on the same frame: a throw within 42 px at the press (tests/test_rom_harness.py), else a fierce
    "throw": [(("F", "hp"), _TAP), (("F",), _TAP)],
    # down + roundhouse: knocks him down when it connects (reach ~70 px; tests/test_rom_harness.py)
    "sweep": [(("D", "hk"), _TAP), (("D",), _TAP)],
    # 12 short taps, 1 frame down, 1 up: the Legs (state 0C) start at frame 18 (tests/test_rom_harness.py)
    "lightning_legs": [(("lk",), 1), ((), 1)] * 12,
    # Ryu and Ken, the same inputs (docs/MOVES.md; tests/test_rom_moves.py): 2 frames per direction, the button on
    # the last one, 2 released.
    # Fierce / roundhouse: every strength "works as defined" the same way; the strongest shows it most (fastest
    # fireball, highest rise, longest spin). Walking forward just before a Hadoken turns it into a Shoryuken on the ROM.
    "hadoken": [(("D",), 2), (("D", "F"), 2), (("F", "hp"), 2), ((), 2)],
    "shoryuken": [(("F",), 2), (("D",), 2), (("D", "F", "hp"), 2), ((), 2)],
    "tatsumaki": [(("D",), 2), (("D", "B"), 2), (("B", "hk"), 2), ((), 2)],
}
# Moves that only travel: resolved from x, so "forward" walks toward him even while the ROM's facing byte lags (the
# ROM does not turn her round while its own forward is held, so she would walk on away from him: tests/test_rom_*.py).
WALKS = frozenset({"forward", "back", "jump_forward"})
assert set(MACROS) == set(CRITERIA) == set(BASICS).union(*SPECIALS.values())


def question(character: str = "chunli") -> Dict:
    """The one question asked at train and play time: the character's own moves. Keep it byte-identical across
    both."""
    return {"type": "choice", "instructions": INSTRUCTIONS, "criteria": {a: CRITERIA[a] for a in moves(character)}}


def expand(action: str) -> List[Tuple[str, ...]]:
    """Per-frame relative tokens for one action."""
    out: List[Tuple[str, ...]] = []
    for tokens, n in MACROS[action]:
        out.extend([tokens] * n)
    return out


def to_physical(tokens: Sequence[str], facing_right: bool) -> List[str]:
    """Relative tokens -> SNES button names (Mesen's)."""
    fwd, back = ("right", "left") if facing_right else ("left", "right")
    names = []
    for t in tokens:
        if t == "U":
            names.append("up")
        elif t == "D":
            names.append("down")
        elif t == "F":
            names.append(fwd)
        elif t == "B":
            names.append(back)
        else:
            names.append(PAD[t])
    return names
