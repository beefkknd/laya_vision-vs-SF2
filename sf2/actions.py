"""The 12 options laya-vision chooses from, and the frame-by-frame inputs the glue code stuffs for each.

Directions are *relative*: F = toward the opponent, B = away. The glue resolves them to LEFT/RIGHT from the
fighters' x positions at decision time, so the model never has to know which side it is on. (The plan's
"left/right" become "forward/back"; "block" is down-back, the crouching guard, so it differs from "back".)
"""
from typing import Dict, List, Sequence, Tuple

from .config import PAD

ACTIONS: List[str] = [
    "idle", "forward", "back", "jump", "crouch",
    "lp", "hp", "lk", "hk",
    "block",
    "hadouken", "shoryuken",
]
INDEX = {a: i for i, a in enumerate(ACTIONS)}

CRITERIA: Dict[str, str] = {
    "idle": "do nothing this moment",
    "forward": "walk toward the opponent",
    "back": "walk away from the opponent (standing guard)",
    "jump": "jump straight up",
    "crouch": "crouch down",
    "lp": "light punch, fast and short",
    "hp": "fierce punch, slow and strong",
    "lk": "light kick, fast low-risk poke",
    "hk": "roundhouse kick, long reach",
    "block": "crouching block against an incoming attack",
    "hadouken": "throw a fireball at the opponent from mid or far range",
    "shoryuken": "dragon punch: anti-air uppercut against a jumping opponent",
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
    "crouch": [(("D",), 4)],
    "lp": [(("lp",), _TAP), ((), _TAP)],
    "hp": [(("hp",), _TAP), ((), _TAP)],
    "lk": [(("lk",), _TAP), ((), _TAP)],
    "hk": [(("hk",), _TAP), ((), _TAP)],
    "block": [(("D", "B"), 6)],
    # quarter circle forward + fierce: D, DF, F+P
    "hadouken": [(("D",), 3), (("D", "F"), 3), (("F", "hp"), 3), ((), 3)],
    # dragon punch + fierce: F, D, DF+P
    "shoryuken": [(("F",), 3), (("D",), 3), (("D", "F", "hp"), 3), ((), 3)],
}
assert set(MACROS) == set(ACTIONS)


def question() -> Dict:
    """The one question asked at train and play time. Keep it byte-identical across both."""
    return {"type": "choice", "instructions": INSTRUCTIONS, "criteria": dict(CRITERIA)}


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
