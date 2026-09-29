"""The game's words, in one place: fighters, ranges, health bars, what the opponent is doing.

These values are part of what laya-vision and text laya were trained on (the notes and the advice lines are built
from them); tests/test_vocab.py pins them. Change one only together with the training data.
"""
from typing import Dict, Tuple

# Character ids as the game keeps them (0x0CD1 player 1, 0x0ED1 player 2; the select grid: top row 0-3, bottom 4-7).
CHARACTERS: Dict[int, str] = {0: "ryu", 1: "honda", 2: "blanka", 3: "guile", 4: "ken", 5: "chunli", 6: "zangief",
                              7: "dhalsim", 10: "balrog", 11: "vega"}
FIGHTERS: Tuple[str, ...] = tuple(CHARACTERS[i] for i in range(8))      # the eight playable ones
IDS: Dict[str, int] = {name: i for i, name in CHARACTERS.items() if i < 8}

# Ranges by |x difference| in game pixels (the SNES screen is 256 wide; a fighter is ~50 wide).
CLOSE, MID = 55, 120
RANGES = ("close", "mid", "far")
RANGE_WORDS = {"close": "up close", "mid": "at mid range", "far": "far away"}


def range_of(gap: int) -> str:
    return "close" if gap < CLOSE else "mid" if gap < MID else "far"


# Health in general words (laya gets no numbers: an exact value per row would be misleading).
FULL_LIFE = 176
BARS = ("full", "high", "half", "low")


def bar(life: int) -> str:
    share = max(0, life if life < 200 else 0) / FULL_LIFE
    return "full" if share >= 0.9 else "high" if share >= 0.6 else "half" if share >= 0.3 else "low"


OPP_STATES = ("jumping", "crouching", "attacking", "standing", "stunned")
