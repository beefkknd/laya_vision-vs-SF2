"""The HUD, read from the screen only: both life bars and the round clock.

Geometry (measured on frames, 2026-10-02; the HUD is fixed and drawn over every sprite):
  bars    rows 35-42; player 1's bar x 32-119 (it empties from the left), player 2's x 136-223 (from the right);
          88 px for 176 life points (2 points a px); the live part is yellow (247, 222, 0), the lost part red
          (247, 66, 0). The bar shows the DRAWN health, which drains 1 point a frame after a hit.
  clock   two digits, tens x 120-127, ones x 128-135, rows 48-59: an orange fill (two dithered colours) and an outline,
          blue (or white: it blinks under 15 s); the glyph masks per digit
          come from calibration frames (``build_digits``; scripts/collect_reader_gate.py --calib-hud).
"""
import os
from dataclasses import dataclass
from functools import lru_cache
from typing import Optional, Tuple

import numpy as np

from .assets import READER_DIR
from .facts import HudFacts

BAR_ROW = 38
BAR_ROWS = slice(35, 43)
BAR_X = ((32, 120), (136, 224))
BAR_PX = 88
YELLOW = (247, 222, 0)
RED = (247, 66, 0)
CLOCK_ROWS = slice(48, 60)
CLOCK_X = ((120, 128), (128, 136))
# fill (two oranges), outline blue, or outline white (the clock blinks under 15 s)
GLYPH_COLOURS = ((247, 107, 66), (255, 156, 107), (24, 66, 173), (231, 231, 231))
DIGITS = os.path.join(READER_DIR, "hud_digits.npz")
MAX_DIGIT_DIST = 10                 # glyph-mask pixels that may differ from the digit's mask (of 96)


def _is(rgb: np.ndarray, colour) -> np.ndarray:
    return (rgb[..., 0] == colour[0]) & (rgb[..., 1] == colour[1]) & (rgb[..., 2] == colour[2])


def bar_fraction(frame: np.ndarray, side: int) -> Tuple[Optional[float], int, int]:
    """(live fraction, yellow px, red px) of bar ``side`` (0 = player 1's, 1 = player 2's); the fraction is None when
    the row is neither yellow nor red (not a fight screen)."""
    x0, x1 = BAR_X[side]
    row = frame[BAR_ROW, x0:x1]
    y, r = int(_is(row, YELLOW).sum()), int(_is(row, RED).sum())
    if y + r < BAR_PX // 2:
        return None, y, r
    return y / BAR_PX, y, r


def glyph_mask(frame: np.ndarray, pos: int) -> np.ndarray:
    """(12, 8) uint8 class per pixel of clock digit ``pos``: 0 not a glyph colour, 1 the fill (two dithered oranges),
    2 the blue outline (the outline is what tells the digits apart: the fill is a solid block)."""
    x0, x1 = CLOCK_X[pos]
    cell = frame[CLOCK_ROWS, x0:x1]
    m = np.zeros(cell.shape[:2], np.uint8)
    m[_is(cell, GLYPH_COLOURS[0]) | _is(cell, GLYPH_COLOURS[1])] = 1
    m[_is(cell, GLYPH_COLOURS[2]) | _is(cell, GLYPH_COLOURS[3])] = 2
    return m


def build_digits(frames: np.ndarray, clock: np.ndarray) -> np.ndarray:
    """(2 positions, 10 digits, 12, 8) uint8 glyph class maps: per position and digit, each pixel's most common class
    over the calibration frames showing it. ``clock``: the value (0-99) each frame shows (calibration labels)."""
    out = np.zeros((2, 10, 12, 8), np.uint8)
    for pos, digit_of in ((0, clock // 10), (1, clock % 10)):
        for d in range(10):
            sel = frames[digit_of == d]
            if len(sel) == 0:
                raise ValueError("no calibration frame shows %d in position %d" % (d, pos))
            maps = np.stack([glyph_mask(f, pos) for f in sel])
            counts = np.stack([(maps == c).sum(0) for c in range(3)])
            out[pos, d] = counts.argmax(0)
    return out


@lru_cache(maxsize=2)
def load_digits(path: str = DIGITS) -> Optional[np.ndarray]:
    if not os.path.exists(path):
        return None
    return np.load(path)["digits"]


def read_clock(frame: np.ndarray, digits: Optional[np.ndarray]) -> Optional[int]:
    if digits is None:
        return None
    value = 0
    for pos in (0, 1):
        dist = (digits[pos] != glyph_mask(frame, pos)[None]).reshape(10, -1).sum(1)
        d = int(dist.argmin())
        if dist[d] > MAX_DIGIT_DIST:
            return None
        value = value * 10 + d
    return value


@dataclass(frozen=True)
class Hud:
    facts: HudFacts
    raw: Tuple[Tuple[int, int], Tuple[int, int]]      # (yellow, red) px per bar


def read_hud(frame: np.ndarray, digits: Optional[np.ndarray] = None) -> Hud:
    bars = [bar_fraction(frame, s) for s in (0, 1)]
    empty = tuple(b[0] is not None and b[1] == 0 for b in bars)
    facts = HudFacts(health=(bars[0][0], bars[1][0]), timer=read_clock(frame, digits), bar_empty=empty)
    return Hud(facts, ((bars[0][1], bars[0][2]), (bars[1][1], bars[1][2])))
