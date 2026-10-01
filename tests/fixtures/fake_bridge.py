"""A deterministic stand-in for sf2.emu.mesen.MesenBridge in a System 1 fight (tests only).

Every RAM row is a pure function of the global frame index f (so the rows do not depend on how play_round batches its
RUNs), and every image carries its own frame index: image[0, 0] = (f >> 16, f >> 8, f) & 0xFF. ``p2_special`` holds f
too (the CPU's p2_special is always 0, nothing reads it). As in the Lua bridge, RUN of n frames returns n + 1 rows
(before frame 0..n) and the images at the requested indices of the same polls. The round ends (result 1) at END."""
from typing import Dict, List, Optional, Sequence

import numpy as np

from sf2.emu.mesen import Obs
from sf2.emu.vs import GROUND_Y, NAMES

END = 700


def row(f: int) -> Dict[str, int]:
    phase = (f // 37) % 9
    p1_state = 0x0E if phase == 4 and f % 37 < 20 else 0x0A if phase == 6 and f % 37 < 12 else 0
    p2_state = (0x00, 0x0A, 0x04, 0x08, 0x0E, 0x02, 0x0A, 0x00, 0x04)[phase]
    air = p2_state == 0x04
    p1_x = 120 + (f // 5) % 140
    p2_x = p1_x + 40 + (f * 7) % 90
    r = {n: 0 for n in NAMES}
    r.update({
        "p1_hp": 176, "p1_life": max(0, 176 - (f // 97) * 9), "p1_x": p1_x, "p1_y": GROUND_Y,
        "p1_state": p1_state, "p1_react": 0x02 if p1_state == 0x0E else 0, "p1_facing": 0x40, "p1_char": 5,
        "p2_hp": 176, "p2_life": max(0, 176 - (f // 131) * 11), "p2_x": p2_x,
        "p2_y": GROUND_Y - (f % 37) if air else GROUND_Y, "p2_state": p2_state,
        "p2_react": 0x06 if phase == 3 else 0, "p2_special": f, "p2_char": 0,
        "timer": 0x99, "result": 1 if f >= END else 0,
        "shot2": 1 if phase in (1, 2) and f % 37 > 15 else 0, "shot2_x": p2_x - (f % 37) * 3,
    })
    return r


def image(f: int) -> np.ndarray:
    im = np.zeros((4, 4, 3), np.uint8)
    im[0, 0] = ((f >> 16) & 0xFF, (f >> 8) & 0xFF, f & 0xFF)
    return im


def frame_of(im: np.ndarray) -> int:
    a, b, c = (int(v) for v in im[0, 0])
    return (a << 16) | (b << 8) | c


class FakeBridge:
    def __init__(self):
        self.f = 0
        self.runs: List[int] = []

    def load_state(self, state: bytes) -> Obs:
        self.f = 0
        return Obs([[row(0)[n] for n in NAMES]])

    def run(self, frames: Sequence[Sequence[str]], caps=(), p2: Optional[Sequence] = None) -> Obs:
        n = len(frames)
        rams = [[row(self.f + k)[nm] for nm in NAMES] for k in range(n + 1)]
        images = {c: image(self.f + c) for c in caps if 0 <= c <= n}
        self.f += n
        self.runs.append(n)
        return Obs(rams, [], images)
