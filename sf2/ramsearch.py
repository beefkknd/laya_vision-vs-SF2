"""Find the six RAM variables from WRAM dumps taken during known phases of play.

Phases (each a list of 128 KiB dumps, in order):
    start      standing at the beginning of the round (you on the left)
    right      after walking right (toward the opponent)
    left       after walking back left about as far
    jump       several dumps during one jump, the last one after landing
    take_hits  while the opponent hits you and you do not attack
    give_hits  while you hit the opponent

Rules:
    my_x   16-bit, rose while walking right, fell back while walking left
    my_y   16-bit, moved during the jump, back to its standing value after it, still while walking
    my_hp  never rises, starts at a life-bar-like value, drops by >=3 (a hit, not a timer tick) in take_hits;
           may also drop in other phases (the CPU attacks whenever it likes)
    opp_hp never rises, drops by hits in give_hits and nowhere else
    opp_x / opp_y  my_x / my_y + (opp_hp - my_hp): both fighters use the same struct layout
"""
from typing import Dict, List, Optional, Tuple

from .ram import Var

HIT = 3


def val(d: bytes, a: int, size: int, signed: bool = False) -> int:
    v = d[a] if size == 1 else d[a] | (d[a + 1] << 8)
    if signed and v >= (1 << (8 * size - 1)):
        v -= 1 << (8 * size)
    return v


def _series(dumps: List[bytes], a: int, size: int) -> List[int]:
    return [val(d, a, size, signed=True) for d in dumps]


def x_candidates(ph: Dict[str, List[bytes]], top: int = 5) -> List[Tuple[int, float]]:
    s, r, l = ph["start"][-1], ph["right"][-1], ph["left"][-1]
    out = []
    for a in range(len(s) - 1):
        vs, vr, vl = val(s, a, 2), val(r, a, 2), val(l, a, 2)
        dr, dl = vr - vs, vr - vl
        if 8 <= dr <= 200 and 8 <= dl <= 200:
            out.append((a, -abs(dr - dl) - abs(vl - vs) / 4))
    return sorted(out, key=lambda t: -t[1])[:top]


def y_candidates(ph: Dict[str, List[bytes]], top: int = 5) -> List[Tuple[int, float]]:
    base = ph["left"][-1]
    out = []
    for a in range(len(base) - 1):
        b = val(base, a, 2)
        if abs(val(ph["start"][-1], a, 2) - b) > 1 or abs(val(ph["right"][-1], a, 2) - b) > 1:
            continue
        js = [val(d, a, 2) for d in ph["jump"]]
        swing = max(abs(v - b) for v in js)
        if swing >= 8 and abs(js[-1] - b) <= 1:
            out.append((a, -abs(swing - 60)))  # SF2 jumps rise a few dozen pixels
    return sorted(out, key=lambda t: -t[1])[:top]


PHASES = ["start", "right", "left", "jump", "take_hits", "give_hits"]  # the order they were recorded in


def hp_candidates(ph: Dict[str, List[bytes]], hit_phase: str, also_falls_in: tuple = (), top: int = 8):
    """(addr, size, score) for values that never rise, fall by hits during ``hit_phase``, and fall nowhere else
    except ``also_falls_in``."""
    order = PHASES
    dumps, phase_of = [], []
    for p in order:
        for d in ph[p]:
            dumps.append(d)
            phase_of.append(p)
    out = []
    n = len(dumps[0])
    for size in (1, 2):
        for a in range(n - size + 1):
            s = _series(dumps, a, size)
            if not 32 <= s[0] <= 512:
                continue
            ok, hits = True, 0
            for i in range(1, len(s)):
                d = s[i - 1] - s[i]
                if d < 0 or (d > 0 and phase_of[i] != hit_phase and phase_of[i] not in also_falls_in):
                    ok = False
                    break
                hits += d >= HIT and phase_of[i] == hit_phase
            if ok and hits:
                out.append((a, size, hits - (0.5 if size == 1 else 0)))
    return sorted(out, key=lambda t: -t[2])[:top]


def analyze(ph: Dict[str, List[bytes]]) -> Tuple[List[Var], Dict]:
    xs, ys = x_candidates(ph), y_candidates(ph)
    mine = hp_candidates(ph, "take_hits", also_falls_in=tuple(PHASES))  # the CPU hits whenever it likes
    theirs = hp_candidates(ph, "give_hits")
    report = {"my_x": xs, "my_y": ys, "my_hp": mine, "opp_hp": theirs}
    best: Optional[Tuple[float, tuple]] = None
    s0 = ph["start"][-1]
    for m, msize, ms in mine:
        for o, osize, os_ in theirs:
            if osize != msize:
                continue
            stride = o - m
            score = ms + os_ + (2 if val(s0, m, msize) == val(s0, o, osize) else 0)
            for x, xs_ in xs[:3]:
                ox = x + stride
                if not 0 <= ox < len(s0) - 1:
                    continue
                gap = val(s0, ox, 2) - val(s0, x, 2)
                sc = score + xs_ / 50 + (3 if 30 <= gap <= 250 else 0)
                for y, _ in ys[:3] or [(None, 0)]:
                    oy = y + stride if y is not None else None
                    sy = sc + (2 if oy is not None and 0 <= oy < len(s0) - 1
                               and abs(val(s0, oy, 2) - val(s0, y, 2)) <= 2 else 0)
                    if best is None or sy > best[0]:
                        best = (sy, (m, o, msize, x, ox, y, oy))
    if best is None:
        return [], report
    m, o, size, x, ox, y, oy = best[1]
    chosen = [Var("my_hp", m, size, True), Var("opp_hp", o, size, True), Var("my_x", x, 2, False),
              Var("opp_x", ox, 2, False)]
    if y is not None:
        chosen += [Var("my_y", y, 2, False), Var("opp_y", oy, 2, False)]
    report["stride"] = o - m
    return chosen, report
