"""Human pad log -> one of the 12 options per decision frame (rung 1 of the teacher ladder).

The recorder logs every frame's held buttons. Here they become relative tokens (F/B from which side you were on),
special-move motions are recognised the same way the game does (in a short input buffer before the punch), and
each decision point (every ``HOLD`` frames) gets the option whose input *started* in its window, else the stick
state held at that frame.
"""
from typing import List, Optional, Sequence, Set, Tuple

from .config import HOLD, PAD

MOTION_WINDOW = 15
_PAD_REV = {v: k for k, v in PAD.items()}
# the 6 buttons fold into the 4 attack options
_ATTACK = {"lp": "lp", "mp": "lp", "hp": "hp", "lk": "lk", "mk": "lk", "hk": "hk"}


def relative(names: Sequence[str], facing_right: bool) -> Tuple[frozenset, Set[str]]:
    """SNES button names (Mesen's) -> (direction tokens, attack names)."""
    fwd, back = ("right", "left") if facing_right else ("left", "right")
    dirs, atk = set(), set()
    for n in names:
        if n == "up":
            dirs.add("U")
        elif n == "down":
            dirs.add("D")
        elif n == fwd:
            dirs.add("F")
        elif n == back:
            dirs.add("B")
        elif n in _PAD_REV:
            atk.add(_PAD_REV[n])
    return frozenset(dirs), atk


def _find_seq(dirs: List[frozenset], patterns: List[Set[str]], end: int, start: int) -> Optional[int]:
    """Latest match of ``patterns`` (exact direction sets, in order) inside dirs[start:end+1]; returns start idx."""
    want = len(patterns) - 1
    idx = end
    first = None
    while want >= 0 and idx >= start:
        if dirs[idx] == frozenset(patterns[want]):
            first = idx
            want -= 1
            # consume the whole run of this direction
            while idx - 1 >= start and dirs[idx - 1] == dirs[idx]:
                idx -= 1
                first = idx
        idx -= 1
    return first if want < 0 else None


def events(frames: Sequence[Tuple[Sequence[str], bool]]) -> List[Tuple[int, str]]:
    """(frame index where the input started, option) for every attack press, specials recognised."""
    dirs, atks = [], []
    for names, facing in frames:
        d, a = relative(names, facing)
        dirs.append(d)
        atks.append(a)
    out = []
    for t in range(len(frames)):
        new = atks[t] - (atks[t - 1] if t else set())
        punch = any(b in ("lp", "mp", "hp") for b in new)
        if not new:
            continue
        lo = max(0, t - MOTION_WINDOW)
        if punch:
            # shoryuken: F, D, DF + P   (punch pressed while still holding DF)
            s = _find_seq(dirs, [{"F"}, {"D"}, {"D", "F"}], t, lo)
            if s is not None and "D" in dirs[t]:
                out.append((s, "shoryuken"))
                continue
            # hadouken: D, DF, F + P
            s = _find_seq(dirs, [{"D"}, {"D", "F"}, {"F"}], t, lo)
            if s is not None:
                out.append((s, "hadouken"))
                continue
        # plain attack: strongest button pressed this frame
        order = ["hp", "hk", "mp", "mk", "lp", "lk"]
        b = next(x for x in order if x in new)
        out.append((t, _ATTACK[b]))
    return out


def stick_option(d: frozenset) -> str:
    if "U" in d:
        return "jump"
    if "D" in d and "B" in d:
        return "block"
    if "D" in d:
        return "crouch"
    if "F" in d:
        return "forward"
    if "B" in d:
        return "back"
    return "idle"


def label_frames(frames: Sequence[Tuple[Sequence[str], bool]], hold: int = HOLD) -> List[Tuple[int, str]]:
    """(decision frame index, option) every ``hold`` frames."""
    ev = sorted(events(frames))
    out, j = [], 0
    for g in range(0, len(frames), hold):
        while j < len(ev) and ev[j][0] < g:
            j += 1
        if j < len(ev) and ev[j][0] < g + hold:
            out.append((g, ev[j][1]))
        else:
            d, _ = relative(frames[g][0], frames[g][1])
            out.append((g, stick_option(d)))
    return out
