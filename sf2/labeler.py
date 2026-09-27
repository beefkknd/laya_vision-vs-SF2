"""Human pad log -> one of the action options per decision frame.

The recorder logs every frame's held buttons. Here they become relative tokens (F/B from which side you were on),
and each decision point (every ``HOLD`` frames) gets the option whose input *started* in its window, else the stick
state held at that frame.
"""
from typing import List, Sequence, Set, Tuple

from .config import HOLD, PAD

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


def events(frames: Sequence[Tuple[Sequence[str], bool]]) -> List[Tuple[int, str]]:
    """(frame index, option) for every attack press. World Warrior Chun-Li has no motion specials, so a motion
    followed by a punch is what its last frame gives. Kick taps stay lk / hk: the Legs are not labelled."""
    rel = [relative(names, facing) for names, facing in frames]
    out = []
    for t in range(len(frames)):
        dirs, atk = rel[t]
        new = atk - (rel[t - 1][1] if t else set())
        if not new:
            continue
        # plain attack: strongest button pressed this frame
        order = ["hp", "hk", "mp", "mk", "lp", "lk"]
        b = next(x for x in order if x in new)
        if b in ("hp", "mp") and dirs & {"F", "B"} and not dirs & {"U", "D"}:
            out.append((t, "throw"))       # toward / back + fierce or strong: a throw up close, else that punch
        elif b == "hk" and "D" in dirs:
            out.append((t, "sweep"))
        else:
            out.append((t, _ATTACK[b]))
    return out


def stick_option(d: frozenset) -> str:
    if "U" in d:
        return "jump_forward" if "F" in d else "jump"
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
