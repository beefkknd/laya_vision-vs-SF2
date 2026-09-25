"""Scripted teacher (rung 2 of the teacher ladder): a dumb, readable Ryu that reads RAM, not pixels.

It returns a *distribution* over the 12 options, which becomes the soft training target. Laya's loss takes soft
targets, so "hadouken 70% / forward 20%" teaches calibrated preferences instead of one arbitrary button.

It can label any frame the student reaches, which is what makes DAgger cheap here: the student plays, the teacher
writes gold on the student's own screens.
"""
from typing import Dict

from .actions import ACTIONS
from .env import Context
from .ram import CLOSE, MID, Fighters

FIREBALL_COOLDOWN = 50  # frames; one fireball on screen at a time
SMOOTH = 0.01


def _dist(weights: Dict[str, float]) -> Dict[str, float]:
    p = {a: SMOOTH for a in ACTIONS}
    for a, w in weights.items():
        p[a] += w
    z = sum(p.values())
    return {a: v / z for a, v in p.items()}


def teacher_policy(f: Fighters, c: Context, character: str = "ryu") -> Dict[str, float]:
    dx = f.dx
    fireball_ready = c.frames_since_fireball >= FIREBALL_COOLDOWN
    if c.my_air:  # jump-in: kick on the way down
        return _dist({"hk": 0.8, "idle": 0.2})
    if c.opp_air and dx < MID and c.dx_trend <= 0:  # anti-air
        if character.lower() == "chunli":
            return _dist({"hk": 0.55, "hp": 0.25, "block": 0.15, "back": 0.05})
        return _dist({"shoryuken": 0.75, "block": 0.15, "hp": 0.1})
    if c.frames_since_hit < 20 and dx < CLOSE + 20:  # just got hit up close: guard
        return _dist({"block": 0.7, "lk": 0.2, "back": 0.1})
    if dx < CLOSE:  # footsies range
        if character.lower() == "chunli":
            return _dist({"hp": 0.35, "hk": 0.3, "lk": 0.2, "block": 0.1, "back": 0.05})
        return _dist({"hp": 0.35, "hk": 0.25, "lk": 0.2, "block": 0.15, "shoryuken": 0.05})
    if dx < MID:
        if character.lower() == "chunli":
            return _dist({"forward": 0.4, "hk": 0.25, "lk": 0.15, "block": 0.1, "crouch": 0.1})
        if fireball_ready:
            return _dist({"hadouken": 0.6, "forward": 0.2, "lk": 0.1, "block": 0.1})
        return _dist({"forward": 0.45, "block": 0.25, "crouch": 0.15, "hk": 0.15})
    if character.lower() == "chunli":
        return _dist({"forward": 0.55, "jump": 0.15, "hk": 0.15, "block": 0.15})
    if fireball_ready:
        return _dist({"hadouken": 0.65, "forward": 0.35})
    return _dist({"forward": 0.7, "jump": 0.1, "block": 0.2})


def argmax(p: Dict[str, float]) -> str:
    return max(p, key=p.get)
