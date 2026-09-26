"""Scripted teacher (rung 2 of the teacher ladder): a dumb, readable World Warrior Chun-Li that reads RAM, not
pixels.

It returns a *distribution* over the action set, which becomes the soft training target. Laya's loss takes soft
targets, so "hk 60% / forward 30%" teaches calibrated preferences instead of one arbitrary button.

It can label any frame the student reaches, which is what makes DAgger cheap here: the student plays, the teacher
writes gold on the student's own screens.
"""
from typing import Dict

from .actions import ACTIONS
from .env import Context
from .ram import CLOSE, MID, Fighters

SMOOTH = 0.01
GROUND_Y = 192
KICK_HEIGHT = 88  # jump-in: a kick pressed this high up (near the apex, ~24 frames in) lands; earlier ones whiff
THROW_RANGE = 40  # toward + fierce throws within 42 px at the press (ROM); he can step a little before it


def _dist(weights: Dict[str, float]) -> Dict[str, float]:
    p = {a: SMOOTH for a in ACTIONS}
    for a, w in weights.items():
        p[a] += w
    z = sum(p.values())
    return {a: v / z for a, v in p.items()}


def teacher_policy(f: Fighters, c: Context, character: str = "chunli") -> Dict[str, float]:
    """Chun-Li's rules; ``character`` is accepted for the callers' signature and ignored."""
    dx = f.dx
    if c.my_air:  # jump-in: wait for the top of the jump, then kick
        if GROUND_Y - f.my_y >= KICK_HEIGHT:
            return _dist({"hk": 0.8, "idle": 0.2})
        return _dist({"idle": 0.7, "hk": 0.3})
    if c.opp_air and dx < MID and c.dx_trend <= 0:  # anti-air
        return _dist({"hk": 0.55, "hp": 0.25, "block": 0.15, "back": 0.05})
    if (f.opp_attacking and dx < MID) or (f.fireball and abs(f.fireball_x - f.my_x) < CLOSE):  # guard
        return _dist({"block": 0.7, "back": 0.1, "hk": 0.1, "jump_forward": 0.1})
    if c.frames_since_hit < 20 and dx < CLOSE + 20:  # just got hit up close: guard
        return _dist({"block": 0.7, "lk": 0.2, "back": 0.1})
    if dx <= THROW_RANGE and f.opp_state in (0x00, 0x02) and not c.opp_air:  # he stands or crouches: throw
        return _dist({"throw": 0.6, "hp": 0.15, "hk": 0.15, "block": 0.1})
    if dx < CLOSE and f.opp_state in (0x00, 0x02) and not c.opp_air:  # walk into throw range
        return _dist({"forward": 0.6, "hp": 0.15, "hk": 0.15, "block": 0.1})
    if dx < CLOSE:  # footsies range
        return _dist({"hp": 0.35, "hk": 0.3, "lk": 0.2, "block": 0.1, "back": 0.05})
    if dx < MID:
        if not f.opp_attacking:  # jump in over his limbs; the in-air rule kicks
            return _dist({"jump_forward": 0.6, "forward": 0.15, "block": 0.15, "hk": 0.1})
        return _dist({"forward": 0.4, "hk": 0.25, "lk": 0.15, "block": 0.1, "crouch": 0.1})
    return _dist({"forward": 0.55, "jump": 0.15, "hk": 0.15, "block": 0.15})


def argmax(p: Dict[str, float]) -> str:
    return max(p, key=p.get)
