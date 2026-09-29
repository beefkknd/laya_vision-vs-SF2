"""A RAM row as sf2.emu.vs reads it (both players and the round fields), with overrides: shared by tests."""
from sf2.emu.vs import GROUND_Y


def row(p1=None, p2=None, **extra):
    base = {"hp": 176, "life": 176, "x": 200, "y": GROUND_Y, "state": 0, "sub": 0, "react": 0, "dizzy": 0,
            "special": 0, "facing": 0x40, "char": 0}
    r = {}
    for p, over in ((1, p1 or {}), (2, dict({"x": 260}, **(p2 or {})))):
        r.update({"p%d_%s" % (p, k): v for k, v in dict(base, **over).items()})
    r.update({"timer": 0x99, "result": 0, "shot1": 0, "shot1_x": 0, "shot2": 0, "shot2_x": 0})
    r.update(extra)
    return r
