"""Mixed stub: actions (RAM-free vocabulary) and note (built from a RAM row)."""
MOVEMENT = {"forward": ()}


def actions(char):
    return dict(MOVEMENT)


def note(me, opp, r, side):
    return "%s %d" % (me, r["p1_x"])
