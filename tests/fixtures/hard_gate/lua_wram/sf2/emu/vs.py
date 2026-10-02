"""Mixed stub: RAM half (VARS, view) and RAM-free half (physical). Its body is not scanned."""
VARS = ["p1_x", "p2_x"]


def view(row):
    return {"a_x": row["p1_x"]}


def physical(tokens, facing_right, pad):
    return [pad.get(t, t) for t in tokens]
