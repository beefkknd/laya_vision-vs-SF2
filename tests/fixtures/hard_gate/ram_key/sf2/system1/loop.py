def gap(bridge, frames):
    rows = bridge.run(frames).rams
    r = rows[-1]
    return abs(r["p2_x"] - r["p1_x"])
