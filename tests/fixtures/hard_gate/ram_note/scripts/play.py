import _path  # noqa: F401
from sf2.data.vs_sweep import actions, note


def main(row):
    return note("chunli", "ryu", row, "left"), actions("chunli")
