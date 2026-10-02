"""Clean runner: screen reader, RAM-free halves of the mixed modules."""
import _path  # noqa: F401
from sf2.data.vs_sweep import MOVEMENT, actions
from sf2.emu.vs import physical
from sf2.screen.reader import read
from sf2.system1.helper import decide


def main():
    return decide(read(None), actions("chunli"), physical(["F"], True, {}), MOVEMENT)
