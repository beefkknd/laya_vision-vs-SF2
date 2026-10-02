import _path  # noqa: F401
from sf2.system1.system1 import System1


def main(table):
    return System1(None, "chunli", advisor=None, oracle=table)
