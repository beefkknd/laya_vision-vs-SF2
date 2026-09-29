"""The RAM map (ram_maps/sf2_snes.txt, SNES Street Fighter II USA): where the fighters' life and positions live.

One variable per line (found with the old RAM finder, tag legacy-dagger, or by hand in Mesen's memory viewer):

    # name   address   size  signed
    my_hp    0x0530    2     1
    opp_hp   0x0730    2     1
    my_x     0x0522    2     0
    ...

Addresses are hex WRAM offsets; SNES bus addresses 7E0000-7FFFFF are accepted and converted.
"""
from dataclasses import dataclass
from typing import List

REQUIRED = ["my_hp", "opp_hp", "my_x", "opp_x", "my_y", "opp_y"]

# Ranges by |x difference| in game pixels (the SNES screen is 256 wide; a fighter is ~50 wide).
CLOSE, MID = 55, 120


@dataclass
class Var:
    name: str
    addr: int
    size: int
    signed: bool


def parse_map(text: str) -> List[Var]:
    out = []
    for line in text.splitlines():
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        name, addr, size, signed = line.split()[:4]
        a = int(addr.lstrip("$"), 16)  # always hex: 0x0530, $0530, 7E0530
        if 0x7E0000 <= a <= 0x7FFFFF:
            a -= 0x7E0000
        out.append(Var(name, a, int(size), signed not in ("0", "false", "no")))
    missing = [n for n in REQUIRED if n not in {v.name for v in out}]
    if missing:
        raise ValueError("RAM map is missing %s" % ", ".join(missing))
    return out


def load_map(path: str) -> List[Var]:
    with open(path) as f:
        return parse_map(f.read())
