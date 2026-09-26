"""RAM map (per cartridge) -> fighter state -> the short text note the model reads next to the screenshot.

The map is a small text file, one variable per line, written by scripts/find_ram.py or by hand from Mesen's
memory viewer:

    # name   address   size  signed
    my_hp    0x0530    2     1
    opp_hp   0x0730    2     1
    my_x     0x0522    2     0
    ...

Addresses are hex WRAM offsets; SNES bus addresses 7E0000-7FFFFF are accepted and converted.
"""
from dataclasses import dataclass
from typing import Dict, List, Optional

REQUIRED = ["my_hp", "opp_hp", "my_x", "opp_x", "my_y", "opp_y"]
OPTIONAL = ["timer"]  # round clock (BCD on the SNES ROM); only "reached zero" is used

# |x difference| in game pixels (the SNES screen is 256 wide; a fighter is ~50 wide). Check on day 1.
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
        raise ValueError("RAM map is missing %s; run scripts/find_ram.py" % ", ".join(missing))
    return out


def load_map(path: str) -> List[Var]:
    with open(path) as f:
        return parse_map(f.read())


def format_map(vars_: List[Var]) -> str:
    lines = ["# name     address  size  signed   (WRAM offsets; 7E0000 + offset on the SNES bus)"]
    lines += ["%-10s 0x%04X   %d     %d" % (v.name, v.addr, v.size, int(v.signed)) for v in vars_]
    return "\n".join(lines) + "\n"


@dataclass
class Fighters:
    my_hp: int
    opp_hp: int
    my_x: int
    opp_x: int
    my_y: int
    opp_y: int
    timer: Optional[int] = None

    @classmethod
    def from_values(cls, names: List[str], values: List[int]) -> "Fighters":
        d: Dict[str, int] = dict(zip(names, values))
        return cls(*(int(d[n]) for n in REQUIRED), *(d.get(n) for n in OPTIONAL))

    @property
    def dx(self) -> int:
        return abs(self.opp_x - self.my_x)

    @property
    def facing_right(self) -> bool:
        return self.my_x <= self.opp_x


def dist_bin(dx: int) -> str:
    return "close" if dx < CLOSE else "mid" if dx < MID else "far"


def pct(hp: int, full: int) -> int:
    return max(0, round(100 * hp / max(1, full)))


def text_state(f: Fighters, me: str, opp: str, last: str, my_air: bool, opp_air: bool, full_hp: int) -> str:
    return ("me=%s opp=%s dist=%s my_hp=%d opp_hp=%d last=%s airborne=%d opp_airborne=%d"
            % (me, opp, dist_bin(f.dx), pct(f.my_hp, full_hp), pct(f.opp_hp, full_hp), last, int(my_air),
               int(opp_air)))
