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
# round clock (BCD seconds), action states, projectile slot (in use, world x), the ROM's round result
OPTIONAL = ["timer", "my_state", "opp_state", "fireball", "fireball_x", "result", "my_react", "opp_react", "my_sub",
            "opp_sub", "my_dizzy", "opp_dizzy", "my_life", "opp_life"]
# Action state (0x0C03 Chun-Li, 0x0E03 Dhalsim), observed on the ROM (harness audit, 2026-09-26); the byte after it
# is a sub-state.
#   00 stand / walk (also while lifted for a throw at y 136, and falling after a KO)   02 crouch
#   04 jump (on the ground for take-off / landing)   06 turning round after the fighters cross
#   08 guard: holding back while an attack comes (+0x43: 1 standing, 2 crouching)   0A attack, on the ground or in the air
#   0C special move (Chun-Li's Lightning Legs, from repeated kicks)
#   0E hit stun and block stun alike; sub-state 02 reeling / knocked into the air, 04 down, 06 getting up,
#      08 dizzy. The hit reaction at +0x4A (0x0C4A / 0x0E4A) tells them apart: 06 / 08 block stun (standing /
#      crouching guard; no life lost, or a Yoga Fire's 4-8 chip), anything else a hit. Dizzy is sub-state 08
#      entered with the flag at +0x89 set (see dizzy()); without it 08 is the last 16 frames of getting up
#   10 winner's pose   12 time-over loser   14 thrown through the air (also the KO fall on some rounds)
# Dhalsim uses the same values when Chun-Li hits (0E), throws or blocks him (08).
HIT_STATE = 0x0E
ATTACK_STATE = 0x0A
JUMP_STATE = 0x04
BLOCK_REACTS = (0x06, 0x08)
DIZZY_SUB = 0x08
THROWN_STATE = 0x14
POSE_STATES = (0x10, 0x12)

# |world x difference| in pixels. Measured from random play (TEACHER.md, 2026-09-25): Chun-Li's normals land below
# CLOSE, Dhalsim's attacks reach up to MID.
CLOSE, MID = 80, 120
# Stage walls in world x, measured on the ROM (tests/fixtures/walls, 2026-09-26): walking back, Chun-Li stops at 53
# and 459. Dhalsim reaches 52 (48 for a frame when thrown). Fighters are also never more than ~212 px apart.
LEFT_WALL, RIGHT_WALL = 53, 459
CORNER = 40  # the wall is less than this far behind you: cornered (about 24 frames of walking back)


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
    my_state: Optional[int] = None
    opp_state: Optional[int] = None
    fireball: Optional[int] = None
    fireball_x: Optional[int] = None
    result: Optional[int] = None
    my_react: Optional[int] = None
    opp_react: Optional[int] = None
    my_sub: Optional[int] = None
    opp_sub: Optional[int] = None
    my_dizzy: Optional[int] = None
    opp_dizzy: Optional[int] = None
    my_life: Optional[int] = None
    opp_life: Optional[int] = None

    @classmethod
    def from_values(cls, names: List[str], values: List[int]) -> "Fighters":
        d: Dict[str, int] = dict(zip(names, values))
        return cls(*(int(d[n]) for n in REQUIRED), *(d.get(n) for n in OPTIONAL))

    @property
    def life(self):
        """(hers, his) true life, which drops by the whole hit on the hit frame; the bars if the map lacks it."""
        return (self.my_hp if self.my_life is None else self.my_life,
                self.opp_hp if self.opp_life is None else self.opp_life)

    @property
    def dx(self) -> int:
        return abs(self.opp_x - self.my_x)

    @property
    def facing_right(self) -> bool:
        return self.my_x <= self.opp_x

    @property
    def opp_attacking(self) -> bool:
        """Dhalsim is in an attack (a limb, a slide, a Yoga Fire throw or a jump attack): time to guard."""
        return self.opp_state == ATTACK_STATE

    @property
    def my_cornered(self) -> bool:
        return cornered(self.my_x, self.facing_right)

    @property
    def opp_cornered(self) -> bool:
        return cornered(self.opp_x, not self.facing_right)


def cornered(x: int, facing_right: bool) -> bool:
    """The stage wall is close behind a fighter at world x facing that way."""
    return x - LEFT_WALL < CORNER if facing_right else RIGHT_WALL - x < CORNER


def dist_bin(dx: int) -> str:
    return "close" if dx < CLOSE else "mid" if dx < MID else "far"


def pct(hp: int, full: int) -> int:
    return max(0, round(100 * hp / max(1, full)))


# Action state -> the word in the note. 08 shows while holding back against his attacks with no life lost (walls
# trace): block. 04 on the ground is take-off / landing, so "jump" comes only from the airborne rule (env.airborne);
# an attack in the air is "jumpattack".
# Anything else (06 turning, 0C special, 10-14 end-of-round poses and throws) is "other".
STATE_WORDS = {0x00: "stand", 0x02: "crouch", 0x04: "stand", 0x08: "block", ATTACK_STATE: "attack", HIT_STATE: "hit"}


def in_block_stun(state: Optional[int], react: Optional[int]) -> bool:
    return state == HIT_STATE and react in BLOCK_REACTS


def dizzy(was: bool, state: Optional[int], sub: Optional[int], flag: Optional[int]) -> bool:
    """Dizzy: 0E with sub-state 08, entered with the flag at +0x89 set. The flag can clear long before the stars
    end, so a fighter stays dizzy (``was``) for as long as it stays in 0E / 08."""
    return state == HIT_STATE and sub == DIZZY_SUB and (bool(flag) or was)


def state_word(state: Optional[int], air: bool, react: Optional[int] = None, dizzy: bool = False) -> str:
    if state == HIT_STATE:
        return "block" if react in BLOCK_REACTS else "dizzy" if dizzy else "hit"
    if air:
        return "jumpattack" if state == ATTACK_STATE else "jump"
    return "stand" if state is None else STATE_WORDS.get(state, "other")


def clock_word(timer: Optional[int]) -> str:
    """Round clock (BCD seconds, 99 at the start): early 99-60, mid 59-30, late 29-0."""
    s = 99 if timer is None else (timer >> 4) * 10 + (timer & 0xF)
    return "early" if s >= 60 else "mid" if s >= 30 else "late"


def text_state(f: Fighters, me: str, opp: str, last: str, my_air: bool, opp_air: bool, full_hp: int,
               dizzy=(False, False)) -> str:
    corner = "me" if f.my_cornered else "opp" if f.opp_cornered else "none"  # never both: they are < 212 px apart
    fireball = dist_bin(abs(f.fireball_x - f.my_x)) if f.fireball else "none"  # how far it is from her
    return ("me=%s %s hp=%d opp=%s %s hp=%d dist=%s facing=%s corner=%s time=%s last=%s fireball=%s"
            % (me, state_word(f.my_state, my_air, f.my_react, dizzy[0]),
               pct(f.my_hp, full_hp), opp,
               state_word(f.opp_state, opp_air, f.opp_react, dizzy[1]),
               pct(f.opp_hp, full_hp), dist_bin(f.dx), "right" if f.facing_right else "left", corner,
               clock_word(f.timer), last, fireball))
