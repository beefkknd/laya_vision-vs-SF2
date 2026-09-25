"""RAM -> fighter state -> the short text note the model reads next to the screenshot.

``health`` / ``enemy_health`` come with the stable-retro integration. The position addresses are the ones used by
linyiLYi/street-fighter-ai on the same ROM (player structs at 0xFF8000 and 0xFF8280, life at +0x42, x at +0x06,
y at +0x0A). Run ``scripts/check_env.py`` once: it walks forward/back and jumps and prints them, so you can see
they move the right way before trusting any label built on them.
"""
from dataclasses import dataclass

from .config import FULL_HP

EXTRA_VARS = {
    "agent_x": {"address": 16744454, "type": ">u2"},
    "agent_y": {"address": 16744458, "type": ">u2"},
    "enemy_x": {"address": 16745094, "type": ">u2"},
    "enemy_y": {"address": 16745098, "type": ">u2"},
}

# |x difference| in game pixels (the Genesis screen is 320 wide; a fighter is ~50 wide)
CLOSE, MID = 60, 140


def register(env) -> None:
    for name, spec in EXTRA_VARS.items():
        env.data.set_variable(name, spec)


@dataclass
class Fighters:
    my_hp: int
    opp_hp: int
    my_x: int
    opp_x: int
    my_y: int
    opp_y: int

    @property
    def dx(self) -> int:
        return abs(self.opp_x - self.my_x)

    @property
    def facing_right(self) -> bool:
        return self.my_x <= self.opp_x


def read(env) -> Fighters:
    v = env.data.lookup_value
    return Fighters(int(v("health")), int(v("enemy_health")), int(v("agent_x")), int(v("enemy_x")),
                    int(v("agent_y")), int(v("enemy_y")))


def dist_bin(dx: int) -> str:
    return "close" if dx < CLOSE else "mid" if dx < MID else "far"


def pct(hp: int) -> int:
    return max(0, round(100 * hp / FULL_HP))


def text_state(f: Fighters, me: str, opp: str, last: str, my_air: bool, opp_air: bool) -> str:
    return ("me=%s opp=%s dist=%s my_hp=%d opp_hp=%d last=%s airborne=%d opp_airborne=%d"
            % (me, opp, dist_bin(f.dx), pct(f.my_hp), pct(f.opp_hp), last, int(my_air), int(opp_air)))
