"""Block data: a block's EFFECT on the character's own health. The other fighter throws a standard attack (the probe);
when it is visible, the character answers with block_high (hold back), block_low (hold down-back) or nothing (idle), and
RAM says what happened to its health:

    blocked   block stun (guarded: at most chip damage)       got_hit   hit stun or thrown (health lost)
    none      the attack never touched it

Probes cover both heights: a standing roundhouse (either block works), a crouching forward and a sweep (low: only
block_low), a jump-in kick (high: only block_high works on a real jump-in). The jump-in reacts to RAM: the attacker
jumps, and presses the kick as it comes back down past DESCEND_Y, so it lands whoever jumps; the character decides once
the attacker, having gone up (above HIGH_Y), comes back down past it. Jumps differ per character, so the collector tries KICK_HEIGHTS in order and
keeps the first that lands on a character who does not block (a jump-in that never lands is recorded as it is).

Record the probe as ``attacker``'s exchange (sf2.emu.vs.record) so a_ = the attacker, d_ = the character.
"""
from typing import Dict, List, Tuple

from .emu.vs import GROUND_Y, Step

HIGH_Y, DESCEND_Y = 150, 150          # world y is smaller when higher (ground 192)
LEAD = 4                              # idle frames before the probe starts (the "a moment ago" frame is before it)
REACT = 3                             # frames after a ground attack starts that the character decides
HOLD = 60                             # frames a block is held: a jump-in decided at take-off lands ~40 frames later
BLOCK_REACTS = (0x06, 0x08)

ANSWERS: Dict[str, Tuple[str, ...]] = {"block_high": ("B",), "block_low": ("D", "B"), "idle": ()}
BLOCKS = ("block_high", "block_low")


def block_steps(action: str) -> Tuple[Step, ...]:
    """The move as a character's action (sf2.vs_sweep.actions): hold the guard direction."""
    return ((ANSWERS[action], HOLD),)


KICK_HEIGHTS = (150, 160, 170, 140, 180, 185, 188)   # jump-in kick heights, tried in order (Dhalsim lands at 185+)


def conds(kick_y: int = DESCEND_Y) -> Dict:
    return {"up": lambda now, start: now["a_y"] < HIGH_Y,                 # the attacker is visibly in the air
            "descending": lambda now, start: now["a_y"] >= HIGH_Y,        # ... and coming back down at me
            "coming_down": lambda now, start: now["a_y"] >= kick_y}       # ... low enough to kick


CONDS = conds()


def probes(attacker_actions: Dict[str, Tuple[Step, ...]]) -> Dict[str, Dict]:
    """probe name -> the attacker's steps and the character's lead-in before it answers."""
    # waits capped at 90 frames: Dhalsim's floaty jump takes ~70 frames and only comes down at ~60 (a 40 cap kicked
    # while he was still high, so his jump-in never landed)
    jump = ((("U", "F"), 4), ("until", "up", (), 90), ("until", "coming_down", (), 90), (("hk",), 2), ((), 20))
    ground = {"s.hk": attacker_actions["hk"], "c.mk": attacker_actions["c.mk"], "sweep": attacker_actions["sweep"]}
    out = {name: {"attacker": (((), LEAD),) + steps, "lead": (((), LEAD + REACT),), "height": "low"
                  if name in ("c.mk", "sweep") else "mid"} for name, steps in ground.items()}
    # the character decides as the attacker comes back down at it (still before the kick): deciding at take-off
    # let a slow jump-in (Dhalsim: ~70 frames, a slow landing kick) outlast the block, which then let go too early
    out["jump_in"] = {"attacker": (((), LEAD),) + jump,
                      "lead": (((), LEAD + 4), ("until", "up", (), 90), ("until", "descending", (), 90)),
                      "height": "high"}
    return out


def decision_frame(rows: List[Dict], probe: str) -> int:
    """The frame index the character answers on (its lead-in is over): fixed for ground probes, the first frame the
    attacker is up for the jump-in."""
    if probe != "jump_in":
        return LEAD + REACT
    up = next(i for i, r in enumerate(rows) if i >= LEAD + 4 and r["a_y"] < HIGH_Y)
    return next(i for i in range(up, len(rows)) if rows[i]["a_y"] >= HIGH_Y)


def outcome(rows: List[Dict]) -> Dict:
    """What happened to the CHARACTER (d_) from its decision on. blocked needs real block stun: the guard pose alone
    (holding back while an attack is out nearby) is not a block, as the dummy data taught."""
    got_hit = any(r["d_state"] == 0x14 or (r["d_state"] == 0x0E and r["d_react"] not in BLOCK_REACTS) for r in rows)
    blocked = not got_hit and any(r["d_state"] == 0x0E and r["d_react"] in BLOCK_REACTS for r in rows)
    lives = [r["d_life"] if r["d_life"] < 200 else 0 for r in rows]
    return {"outcome": "got_hit" if got_hit else "blocked" if blocked else "none",
            "damage_taken": max(0, lives[0] - min(lives)),
            "airborne_attacker": any(r["a_y"] < GROUND_Y for r in rows)}
