"""The game log System 2 will review: one line per action System 1 took, from that decision to the next one (so the
opponent's reaction, and any punish while System 1 could not act, belong to the action that led to it), and one
line per game.

actions.jsonl, per action:
    who / when:    game, frame, clock (seconds left), me, opp, side, gap, range, my_life, opp_life
    before:        my_state, opp_state (named), opp_air
    decision:      action, kind, p_hit (model, for the chosen attack), predicted, top3 (model P(hit))
    result:        actual (hit / whiff / blocked / none), dealt
    reaction:      opp_reaction (the opponent's states in order), opp_attacked, opp_blocked, i_was_hit, taken
    after:         gap_after, my_life_after, opp_life_after, frames (to the next decision), images
    scores:        laya-vision's score for every move (P(hit), P(blocked) for a block); top3 is its first three
    his move:      opp_move (fireball / uppercut / hurricane / slap / throw / jump_attack / normal / none: his attack
                   episode overlapping this window, sf2.system1.opp_moves), opp_shot (his projectile out at the decision)
games.jsonl, per game: result, frames, clock_end, lives at the end, dealt, taken, action counts, outcome counts.
"""
import collections
from typing import Dict, List

from ..vocab import range_of
from ..data.vs_defense import BLOCKS
from ..data.vs_sweep import MOVEMENT

STATE = {0x00: "stand", 0x02: "crouch", 0x04: "jump", 0x06: "turn", 0x08: "guard", 0x0A: "attack",
         0x0C: "special", 0x0E: "hit_stun", 0x10: "win_pose", 0x12: "timeover", 0x14: "thrown"}
BLOCK_REACTS = (0x06, 0x08)


def name(state: int) -> str:
    return STATE.get(state, "0x%02X" % state)


def clock(timer_bcd: int) -> int:
    return (timer_bcd >> 4) * 10 + (timer_bcd & 0x0F)


def _life(v: int) -> int:
    return v if v < 200 else 0          # a KO wraps the life byte to ~255


def _drops(rows: List[Dict], key: str) -> int:
    return sum(max(0, _life(a[key]) - _life(b[key])) for a, b in zip(rows, rows[1:]))


def _sequence(states: List[int]) -> List[str]:
    out: List[str] = []
    for s in states:
        if not out or out[-1] != name(s):
            out.append(name(s))
    return out


def action_entry(game: int, frame: int, me: str, opp: str, before: Dict, rows: List[Dict], decision: Dict,
                 actual: str) -> Dict:
    """``before``: the RAM row at the decision; ``rows``: every row after it up to the next decision."""
    after = rows[-1]
    gap = abs(before["p2_x"] - before["p1_x"])
    hit_me = [r for r in rows if r["p1_state"] == 0x14 or (r["p1_state"] == 0x0E and r["p1_react"] not in BLOCK_REACTS)]
    top3 = sorted(decision["probs"].items(), key=lambda kv: -kv[1])[:3]
    return {
        "game": game, "frame": frame, "clock": clock(before["timer"]), "me": me, "opp": opp,
        "side": "left" if before["p1_x"] < before["p2_x"] else "right", "gap": gap, "range": range_of(gap),
        "my_life": _life(before["p1_life"]), "opp_life": _life(before["p2_life"]),
        "my_state": name(before["p1_state"]), "opp_state": name(before["p2_state"]), "opp_air": before["p2_y"] != 192,
        "action": decision["action"], "kind": "movement" if decision["action"] in MOVEMENT else
        "defense" if decision["action"] in BLOCKS else "attack",
        "p_hit": decision["p_hit"], "predicted": decision["predicted"], "top3": [[a, p] for a, p in top3],
        "actual": actual, "dealt": _drops([before] + rows, "p2_life"),
        "opp_reaction": _sequence([r["p2_state"] for r in rows]),
        "opp_attacked": any(r["p2_state"] in (0x0A, 0x0C) for r in rows),
        "opp_blocked": any(r["p2_state"] == 0x08 or (r["p2_state"] == 0x0E and r["p2_react"] in BLOCK_REACTS)
                           for r in rows),
        "i_was_hit": bool(hit_me), "taken": _drops([before] + rows, "p1_life"),
        "gap_after": abs(after["p2_x"] - after["p1_x"]), "my_life_after": _life(after["p1_life"]),
        "opp_life_after": _life(after["p2_life"]), "frames": len(rows),
        "scores": dict(decision["probs"]),
    }


def game_entry(game: int, result: str, frames: int, last: Dict, actions: List[Dict]) -> Dict:
    return {
        "game": game, "result": result, "frames": frames, "clock_end": clock(last["timer"]),
        "my_life_end": _life(last["p1_life"]), "opp_life_end": _life(last["p2_life"]),
        "dealt": sum(a["dealt"] for a in actions), "taken": sum(a["taken"] for a in actions),
        "actions": len(actions), "by_action": dict(collections.Counter(a["action"] for a in actions)),
        "outcomes": dict(collections.Counter(a["actual"] for a in actions if a["kind"] == "attack")),
        "times_hit": sum(a["i_was_hit"] for a in actions),
    }
