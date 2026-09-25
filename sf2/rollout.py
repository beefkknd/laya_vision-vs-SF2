"""Shared bookkeeping for rollouts: after-the-fact damage windows, whiffs, and the gate numbers.

Every decision row's ``meta`` carries: episode, round, frame, frames (how long the action ran), action (what was
executed), dmg_for / dmg_against (during that action), and round_result once the round is over.
"""
from collections import Counter, defaultdict
from typing import Dict, List

from .config import NEXT_WINDOW, WHIFF_WINDOW

BIG_HIT = 20  # life points (of 176) taken inside the next window ~ a combo or knockdown


def annotate(rows: List[Dict], window: int = NEXT_WINDOW, whiff_window: int = WHIFF_WINDOW) -> List[Dict]:
    """Add dmg_for_next / dmg_against_next (the next ``window`` frames), whiff and hot flags to each row's meta."""
    by_round = defaultdict(list)
    for r in rows:
        m = r["meta"]
        by_round[(m["episode"], m["round"])].append(r)
    for rs in by_round.values():
        rs.sort(key=lambda r: r["meta"]["frame"])
        for i, r in enumerate(rs):
            m = r["meta"]
            f0 = m["frame"]
            fwd = agn = fwd_w = 0
            for s in rs[i:]:
                f = s["meta"]["frame"]
                if f < f0 + window:
                    fwd += s["meta"]["dmg_for"]
                    agn += s["meta"]["dmg_against"]
                if f < f0 + whiff_window:
                    fwd_w += s["meta"]["dmg_for"]
                else:
                    break
            m["dmg_for_next"], m["dmg_against_next"] = fwd, agn
            m["whiff"] = m["action"] in ("hadouken", "shoryuken") and fwd_w == 0
            m["hot"] = agn > 0 or m["whiff"]          # hits taken (incl. knockdowns) and whiffed specials
            m["big_hit"] = agn >= BIG_HIT
    return rows


def gate(rows: List[Dict], rounds: List[Dict]) -> Dict:
    """Round win rate, damage per round, special-move whiff rates, action mix."""
    n = len(rounds)
    wins = sum(r["winner"] == "me" for r in rounds)
    dealt = sum(r["dmg_for"] for r in rounds)
    taken = sum(r["dmg_against"] for r in rounds)
    acts = Counter(r["meta"]["action"] for r in rows)
    out = {
        "rounds": n,
        "round_win_rate": wins / n if n else 0.0,
        "dmg_dealt_per_round": dealt / n if n else 0.0,
        "dmg_taken_per_round": taken / n if n else 0.0,
        "decisions": len(rows),
        "action_mix": {a: round(c / max(1, len(rows)), 3) for a, c in acts.most_common()},
    }
    for sp in ("hadouken", "shoryuken"):
        tries = [r for r in rows if r["meta"]["action"] == sp]
        out[sp + "_whiff_rate"] = sum(r["meta"]["whiff"] for r in tries) / len(tries) if tries else None
    agree = [r for r in rows if r["meta"].get("teacher_action")]
    if agree:
        out["teacher_agreement"] = sum(r["meta"]["action"] == r["meta"]["teacher_action"] for r in agree) / len(agree)
    return out
