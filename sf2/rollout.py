"""Shared bookkeeping for rollouts: after-the-fact damage windows, hot flags, and the gate numbers.

Every decision row's ``meta`` carries: episode, round, frame, frames (how long the action ran), action (what was
executed), dmg_for / dmg_against (during that action), and round_result once the round is over.
"""
import math
from collections import Counter, defaultdict
from typing import Dict, List

from .config import NEXT_WINDOW, ROUND_LIFE

BIG_HIT = 20  # life points (of 176) taken inside the next window ~ a combo or knockdown


def annotate(rows: List[Dict], window: int = NEXT_WINDOW) -> List[Dict]:
    """Add dmg_for_next / dmg_against_next (the next ``window`` frames) and hot flags to each row's meta."""
    by_round = defaultdict(list)
    for r in rows:
        m = r["meta"]
        by_round[(m["episode"], m["round"])].append(r)
    for rs in by_round.values():
        rs.sort(key=lambda r: r["meta"]["frame"])
        for i, r in enumerate(rs):
            m = r["meta"]
            f0 = m["frame"]
            fwd = agn = 0
            for s in rs[i:]:
                if s["meta"]["frame"] >= f0 + window:
                    break
                fwd += s["meta"]["dmg_for"]
                agn += s["meta"]["dmg_against"]
            m["dmg_for_next"], m["dmg_against_next"] = fwd, agn
            m["hot"] = agn > 0                          # hits taken (incl. knockdowns)
            m["big_hit"] = agn >= BIG_HIT
    return rows


def gate(rows: List[Dict], rounds: List[Dict]) -> Dict:
    """Round win rate, damage per round, action mix."""
    n = len(rounds)
    wins = sum(r["winner"] == "me" for r in rounds)
    dealt = sum(r["dmg_for"] for r in rounds)
    taken = sum(r["dmg_against"] for r in rounds)
    acts = Counter(r["meta"]["action"] for r in rows)
    matches = defaultdict(list)
    for r in rounds:  # a match's signature: every round's damage, winner and end frame
        matches[r["episode"]].append((r["round"], r["dmg_for"], r["dmg_against"], r["winner"], r.get("end_frame")))
    def sd(xs):
        mean = sum(xs) / len(xs) if xs else 0.0
        return math.sqrt(sum((x - mean) ** 2 for x in xs) / (len(xs) - 1)) if len(xs) > 1 else 0.0

    dealt_sd = sd([r["dmg_for"] for r in rounds])
    net_sd = sd([r["dmg_for"] - r["dmg_against"] for r in rounds])
    dealt_per_round = dealt / n if n else 0.0
    taken_per_round = taken / n if n else 0.0
    out = {
        "rounds": n,
        "round_win_rate": wins / n if n else 0.0,
        "dmg_dealt_per_round": dealt_per_round,
        "dmg_taken_per_round": taken_per_round,
        # Early learning metric: 100 means a full opponent life bar per round.
        # It remains informative before the policy can reliably win rounds.
        "damage_score": 100.0 * dealt_per_round / ROUND_LIFE,
        "net_damage_per_round": dealt_per_round - taken_per_round,
        "matches": len(matches),
        "distinct_matches": len({tuple(v) for v in matches.values()}),
        "dmg_dealt_sd": dealt_sd,
        "damage_score_se": 100.0 * dealt_sd / ROUND_LIFE / math.sqrt(n) if n else 0.0,
        "net_damage_se": net_sd / math.sqrt(n) if n else 0.0,
        "decisions": len(rows),
        "action_mix": {a: round(c / max(1, len(rows)), 3) for a, c in acts.most_common()},
    }
    agree = [r for r in rows if r["meta"].get("teacher_action")]
    if agree:
        out["teacher_agreement"] = sum(r["meta"]["action"] == r["meta"]["teacher_action"] for r in agree) / len(agree)
    return out
