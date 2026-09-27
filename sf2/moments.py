"""Moments: the decisions System 1 hands to System 2 (docs/TWO_SYSTEM_PLAN.md §2.3), pulled from a rollout.

Only controllable decisions are flagged, and each at most once:

- ``surprised``: the last controllable decision before she takes a hit. A combo (hits less than a window apart) is
  one episode and one moment.
- ``unsure``: laya's top-2 margin is below ``unsure_margin`` (tune it with ``margin_for_share`` to flag ~5%).
- ``audit``: a random ``audit_rate`` share of the remaining decisions.

    python scripts/moments.py rollouts/<name>      # -> out/moments/<name>.jsonl
"""
import random
from collections import defaultdict
from typing import Dict, List

from . import contract as C
from .config import FPS, NEXT_WINDOW

BEFORE = FPS  # the notes of the second before a moment
TOP = 5       # probabilities kept per moment


def _rounds(rows: List[Dict]):
    by = defaultdict(list)
    for r in rows:
        by[(r["meta"]["episode"], r["meta"]["round"])].append(r)
    for key in sorted(by):
        yield sorted(by[key], key=lambda r: r["meta"]["frame"])


def _margin(r: Dict) -> float:
    return C.confidence(r["meta"]["student_probs"])[0]


def _surprises(rs: List[Dict], window: int) -> List[int]:
    """Indexes of the last controllable decision before each hit episode."""
    out, last_hit = [], None
    for i, r in enumerate(rs):
        if r["meta"]["dmg_against"] <= 0:
            continue
        frame = r["meta"]["frame"]
        new = last_hit is None or frame - last_hit >= window
        last_hit = frame
        if not new:
            continue
        before = [j for j in range(i) if rs[j]["meta"].get("controllable")]
        if before and before[-1] not in out:
            out.append(before[-1])
    return out


def _record(rs: List[Dict], i: int, why: str) -> Dict:
    r, m = rs[i], rs[i]["meta"]
    f = m["frame"]
    probs = dict(sorted(m["student_probs"].items(), key=lambda kv: -kv[1])[:TOP])
    notes_before = [s["state_text"] for s in rs[:i] if f - BEFORE <= s["meta"]["frame"] < f]
    return C.moment_record(match=m["episode"], round=m["round"], frame=f, why=why, notes_before=notes_before,
                           note=r["state_text"], probs=probs, played=m["action"], dealt=int(m["dmg_for_next"]),
                           taken=int(m["dmg_against_next"]), frames=list(r["images"]),
                           margin=round(_margin(r), 4))


def extract(rows: List[Dict], rng: random.Random, unsure_margin: float = C.UNSURE_MARGIN,
            audit_rate: float = C.AUDIT_RATE, window: int = NEXT_WINDOW) -> List[Dict]:
    out = []
    for rs in _rounds([r for r in rows if "student_probs" in r["meta"]]):
        why = {i: "surprised" for i in _surprises(rs, window)}
        for i, r in enumerate(rs):
            if i in why or not r["meta"].get("controllable"):
                continue
            if _margin(r) < unsure_margin:
                why[i] = "unsure"
            elif rng.random() < audit_rate:
                why[i] = "audit"
        out += [_record(rs, i, w) for i, w in sorted(why.items())]
    return out


def margin_for_share(rows: List[Dict], share: float) -> float:
    """The top-2 margin below which ``share`` of the controllable decisions fall."""
    ms = sorted(_margin(r) for r in rows if r["meta"].get("controllable") and "student_probs" in r["meta"])
    return ms[min(len(ms) - 1, int(share * len(ms)))] if ms else 0.0
