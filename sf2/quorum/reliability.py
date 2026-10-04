"""Per-(when, voter) reliability: the "dance length" multiplier of each vote.

State = ``{"rel": {when: {voter: [weight, n]}}}``, JSON-serialisable and kept in its own file (--carry-quorum /
--save-quorum), because value_table's clone/merge keep only their own keys. A voter that proposed the move that was
played is updated from that move's outcome (multiplicative weights): w <- clamp(w * exp(eta * x)), with x the net hp
squashed into (-1, 1). Voters that proposed something else are not updated: their outcome was not observed.
"""
import math
from typing import Dict, Iterable, Sequence

from .config import QuorumConfig

State = Dict


def blank() -> State:
    return {"rel": {}}


def squash(net: float, scale: float) -> float:
    return net / (abs(net) + scale)


def weight(state: State, when: str, voter: str, cfg: QuorumConfig) -> float:
    """The voter's weight in ``when``; a split child falls back to its base cell, then to the voter's prior."""
    rel = state.get("rel", {})
    s = rel.get(when, {}).get(voter)
    if s is None and when.count("|") >= 3:
        s = rel.get("|".join(when.split("|")[:3]), {}).get(voter)
    return float(s[0]) if s else float(cfg.priors.get(voter, 1.0))


def update(state: State, when: str, voters: Iterable[str], net: float, cfg: QuorumConfig) -> None:
    """In place: every voter in ``voters`` (they all proposed the played move) moves by the outcome ``net``."""
    x = squash(net, cfg.net_scale)
    cell = state.setdefault("rel", {}).setdefault(when, {})
    for v in set(voters):
        w0, n = cell.get(v, [cfg.priors.get(v, 1.0), 0])
        w = min(cfg.w_max, max(cfg.w_min, w0 * math.exp(cfg.eta * x)))
        cell[v] = [w, n + 1]


def credit(state: State, decisions: Sequence[Dict], drows: Sequence[Dict], cfg: QuorumConfig) -> State:
    """Fold one round into the reliability state (returns it, updated in place). ``decisions`` are the round's
    decisions.jsonl records and ``drows`` the matching screen_evidence rows (one per decision, same order: the
    delayed-hit-corrected dealt / taken). Only decisions that carry a ``quorum`` record count."""
    for dec, row in zip(decisions, drows):
        q = dec.get("quorum")
        if not q or not dec.get("when"):
            continue
        played = dec.get("action")
        backers = [p[0] for p in q.get("proposals", []) if p[1] == played]
        if backers:
            net = (row.get("dealt", 0) or 0) - (row.get("taken", 0) or 0)
            update(state, dec["when"], backers, net, cfg)
    return state


def merge(states: Sequence[State]) -> State:
    """Pool parallel workers: per (when, voter), n adds up and the weight is the n-weighted mean of log weights
    (the multiplicative update is additive in log space)."""
    acc: Dict[str, Dict[str, list]] = {}
    for st in states:
        for when, cell in st.get("rel", {}).items():
            for v, (w, n) in cell.items():
                a = acc.setdefault(when, {}).setdefault(v, [0.0, 0])
                a[0] += n * math.log(max(w, 1e-12))
                a[1] += n
    out = blank()
    for when, cell in acc.items():
        out["rel"][when] = {v: [math.exp(s / n) if n else 1.0, n] for v, (s, n) in cell.items()}
    return out
