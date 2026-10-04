"""Proposals, the table voter, and the weighted tally that turns votes into a vote share.

    S(a) = sum over voters v that proposed a of  confidence_v * weight_v(when)
           + beta  * shrink(n) * max(0,  squash(mean))      recruitment
           - gamma * shrink(n) * max(0, -squash(mean))      cross-inhibition
    shrink(n) = n / (n + k);  share(top) = max(S(top), 0) / sum of max(S(a), 0) over the candidates

Pure: no models, no I/O. ``cell`` is value_table's per-action stats for the 'when' ({action: [n, sum, sumsq]}).
"""
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from . import reliability as R
from .config import QuorumConfig


@dataclass(frozen=True)
class Proposal:
    voter: str
    action: str
    confidence: float


def _stats(cell: Dict[str, List], action: str) -> Tuple[int, float]:
    s = cell.get(action)
    if not s or not s[0]:
        return 0, 0.0
    return int(s[0]), s[1] / s[0]


def table_proposal(cell: Dict[str, List], actions: Sequence[str], cfg: QuorumConfig) -> Optional[Proposal]:
    """The table's vote: its best-mean followable move with at least ``table_min_n`` samples and a positive mean
    (it never votes for the least-bad move -- the defensive drift that sank the pure table). Confidence grows with
    evidence: n / (n + k). None = abstain."""
    best = None
    for a in actions:
        n, m = _stats(cell, a)
        if n >= cfg.table_min_n and m > 0 and (best is None or m > best[1]):
            best = (a, m, n)
    if best is None:
        return None
    a, _, n = best
    return Proposal("table", a, n / (n + cfg.k))


def score(proposals: Sequence[Proposal], cell: Dict[str, List], when: str, rel: R.State,
          cfg: QuorumConfig) -> Dict[str, Dict[str, float]]:
    """Per candidate action: {"votes", "recruit", "inhibit", "score"}. Candidates = the distinct proposed actions."""
    out: Dict[str, Dict[str, float]] = {}
    for p in proposals:
        c = out.setdefault(p.action, {"votes": 0.0, "recruit": 0.0, "inhibit": 0.0, "score": 0.0})
        c["votes"] += p.confidence * R.weight(rel, when, p.voter, cfg)
    for a, c in out.items():
        n, m = _stats(cell, a)
        shrink = n / (n + cfg.k)
        x = R.squash(m, cfg.net_scale)
        c["recruit"] = cfg.beta * shrink * max(0.0, x)
        c["inhibit"] = cfg.gamma * shrink * max(0.0, -x)
        c["score"] = c["votes"] + c["recruit"] - c["inhibit"]
    return out


def ranking(scores: Dict[str, Dict[str, float]]) -> Tuple[List[str], float]:
    """(candidates best first, the top candidate's vote share). Ties break by name so a replay is deterministic."""
    order = sorted(scores, key=lambda a: (-scores[a]["score"], a))
    pos = sum(max(0.0, c["score"]) for c in scores.values())
    share = max(0.0, scores[order[0]]["score"]) / pos if order and pos > 0 else 0.0
    return order, share
