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
from ..system1.value_table import _separated, mean as _mean


NEG_CONF_CAP = 0.9   # a net-negative "least-bad" table vote is held below ~1.0 on purpose: an exploit
                     # voter must never become a lone near-certain dictator in a LOSING cell (the vote
                     # that a KNOWN-GOOD move earns, n/(n+k)->1, is not granted to a merely less-bad one).


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


def table_proposal(cell: Dict[str, List], actions: Sequence[str], cfg: QuorumConfig,
                   laya_move: Optional[str] = None) -> Optional[Proposal]:
    """The table's vote. Two cases, both keyed on the covered argmax (n >= ``table_min_n``):

    - positive argmax  -> vote it, confidence n/(n+k)  (UNCHANGED behaviour).
    - covered argmax NONPOSITIVE -> the old rule abstained, leaving laya UNOPPOSED in every hard
      cell (she over-blocked into jump-ins and pokes). Now vote the least-bad covered move IFF it
      is Welch-``_separated`` strictly ABOVE ``laya_move`` -- i.e. the table has evidence that a
      concrete move beats what laya would do. Confidence = min(NEG_CONF_CAP, n/(n+k) * margin)
      where margin = (m - mean(laya))/net_scale: it fades on both evidence and separation and is
      capped below ~1.0, so it cannot recreate the frontier near-certain-vote stacking bug.
      Abstains if laya's move isn't given, isn't covered, is itself the argmax, or the argmax
      doesn't separate from it.

    ``None`` = abstain."""
    cov = [(a, cell[a][0], _mean(cell[a])) for a in actions
           if a in cell and cell[a][0] >= cfg.table_min_n]
    if not cov:
        return None
    # Pick by SHRUNK mean (mean * n/(n+k)), not raw mean: a 3-sample outlier (+36.7) must not beat a
    # well-sampled move (+34.8 over n=871) and then vote weakly off its tiny n. The chosen move votes
    # at its OWN n/(n+k) -- strong only when well-sampled. (Shrinkage pulls toward 0, so in an
    # all-negative cell it is optimistic about a low-n move; the nonpositive branch below is guarded by
    # the Welch _separated test, which a tiny-n move fails -> abstain, never a bad vote.)
    a, n, m = max(cov, key=lambda t: t[2] * t[1] / (t[1] + cfg.k))
    if m > 0:
        return Proposal("table", a, n / (n + cfg.k))
    laya_s = cell.get(laya_move) if laya_move else None
    if laya_s is None or laya_s[0] < cfg.table_min_n or a == laya_move or not _separated(cell[a], laya_s):
        return None
    # Confidence fades on BOTH evidence and the separation MARGIN. Dropping the margin (so a separated
    # move votes min(cap, n/(n+k)) ~ 0.9) helped Zangief (+11pt) but CRATERED honda 92->18.8% in a
    # cross-character regression: honda's one-step "least-bad" moves are traps (misleading one-step
    # means), and a strong table vote overrode laya's block that was carrying it. So the aggressive
    # vote is NOT globally safe -- the conservative margin stays the default; scope the drop per-char.
    margin = min(1.0, (m - _mean(laya_s)) / cfg.net_scale)    # >0 since _separated implies m > mean(laya_s)
    return Proposal("table", a, min(NEG_CONF_CAP, (n / (n + cfg.k)) * margin))


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
