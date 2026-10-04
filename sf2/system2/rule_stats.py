"""Pooled per-rule statistics for the live short memory (Step 1B of docs/plan_review_framework.md).

The bug this fixes: each career BLOCK is a fresh play_loop_screen subprocess with `all_rows=[]`, so
`lessons.condition_evidence` never sees more than one block's decisions and `MIN_TRIES=20` on both sides is
almost never reached -> no rule ever graduates -> endless churn. Here each rule carries a small `stats` dict
IN its registry entry (beside `rounds`), folded after every round and CARRIED across blocks/restarts by
`--save-registry`/`--carry`. `evidence_from_stats` reproduces `condition_evidence` exactly, but over the pooled
sums, so the evidence accumulates and graduation can actually fire.

Pure: no I/O, returns NEW dicts (owner immutability). `tally` folds one round; the loop writes the result back.
"""
from typing import Dict, Optional, Sequence

from ..system1.advice import opp_doing
from .move_coach import MIN_TRIES

Stats = Dict[str, float]

_ZERO: Stats = {"applicable": 0, "followed": 0, "n_mine": 0, "sum_mine": 0.0, "sumsq_mine": 0.0,
                "n_rest": 0, "sum_rest": 0.0, "sumsq_rest": 0.0,
                "rounds_fired": 0, "wins_fired": 0, "rounds_idle": 0, "wins_idle": 0}


def blank() -> Stats:
    return dict(_ZERO)


def _matches(row: Dict, rng: Optional[str], when: Optional[str]) -> bool:
    """A decision row is in the claim's CONDITION -- same filter lessons.condition_evidence uses."""
    return (rng is None or row.get("range") == rng) and (when is None or opp_doing(row) == when)


def tally(stats: Optional[Stats], rows: Sequence[Dict], round_summary: Dict, claim: Dict) -> Stats:
    """Fold ONE round's decision rows (``rows`` = this round's drows, carrying range/action/dealt/taken and the
    opp-state fields ``opp_doing`` reads) into a NEW stats dict for ``claim``. Accumulates the same mine/rest net-hp
    sums ``condition_evidence`` computes, plus applicable/followed counts and the fired-vs-idle round outcome."""
    st = dict(stats) if stats else blank()
    rng, when, move = claim.get("range"), claim.get("when"), claim.get("move")
    fired = False
    for a in rows:
        if not _matches(a, rng, when):
            continue
        st["applicable"] += 1
        net = (a.get("dealt", 0) or 0) - (a.get("taken", 0) or 0)
        if a.get("action") == move:
            st["followed"] += 1
            st["n_mine"] += 1
            st["sum_mine"] += net
            st["sumsq_mine"] += net * net
            fired = True
        else:
            st["n_rest"] += 1
            st["sum_rest"] += net
            st["sumsq_rest"] += net * net
    won = 1 if round_summary.get("result") == "win" else 0
    if fired:
        st["rounds_fired"] += 1
        st["wins_fired"] += won
    else:
        st["rounds_idle"] += 1
        st["wins_idle"] += won
    return st


def _var(n: float, s: float, ss: float) -> float:
    """Sample variance from n, sum, sumsq (== sum((x-mean)^2)/(n-1)); inf for n<=1 (matches lessons._mean_var)."""
    return (ss - s * s / n) / (n - 1) if n > 1 else float("inf")


def evidence_from_stats(st: Stats) -> Dict:
    """Reproduce ``lessons.condition_evidence``'s output dict from the pooled sums (same MIN_TRIES gate, same Welch
    half-width, same better/worse/unclear/few rule) -- so graduation over pooled stats matches the block-local scorer."""
    nm, sm, qm = st.get("n_mine", 0), st.get("sum_mine", 0.0), st.get("sumsq_mine", 0.0)
    nr, sr, qr = st.get("n_rest", 0), st.get("sum_rest", 0.0), st.get("sumsq_rest", 0.0)
    net = sm / nm if nm else 0.0
    base = sr / nr if nr else 0.0
    out = {"tries": nm, "others": nr, "net": net, "base": base, "diff": 0.0, "lo": 0.0, "hi": 0.0, "cls": "few"}
    if nm < MIN_TRIES or nr < MIN_TRIES:
        return out
    half = 1.96 * (_var(nm, sm, qm) / nm + _var(nr, sr, qr) / nr) ** 0.5
    d = net - base
    return dict(out, diff=d, lo=d - half, hi=d + half,
                cls="better" if d - half > 0 else "worse" if d + half < 0 else "unclear")
