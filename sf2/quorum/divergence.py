"""Divergence study: what a SEEDED character learned that differs from its prior (owner 2026-10-05).

When ken is seeded from ryu and grown via merge(workers, shared=ryu_seed), the grown table's per-cell moments are
the seed's PLUS ken's own. So ken's own contribution is recoverable by subtraction:
    own_n    = grown.n   - prior.n
    own_sum  = grown.sum - prior.sum
    own_mean = own_sum / own_n            (ken's observed value, free of ryu's prior)
    delta    = own_mean - prior_mean      (how ken revised ryu's value here)
Cells with enough ken samples and a large |delta| are where ken GENUINELY differs from ryu -- "what is missing/
different for ken", which a transferred table otherwise hides. Cells ryu never had (prior_mean None) are ken-new.
Pure arithmetic; no I/O. ``prior`` and ``grown`` are value_table ``cells`` dicts ({when: {move: [n, sum, sumsq]}})."""
import math
from typing import Dict, List, Optional


def _nsum(cell: Dict, move: str):
    s = cell.get(move)
    return (s[0], s[1]) if s else (0, 0.0)


def divergence(prior: Dict[str, Dict[str, List]], grown: Dict[str, Dict[str, List]],
               min_own_n: int = 4) -> List[Dict]:
    """Rows for every (when, move) the seeded character sampled at least ``min_own_n`` times of its own, ranked by
    evidence-weighted |delta| (|delta| * sqrt(own_n)) so a big, well-sampled revision ranks above a noisy one."""
    out: List[Dict] = []
    for when, moves in grown.items():
        pcell = prior.get(when, {})
        for move, g in moves.items():
            gn, gs = g[0], g[1]
            pn, ps = _nsum(pcell, move)
            own_n = gn - pn
            if own_n < min_own_n:
                continue
            own_mean = (gs - ps) / own_n
            prior_mean = (ps / pn) if pn else None
            delta = own_mean - (prior_mean if prior_mean is not None else 0.0)
            out.append({"when": when, "move": move, "own_n": own_n,
                        "own_mean": own_mean, "prior_mean": prior_mean, "delta": delta,
                        "kind": "divergent" if prior_mean is not None else "ken-new",
                        "weight": abs(delta) * math.sqrt(own_n)})
    out.sort(key=lambda r: -r["weight"])
    return out
