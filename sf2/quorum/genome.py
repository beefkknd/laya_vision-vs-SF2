"""Choice 3: the evolution loop's genome. Bounds, sampling, mutation and fitness for QuorumConfig numbers.

The loop itself is scripts/quorum_evolve.py; this module is pure so it can be tested and reused.

Fitness (docs/plan_bee_quorum.md):  F = mean net hp per round + lambda_w * round win rate
                                        - lambda_q * max(0, escalation rate - target)
The escalation penalty stops evolution from raising theta until Qwen decides everything (which deletes System 1);
the win-rate term stops it from winning on margin by turtling.
"""
import math
import random
from typing import Dict, List, Sequence

from .config import VOTERS, QuorumConfig

# gene -> (low, high, log-scale?)
BOUNDS: Dict[str, tuple] = {
    "theta": (0.30, 0.90, False),
    "epsilon": (0.0, 0.20, False),
    "beta": (0.0, 2.0, False),
    "gamma": (0.0, 2.0, False),
    "k": (1.0, 50.0, True),
    "eta": (0.01, 0.5, True),
    "net_scale": (2.0, 40.0, True),
}
PRIOR_BOUNDS = (0.2, 3.0)
GENES: List[str] = list(BOUNDS) + ["prior_" + v for v in VOTERS]


def _bounds(gene: str) -> tuple:
    return BOUNDS[gene] if gene in BOUNDS else (PRIOR_BOUNDS[0], PRIOR_BOUNDS[1], True)


def encode(cfg: QuorumConfig) -> Dict[str, float]:
    g = {k: float(getattr(cfg, k)) for k in BOUNDS}
    g.update({"prior_" + v: float(cfg.priors[v]) for v in VOTERS})
    return g


def decode(genes: Dict[str, float], base: QuorumConfig) -> QuorumConfig:
    d = base.to_dict()
    for k, v in genes.items():
        lo, hi, _ = _bounds(k)
        v = min(hi, max(lo, float(v)))
        if k.startswith("prior_"):
            d["priors"][k[len("prior_"):]] = v
        else:
            d[k] = v
    return QuorumConfig.from_dict(d)


def mutate(genes: Dict[str, float], rng: random.Random, sigma: float = 0.15) -> Dict[str, float]:
    """Gaussian step of ``sigma`` x the gene's range (in log space for the log-scale genes), clipped to bounds."""
    out = {}
    for k, v in genes.items():
        lo, hi, log = _bounds(k)
        if log:
            span = math.log(hi) - math.log(lo)
            v = math.exp(math.log(max(v, lo)) + rng.gauss(0, sigma * span))
        else:
            v = v + rng.gauss(0, sigma * (hi - lo))
        out[k] = min(hi, max(lo, v))
    return out


def fitness(rounds: Sequence[Dict], escalation_rate: float, win_weight: float = 20.0, qwen_weight: float = 50.0,
            qwen_target: float = 0.10) -> float:
    """``rounds``: per-round summaries with ``dealt`` / ``taken`` / ``result`` (screen_evidence / verdict)."""
    if not rounds:
        return float("-inf")
    net = sum((r.get("dealt", 0) or 0) - (r.get("taken", 0) or 0) for r in rounds) / len(rounds)
    wins = sum(1 for r in rounds if r.get("result") == "win") / len(rounds)
    return net + win_weight * wins - qwen_weight * max(0.0, escalation_rate - qwen_target)
