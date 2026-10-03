"""Measurement seam for the outcome loop: assemble a BlockStat from two arms over a seed block.

PURE given an injected run_arm(rules, seed) -> (hp_margin, win, decisions) - the game-shelling runner
lives in scripts/outcome_loop.py so this stays unit-testable without a ROM or emulator. Independent
sampling across seeds (pairing was shown not to reduce variance), Welch two-sample 95% CI on the
candidate-minus-incumbent hp margin, candidate wins, and the candidate's pooled fire_rate/follows.
"""
import math
import statistics as st

from sf2.system2.coverage import coverage
from sf2.system2.promotion import BlockStat


def welch_delta(base, cand):
    """(delta, lo, hi) for mean(cand) - mean(base) with a Welch two-sample 95% CI."""
    na, nb = len(base), len(cand)
    if na == 0 or nb == 0:
        raise ValueError("both arms need at least one sample")
    ma, mb = st.mean(base), st.mean(cand)
    va = st.variance(base) if na > 1 else 0.0
    vb = st.variance(cand) if nb > 1 else 0.0
    se = math.sqrt(va / na + vb / nb)
    d = mb - ma
    return d, d - 1.96 * se, d + 1.96 * se


def assemble(base, cand):
    """Pure stat assembly. base/cand are equal-length lists of (hp_margin, win, decisions), one per
    seed. Returns a BlockStat for candidate-minus-incumbent. fire_rate/follows are pooled over ALL
    candidate decision frames. Separated from the runner so the CLI can run games in parallel and
    still feed this one tested assembler."""
    if not base or not cand:
        raise ValueError("need at least one sample in each arm")
    if len(base) != len(cand):
        raise ValueError("arms must have the same number of seeds: %d vs %d" % (len(base), len(cand)))
    bh = [b[0] for b in base]
    ch = [c[0] for c in cand]
    cw = sum(c[1] for c in cand)
    fire, follows = coverage([d for c in cand for d in c[2]])
    d, lo, hi = welch_delta(bh, ch)
    return BlockStat(delta=d, lo=lo, hi=hi, cand_wins=cw, n=len(cand), fire_rate=fire, follows=follows)


def measure_block(incumbent_rules, candidate_rules, seeds, run_arm):
    """Serial convenience: run both arms over `seeds` with run_arm(rules, seed) -> (hp, win, decisions)
    and assemble() the BlockStat. The CLI uses a parallel runner + assemble() directly instead."""
    seeds = list(seeds)
    if not seeds:
        raise ValueError("need at least one seed")
    base = [run_arm(tuple(incumbent_rules), s) for s in seeds]
    cand = [run_arm(tuple(candidate_rules), s) for s in seeds]
    return assemble(base, cand)
