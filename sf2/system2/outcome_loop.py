"""Outcome-driven loop orchestration (step C). PURE mechanics: the measurer (runs games -> BlockStat)
and the proposer (Qwen -> candidate playbooks) are INJECTED, so promote/keep, held-out gating, seed
rotation and the evidence ledger are provable without a game or a model call.

The policy is text-laya reading short memory (feedback_textlaya_is_the_player); this loop only SELECTS
which short-memory playbook survives, by in-game OUTCOME. Per the plan (docs/plan_outcome_loop.md):
  - a candidate replaces the incumbent only if decide() promotes it (dev-significant + held replicates);
  - the expensive held-out block is run only when the dev result warrants it (cost/noise control);
  - at most ONE promotion per round, the best by held-out delta (noise control);
  - dev and held seed blocks are DISJOINT and the held block rotates per round; the TERMINAL block is
    reserved and never touched in-loop (winner's-curse guard).

Types here are immutable; run_round/run_session return new state rather than mutating inputs.
"""
from dataclasses import dataclass
from typing import Callable, Optional, Sequence, Tuple

from sf2.system2.promotion import Cfg, coverage_gated, decide, is_gain, warrants_held


@dataclass(frozen=True)
class Playbook:
    """A short-memory version: an id and the advice lines it carries (the candidate text-laya plays)."""
    id: str
    rules: Tuple[str, ...]


@dataclass(frozen=True)
class SeedBlocks:
    """Seed discipline for a session. dev/held are pools of blocks rotated per round and must be
    mutually disjoint (no peeking leakage); terminal is reserved and never used in-loop."""
    dev: Tuple[Tuple[int, ...], ...]
    held: Tuple[Tuple[int, ...], ...]
    terminal: Tuple[int, ...]

    def __post_init__(self):
        dev_s = {s for blk in self.dev for s in blk}
        held_s = {s for blk in self.held for s in blk}
        term_s = set(self.terminal)
        if dev_s & held_s:
            raise ValueError("dev and held pools share seeds %s" % sorted(dev_s & held_s))
        if term_s & (dev_s | held_s):
            raise ValueError("terminal overlaps dev/held seeds %s" % sorted(term_s & (dev_s | held_s)))
        if not self.dev or not self.held:
            raise ValueError("need at least one dev block and one held block")


@dataclass(frozen=True)
class RoundResult:
    new_incumbent: Playbook
    promoted: bool
    rows: Tuple[dict, ...]


def blocks_for_round(sb: SeedBlocks, k: int):
    """(dev_seeds, held_seeds) for round k, rotating through the disjoint pools."""
    return sb.dev[k % len(sb.dev)], sb.held[k % len(sb.held)]


def _ledger_row(cand: Playbook, dev, held, dec, cfg):
    return {"candidate": cand.id, "rules": list(cand.rules),
            "verdict": dec.verdict, "ceiling": dec.ceiling, "reason": dec.reason,
            "dev_delta": dev.delta, "dev_ci": [dev.lo, dev.hi], "dev_wins": dev.cand_wins,
            "dev_fire": dev.fire_rate, "dev_follows": dev.follows,
            "dev_significant": is_gain(dev),              # CI excludes 0 and positive
            "coverage_gated": coverage_gated(dev, cfg),   # NEAR-MISS: significant but fires < floor
            "held_delta": (held.delta if held else None),
            "held_ci": ([held.lo, held.hi] if held else None),
            "held_wins": (held.cand_wins if held else None)}


# measure(incumbent, candidate, seeds) -> BlockStat (candidate-minus-incumbent on those seeds)
Measure = Callable[[Playbook, Playbook, Sequence[int]], object]


def run_round(incumbent: Playbook, candidates: Sequence[Playbook], measure: Measure,
              dev_seeds: Sequence[int], held_seeds: Sequence[int], cfg: Cfg = Cfg()) -> RoundResult:
    """Evaluate a few candidates against the incumbent; promote at most one (best held-out delta)."""
    rows = []
    winner: Optional[Playbook] = None
    winner_held_delta = None
    for cand in candidates:
        dev = measure(incumbent, cand, dev_seeds)
        held = measure(incumbent, cand, held_seeds) if warrants_held(dev, cfg) else None
        dec = decide(dev, held, cfg)
        rows.append(_ledger_row(cand, dev, held, dec, cfg))
        if dec.verdict == "promote" and (winner_held_delta is None or held.delta > winner_held_delta):
            winner, winner_held_delta = cand, held.delta
    return RoundResult(winner or incumbent, winner is not None, tuple(rows))


def run_session(opp: str, incumbent: Playbook, proposer, measure: Measure, seeds: SeedBlocks,
                n_rounds: int, ledger_write, cfg: Cfg = Cfg()) -> Playbook:
    """Run n_rounds of propose -> evaluate -> maybe-promote; append every candidate's row to the
    ledger (the system-of-record). Returns the final incumbent. The terminal block is NOT touched
    here - the caller runs the terminal untouched test on the returned playbook separately."""
    for k in range(n_rounds):
        dev_seeds, held_seeds = blocks_for_round(seeds, k)
        candidates = proposer(opp, incumbent, k)
        rr = run_round(incumbent, candidates, measure, dev_seeds, held_seeds, cfg)
        for row in rr.rows:
            ledger_write({**row, "round": k, "opp": opp, "incumbent": incumbent.id})
        incumbent = rr.new_incumbent
    return incumbent
