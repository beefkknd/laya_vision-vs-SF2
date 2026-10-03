"""Keep/drop decision for the outcome-driven loop (step C). PURE: no I/O, no games, no Qwen.

A candidate playbook replaces the incumbent ONLY IF, against the incumbent, it:
  1. clears the EFFECTIVE-COVERAGE gate - it actually fires (fire_rate >= Cfg.min_fire); a rule that
     does not fire cannot be the reason for a win (plan principle 5), and
  2. is significantly BETTER on the DEV block (hp-margin delta > 0 with its 95% CI excluding 0), and
  3. REPLICATES that on a FRESH held-out block (same test). dev alone is winner's curse (principle 3).

Anything short of all three KEEPS the incumbent. Orthogonally, "fires enough + followed but still no
win" raises the CEILING diagnostic: the limit is the 40-frame executor / move set, not the short
memory (plan principle 1 / KILL). A candidate that never fired is a coverage miss, NOT a ceiling.

The decision is over pre-summarised stats (delta + Welch CI + wins + fire/follows), so this module
never runs a game; scripts/measure_rule.py produces the BlockStat numbers it judges.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class BlockStat:
    """One measured block: candidate-minus-incumbent hp-margin delta with its Welch 95% CI, the
    candidate's wins over n seeds, and the candidate's fire-rate / follows_rule (both 0..1)."""
    delta: float
    lo: float
    hi: float
    cand_wins: int
    n: int
    fire_rate: float
    follows: float


@dataclass(frozen=True)
class Cfg:
    min_fire: float = 0.40        # effective-coverage floor: below this the candidate has no grip
    ceiling_fire: float = 0.50    # CEILING needs the candidate to actually be firing...
    ceiling_follows: float = 0.70  # ...and be followed...
    ceiling_win_frac: float = 0.25  # ...yet (nearly) never win


@dataclass(frozen=True)
class Decision:
    verdict: str   # "promote" | "keep"
    ceiling: bool  # executor/move-set ceiling diagnostic (not the memory)
    reason: str


PROMOTE = "promote"
KEEP = "keep"


def _validate(b):
    if b.n <= 0:
        raise ValueError("BlockStat.n must be > 0 (no seeds = nothing measured)")
    if not (b.lo <= b.delta <= b.hi):
        raise ValueError("CI must bracket delta: lo=%.3f delta=%.3f hi=%.3f" % (b.lo, b.delta, b.hi))
    if not (0.0 <= b.fire_rate <= 1.0 and 0.0 <= b.follows <= 1.0):
        raise ValueError("fire_rate/follows must be in [0,1]")
    if not (0 <= b.cand_wins <= b.n):
        raise ValueError("cand_wins must be in [0, n]")


def _sig_better(b):
    """Candidate beats the incumbent with the CI excluding 0."""
    return b.delta > 0 and b.lo > 0


def _ceiling(b, cfg):
    """Fires enough AND is followed AND (nearly) never wins AND is not beating the incumbent."""
    return (b.fire_rate >= cfg.ceiling_fire and b.follows >= cfg.ceiling_follows
            and b.cand_wins <= int(cfg.ceiling_win_frac * b.n) and b.delta <= 0)


def decide(dev, held, cfg=Cfg()):
    """dev: BlockStat for the dev block. held: BlockStat for a FRESH held-out block, or None.
    Returns a Decision. Promotes only on dev-significant AND held-replicated, above the fire floor."""
    _validate(dev)
    if held is not None:
        _validate(held)

    # 1. effective-coverage gate - a candidate that barely fires cannot earn promotion, whatever its delta
    if dev.fire_rate < cfg.min_fire:
        return Decision(KEEP, _ceiling(dev, cfg),
                        "candidate fires %.0f%% < %.0f%% floor: no grip (coverage miss)"
                        % (100 * dev.fire_rate, 100 * cfg.min_fire))

    # 2. must be significantly better on dev
    if not _sig_better(dev):
        return Decision(KEEP, _ceiling(dev, cfg),
                        "dev not better: delta %+.1f CI[%+.1f,%+.1f]" % (dev.delta, dev.lo, dev.hi))

    # 3. must replicate on a FRESH held-out block (winner's-curse guard)
    if held is None:
        return Decision(KEEP, False,
                        "dev-significant (%+.1f) but no held-out block to confirm (winner's curse)" % dev.delta)
    if not _sig_better(held):
        return Decision(KEEP, _ceiling(held, cfg),
                        "dev-significant (%+.1f) but did NOT replicate held-out: delta %+.1f CI[%+.1f,%+.1f]"
                        % (dev.delta, held.delta, held.lo, held.hi))

    return Decision(PROMOTE, False,
                    "beats incumbent on dev (%+.1f) AND held-out (%+.1f); fires %.0f%% follows %.0f%%"
                    % (dev.delta, held.delta, 100 * dev.fire_rate, 100 * dev.follows))
