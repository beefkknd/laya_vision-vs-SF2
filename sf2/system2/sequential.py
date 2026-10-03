"""Option D: screen -> powered sequential confirm, with the WIN-based stop. PURE (no games/network).

Resolves the metric tension the two reviewers (Fable + gpt6) raised and folds in the owner's objective:
WINNING is the goal, not HP-margin maximization. While still climbing, HP-margin is the LEARNING signal
(continuous, high-information); win-rate at n~=12 is too coarse to learn from but IS the goal / stop
signal. Once the incumbent reliably beats the opponent it is SOLVED -> freeze the lesson set (chasing
pure knockout margin would overfit the CPU's fixed patterns - "the game shrinks").

Design (both reviewers converged):
  SCREEN (dev block, 12 seeds): dev is selection-biased, so its significance is NOT evidence - it only
    screens. Pass iff fires >= min_fire AND the one-sided lower bound (delta - z_screen*se) > 0.
  CONFIRM (FRESH seeds, pooled across up to max_looks blocks, one-sided): after each block, on the
    POOLED confirm data only (never the dev data), z = delta/se. PROMOTE if z >= confirm_z AND a win
    veto passes (pooled cand wins >= pooled inc wins - win_tol). FUTILITY-REJECT if the pooled gain is
    too small (pooled delta < futility_delta, or the one-sided upper bound < futility_upper). Otherwise
    CONTINUE with another fresh block; after max_looks without promotion, REJECT (directional-unconfirmed).
  SOLVED: incumbent round win-rate >= win_target -> opponent beaten, freeze.

Metric: HP margin gates (continuous, ~4x the info of a 12-game binomial); win-rate is a VETO only, never
a promoter. Unit of analysis is the SEED (one game-mean per seed), so samples are independent.
"""
import math
import statistics as st
from dataclasses import dataclass
from typing import List, Sequence, Tuple


@dataclass(frozen=True)
class Block:
    """One measured block, per-seed arrays so blocks can be POOLED for the sequential confirm.
    inc_hp/cand_hp: per-seed hp margin; inc_wins/cand_wins: rounds won; fire_rate/follows: candidate's."""
    inc_hp: Tuple[float, ...]
    cand_hp: Tuple[float, ...]
    inc_wins: int
    cand_wins: int
    fire_rate: float
    follows: float


@dataclass(frozen=True)
class SeqCfg:
    min_fire: float = 0.40
    z_screen: float = 1.645    # one-sided 95% for the dev screen
    confirm_z: float = 2.0     # ~Pocock one-sided over up to max_looks
    max_looks: int = 3
    futility_delta: float = 10.0   # pooled mean gain below this -> not worth confirming
    futility_upper: float = 15.0   # one-sided upper bound below this -> reject
    win_tol: int = 2           # win veto: pooled cand wins >= pooled inc wins - win_tol
    win_target: float = 0.60   # SOLVED: incumbent round win-rate >= this -> freeze


@dataclass(frozen=True)
class SeqDecision:
    verdict: str   # "promote" | "continue" | "reject"
    z: float
    delta: float
    reason: str


def _welch(inc: Sequence[float], cand: Sequence[float]) -> Tuple[float, float]:
    """(delta = mean(cand)-mean(inc), standard error). Needs >= 2 samples per arm for a variance."""
    na, nb = len(inc), len(cand)
    if na < 2 or nb < 2:
        raise ValueError("need >= 2 samples per arm for a variance (got %d, %d)" % (na, nb))
    se = math.sqrt(st.variance(inc) / na + st.variance(cand) / nb)
    return st.mean(cand) - st.mean(inc), se


def screen(dev: Block, cfg: SeqCfg = SeqCfg()) -> Tuple[bool, str]:
    """Cheap dev screen: fires enough AND a one-sided lower bound above 0. Dev significance is NOT taken
    as proof (selection bias) - this only decides whether the candidate is worth a fresh confirm."""
    if dev.fire_rate < cfg.min_fire:
        return False, "fires %.0f%% < %.0f%% floor" % (100 * dev.fire_rate, 100 * cfg.min_fire)
    delta, se = _welch(dev.inc_hp, dev.cand_hp)
    lower = delta - cfg.z_screen * se
    if lower <= 0:
        return False, "one-sided lower bound %+.1f <= 0 (delta %+.1f se %.1f)" % (lower, delta, se)
    return True, "screened: delta %+.1f lower %+.1f fire %.0f%%" % (delta, lower, 100 * dev.fire_rate)


def _pool(blocks: Sequence[Block]):
    inc = [h for b in blocks for h in b.inc_hp]
    cand = [h for b in blocks for h in b.cand_hp]
    iw = sum(b.inc_wins for b in blocks)
    cw = sum(b.cand_wins for b in blocks)
    return inc, cand, iw, cw


def confirm(blocks: List[Block], cfg: SeqCfg = SeqCfg()) -> SeqDecision:
    """Sequential confirm on FRESH blocks, pooled. Call with 1 block, then 2, ... as blocks arrive.
    Returns promote / continue / reject. 'continue' means run another fresh block (until max_looks)."""
    if not blocks:
        raise ValueError("need at least one confirm block")
    inc, cand, iw, cw = _pool(blocks)
    delta, se = _welch(inc, cand)
    z = delta / se if se > 0 else (math.inf if delta > 0 else -math.inf)
    upper = delta + cfg.z_screen * se
    win_ok = cw >= iw - cfg.win_tol

    if delta < cfg.futility_delta or upper < cfg.futility_upper:
        return SeqDecision("reject", z, delta, "futility: pooled delta %+.1f (upper %+.1f) too small" % (delta, upper))
    if z >= cfg.confirm_z and win_ok:
        return SeqDecision("promote", z, delta, "confirmed: pooled z %.2f delta %+.1f wins %d>=%d-%d"
                           % (z, delta, cw, iw, cfg.win_tol))
    if z >= cfg.confirm_z and not win_ok:
        return SeqDecision("reject", z, delta, "margin gain but win veto: cand wins %d < inc %d - %d" % (cw, iw, cfg.win_tol))
    if len(blocks) >= cfg.max_looks:
        return SeqDecision("reject", z, delta, "directional-unconfirmed after %d looks (z %.2f delta %+.1f)"
                           % (len(blocks), z, delta))
    return SeqDecision("continue", z, delta, "underpowered (z %.2f delta %+.1f): run another confirm block" % (z, delta))


def solved(inc_wins: int, n: int, cfg: SeqCfg = SeqCfg()) -> bool:
    """The opponent is BEATEN - the incumbent's round win-rate meets the target - so freeze the lesson
    set (stop proposing/promoting) until the opponent ranks up. The owner's win objective, not margin."""
    if n <= 0:
        raise ValueError("n must be > 0")
    return inc_wins / n >= cfg.win_target
