"""Measurer stat-assembly, test-first and PURE (run_arm injected -> no games here).

measure_block runs both arms over the seeds via the injected run_arm(rules, seed) -> (hp, win,
decisions), then assembles a BlockStat: Welch 95% CI on candidate-minus-incumbent hp margin, candidate
wins, and the candidate's pooled fire_rate/follows via coverage(). This is the seam the pure keep/drop
core (sf2.system2.promotion.decide) consumes; the game-shelling run_arm lives in scripts/outcome_loop.py.
"""
from sf2.system2.measurer import welch_delta, measure_block


def _dec(fire, follows):
    return {"prompt_lines": (["r"] if fire else []), "follows_rule": follows}


def test_welch_delta_sign_and_bracket():
    d, lo, hi = welch_delta([0, 0, 0, 0], [10, 10, 10, 10])  # candidate clearly higher, no variance
    assert d == 10.0 and lo <= d <= hi


def test_welch_delta_spans_zero_when_noisy():
    d, lo, hi = welch_delta([0, 20, -20, 5], [3, -15, 25, -2])
    assert lo < 0 < hi  # overlapping noise -> CI straddles 0


def test_measure_block_assembles_blockstat():
    # incumbent loses (hp -10 each), candidate wins (hp +20 each, wins all), fires 2/3, follows 1/2 of fired
    def run_arm(rules, seed):
        if rules == ("inc",):
            return (-10.0, 0, [_dec(False, False)])
        return (20.0, 1, [_dec(True, True), _dec(True, False), _dec(False, False)])

    bs = measure_block(("inc",), ("cand",), [0, 1, 2, 3], run_arm)
    assert bs.n == 4
    assert bs.cand_wins == 4
    assert bs.delta == 30.0          # 20 - (-10)
    assert bs.lo == bs.hi == 30.0    # zero variance both arms
    assert bs.fire_rate == 2 / 3     # pooled candidate decisions: 2 of 3 fired
    assert bs.follows == 0.5         # of the 2 fired, 1 followed


def test_measure_block_runs_every_seed_for_both_arms():
    seen = []

    def run_arm(rules, seed):
        seen.append((rules, seed))
        return (1.0, 1, [_dec(True, True)])

    measure_block(("inc",), ("cand",), [7, 8], run_arm)
    assert sorted(seen) == sorted([(("inc",), 7), (("inc",), 8), (("cand",), 7), (("cand",), 8)])


def test_measure_block_rejects_empty_seeds():
    import pytest
    with pytest.raises(ValueError):
        measure_block(("inc",), ("cand",), [], lambda r, s: (0.0, 0, []))
