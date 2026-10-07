"""The table voter must put a competitor on the ballot in ALL-NEGATIVE cells.

Root cause (verified in frozen bees-off play, 2026-10-06): where every covered move has mean<=0,
`table_proposal` abstained (its old `mean>0` rule), so laya's `block_high` was the ONLY candidate
and played unopposed at share 1.0 -- Honda over-blocked into jump-ins and pokes. Fix: in a
negative cell, vote the covered argmax IFF it is Welch-`_separated` ABOVE laya's move. Confidence
fades on evidence and on the separation margin, never 1.0 (so it cannot recreate the frontier
stacking bug). Positive-mean cells are UNCHANGED.
"""
from sf2.quorum.config import QuorumConfig
from sf2.quorum.tally import table_proposal, NEG_CONF_CAP


def _eq(n, mean):
    """A Welford accumulator [n, sum, sumsq] for n identical samples == mean (variance 0)."""
    return [n, n * mean, n * mean * mean]


def _spread(n, mean, var):
    """n samples with the given mean and (approx) variance, for the 'not separated' cases."""
    # sumsq = var*(n-1) + sum*sum/n
    s = n * mean
    return [n, s, var * (n - 1) + s * s / n]


CFG = QuorumConfig()          # table_min_n=3, k=8, net_scale=10
ACTS = ["block_high", "cl.hk", "sumo_headbutt"]


def test_abstains_when_all_negative_and_no_laya_move_given():
    # backward-compatible call (old signature): still abstains in an all-negative cell.
    cell = {"block_high": _eq(100, -9.5), "cl.hk": _eq(100, -0.9)}
    assert table_proposal(cell, ACTS, CFG) is None


def test_votes_separated_least_bad_move_over_laya_in_negative_cell():
    # close|jumping shape: laya blocks (-9.5); cl.hk (-0.9) is covered and confidently better.
    cell = {"block_high": _eq(100, -9.5), "cl.hk": _eq(100, -0.9)}
    p = table_proposal(cell, ACTS, CFG, laya_move="block_high")
    assert p is not None and p.voter == "table" and p.action == "cl.hk"
    assert 0.0 < p.confidence < 1.0                       # never the frontier 1.0; faded by n and margin


def test_positive_cell_is_unchanged_by_the_fix():
    # a positive argmax still wins with the old confidence n/(n+k), regardless of laya_move.
    cell = {"block_high": _eq(100, -2.0), "cl.hk": _eq(100, 5.0)}
    base = table_proposal(cell, ACTS, CFG)
    withl = table_proposal(cell, ACTS, CFG, laya_move="block_high")
    assert base is not None and base.action == "cl.hk"
    assert withl == base                                 # identical proposal with or without laya_move
    assert abs(base.confidence - 100 / (100 + CFG.k)) < 1e-9


def test_abstains_when_least_bad_is_not_separated_from_laya():
    # cl.hk barely edges block but with wide variance + few samples -> not Welch-separated -> no noise vote.
    cell = {"block_high": _spread(100, -5.0, 1.0), "cl.hk": _spread(4, -4.8, 400.0)}
    assert table_proposal(cell, ACTS, CFG, laya_move="block_high") is None


def test_abstains_when_laya_already_picks_the_covered_argmax():
    cell = {"block_high": _eq(100, -9.5), "cl.hk": _eq(100, -0.9)}
    assert table_proposal(cell, ACTS, CFG, laya_move="cl.hk") is None


def test_abstains_when_layas_move_is_not_covered():
    # laya picks a move the table has barely seen -> cannot establish the table is better -> abstain.
    cell = {"cl.hk": _eq(100, -0.9), "block_high": _eq(100, -9.5), "sumo_headbutt": _eq(1, -0.1)}
    assert table_proposal(cell, ACTS, CFG, laya_move="sumo_headbutt") is None


def test_prefers_well_sampled_move_over_a_tiny_n_outlier():
    # Zangief close|standing: throw_F+hp (n=871, +34.8) is the real grab; throw_F+hk (n=3, +36.7) is
    # 3-sample noise. Raw-mean argmax picked the noise and voted it weakly (conf 3/11=0.27) so block
    # won; shrunk mean (mean * n/(n+k)) picks the well-sampled grab and votes it ~0.99.
    cell = {"throw_F+hp": _eq(871, 34.8), "throw_F+hk": _eq(3, 36.7), "block_high": _eq(400, -3.2)}
    p = table_proposal(cell, ["throw_F+hp", "throw_F+hk", "block_high"], CFG, laya_move="block_high")
    assert p is not None and p.action == "throw_F+hp"
    assert p.confidence > 0.9                            # strong: selected by evidence, not by a 3-sample fluke


def test_separated_negative_move_vote_fades_on_the_margin():
    # The conservative margin stays (dropping it cratered honda 92->18.8% cross-character). A move
    # separated by only ~3 HP votes weakly (n/(n+k) * 3/net_scale), deferring to laya -- that caution
    # is what keeps honda's misleading one-step "least-bad" moves from overriding her block.
    cell = {"double_lariat": _eq(1162, -4.7), "block_high": _eq(857, -7.7)}
    p = table_proposal(cell, ["double_lariat", "block_high"], CFG, laya_move="block_high")
    assert p is not None and p.action == "double_lariat"
    assert abs(p.confidence - (1162 / 1170) * 0.3) < 0.02      # ~0.30, faded by the 3-HP margin


def test_laya_base_prior_default_is_one():
    # the laya down-weight is NOT global (it cratered honda 92->18.8%): default stays 1.0, scoped per-char.
    from sf2.quorum.config import QuorumConfig as QC
    assert QC().priors["laya"] == 1.0 and QC().priors["table"] == 1.0


def test_negative_vote_is_capped_below_one_even_at_huge_n_and_margin():
    # invariant (blind-review finding): a net-negative "least-bad" vote must never approach 1.0 at
    # rest. Uncapped, n/(n+k)*margin here would be ~0.999; the cap holds it to NEG_CONF_CAP.
    cell = {"block_high": _eq(10000, -15.0), "cl.hk": _eq(10000, -1.0)}   # gap 14 -> margin clamps to 1
    p = table_proposal(cell, ACTS, CFG, laya_move="block_high")
    assert p is not None and p.action == "cl.hk"
    assert p.confidence <= NEG_CONF_CAP < 1.0


def test_seeded_red_poisoned_argmax_is_not_proposed():
    # Fable's seeded fault: if the 'answer' move is actually terrible (-20), it must NOT be proposed.
    cell = {"block_high": _eq(100, -9.5), "cl.hk": _eq(100, -20.0)}
    # cl.hk is now worse than laya's block -> block is the argmax == laya_move -> abstain.
    assert table_proposal(cell, ACTS, CFG, laya_move="block_high") is None
