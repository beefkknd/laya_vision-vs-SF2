"""Divergence study (owner 2026-10-05): when a character's table is SEEDED from a related one (ryu -> ken) and then
grown, the merge (shared=seed) means the grown table's per-cell counts = seed counts + the new character's OWN samples.
So we can recover what the new character actually learned ON TOP of the prior: ken_own_n = grown.n - prior.n,
ken_own_mean = (grown.sum - prior.sum)/ken_own_n. The DELTA (ken_own_mean - prior_mean), weighted by ken_own_n, is
"what is truly different for ken" -- the thing a transferred table otherwise hides. Pure arithmetic over two tables."""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(__file__))

from sf2.quorum.divergence import divergence                             # noqa: E402


def acc(n, mean):
    return [n, n * mean, n * mean * mean]


def test_recovers_the_new_characters_own_samples_and_delta():
    prior = {"mid|standing|0": {"hadoken_lp": acc(10, 2.0)}}              # ryu liked hadoken here (+2, n=10)
    grown = {"mid|standing|0": {"hadoken_lp": acc(15, 0.0)}}             # grown = ryu's 10 + ken's 5 ; grown mean 0
    d = divergence(prior, grown, min_own_n=3)
    row = next(r for r in d if r["move"] == "hadoken_lp")
    assert row["own_n"] == 5                                              # 15 - 10
    assert row["own_mean"] == pytest.approx(-4.0)                         # (0 - 20)/5
    assert row["prior_mean"] == pytest.approx(2.0)
    assert row["delta"] == pytest.approx(-6.0)                            # ken dislikes what ryu liked here


def test_skips_cells_the_new_character_barely_touched():
    prior = {"w": {"m": acc(20, 3.0)}}
    grown = {"w": {"m": acc(21, 3.0)}}                                    # only 1 ken sample -> below min_own_n
    assert divergence(prior, grown, min_own_n=3) == []


def test_flags_cells_ken_explored_that_ryu_never_had_as_new():
    prior = {"w": {}}
    grown = {"w": {"shoryuken_hp": acc(8, 5.0)}}                          # ryu had no data; ken's own
    d = divergence(prior, grown, min_own_n=3)
    row = next(r for r in d if r["move"] == "shoryuken_hp")
    assert row["own_n"] == 8 and row["prior_mean"] is None and row["kind"] == "ken-new"


def test_ranked_by_evidence_weighted_absolute_delta():
    prior = {"w": {"a": acc(10, 0.0), "b": acc(10, 0.0)}}
    grown = {"w": {"a": acc(14, -7.0 * 14 / 4 / 1), "b": acc(60, 0.0)}}   # a: big delta few samples; b: no delta many
    # construct a clean: a own_n=4 own_mean large negative; b own_n=50 own_mean ~0
    prior = {"w": {"a": acc(10, 0.0), "b": acc(10, 0.0)}}
    grown = {"w": {"a": [14, -40.0, 400.0], "b": [60, 0.0, 0.0]}}         # a: own 4 samples sum -40 -> mean -10 ; b: own 50 mean 0
    d = divergence(prior, grown, min_own_n=3)
    assert d[0]["move"] == "a"                                            # evidence-weighted |delta| ranks 'a' first
