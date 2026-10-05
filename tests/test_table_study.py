"""The trunk-study -> bee-suggestion process, formalized (owner 2026-10-05: "a good lesson/process to formulate in a
way to quickly call to analyze"). sf2/quorum/table_study.py reads ONE value table's cells and reports what the table
knows (the thick trunk), where it is blind/thin, and the fireball-slice coverage -- then suggests which bees to turn
on to fill the gaps. Pure: table stats only, no models, no I/O. This is the exact analysis run by hand on table A
(rollouts/loop_screen/A_20261005_084309: 25 contexts, 7 blind, fb=1 a near-empty slice)."""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(__file__))

from sf2.quorum import bees                                              # noqa: E402
from sf2.quorum.config import VOTERS                                     # noqa: E402
from sf2.quorum.table_study import study, suggest_bees                   # noqa: E402


def acc(n, mean):
    return [n, n * mean, n * mean * mean]


def table(cells):
    return {"cells": cells, "shadow": {}, "depth": {}}


# ---------------------------------------------------------------- the roster lines up with the config
def test_bee_roster_matches_the_configured_voters():
    assert set(bees.BEES) == set(VOTERS)                                # one source of truth, no drift
    assert {b["role"] for b in bees.BEES.values()} <= set(bees.ROLES)
    assert bees.BEES["laya"]["role"] == "base" and bees.BEES["table"]["role"] == "exploit"
    assert bees.BEES["frontier"]["role"] == "explore" and bees.BEES["fireball"]["role"] == "explore"


# ---------------------------------------------------------------- study: trunk / blind / coverage
def test_study_finds_the_thick_trunk_winner_per_context():
    cells = {"close|standing|0": {"throw_F+mp": acc(58, 25.0), "block_high": acc(40, -2.0)}}
    a = study(cells, confident_n=20)
    trunk = dict((w, (m, mv)) for w, mv, m, n in a["trunk"])
    assert trunk["close|standing|0"] == (pytest.approx(25.0), "throw_F+mp")   # best confident POSITIVE move


def test_study_flags_blind_contexts_with_no_confident_move():
    cells = {"mid|attacking|1": {"block_high": acc(9, -1.0), "s.mk": acc(4, 2.0)},     # all < confident_n -> blind
             "mid|standing|0": {"s.mp": acc(30, 3.0)}}                                 # confident -> not blind
    a = study(cells, confident_n=20)
    assert [w for w, _ in a["blind"]] == ["mid|attacking|1"]


def test_study_measures_the_fireball_slice_coverage():
    cells = {"mid|standing|0": {"s.mp": acc(400, 3.0)}, "mid|attacking|1": {"block_high": acc(20, -1.0)}}
    fb = study(cells, confident_n=20)["fb"]
    assert fb["fb1_n"] == 20 and fb["fb0_n"] == 400
    assert fb["fb1_share"] == pytest.approx(20 / 420)


def test_study_lists_promising_but_thin_cells():
    cells = {"mid|stunned|0": {"jf.hk_s.mp_s.hp": acc(11, 5.8), "block_high": acc(50, -0.8)}}
    prom = study(cells, confident_n=20, min_n=4)["promising"]
    assert ("mid|stunned|0", "jf.hk_s.mp_s.hp") in [(w, mv) for w, mv, m, n in prom]   # positive mean, under-sampled


# ---------------------------------------------------------------- suggest_bees: the gap -> bee mapping
def test_suggests_fireball_bee_when_the_fb1_slice_is_under_explored():
    cells = {"mid|standing|0": {"s.mp": acc(400, 3.0)}, "mid|attacking|1": {"block_high": acc(10, -1.0)}}
    s = suggest_bees(study(cells, confident_n=20), fb_share_floor=0.15)
    assert s["fireball"] is True and any("fireball" in n.lower() for n in s["notes"])


def test_does_not_suggest_fireball_when_the_slice_is_well_covered():
    cells = {"mid|standing|0": {"s.mp": acc(200, 3.0)}, "mid|standing|1": {"s.mp": acc(200, 3.0)}}
    s = suggest_bees(study(cells, confident_n=20), fb_share_floor=0.15)
    assert s["fireball"] is False


def test_suggests_frontier_bee_whenever_thin_or_blind_cells_exist():
    cells = {"mid|standing|0": {"s.mp": acc(5, 3.0)}}                    # one thin cell
    s = suggest_bees(study(cells, confident_n=20))
    assert s["frontier"] is True


def test_retires_category_forcing_bees_by_default():
    s = suggest_bees(study({"mid|standing|0": {"s.mp": acc(5, 3.0)}}, confident_n=20))
    assert s["flavors"] == {}                                           # the gap-fill process never re-forces a category
