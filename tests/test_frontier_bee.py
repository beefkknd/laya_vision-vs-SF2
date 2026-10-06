"""Gap-filling bees (sf2/quorum/frontier.py), owner 2026-10-05 after A showed force-combo bees hurt (65% < 71%):
the quorum should FILL the table's thin/blind cells, not force a category. Two pure voters:
  frontier  -> votes the least-sampled followable action; confidence k/(k+covered_n) fades as the table gains a
               confident positive winner, so it LEADS in blind/thin cells and DEFERS to the thick trunk.
  fireball  -> the same, gated to fireball-up (when ends '|1'), the table's most under-explored slice (A: fb=1 had
               273 samples vs fb=0's 7845). At fb=1 both bees fire -> double the exploration push there.
The table study that motivated this: rollouts/loop_screen/A_20261005_084309 (25 contexts, 7 blind, fb=1 a blind slice)."""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(__file__))

from sf2.quorum.config import QuorumConfig                                 # noqa: E402
from sf2.quorum.frontier import fireball_proposal, frontier_proposal, pressure_proposal   # noqa: E402
from sf2.quorum.tally import Proposal, ranking, score                      # noqa: E402
from sf2.quorum import reliability as R                                    # noqa: E402

WHEN0 = "mid|standing|0"
WHEN1 = "mid|attacking|1"      # a fireball-up context (also opp-attacking)
WHENP = "close|attacking|0"   # opponent attacking, no fireball -> the pressured slice


def acc(n, mean):
    return [n, n * mean, n * mean * mean]


def cfg(**kw):
    return QuorumConfig(mode="vote", **kw)


# -------------------------------------------------------------------- frontier: WHAT it proposes
def test_frontier_proposes_the_least_sampled_followable_action():
    cell = {"s.mk": acc(30, 3.0), "s.mp": acc(5, 1.0)}        # c.* untried (absent)
    p = frontier_proposal(cell, ["s.mk", "s.mp", "lightning_legs"], cfg())
    assert p.voter == "frontier" and p.action == "lightning_legs"   # n=0, the true gap


def test_frontier_ties_break_by_name_for_deterministic_replay():
    p = frontier_proposal({}, ["s.mp", "block_high", "s.mk"], cfg())   # all untried
    assert p.action == "block_high"                                     # min by name


def test_frontier_abstains_when_there_are_no_actions():
    assert frontier_proposal({}, [], cfg()) is None


def test_frontier_off_by_config_is_silent():
    assert frontier_proposal({}, ["s.mp"], cfg(frontier=False)) is None


# -------------------------------------------------------------------- frontier: HOW LOUD (coverage schedule)
def test_frontier_confidence_is_full_in_a_blind_cell_and_fades_as_a_winner_accrues():
    k = 8.0
    blind = frontier_proposal({}, ["s.mp", "s.mk"], cfg(k=k)).confidence
    assert blind == pytest.approx(1.0)                                  # k/(k+0)
    # a confident POSITIVE winner (n=24) gives the table a grip -> frontier quiets to k/(k+24)
    held = frontier_proposal({"s.mp": acc(24, 5.0)}, ["s.mp", "s.mk"], cfg(k=k, table_min_n=3)).confidence
    assert held == pytest.approx(k / (k + 24))
    assert held < blind


def test_frontier_stays_loud_where_the_only_samples_are_negative():
    # a cell full of losing moves is still a GAP (nothing good found) -> keep exploring, confidence 1.0
    cell = {"block_high": acc(50, -3.0)}
    assert frontier_proposal(cell, ["block_high", "s.mk"], cfg()).confidence == pytest.approx(1.0)


# -------------------------------------------------------------------- frontier in the tally: lead vs defer
def test_frontier_leads_in_a_blind_cell_so_the_gap_gets_explored():
    c = cfg()
    cell = {"block_high": acc(2, 0.0)}                                   # laya's move already (barely) tried; still blind
    fr = frontier_proposal(cell, ["block_high", "s.mk", "lightning_legs"], c)
    assert fr.action != "block_high" and fr.confidence == pytest.approx(1.0)   # a genuine gap, voted loudly
    order, _ = ranking(score([Proposal("laya", "block_high", 0.45), fr], cell, WHEN0, R.blank(), c))
    assert order[0] == fr.action                                        # frontier's gap target beats laya's low-conf block


def test_frontier_defers_to_the_thick_trunk_in_a_confident_cell():
    c = cfg(beta=0.5, k=8.0, table_min_n=3)
    cell = {"throw_F+mp": acc(58, 25.0)}                                 # a real thick-trunk winner
    table = Proposal("table", "throw_F+mp", 58 / (58 + c.k))
    props = [Proposal("laya", "block_high", 0.45), table, frontier_proposal(cell, ["throw_F+mp", "block_high", "s.mk"], c)]
    order, _ = ranking(score(props, cell, WHEN0, R.blank(), c))
    assert order[0] == "throw_F+mp"                                     # the table's grip holds; frontier does not overturn


# -------------------------------------------------------------------- fireball bee: gated to fb=1
def test_fireball_bee_fires_only_when_a_fireball_is_out():
    assert fireball_proposal({}, ["s.mk"], WHEN1, cfg()) is not None     # ...|1
    assert fireball_proposal({}, ["s.mk"], WHEN0, cfg()) is None         # ...|0
    assert fireball_proposal({}, ["s.mk"], WHEN1, cfg(fireball=False)) is None


def test_fireball_respects_his_label_split_keys():
    assert fireball_proposal({}, ["s.mk"], "close|standing|1|stand", cfg()) is not None
    assert fireball_proposal({}, ["s.mk"], "close|standing|0|stand", cfg()) is None


def test_both_bees_push_the_same_fb1_gap_target_doubling_the_vote():
    c = cfg()
    fr = frontier_proposal({}, ["s.mk", "lightning_legs"], c)
    fb = fireball_proposal({}, ["s.mk", "lightning_legs"], WHEN1, c)
    assert fr.action == fb.action                                        # same least-sampled target
    s = score([fr, fb], {}, WHEN1, R.blank(), c)
    only_fr = score([fr], {}, WHEN1, R.blank(), c)
    assert s[fr.action]["votes"] == pytest.approx(2 * only_fr[fr.action]["votes"])


# -------------------------------------------------------------------- pressure bee: gated to opp-attacking
def test_pressure_bee_fires_only_when_the_opponent_is_attacking():
    assert pressure_proposal({}, ["s.mk"], WHENP, cfg()) is not None     # close|attacking -> pressured
    assert pressure_proposal({}, ["s.mk"], WHEN1, cfg()) is not None     # mid|attacking|1 -> also attacking
    assert pressure_proposal({}, ["s.mk"], WHEN0, cfg()) is None         # mid|standing -> not pressured
    assert pressure_proposal({}, ["s.mk"], WHENP, cfg(pressure=False)) is None


def test_pressure_respects_his_label_split_keys():
    assert pressure_proposal({}, ["s.mk"], "close|attacking|0|attack", cfg()) is not None
    assert pressure_proposal({}, ["s.mk"], "close|standing|0|stand", cfg()) is None


def test_pressure_bee_boosts_the_least_sampled_gap_loud_where_blind():
    p = pressure_proposal({}, ["block_high", "shoryuken_hp"], WHENP, cfg())
    assert p.voter == "pressure" and p.confidence == pytest.approx(1.0)  # blind pressured cell -> loud


# -------------------------------------------------------------------- config: the new pinned voter set
def test_gapfill_bee_set_is_pinned():
    from sf2.quorum.config import FLAVORS, VOTERS
    assert set(VOTERS) == {"laya", "table", "frontier", "fireball", "pressure",
                           "punish", "vs_crouch", "antiair"}            # posture-slice explore bees added
    assert FLAVORS == {}                                                # no category-forcing flavours by default
    c = QuorumConfig()
    assert c.frontier is True and c.fireball is True and c.pressure is True
    assert c.punish is True and c.vs_crouch is True and c.antiair is True
    assert set(c.priors) >= {"frontier", "fireball", "pressure", "punish", "vs_crouch", "antiair"}
