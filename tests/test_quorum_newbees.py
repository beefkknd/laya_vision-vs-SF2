"""Three new posture-slice explore bees (sf2/quorum/frontier.py), each the frontier push gated to one value of the
'when' key's posture field (field 1, the OPPONENT's posture/action):
  punish    -> opponent STUNNED  (field 1 == 'stunned')  -> punish window
  vs_crouch -> opponent CROUCHING (field 1 == 'crouching')
  antiair   -> opponent JUMPING  (field 1 == 'jumping')  -> anti-air
Each behaves EXACTLY like pressure_proposal: fires the least-sampled followable action ONLY in its slice, abstains
otherwise, and abstains when its own cfg flag is False. Mirrors tests/test_frontier_bee.py."""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(__file__))

from sf2.quorum.config import QuorumConfig                                 # noqa: E402
from sf2.quorum.frontier import antiair_proposal, punish_proposal, vs_crouch_proposal   # noqa: E402

# (bee function, its posture value, matching when key, a non-matching when key)
BEES = [
    (punish_proposal,    "stunned",   "mid|stunned|0",   "mid|standing|0"),
    (vs_crouch_proposal, "crouching", "close|crouching|0", "close|standing|0"),
    (antiair_proposal,   "jumping",   "far|jumping|1",   "far|standing|1"),
]
FLAGS = {punish_proposal: "punish", vs_crouch_proposal: "vs_crouch", antiair_proposal: "antiair"}
NAMES = {punish_proposal: "punish", vs_crouch_proposal: "vs_crouch", antiair_proposal: "antiair"}


def acc(n, mean):
    return [n, n * mean, n * mean * mean]


def cfg(**kw):
    return QuorumConfig(mode="vote", **kw)


@pytest.mark.parametrize("fn,value,match,nomatch", BEES)
def test_bee_fires_only_in_its_posture_slice(fn, value, match, nomatch):
    assert fn({}, ["s.mk"], match, cfg()) is not None                    # posture matches -> fires
    assert fn({}, ["s.mk"], nomatch, cfg()) is None                      # posture differs -> abstains


@pytest.mark.parametrize("fn,value,match,nomatch", BEES)
def test_bee_abstains_when_its_flag_is_off(fn, value, match, nomatch):
    assert fn({}, ["s.mk"], match, cfg(**{FLAGS[fn]: False})) is None


@pytest.mark.parametrize("fn,value,match,nomatch", BEES)
def test_bee_proposes_the_least_sampled_followable_action_with_its_own_name(fn, value, match, nomatch):
    cell = {"s.mk": acc(30, 3.0), "s.mp": acc(5, 1.0)}                   # lightning_legs untried (the gap)
    p = fn(cell, ["s.mk", "s.mp", "lightning_legs"], match, cfg())
    assert p.voter == NAMES[fn] and p.action == "lightning_legs"


@pytest.mark.parametrize("fn,value,match,nomatch", BEES)
def test_bee_is_loud_where_blind(fn, value, match, nomatch):
    p = fn({}, ["block_high", "shoryuken_hp"], match, cfg())             # blind cell in its slice
    assert p.confidence == pytest.approx(1.0)


@pytest.mark.parametrize("fn,value,match,nomatch", BEES)
def test_bee_respects_his_label_split_keys(fn, value, match, nomatch):
    assert fn({}, ["s.mk"], match + "|stand", cfg()) is not None         # 4-field key, posture still matches
    assert fn({}, ["s.mk"], "close|standing|0|stand", cfg()) is None


@pytest.mark.parametrize("fn,value,match,nomatch", BEES)
def test_bee_abstains_with_no_actions(fn, value, match, nomatch):
    assert fn({}, [], match, cfg()) is None


def test_bees_target_distinct_posture_values():
    assert {b[1] for b in BEES} == {"stunned", "crouching", "jumping"}   # no overlap with pressure's 'attacking'
