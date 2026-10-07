"""Purposed counter-bees (frontier.COUNTERS): each pushes ONE specific move in ONE named weakness
context, confidence fading on THAT move's own n, TRAIN-only, off by default. Designed from Zangief's
frozen-play ledger (can't get in vs fireballs; bleeds to jump-ins). These pin: the gate + move, the
own-n fade, default-off, and the eval-stage ban.
"""
from sf2.quorum.config import QuorumConfig
from sf2.quorum.frontier import counter_proposals, COUNTERS


def _eq(n, mean):
    return [n, n * mean, n * mean * mean]


TRAIN = QuorumConfig(counters=True)                 # train stage, counters on
ACTS = ["jump_forward", "spinning_piledriver", "s.mp", "block_high", "double_lariat"]


def test_approach_bee_fires_in_far_attacking_with_jump_forward():
    cell = {"jump_forward": _eq(4, -2.0), "block_high": _eq(100, -7.9)}
    props = counter_proposals(cell, ACTS, "far|attacking|0", TRAIN)
    byname = {p.voter: p for p in props}
    assert "approach" in byname and byname["approach"].action == "jump_forward"
    assert byname["approach"].confidence == TRAIN.k / (TRAIN.k + 4)   # fades on jump_forward's own n


def test_airgrab_and_airpoke_fire_in_their_own_gates_only():
    close_jump = counter_proposals({"spinning_piledriver": _eq(10, 0.0)}, ACTS, "close|jumping|0", TRAIN)
    mid_jump = counter_proposals({"s.mp": _eq(10, 1.9)}, ACTS, "mid|jumping|0", TRAIN)
    assert [p.voter for p in close_jump] == ["airgrab"] and close_jump[0].action == "spinning_piledriver"
    assert [p.voter for p in mid_jump] == ["airpoke"] and mid_jump[0].action == "s.mp"
    # airgrab must NOT fire at mid, airpoke must NOT fire at close
    assert "airgrab" not in {p.voter for p in mid_jump}
    assert "airpoke" not in {p.voter for p in close_jump}


def test_loud_when_untried_fades_as_sampled():
    untried = counter_proposals({}, ACTS, "far|attacking|0", TRAIN)[0]         # n=0 -> k/(k+0)=1.0, loud
    tried = counter_proposals({"jump_forward": _eq(72, -1.0)}, ACTS, "far|attacking|0", TRAIN)[0]
    assert untried.confidence == 1.0 and tried.confidence < 0.2                 # loud to force the first tries, then quiet


def test_off_by_default_and_when_move_not_followable():
    assert counter_proposals({"jump_forward": _eq(4, -2.0)}, ACTS, "far|attacking|0", QuorumConfig()) == []   # default off
    assert counter_proposals({}, ["block_high"], "far|attacking|0", TRAIN) == []     # jump_forward not in actions


def test_counters_are_banned_at_eval_stage():
    # the invariant: an eval config cannot carry counters; and an eval config yields no counter votes.
    import pytest
    with pytest.raises(ValueError):
        QuorumConfig(stage="eval", epsilon=0.0, counters=True,
                     **{b: False for b in ("frontier", "fireball", "pressure", "punish", "vs_crouch", "antiair")})
    ev = QuorumConfig.for_eval(TRAIN)
    assert ev.counters is False
    assert counter_proposals({"jump_forward": _eq(4, -2.0)}, ACTS, "far|attacking|0", ev) == []


def test_counters_table_moves_are_sane():
    assert COUNTERS["approach"][1] == "jump_forward"
    assert COUNTERS["airgrab"][1] == "spinning_piledriver"
    assert COUNTERS["airpoke"][1] == "s.mp"
