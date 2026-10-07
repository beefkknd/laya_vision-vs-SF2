"""The bee admission gate (sf2.quorum.bee_check): a RED/GREEN verdict on a counter-bee from the TABLE
alone, so a mute or losing bee is caught in milliseconds instead of a 90-minute round. Seen-red twins
are the two Zangief bees that wasted R3: far->double_lariat (mute, n=1162) and mid|standing->walk_forward
(loud on a -20 move vs a +27 move). Green cases are the under-sampled class-C bees."""
from sf2.quorum.bee_check import check_bee
from sf2.quorum.config import QuorumConfig

CFG = QuorumConfig()


def _eq(n, mean):
    return [n, n * mean, n * mean * mean]


def test_red_mute_when_move_is_already_thick():
    # far|attacking double_lariat at n=1162 -> bee conf 0.007 -> MUTE (the R3 waste).
    cells = {"far|attacking|0": {"double_lariat": _eq(1162, -4.7), "block_high": _eq(857, -7.7)}}
    v, why = check_bee(cells, ["far", "attacking", "double_lariat"], CFG)
    assert v == "RED" and "MUTE" in why


def test_red_loser_when_loud_move_is_clearly_worse():
    # mid|standing walk_forward is loud (n=4) but -20.5 where SPD is +27 -> LOSER (the other R3 waste).
    cells = {"mid|standing|0|block": {"walk_forward": _eq(4, -20.5), "spinning_piledriver": _eq(30, 27.0)}}
    v, why = check_bee(cells, ["mid", "standing", "walk_forward"], CFG)
    assert v == "RED" and "LOSER" in why


def test_red_when_gate_never_fires():
    v, why = check_bee({"close|standing|0": {"throw_F+hp": _eq(100, 34.0)}}, ["far", "jumping", "crouch"], CFG)
    assert v == "RED" and "never fires" in why


def test_green_under_sampled_positive_move():
    # close|stunned cl.lk +4.4 at n=5 -> loud and positive: a real class-C bee.
    cells = {"close|stunned|0": {"cl.lk": _eq(5, 4.4), "block_high": _eq(60, -4.5)}}
    v, why = check_bee(cells, ["close", "stunned", "cl.lk"], CFG)
    assert v == "GREEN" and "positive" in why


def test_green_under_sampled_least_bad_move():
    # mid|attacking|1 jump_up -3.8 at n=17, less bad than block -6.7 -> GREEN (least-bad).
    cells = {"mid|attacking|1": {"jump_up": _eq(17, -3.8), "block_high": _eq(45, -6.7)}}
    v, why = check_bee(cells, ["mid", "attacking", "jump_up"], CFG)
    assert v == "GREEN"


def test_red_catastrophic_loser_even_without_a_covered_best():
    # the real mid|standing|block cell had NO covered move, but walk_forward -20.5 is catastrophic ->
    # RED by the absolute floor (it slipped through GREEN before this guard).
    cells = {"mid|standing|0|block": {"walk_forward": _eq(4, -20.5), "block_high": _eq(9, -9.1)}}
    v, why = check_bee(cells, ["mid", "standing", "walk_forward"], CFG)
    assert v == "RED" and "LOSER" in why


def test_green_blind_cell_gets_coverage_benefit():
    cells = {"far|standing|1": {"block_high": _eq(6, -10.7)}}       # double_lariat unseen here (blind)
    v, why = check_bee(cells, ["far", "standing", "double_lariat"], CFG)
    assert v == "GREEN" and "blind" in why
