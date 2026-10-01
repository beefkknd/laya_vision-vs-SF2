"""The movement label (docs/prereg_movement.md; sf2.data.movement): his state at the displayed frame (lag 1), one of
eight general answers, "unknown" where the rows do not decide it. Rows are real-shaped (tests/ram_rows.row: every
field play_system1 --ram-log stores); a real-data invariant runs when the U collection is on disk."""
import os

import pytest
from ram_rows import row

from sf2.data import movement as M
from sf2.data.perception import ATTACK, LAG, decode, phase
from sf2.emu.vs import GROUND_Y

ME_X = 200          # me (player 1) on the left, him to my right


def rows_with(t_row, prev_row=None, n=10, t=None):
    """n + 1 neutral rows (him standing still at x 260) with row t replaced (and row t - 4, if given)."""
    rows = [row({"x": ME_X}, {"x": 260}) for _ in range(n + 1)]
    t = n - LAG if t is None else t
    rows[t] = t_row
    if prev_row is not None:
        rows[t - 4] = prev_row
    return rows, n


def his(**p2):
    return row({"x": ME_X}, dict({"x": 260}, **p2))


def label(rows, n):
    return M.movement_at(rows, n)


# ---- the answers and the question ------------------------------------------------------------------------------------

def test_eight_answers_in_the_prereg_order():
    assert M.ANSWERS == ("standing", "walking toward me", "walking away", "crouching", "jumping", "attacking",
                         "being hit", "blocking")


def test_the_question_is_the_prereg_wording_general_and_short():
    q = M.movement_question()
    assert q["type"] == "choice" and q["instructions"] == "What is he doing right now?"
    assert tuple(q["criteria"]) == M.ANSWERS
    assert len(set(q["criteria"].values())) == 8 and all(len(v) <= 45 for v in q["criteria"].values())
    from sf2.vocab import FIGHTERS
    text = repr(q).lower()
    assert not [c for c in FIGHTERS if c in text]                       # no opponent names
    assert M.movement_question() == M.movement_question()               # byte-identical at train and eval time


# ---- states ----------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("p2,want", [
    ({"state": 0x02}, "crouching"),
    ({"state": 0x04}, "jumping"),                                   # jump state, still on the ground (take-off)
    ({"state": 0x04, "y": GROUND_Y - 40}, "jumping"),
    ({"state": 0x00, "y": GROUND_Y - 3}, "jumping"),                # y above ground
    ({"state": 0x02, "y": GROUND_Y - 3}, "jumping"),
    ({"state": 0x08}, "blocking"),                                  # guard
    ({"state": 0x0A}, "attacking"),
    ({"state": 0x0C}, "attacking"),
    ({"state": 0x0A, "y": GROUND_Y - 30}, "attacking"),             # a jump attack is attacking
    ({"state": 0x0E, "react": 0x0E}, "being hit"),
    ({"state": 0x0E, "react": 0x00}, "being hit"),
    ({"state": 0x0E, "react": 0x14}, "being hit"),
    ({"state": 0x0E, "react": 0x0E, "y": GROUND_Y - 20}, "being hit"),   # knocked into the air
    ({"state": 0x14}, "being hit"),                                 # thrown
    ({"state": 0x14, "y": GROUND_Y - 20}, "being hit"),
    ({"state": 0x0E, "react": 0x06}, "blocking"),                   # hit stun with a block react
    ({"state": 0x0E, "react": 0x08}, "blocking"),
])
def test_state_at_the_displayed_frame(p2, want):
    rows, n = rows_with(his(**p2))
    assert label(rows, n) == want


def test_a_state_the_prereg_does_not_name_is_unknown():
    rows, n = rows_with(his(state=0x06))                            # turning round, on the ground
    assert label(rows, n) == M.UNKNOWN


def test_lag_1_reads_the_row_before_the_decision_row():
    rows, n = rows_with(his(state=0x0A))
    rows[n] = his(state=0x02)                                        # the decision row itself: not displayed yet
    assert label(rows, n) == "attacking"
    assert M.movement(rows, n) == "crouching"                        # movement() takes the displayed row t itself


# ---- standing vs walking (x between the two displayed frames, t - 4 -> t) ---------------------------------------------

@pytest.mark.parametrize("prev_x,now_x,want", [
    (260, 260, "standing"), (261, 260, "standing"), (259, 260, "standing"),
    (262, 260, "walking toward me"), (267, 260, "walking toward me"),     # x falls toward me (I am at 200)
    (258, 260, "walking away"), (250, 260, "walking away"),
])
def test_walking_needs_2_px_between_the_displayed_frames(prev_x, now_x, want):
    rows, n = rows_with(his(x=now_x), his(x=prev_x))
    assert label(rows, n) == want


def test_toward_and_away_follow_which_side_i_am_on():
    me_right = lambda x: row({"x": 400}, {"x": x})
    rows, n = rows_with(me_right(300), me_right(290))
    assert label(rows, n) == "walking toward me"
    rows, n = rows_with(me_right(300), me_right(310))
    assert label(rows, n) == "walking away"


def test_only_the_two_displayed_frames_count_not_the_rows_between():
    rows, n = rows_with(his(x=260), his(x=260))
    t = n - LAG
    rows[t - 2] = his(x=280)                                        # a wobble between the frames: not seen
    assert label(rows, n) == "standing"


def test_movement_of_a_non_standing_state_does_not_make_it_walking():
    rows, n = rows_with(his(state=0x02, x=260), his(state=0x02, x=270))
    assert label(rows, n) == "crouching"


# ---- unknown: missing rows, impossible x ---------------------------------------------------------------------------------

def test_missing_displayed_row_is_unknown():
    rows = [his() for _ in range(3)]
    assert label(rows, 0) == M.UNKNOWN                              # t = -1
    assert M.movement(rows, 5) == M.UNKNOWN


def test_standing_without_the_earlier_frame_is_unknown_but_other_states_stand():
    rows = [his() for _ in range(5)]
    assert label(rows, 3) == M.UNKNOWN                              # t = 2, t - 4 = -2
    rows[2] = his(state=0x0A)
    assert label(rows, 3) == "attacking"


@pytest.mark.parametrize("now,prev", [({"x": 65369}, {"x": 260}), ({"x": 260}, {"x": 65369})])
def test_an_impossible_his_x_makes_standing_unknown(now, prev):
    rows, n = rows_with(his(**now), his(**prev))
    assert label(rows, n) == M.UNKNOWN


def test_an_impossible_my_x_makes_standing_unknown():
    rows, n = rows_with(row({"x": 65369}, {"x": 260}), his(x=260))
    assert label(rows, n) == M.UNKNOWN


def test_an_impossible_x_does_not_touch_a_state_that_reads_no_x():
    rows, n = rows_with(his(state=0x0A, x=65369))
    assert label(rows, n) == "attacking"


def test_walking_on_my_x_has_no_direction_unknown():
    rows, n = rows_with(row({"x": 260}, {"x": 260}), row({"x": 260}, {"x": 250}))
    assert label(rows, n) == M.UNKNOWN


# ---- real data: agrees with perception.phase where the rules are the same ----------------------------------------------

REAL = os.path.join("rollouts", "u_perception", "p_vs_ryu", "ken", "ram.jsonl")


@pytest.mark.skipif(not os.path.exists(REAL), reason="the U collection is not on disk")
def test_real_rows_agree_with_phase_on_hit_block_and_attack():
    import json
    seen = set()
    with open(REAL) as f:
        for i, line in enumerate(f):
            if i >= 400:
                break
            rows, n = decode(json.loads(line))
            t = n - LAG
            got = M.movement_at(rows, n)
            ph = phase(rows, t, {})
            seen.add(got)
            assert got in M.ANSWERS + (M.UNKNOWN,)
            if ph in ("being hit", "blocking"):
                assert got == ph
            assert (got == "attacking") == (rows[t]["p2_state"] in ATTACK)
    assert {"standing", "attacking", "jumping"} <= seen
