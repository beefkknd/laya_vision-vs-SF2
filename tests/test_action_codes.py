"""The action label (sf2.data.action_codes): per fighter, codes for run states (attack / special / jump: the run's
first attack ID, or its no-box class) and per-row codes for the rest, episodes, and stages."""
import pytest
from ram_rows import row

from sf2.data import action_codes as A
from sf2.emu.vs import GROUND_Y

NONE = {"aid": 0, "mclass": 0xFF, "sclass": 0xFF}


def r(p1=None, p2=None, **extra):
    out = row(dict(NONE, **(p1 or {})), dict(NONE, **(p2 or {})), **extra)
    return out


def still(n, p1=None, p2=None, **extra):
    return [r(p1, p2, **extra) for _ in range(n)]


def lead(n=A.FIRST_T):
    """Standing rows before the rows under test (rows before FIRST_T are never labelled)."""
    return still(n, {"x": 200}, {"x": 300})


def codes(rows, p):
    return [(e.start, e.end, e.code, e.reason, e.cut_end) for e in A.episodes_of(rows, p)]


# ---- run states --------------------------------------------------------------------------------------------------

def test_attack_run_gets_its_first_nonzero_id_over_the_whole_run():
    att = [r(p2={"state": 0x0A, "x": 300, "aid": a}) for a in (0, 0, 5, 5, 6, 0)]
    rows = lead() + att + still(3, {"x": 200}, {"x": 300})
    eps = A.episodes_of(rows, 2)
    hit = [e for e in eps if e.reason == "id"]
    assert len(hit) == 1 and (hit[0].start, hit[0].end, hit[0].code) == (4, 9, 5)


def test_jump_with_an_attack_is_the_attack_whole_jump():
    up = {"state": 0x04, "y": GROUND_Y - 20, "x": 300}
    rows = lead() + [r(p2=dict(up, aid=a)) for a in (0, 0, 0, 25, 25, 0, 0)] + still(2, {"x": 200}, {"x": 300})
    ep = [e for e in A.episodes_of(rows, 2) if e.start == 4][0]
    assert (ep.end, ep.code, ep.reason) == (10, 25, "id")


def test_jump_without_attack_is_jump():
    up = {"state": 0x04, "y": GROUND_Y - 20, "x": 300}
    rows = lead() + [r(p2=up) for _ in range(6)] + still(2, {"x": 200}, {"x": 300})
    ep = [e for e in A.episodes_of(rows, 2) if e.start == 4][0]
    assert (ep.end, ep.code) == (9, A.JUMP)


def test_no_box_throw_projectile_special_cut():
    base = {"state": 0x0A, "x": 300}
    throw = [r(p2=dict(base, mclass=0x22))] + [r(p2=dict(base, mclass=0x06)) for _ in range(5)]
    shot = [r(p2=base, shot2=1 if i == 3 else 0) for i in range(6)]
    spec = [r(p2={"state": 0x0C, "x": 300, "sclass": 0xFF})] + [r(p2={"state": 0x0C, "x": 300, "sclass": 0x09})
                                                                 for _ in range(5)]
    cut = [r(p2=base) for _ in range(5)]
    for seg, want in ((throw, (A.THROW, "throw")), (shot, (A.PROJECTILE, "projectile")),
                      (spec, (A.SPECIAL_BASE + 9, "special")), (cut, (None, "cut"))):
        rows = lead() + seg + still(2, {"x": 200}, {"x": 300})
        ep = [e for e in A.episodes_of(rows, 2) if e.start == 4][0]
        assert (ep.code, ep.reason) == want


def test_the_other_players_shot_slot_is_not_his_projectile():
    rows = lead() + [r(p2={"state": 0x0A, "x": 300}, shot1=1) for _ in range(6)] + still(2, {"x": 200}, {"x": 300})
    ep = [e for e in A.episodes_of(rows, 2) if e.start == 4][0]
    assert ep.code is None and ep.reason == "cut"


def test_throw_class_read_after_the_runs_first_row():
    # the first row's MOVE_CLASS is left over from before: a single throw-class first row must not decide
    rows = lead() + [r(p2={"state": 0x0A, "x": 300, "mclass": m}) for m in (0x06, 0x06, 0x22, 0x22, 0x33)] + still(
        2, {"x": 200}, {"x": 300})
    ep = [e for e in A.episodes_of(rows, 2) if e.start == 4][0]
    assert ep.code is None


def test_id_above_the_max_is_unknown_not_a_reserved_code():
    rows = lead() + [r(p2={"state": 0x0A, "x": 300, "aid": A.ATTACK_ID_MAX + 1}) for _ in range(4)] + still(
        2, {"x": 200}, {"x": 300})
    ep = [e for e in A.episodes_of(rows, 2) if e.start == 4][0]
    assert (ep.code, ep.reason) == (None, "id_range")
    rows = lead() + [r(p2={"state": 0x0A, "x": 300, "aid": A.ATTACK_ID_MAX}) for _ in range(4)] + still(
        2, {"x": 200}, {"x": 300})
    assert [e for e in A.episodes_of(rows, 2) if e.start == 4][0].code == A.ATTACK_ID_MAX


def test_special_class_beyond_the_reserved_range_is_unknown():
    rows = lead() + [r(p2={"state": 0x0C, "x": 300, "sclass": 0x20}) for _ in range(4)] + still(
        2, {"x": 200}, {"x": 300})
    ep = [e for e in A.episodes_of(rows, 2) if e.start == 4][0]
    assert ep.code is None and ep.reason == "special_range"


def test_back_to_back_runs_of_two_states_are_two_episodes():
    rows = lead() + [r(p2={"state": 0x0A, "x": 300, "aid": 7}) for _ in range(4)] + [
        r(p2={"state": 0x0C, "x": 300, "aid": 50}) for _ in range(4)] + still(2, {"x": 200}, {"x": 300})
    got = [(e.start, e.end, e.code) for e in A.episodes_of(rows, 2) if e.code in (7, 50)]
    assert got == [(4, 7, 7), (8, 11, 50)]


# ---- per-row codes -----------------------------------------------------------------------------------------------

@pytest.mark.parametrize("p2,want", [
    ({"state": 0x14}, A.THROWN), ({"state": 0x0E, "react": 0x06}, A.BLOCK), ({"state": 0x0E, "react": 0x08}, A.BLOCK),
    ({"state": 0x0E, "react": 0x0E}, A.HIT), ({"state": 0x08}, A.BLOCK), ({"state": 0x02}, A.CROUCH),
    ({"state": 0x00, "y": GROUND_Y - 5}, A.JUMP), ({"state": 0x0E, "react": 0x0E, "y": GROUND_Y - 30}, A.HIT),
    ({"state": 0x06}, None), ({"state": 0x00}, A.STAND), ({"state": 0x08, "y": GROUND_Y - 20}, A.BLOCK),
    ({"state": 0x14, "y": GROUND_Y - 20}, A.THROWN)])
def test_frame_code(p2, want):
    rows = still(6, {"x": 200}, dict({"x": 300}, **p2))
    assert A.frame_code(rows, 5, 2) == want


def test_walk_toward_and_away_for_each_player():
    # him: x 300 -> 296 with me at 200 = toward; she: x 200 -> 204 with him at 300 = toward
    rows = [r({"x": 200 + i}, {"x": 300 - i}) for i in range(8)]
    assert A.frame_code(rows, 6, 2) == A.WALK_TOWARD and A.frame_code(rows, 6, 1) == A.WALK_TOWARD
    rows = [r({"x": 200 - i}, {"x": 300 + i}) for i in range(8)]
    assert A.frame_code(rows, 6, 2) == A.WALK_AWAY and A.frame_code(rows, 6, 1) == A.WALK_AWAY
    rows = [r({"x": 200}, {"x": 300 + (i % 2)}) for i in range(8)]
    assert A.frame_code(rows, 6, 2) == A.STAND


def test_walk_needs_a_possible_x_and_a_side():
    rows = [r({"x": 200}, {"x": 300 - i}) for i in range(8)]
    rows[6] = r({"x": 65369}, {"x": 294})
    assert A.frame_code(rows, 6, 2) is None
    rows = [r({"x": 290}, {"x": 300 - 2 * i}) for i in range(8)]
    assert A.frame_code(rows, 5, 2) is None          # him at x 290 = my x: no side


def test_rows_before_first_t_never_labelled_and_episodes_split_by_code():
    rows = still(8, {"x": 200}, {"x": 300}) + still(5, {"x": 200}, {"x": 300, "state": 0x02})
    got = codes(rows, 2)
    assert got == [(4, 7, A.STAND, "frame", False), (8, 12, A.CROUCH, "frame", True)]


def test_guard_then_block_react_is_one_block_episode():
    rows = lead() + still(3, {"x": 200}, {"x": 300, "state": 0x08}) + still(
        3, {"x": 200}, {"x": 300, "state": 0x0E, "react": 0x06}) + still(2, {"x": 200}, {"x": 300})
    assert (4, 9, A.BLOCK, "frame", False) in codes(rows, 2)


def test_unknown_rows_break_episodes():
    rows = lead() + still(3, {"x": 200}, {"x": 300, "state": 0x02}) + still(1, {"x": 200}, {"x": 300, "state": 0x06}) \
        + still(3, {"x": 200}, {"x": 300, "state": 0x02})
    got = codes(rows, 2)
    assert [g[2] for g in got] == [A.CROUCH, None, A.CROUCH]


def test_last_episode_flagged_cut_end():
    rows = lead() + still(5, {"x": 200}, {"x": 300, "state": 0x02})
    assert codes(rows, 2)[-1][-1] is True


# ---- stages and labels -------------------------------------------------------------------------------------------

def test_stage_thirds():
    assert [A.stage_of(t, 10, 18) for t in range(10, 19)] == [1, 1, 1, 2, 2, 2, 3, 3, 3]
    assert [A.stage_of(t, 0, 1) for t in (0, 1)] == [1, 2]
    assert A.stage_of(0, 0, 0) == 1
    with pytest.raises(ValueError):
        A.stage_of(19, 10, 18)


def test_label_and_labels_at():
    assert A.label("chunli", 5, 2) == "chunli act05 stg2"
    rows = lead() + still(6, {"x": 200}, {"x": 300, "state": 0x02}) + still(1, {"x": 200}, {"x": 300})
    lab = A.labels_at(rows, 2)
    assert [lab[t][1] for t in range(4, 10)] == [1, 1, 2, 2, 3, 3]
    assert all(t not in lab for t in range(4))


def test_bad_player_and_table_reserved_codes_distinct():
    with pytest.raises(ValueError):
        A.ActorTrack(3)
    reserved = list(A.RESERVED)
    assert len(set(reserved)) == len(reserved) and min(reserved) > A.ATTACK_ID_MAX
    assert A.SPECIAL_BASE > max(reserved) and A.SPECIAL_MAX < 100
    t = A.table()
    assert t["ram"] == {"p1_aid": "0x0C3E", "p1_mclass": "0x0CBA", "p1_sclass": "0x0C49", "p2_aid": "0x0E3E",
                        "p2_mclass": "0x0EBA", "p2_sclass": "0x0E49"}


def test_lessons_table_matches_the_code():
    import json
    import os
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "lessons", "action_codes.json")
    assert json.load(open(path)) == json.loads(json.dumps(A.table()))
