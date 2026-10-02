"""docs/eye_questions_v1.md, "Datasets v1": the eye's one frame pool (sf2.data.eye_pool). Every image pair on disk,
labelled from RAM at the displayed row t (lag 1): q1 fireball yes / no (a fireball flight's slot on AND drawn at t;
blink-hidden, before spawn, after impact and yoga flame are "no"), the q1 hard-negative tags, q3 act, q4 air (incl.
take-off / landing), q5 close / far by the poke band of all characters."""
import pytest

from sf2.data import eye_pool as P
from sf2.emu.vs import GROUND_Y
from test_pairs_labels import prow


def rows_with(n=60, flights=(), hide=(), hide_bytes=True, p1=None, p2=None):
    """n rows; flights (slot, start, end) turn the slot on; rows in ``hide`` have the blink bit of every on slot."""
    out = []
    for k in range(n):
        r = prow(dict({"x": 200}, **(p1(k) if p1 else {})), dict({"x": 300}, **(p2(k) if p2 else {})))
        if hide_bytes:
            r.update(shot1_hide=0, shot2_hide=0)
        for s, a, e in flights:
            if a <= k <= e:
                r["shot%d" % s], r["shot%d_x" % s] = 1, 220 + k - a
                if k in hide and hide_bytes:
                    r["shot%d_hide" % s] = 1
        out.append(r)
    return out


def words_with(n, presses):
    """presses: (slot, word, first row, last row) -> PM.pressed_words-shaped {slot: [word per row]}."""
    w = {1: [None] * n, 2: [None] * n}
    for s, word, a, e in presses:
        for u in range(a, e + 1):
            w[s][u] = word
    return w


CHARS = {1: "ryu", 2: "ken"}


def test_disk_pairs_need_both_images_four_captures_apart():
    names = ["g0003_k00010.png", "g0003_k00014.png", "g0003_k00015.png", "g0004_k00020.png", "g0004_k00024.png",
             "junk.txt", "g0005_k00030.png"]
    assert P.disk_pairs(names) == {3: [13], 4: [23], 5: []}


def test_flight_kinds_from_the_word_pressed_at_the_flights_first_row():
    rows = rows_with(40, flights=[(1, 5, 10), (2, 20, 25), (1, 30, 33)])
    words = words_with(40, [(1, "hadoken", 3, 6), (2, "lp", 18, 22), (1, "shoryuken", 29, 31)])
    k = P.flight_kinds(rows, words, CHARS)
    assert k[1][4] is None and k[1][5] == k[1][10] == "fireball" and k[1][11] is None
    assert k[2][20] == "other" and k[1][30] == "other"
    flame = P.flight_kinds(rows, words_with(40, [(1, "yoga_flame", 5, 5)]), {1: "dhalsim", 2: "ken"})
    assert flame[1][7] == "yoga_flame"
    fire = P.flight_kinds(rows, words_with(40, [(1, "yoga_fire", 5, 5)]), {1: "dhalsim", 2: "ken"})
    assert fire[1][7] == "fireball"


def test_drawn_needs_the_slot_on_and_the_blink_bit_clear_and_is_unknown_without_the_byte():
    r = rows_with(12, flights=[(1, 2, 10)], hide={5})
    assert P.drawn(r[4], 1) is True and P.drawn(r[5], 1) is False and P.drawn(r[1], 1) is False
    old = rows_with(12, flights=[(1, 2, 10)], hide_bytes=False)
    assert P.drawn(old[4], 1) is None and P.drawn(old[1], 1) is False


def _kinds(rows, presses, chars=CHARS):
    return P.flight_kinds(rows, words_with(len(rows), presses), chars)


def test_fire_answer_over_both_shown_frames():
    rows = rows_with(40, flights=[(1, 10, 30)], hide={11, 15, 19})
    kinds = _kinds(rows, [(1, "hadoken", 9, 10)])
    assert P.fire_answer(rows, kinds, 12) == "yes"
    assert P.fire_why(rows, kinds, 15) == (None, "hidden_both")    # active, blinked off in rows 11 and 15
    assert P.fire_frames(rows, kinds, 19) == [] and P.fire_answer(rows, kinds, 19) is None
    assert P.fire_answer(rows, kinds, 16) == "yes" and P.fire_frames(rows, kinds, 16) == ["n-4", "n"]
    assert P.fire_frames(rows, kinds, 23) == ["n"]
    assert P.fire_frames(rows, kinds, 34) == ["n-4"] and P.fire_answer(rows, kinds, 34) == "yes"   # gone at 34
    assert P.fire_answer(rows, kinds, 9) == "no"                     # spawns at row 10: rows 5 / 9 show none
    assert P.fire_answer(rows, kinds, 39) == "no" and P.fire_answer(rows, kinds, 5) == "no"
    other = _kinds(rows, [(1, "shoryuken", 9, 10)])
    assert P.fire_why(rows, other, 12) == (None, "unknown_active")
    flame = _kinds(rows, [(1, "yoga_flame", 9, 10)], {1: "dhalsim", 2: "ken"})
    assert P.fire_answer(rows, flame, 12) == "no"            # yoga flame is not a fireball
    old = rows_with(40, flights=[(1, 10, 30)], hide_bytes=False)
    assert P.fire_answer(old, kinds, 12) is None and P.fire_answer(old, kinds, 39) is None


def test_hard_tags_before_spawn_after_impact_pose_yoga_flame():
    rows = rows_with(80, flights=[(1, 20, 40)])
    kinds = _kinds(rows, [(1, "hadoken", 18, 20)])
    none = {1: "moving", 2: "moving"}
    assert P.hard_tags(rows, kinds, 12, none) == ["before_spawn"]
    assert P.hard_tags(rows, kinds, 7, none) == []                 # 13 rows before: not near
    assert P.hard_tags(rows, kinds, 50, none) == ["after_impact"]  # flight ended at 40, rows 46 / 50 off
    assert P.hard_tags(rows, kinds, 52, none) == ["after_impact"]
    assert P.hard_tags(rows, kinds, 53, none) == []
    assert P.hard_tags(rows, kinds, 60, {1: "projectile", 2: "moving"}) == ["pose"]
    flame = _kinds(rows, [(1, "yoga_flame", 18, 20)], {1: "dhalsim", 2: "ken"})
    assert P.hard_tags(rows, flame, 25, none) == ["yoga_flame"]


def test_pose_is_projectile_only_in_an_attack_state_with_the_projectile_word():
    rows = rows_with(10, p1=lambda k: {"state": 0x0C} if k >= 5 else {})
    assert P.pose_of(rows, 6, 1, "ryu", "hadoken", "special") == "projectile"
    assert P.pose_of(rows, 6, 1, "ryu", "shoryuken", "special") == "special"
    assert P.pose_of(rows, 3, 1, "ryu", "hadoken", "stand") == "moving"
    assert P.pose_of(rows, 6, 1, "blanka", "hadoken", "special") == "special"
    assert P.pose_of(rows, 3, 1, "ryu", None, "unknown") == "unknown"


@pytest.mark.parametrize("x1, x2, want", [(200, 272, "close"), (200, 273, "far"), (300, 228, "close"),
                                          (200, 600, None)])
def test_dist_is_the_gap_against_one_band(x1, x2, want):
    assert P.dist_answer(prow({"x": x1}, {"x": x2}), 72) == want


def test_act_of_the_grid_movements():
    assert [P.act_of(m) for m in ("stand", "jump", "attack", "special", "block", "unknown")] == [
        "moving", "moving", "attack", "special", "moving", None]


def test_game_facts_by_side_with_take_off_and_episode():
    def p1(k):
        return {"state": 0x04, "y": GROUND_Y - 20} if k >= 10 else {}
    rows = rows_with(30, p1=p1)
    fs = {f["t"]: f for f in P.game_facts(rows, [], CHARS, [12, 20, 3], 72)}
    assert sorted(fs) == [12, 20]                                # t - 4 must exist
    f = fs[12]["fighters"][1]
    assert f["side"] == "left" and f["air"] == "air" and f["air_prev"] == "ground" and f["act"] == "moving"
    assert f["in_episode"] is False and fs[20]["fighters"][1]["in_episode"] is True
    assert fs[20]["fighters"][2]["side"] == "right" and fs[20]["dist"] == "far" and fs[20]["fire"] == "no"


def test_a_fireball_off_the_screen_or_at_its_edge_is_neither_answer():
    rows = rows_with(40, flights=[(1, 10, 30)])                   # fighters at 200 / 300: screen 122 .. 378
    kinds = _kinds(rows, [(1, "hadoken", 9, 10)])
    assert P.camera_x(rows[12]) == 122 and P.fire_why(rows, kinds, 12) == ("yes", "drawn_on_screen")
    for x, want in ((129, None), (130, "yes"), (128, None), (371, None), (370, "yes"), (500, None)):
        rows[12]["shot1_x"] = x
        assert P.fire_answer(rows, kinds, 12) == want, x
    assert P.fire_why(rows, kinds, 12)[1] == "off_screen"
    near_wall = prow({"x": 60}, {"x": 120})                        # the camera stops at the stage's edge
    assert P.camera_x(near_wall) == 32
    far_wall = prow({"x": 400}, {"x": 460})
    assert P.camera_x(far_wall) == 224
