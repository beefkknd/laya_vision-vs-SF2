"""VS BATTLE move checks and measurements on synthetic rows: each check is shown passing AND failing."""
from sf2.vs import GROUND_Y, view
from sf2.vs_metrics import measure, reach
from sf2.vs_moves import MOVESETS, combo, connected, toward


def row(p1=None, p2=None, **extra):
    base = {"hp": 176, "life": 176, "x": 200, "y": GROUND_Y, "state": 0, "sub": 0, "react": 0, "dizzy": 0,
            "special": 0, "facing": 0x40, "char": 0}
    r = {}
    for p, over in ((1, p1 or {}), (2, dict({"x": 260}, **(p2 or {})))):
        r.update({"p%d_%s" % (p, k): v for k, v in dict(base, **over).items()})
    r.update({"timer": 0x99, "result": 0, "shot1": 0, "shot1_x": 0, "shot2": 0, "shot2_x": 0})
    r.update(extra)
    return r


def viewed(rows, attacker=1):
    return [view(r, attacker) for r in rows]


def move(char, name):
    return next(m for m in MOVESETS[char]() if m.name == name)


def test_view_swaps_roles_for_player_2():
    v = view(row(p1={"x": 10}, p2={"x": 90}), 2)
    assert (v["a_x"], v["d_x"], v["attacker"]) == (90, 10, 2)


def test_toward_is_relative_to_side():
    assert toward(viewed([row(), row(p1={"x": 230})])) == 30          # p1 left walks right
    assert toward(viewed([row(), row(p2={"x": 230})], 2)) == 30       # p2 right walks left
    assert toward(viewed([row(), row(p2={"x": 290})], 2)) == -30


def test_walk_forward_check_fails_when_standing_still():
    m = move("ryu", "walk_forward")
    assert m.check(viewed([row(), row(p1={"x": 225})]))
    assert not m.check(viewed([row(), row(p1={"x": 205})]))


def test_close_normal_needs_attack_state_and_contact():
    m = move("ryu", "cl.hp")
    hit = [row(), row(p1={"state": 0x0A}), row(p1={"state": 0x0A}, p2={"life": 150, "state": 0x0E})]
    whiff = [row(), row(p1={"state": 0x0A}), row(p1={"state": 0x0A})]
    assert m.check(viewed(hit)) and not m.check(viewed(whiff))
    assert not m.check(viewed([row(), row(p2={"life": 150})]))   # damage without an attack state is not the move


def test_special_id_checked_for_player_1_only():
    m = move("ryu", "shoryuken_hp")
    up = row(p1={"state": 0x0C, "special": 4, "y": 120})
    wrong = row(p1={"state": 0x0C, "special": 0, "y": 120})
    assert m.check(viewed([row(), up])) and not m.check(viewed([row(), wrong]))
    assert m.check(viewed([row(), row(p2={"state": 0x0C, "special": 0, "y": 120})], 2))


def test_throw_needs_thrown_state():
    m = move("chunli", "throw_F+hp")
    assert m.check(viewed([row(), row(p2={"state": 0x14})]))
    assert not m.check(viewed([row(), row(p2={"state": 0x0E, "life": 150})]))


def test_block_passes_only_without_damage():
    m = move("ryu", "block_high")   # measured fighter defends: the other (p2) attacks
    guard = [row(), row(p1={"state": 0x08}, p2={"state": 0x0A})]
    hit = [row(), row(p1={"state": 0x0E, "life": 140}, p2={"state": 0x0A})]
    assert m.check(viewed(guard, 2)) and not m.check(viewed(hit, 2))


def test_combo_breaks_when_the_defender_recovers_between_hits():
    hit = lambda life, st=0x0E: row(p2={"life": life, "state": st})  # noqa: E731
    assert combo(viewed([row(), hit(160), hit(160), hit(140)]), 2)
    assert not combo(viewed([row(), hit(160), hit(160, 0), hit(140)]), 2)
    assert not combo(viewed([row(), hit(160)]), 2)


def test_measure_startup_damage_and_projectile():
    rows = [row(), row(p1={"state": 0x0C}, shot1=1, shot1_x=210), row(p1={"state": 0x0C}, shot1=1, shot1_x=214),
            row(shot1=1, shot1_x=218), row(shot1=1, shot1_x=222), row(shot1=1, shot1_x=226),
            row(p2={"life": 150, "state": 0x0E}), row()]
    m = measure(viewed(rows))
    assert (m["startup"], m["damage"], m["hits"], m["duration"]) == (6, 26, 1, 2)
    assert m["shot_speed"] == 4.0 and m["shot_frames"] == 5
    assert connected(viewed(rows)) and not connected(viewed([row(), row()]))


def test_reach_is_widest_connecting_gap():
    assert reach([{"gap": 40, "connected": True}, {"gap": 70, "connected": True}, {"gap": 90, "connected": False}]) \
        == {"reach_max": 70, "reach_min": 40, "sweep_hits": 2, "sweep_tries": 3}
    assert reach([{"gap": 90, "connected": False}])["reach_max"] is None


# ---------------------------------------------------------------------------------------------------- stage-1 sweep
import numpy as np  # noqa: E402

from sf2.frames import HUD_ROWS, mirror_frame, model_frame  # noqa: E402
from sf2.vs_sweep import actions, mirror_record, outcome  # noqa: E402


def test_twenty_actions_per_character():
    assert len(actions("ryu")) == len(actions("chunli")) == 20


def test_model_frame_blanks_the_hud_and_mirror_is_a_whole_flip():
    img = np.arange(224 * 256 * 3, dtype=np.uint32).reshape(224, 256, 3).astype(np.uint8)
    f = model_frame(img)
    assert f.shape == (256, 256, 3) and not f[224:].any()                 # padded, not resized
    assert not f[:HUD_ROWS].any() and (f[HUD_ROWS:224] == img[HUD_ROWS:]).all()
    assert (mirror_frame(img) == f[:, ::-1]).all()
    assert img[:HUD_ROWS].any()                                          # the input is not changed


def test_mirror_record_swaps_side_fields_not_action():
    rec = {"id": "x", "side": "left", "facing": "right", "dx": 40, "action": "forward",
           "buttons": [["right"], ["right", "y"]], "state_text": "me=ryu side=left dx=+40 opp=chunli"}
    m = mirror_record(rec)
    assert (m["side"], m["facing"], m["dx"], m["action"]) == ("right", "left", -40, "forward")
    assert m["buttons"] == [["left"], ["left", "y"]]
    assert m["state_text"] == "me=ryu side=right dx=-40 opp=chunli" and m["mirrored"]
    assert rec["side"] == "left"                                          # the original is not changed


def test_outcome_labels():
    def rs(*d2, a_state=0x0A):
        return [view(row(p1={"state": a_state}, p2=d), 1) for d in ({},) + d2]
    assert outcome(rs({"state": 0x0E, "react": 0x00, "life": 150}), "hp")["outcome"] == "hit"
    assert outcome(rs({"state": 0x0E, "react": 0x06, "life": 170}), "hadoken")["outcome"] == "blocked"   # chip
    assert outcome(rs({"state": 0x08}, {"state": 0x0E, "react": 0x08}), "hp")["outcome"] == "blocked"
    guard_only = outcome(rs({"state": 0x08}), "lp")         # proximity guard, the jab fell short
    assert guard_only["outcome"] == "whiff" and guard_only["guard_pose"]
    assert outcome(rs({"state": 0x14}), "throw")["outcome"] == "hit"
    assert outcome(rs({}), "hp")["outcome"] == "whiff"
    assert outcome(rs({}, a_state=0), "hp")["executed"] is False
    assert outcome(rs({}, a_state=0), "forward")["outcome"] == "none"


def test_train_val_split_never_shares_a_frame():
    import importlib.util
    import random
    import sys
    sys.path.insert(0, "scripts")                                          # train.py imports _path from there
    spec = importlib.util.spec_from_file_location("train", "scripts/train.py")
    train = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(train)
    rows = [{"dataset": c, "state": {"images": ["frames/%s%s_%d_prev.png" % (m, s, i), "frames/%s%s_%d_now.png" % (m, s, i)]},
             "id": (c, s, i, a, m)} for c in ("ryu", "ken") for s in ("left_close", "left_far") for i in range(10)
            for a in range(20) for m in ("", "mirror_")]
    tr, va = train.split_by_position(rows, random.Random(0))
    assert va and len(tr) + len(va) == len(rows)
    assert not {train.position(x) for x in tr} & {train.position(x) for x in va}
    assert all(len([x for x in va if train.position(x) == k]) == 40 for k in {train.position(x) for x in va})
    per = {c: len({train.position(x) for x in va if x["dataset"] == c}) for c in ("ryu", "ken")}
    assert per["ryu"] == per["ken"] >= 1                                    # every character equally
