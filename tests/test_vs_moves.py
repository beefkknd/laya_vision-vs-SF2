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


def test_twenty_static_actions_plus_two_blocks_per_character():
    from sf2.vs_sweep import static_actions
    assert len(static_actions("ryu")) == len(static_actions("chunli")) == 20
    assert len(actions("ryu")) == len(actions("chunli")) == 22
    assert {"block_high", "block_low"} <= set(actions("dhalsim")) and "block_high" not in static_actions("dhalsim")


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


def test_game_log_puts_the_opponents_reaction_on_the_action_that_caused_it():
    from sf2.game_log import action_entry, clock
    def raw(p1=None, p2=None, timer=0x87):
        r = row(p1=p1, p2=p2, timer=timer)
        return r
    before = raw()
    rows = [raw(p1={"state": 0x0A}), raw(p1={"state": 0x0A}, p2={"state": 0x0A}),           # I whiff, he attacks
            raw(p1={"state": 0x0E, "react": 0x00, "life": 150}, p2={"state": 0x0A}),         # he hits me
            raw(p1={"state": 0x0E, "life": 150}), raw(p1={"life": 150})]
    d = {"action": "hp", "p_hit": 0.9, "predicted": "hit", "probs": {"hp": 0.9, "lp": 0.2}}
    e = action_entry(3, 120, "ryu", "ken", before, rows, d, "whiff")
    assert (e["game"], e["clock"], e["me"], e["opp"], e["side"]) == (3, 87, "ryu", "ken", "left")
    assert e["actual"] == "whiff" and e["dealt"] == 0
    assert e["i_was_hit"] and e["taken"] == 26 and e["opp_attacked"] and not e["opp_blocked"]
    assert e["opp_reaction"] == ["stand", "attack", "stand"] and e["my_life_after"] == 150
    assert clock(0x99) == 99 and e["top3"][0] == ["hp", 0.9]


def test_short_memory_goes_into_the_prompt_and_bad_memory_is_refused():
    from sf2.memory import check, prompt_text
    note = "me=ryu opp=ken dist=mid side=left dx=+70"
    assert prompt_text(note, None) == note and prompt_text(note, {"opp": "ken", "lessons": []}) == note
    les = [{"text": "use more sweep at mid range", "kind": "use_more", "action": "sweep", "range": "mid",
            "evidence": {"tries": 11, "count": 7, "rate": 0.64, "refs": ["g1f40"]}}]
    assert prompt_text(note, {"opp": "ken", "lessons": les * 7}).count("use more sweep") == 5   # at most 5 lessons
    assert prompt_text(note, {"opp": "ken", "lessons": les}) == note + "\nmemory vs ken: use more sweep at mid range"
    acts = ["lp", "sweep", "hadoken"]
    assert check({"lessons": les}, acts) == []
    bad = [dict(les[0], action="flying_kick"), dict(les[0], kind="vibes"),
           dict(les[0], evidence={"tries": 3, "count": 5, "refs": ["g1f1"]}), dict(les[0], evidence={})]
    assert all(check({"lessons": [b]}, acts) for b in bad)


def test_system2_vet_keeps_laya_lessons_short_unique_and_backed():
    from sf2.system2 import vet
    def act(action, rng, actual, hit_me=False, opp_state="stand"):
        return {"kind": "attack", "action": action, "range": rng, "actual": actual, "i_was_hit": hit_me,
                "opp_state": opp_state, "game": 0, "frame": 0}
    acts = [act("spinning_bird_kick", r, "whiff", True) for r in ("close", "mid", "far") for _ in range(4)]
    acts += [act("lp", "close", "hit") for _ in range(5)]
    L = lambda text, kind, action, rng, claim: dict(text=text, kind=kind, action=action, range=rng, claim=claim)  # noqa
    kept, rej = vet([L("avoid spinning_bird_kick: he punishes it", "avoid", "spinning_bird_kick", None, "punished"),
                     L("avoid spinning_bird_kick at mid", "avoid", "spinning_bird_kick", "mid", "punished"),   # repeat
                     L("use more lp up close: lands 57%", "use_more", "lp", "close", "lands"),                 # number
                     L("use more lp up close, it is by far your most dependable poke there", "use_more", "lp",
                       "close", "lands"),                                                                    # too long
                     L("use more lp up close", "use_more", "lp", "close", "lands"),
                     L("use more mp at mid", "use_more", "mp", "mid", "lands")], acts, 5)                     # no tries
    assert [k["text"] for k in kept] == ["avoid spinning_bird_kick: he punishes it", "use more lp up close"]
    assert kept[0]["evidence"]["tries"] == 12 and kept[0]["evidence"]["count"] == 12
    whys = [r["why"] for r in rej]
    assert any("repeats" in w for w in whys) and any("numbers" in w for w in whys)
    assert any("characters" in w for w in whys) and any("tries" in w for w in whys)


def test_short_memory_evidence_uses_all_games_until_this_opponent_has_enough():
    from sf2.system2 import vet
    a = lambda act, res, st="stand": {"kind": "attack", "action": act, "range": "close", "actual": res,  # noqa: E731
                                       "i_was_hit": False, "opp_state": st, "game": 0, "frame": 0}
    all_games = [a("lp", "hit") for _ in range(20)]
    vs_him = [a("lp", "whiff"), a("hp", "hit", "jump"), a("hp", "hit", "jump"), a("hp", "hit", "jump")]
    L = lambda text, kind, action, claim: dict(text=text, kind=kind, action=action, range=None, claim=claim)  # noqa
    kept, rej = vet([L("use more lp up close", "use_more", "lp", "lands"),
                     L("when he jumps in, use hp", "counter", "hp", "counter:jump"),
                     L("he jumps in a lot", "opponent_habit", None, "habit:jump")], vs_him, 5, fallback=all_games)
    by = {k["text"]: k["evidence"] for k in kept}
    assert by["use more lp up close"]["scope"] == "all opponents" and by["use more lp up close"]["tries"] == 20
    assert by["when he jumps in, use hp"]["scope"] == "this opponent" and by["when he jumps in, use hp"]["tries"] == 3
    assert by["he jumps in a lot"]["tries"] == 4                          # habits: his games only, never the fallback


def test_a_kept_short_memory_must_meet_todays_rules():
    from sf2.system2 import fits_laya
    ok = {"text": "use more lp up close", "kind": "use_more", "action": "lp", "range": "close", "claim": "lands"}
    old_style = {"text": "ryu punishes my spinning_bird_kick at mid range (70%)", "kind": "avoid",
                 "action": "spinning_bird_kick", "range": "mid"}                    # no claim, numbers: the old builder
    assert fits_laya({"lessons": [ok]})
    assert not fits_laya({"lessons": [ok, old_style]})
    assert not fits_laya({"lessons": [ok, dict(ok, text="use more lp up close, really")]})        # the same move twice
    assert not fits_laya({"lessons": [dict(ok, text="x" * 61)]}) and not fits_laya({"lessons": []}) and not fits_laya(None)


def test_revise_prompt_shows_which_lessons_system1_followed():
    from sf2.system2_prompts import followed
    a = lambda act, rng, res, punished=False: {"action": act, "range": rng, "actual": res, "i_was_hit": punished}  # noqa
    recent = [a("sweep", "close", "whiff", True), a("sweep", "close", "whiff"), a("hp", "mid", "hit")]
    mem = {"lessons": [{"text": "use more lp up close", "kind": "use_more", "action": "lp", "range": "close"},
                       {"text": "avoid sweep up close", "kind": "avoid", "action": "sweep", "range": "close"},
                       {"text": "he jumps in at far", "kind": "opponent_habit", "action": None, "range": "far"}]}
    lines = followed(recent, mem)
    assert lines[0] == '- IGNORED (never used) "use more lp up close": lp at close used 0 times'
    assert lines[1] == '- IGNORED (used anyway) "avoid sweep up close": sweep at close used 2 times (whiff 2; punished 1)'
    assert len(lines) == 2                                          # habits have no move to follow


def test_system2_reply_checks_flag_each_broken_rule():
    """The checks the live Qwen prompt tests gate on, seen red on canned bad replies."""
    import json
    from sf2.system2_checks import reply_problems
    logs = json.load(open("tests/fixtures/system2/chunli_logs.json"))
    ryu = logs["ryu"]["actions"]
    every = [a for v in logs.values() for a in v["actions"]]
    good = {"text": "use more lp up close", "kind": "use_more", "action": "lp", "range": "close", "claim": "lands"}
    assert reply_problems("chunli", {"lessons": [good]}, 5, ryu, every) == []
    bad = {
        "no lessons key": {"notes": "x"},
        "over the limit": {"lessons": [good, dict(good, action="c.lk", text="use more c.lk up close")] * 3},
        "not her move": {"lessons": [dict(good, action="hadoken", text="use more hadoken")]},
        "numbers": {"lessons": [dict(good, text="use more lp up close: 57%")]},
        "too long": {"lessons": [dict(good, text="use more lp up close because it is by far the best poke you have")]},
        "repeat": {"lessons": [good, dict(good, text="lp up close again")]},
        "use and avoid": {"lessons": [good, dict(good, kind="avoid", claim="whiffs", text="avoid lp up close")]},
        "not backed": {"lessons": [dict(good, action="hp", text="use more hp up close")]},
    }
    for name, reply in bad.items():
        assert reply_problems("chunli", reply, 5, ryu, every), name


def test_block_outcome_needs_real_block_stun():
    from sf2 import vs_defense as D
    def r(d_state=0, react=0, life=176, a_y=192):
        return {"d_state": d_state, "d_react": react, "d_life": life, "a_y": a_y}
    assert D.outcome([r(), r(0x08), r(0x0E, 0x06), r()])["outcome"] == "blocked"                       # block stun
    assert D.outcome([r(), r(0x08), r(0x08), r()])["outcome"] == "none"          # guard pose alone: not a block
    hit = D.outcome([r(), r(0x0E, 0x00, 150), r(0x0E, 0x00, 150)])
    assert hit["outcome"] == "got_hit" and hit["damage_taken"] == 26
    assert D.outcome([r(), r(0x14, 0, 140)])["outcome"] == "got_hit"                                  # thrown
    assert D.outcome([r(), r(0x0E, 0x08, 172)])["damage_taken"] == 4                                  # chip still blocked
    assert D.decision_frame([r(a_y=192)] * 9 + [r(a_y=140)] * 5 + [r(a_y=155)], "jump_in") == 14   # coming down
    assert D.decision_frame([], "sweep") == D.LEAD + D.REACT
