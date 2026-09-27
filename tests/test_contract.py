"""Step 0 of the two-system plan (docs/TWO_SYSTEM_PLAN.md): the formats System 1 and any System 2 share."""
import json
import random

import pytest

from sf2 import contract as C
from sf2.actions import ACTIONS

NOTE = ("me=chunli stand hp=80 opp=ryu jump hp=45 dist=mid facing=right corner=none time=early last=hk "
        "fireball=none")


# --- the note as named fields --------------------------------------------------------------------------------------

def test_note_parses_into_named_fields():
    s = C.parse_note(NOTE)
    assert s == {"me": "chunli", "me_state": "stand", "me_hp": 80, "opp": "ryu", "opp_state": "jump", "opp_hp": 45,
                 "dist": "mid", "facing": "right", "corner": "none", "time": "early", "last": "hk",
                 "fireball": "none"}


def test_note_from_the_real_text_state_parses():
    from sf2 import ram

    f = ram.Fighters(my_hp=176, opp_hp=88, my_x=200, opp_x=300, my_y=192, opp_y=192, timer=0x45, my_state=0,
                     opp_state=ram.ATTACK_STATE, fireball=0)
    s = C.parse_note(ram.text_state(f, "chunli", "dhalsim", "forward", False, False, 176))
    assert (s["me_state"], s["opp_state"], s["opp_hp"], s["dist"], s["time"]) == ("stand", "attack", 50, "mid", "mid")


@pytest.mark.parametrize("bad", ["", "me=chunli stand hp=80", NOTE.replace("dist=mid", "dist=near"),
                                 NOTE.replace("hp=80", "hp=eighty")])
def test_malformed_note_is_rejected(bad):
    with pytest.raises(ValueError):
        C.parse_note(bad)


# --- situation rules -----------------------------------------------------------------------------------------------

def test_rule_parses_conditions_move_and_options():
    r = C.parse_rule("opp=ryu opp_state=jump dist=mid -> hp  weight=0.6 author=you  # anti-air with fierce")
    assert r.when == (("opp", "=", "ryu"), ("opp_state", "=", "jump"), ("dist", "=", "mid"))
    assert (r.move, r.weight, r.author, r.reason) == ("hp", 0.6, "you", "anti-air with fierce")


def test_rule_defaults():
    r = C.parse_rule("opp_state=jumpattack -> block")
    assert (r.weight, r.author, r.reason) == (C.DEFAULT_WEIGHT, "you", "")


def test_rule_matches_only_when_every_condition_holds():
    r = C.parse_rule("opp=ryu opp_state=jump dist=mid -> hp")
    s = C.parse_note(NOTE)
    assert r.matches(s)
    assert not r.matches({**s, "dist": "far"})
    assert not r.matches({**s, "opp": "dhalsim"})


def test_hp_rules_compare_numbers():
    s = C.parse_note(NOTE)  # me_hp 80, opp_hp 45
    assert C.parse_rule("opp_hp<50 -> forward").matches(s)
    assert not C.parse_rule("me_hp<=30 -> block").matches(s)
    assert C.parse_rule("me_hp>=80 opp_hp>44 -> hk").matches(s)


@pytest.mark.parametrize("bad, why", [
    ("opp=ryu -> fireball", "move"),                 # not one of Chun-Li's 14 moves
    ("height=tall -> hp", "field"),                  # no such field in the note
    ("dist=near -> hp", "value"),                    # dist is close / mid / far
    ("opp=akuma -> hp", "value"),                    # not a World Warrior character
    ("dist<mid -> hp", "operator"),                  # < and > only on the hp fields
    ("-> hp", "condition"),                          # a rule that matches everything is not a situation
    ("opp=ryu hp", "->"),
    ("opp=ryu -> hp weight=2", "weight"),            # weight is 0..1
    ("opp=ryu -> hp colour=red", "option"),
    ("opp=ryu opp=ken -> hp", "twice"),
])
def test_invalid_rules_are_rejected_with_the_reason(bad, why):
    with pytest.raises(ValueError, match=why):
        C.parse_rule(bad)


def test_rule_round_trips_through_its_text():
    line = "opp=ryu opp_state=jump dist=mid -> hp weight=0.6 author=qwen # anti-air"
    assert C.parse_rule(str(C.parse_rule(line))) == C.parse_rule(line)


def test_rules_file_skips_blanks_and_comments_and_names_the_bad_line():
    rules = C.parse_rules("# my playbook\n\nopp_state=jump -> hp\nopp_state=hit dist=close -> lk\n")
    assert [r.move for r in rules] == ["hp", "lk"]
    with pytest.raises(ValueError, match="line 2"):
        C.parse_rules("opp_state=jump -> hp\ndist=near -> hp\n")


# --- moment records ------------------------------------------------------------------------------------------------

def _moment(**kw):
    m = dict(match=3, round=2, frame=1432, why="surprised", notes_before=[NOTE], note=NOTE,
             probs={"hp": 0.31, "hk": 0.29, "block": 0.12}, played="hk", dealt=0, taken=18,
             frames=["t-4.png", "t.png"])
    m.update(kw)
    return m


def test_moment_record_validates_and_gets_an_id():
    rec = C.moment_record(**_moment())
    assert rec["id"] == "m3r2f1432"
    assert json.loads(json.dumps(rec)) == rec  # one JSON line


@pytest.mark.parametrize("change, why", [
    ({"why": "bored"}, "why"),
    ({"played": "fireball"}, "played"),
    ({"probs": {"fireball": 0.5}}, "probs"),
    ({"probs": {"hp": 0.8, "hk": 0.7}}, "probs"),     # more than 1 in total
    ({"taken": -3}, "taken"),
    ({"note": "garbage"}, "note"),
    ({"frames": []}, "frames"),
])
def test_invalid_moment_is_rejected(change, why):
    with pytest.raises(ValueError, match=why):
        C.moment_record(**_moment(**change))


def test_moment_advice_is_validated():
    a = C.parse_advice({"moment": "m3r2f1432", "move": "hp", "reason": "he jumped from mid range"})
    assert (a.moment, a.move, a.author) == ("m3r2f1432", "hp", "you")
    with pytest.raises(ValueError, match="move"):
        C.parse_advice({"moment": "m3r2f1432", "move": "shoryuken"})
    with pytest.raises(ValueError, match="moment"):
        C.parse_advice({"move": "hp"})


# --- why a decision is flagged -------------------------------------------------------------------------------------

def test_confidence_margin_and_entropy():
    probs = {a: 0.0 for a in ACTIONS} | {"hp": 0.5, "hk": 0.3, "block": 0.2}
    margin, entropy = C.confidence(probs)
    assert margin == pytest.approx(0.2)
    assert 0 < entropy < C.confidence({a: 1 / len(ACTIONS) for a in ACTIONS})[1]


def test_flag_priority_surprised_then_unsure_then_audit():
    rng = random.Random(0)
    assert C.flag(margin=0.01, taken_next=12, round_lost=False, rng=rng) == "surprised"
    assert C.flag(margin=0.9, taken_next=0, round_lost=True, rng=rng) == "surprised"
    assert C.flag(margin=0.01, taken_next=0, round_lost=False, rng=rng) == "unsure"
    assert C.flag(margin=0.9, taken_next=0, round_lost=False, rng=random.Random(0), audit_rate=1.0) == "audit"
    assert C.flag(margin=0.9, taken_next=0, round_lost=False, rng=rng, audit_rate=0.0) is None


def test_audit_rate_is_roughly_the_requested_share():
    rng = random.Random(7)
    n = sum(C.flag(margin=0.9, taken_next=0, round_lost=False, rng=rng) == "audit" for _ in range(20000))
    assert 150 < n < 250  # 1% of 20,000
