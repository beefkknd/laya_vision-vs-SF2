"""Moments: the decisions System 1 hands to System 2 (docs/TWO_SYSTEM_PLAN.md §2.3), pulled from a rollout."""
import random

from sf2 import contract as C
from sf2 import moments as MO

NOTE = ("me=chunli stand hp=80 opp=ryu jump hp=45 dist=mid facing=right corner=none time=early last=hk "
        "fireball=none")


def _row(frame, probs=None, taken=0, controllable=True, action="hk", episode=0, rnd=0):
    probs = probs or {"hk": 0.9, "hp": 0.1}
    return {"images": ["a_prev.png", "a.png"], "state_text": NOTE,
            "meta": {"episode": episode, "round": rnd, "frame": frame, "action": action, "controllable": controllable,
                     "student_probs": probs, "dmg_for": 0, "dmg_against": taken, "dmg_for_next": 0,
                     "dmg_against_next": taken}}


def test_one_surprise_per_hit_on_the_last_decision_she_could_act_on():
    rows = [_row(0), _row(4), _row(8, controllable=False), _row(12, taken=20, controllable=False), _row(16)]
    got = MO.extract(rows, random.Random(0), unsure_margin=0.0, audit_rate=0.0)
    assert [(m["frame"], m["why"]) for m in got] == [(4, "surprised")]


def test_a_combo_is_one_episode_not_one_surprise_per_hit():
    rows = [_row(0), _row(4, taken=10), _row(8, taken=10, controllable=False), _row(12, taken=10, controllable=False),
            _row(60), _row(64, taken=5)]
    got = MO.extract(rows, random.Random(0), unsure_margin=0.0, audit_rate=0.0)
    assert [m["frame"] for m in got if m["why"] == "surprised"] == [0, 60]


def test_unsure_flags_small_margins_and_audit_samples_the_rest():
    rows = [_row(f, probs={"hk": 0.45, "hp": 0.40}) for f in range(0, 40, 4)]
    got = MO.extract(rows, random.Random(0), unsure_margin=0.1, audit_rate=0.0)
    assert {m["why"] for m in got} == {"unsure"} and len(got) == len(rows)
    confident = [_row(f) for f in range(0, 4000, 4)]
    got = MO.extract(confident, random.Random(3), unsure_margin=0.1, audit_rate=0.01)
    assert {m["why"] for m in got} == {"audit"} and 3 <= len(got) <= 25


def test_records_are_valid_moment_records_with_notes_before():
    rows = [_row(f) for f in range(0, 80, 4)] + [_row(80, taken=12)]
    got = MO.extract(rows, random.Random(0), unsure_margin=0.0, audit_rate=0.0)
    m = got[0]
    assert C.moment_record(**{k: v for k, v in m.items() if k != "id"})["id"] == m["id"]
    assert len(m["notes_before"]) >= 10 and m["taken"] == 0 and m["played"] == "hk"


def test_uncontrollable_decisions_are_never_flagged():
    rows = [_row(f, probs={"hk": 0.5, "hp": 0.5}, controllable=False) for f in range(0, 40, 4)]
    assert MO.extract(rows, random.Random(0), unsure_margin=0.2, audit_rate=1.0) == []


def test_margin_for_a_target_share():
    rows = [_row(f, probs={"hk": 0.5 + f / 800, "hp": 0.5 - f / 800}) for f in range(0, 400, 4)]  # margins 0..0.99
    m = MO.margin_for_share(rows, 0.05)
    share = sum(C.confidence(r["meta"]["student_probs"])[0] < m for r in rows) / len(rows)
    assert 0.03 <= share <= 0.07
