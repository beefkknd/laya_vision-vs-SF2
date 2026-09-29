"""Qwen picks good and bad moves from evidence code computes (sf2.system2.move_coach): the evidence and its classes,
the signal Qwen reads, reading Qwen's reply, the advice lines, and the mechanical grades of Q1 (identify) and Q2
(keep the good, drop the wrong, game after game)."""
import random

import pytest

from sf2.system1.advice import read
from sf2.system2.move_coach import (classify, evidence, grade, grade_update, lines, parse_reply, signal)

MOVES = ["sweep", "c.mk", "mp", "hp", "throw", "spinning_bird_kick"]


def atk(move, rng, net):
    return {"action": move, "range": rng, "kind": "attack", "dealt": max(net, 0), "taken": max(-net, 0),
            "actual": "hit" if net > 0 else "whiff"}


def rows():
    r = random.Random(0)
    out = []
    out += [atk("sweep", "close", -11 + r.randint(-3, 3)) for _ in range(60)]      # clearly bad, biggest drain
    out += [atk("c.mk", "mid", 4 + r.randint(-3, 3)) for _ in range(50)]          # clearly good
    out += [atk("mp", "close", -2 + r.randint(-3, 3)) for _ in range(30)]         # clearly bad, small
    out += [atk("hp", "close", r.choice([-20, 20])) for _ in range(40)]           # unclear: big spread, mean ~0
    out += [atk("throw", "close", 15) for _ in range(5)]                          # too few tries
    return out


def test_evidence_classes():
    ev = {(e["move"], e["range"]): e for e in evidence(rows())}
    assert ev[("sweep", "close")]["cls"] == "bad" and ev[("c.mk", "mid")]["cls"] == "good"
    assert ev[("mp", "close")]["cls"] == "bad" and ev[("hp", "close")]["cls"] == "unclear"
    assert ev[("throw", "close")]["cls"] == "few"
    assert ev[("sweep", "close")]["total"] < ev[("mp", "close")]["total"] < 0
    assert classify(30, -1.0, 0.5) == "unclear" and classify(30, 0.2, 3.0) == "good" and classify(5, 1, 2) == "few"


def test_signal_shows_every_tried_move_in_a_given_order_and_level():
    ev = evidence(rows())
    a = signal(ev, "table", random.Random(1))
    b = signal(ev, "table", random.Random(2))
    assert sorted(a.splitlines()) == sorted(b.splitlines()) and a != b          # same facts, other order
    assert "move sweep, range close:" in a and "clearly" not in a       # move and range as separate fields
    assert "clearly bad" in signal(ev, "classes", random.Random(1))


def test_parse_reply_keeps_known_picks_and_reports_the_rest():
    ev = evidence(rows())
    reply = {"use_more": [{"move": "c.mk", "range": "mid"}, {"move": "fireball", "range": "far"}],
             "avoid": [{"move": "sweep", "range": "close"}, {"move": "sweep", "range": "close"}], "why": "x"}
    picks, problems = parse_reply(reply, ev)
    assert picks == {"use_more": [("c.mk", "mid")], "avoid": [("sweep", "close")]}
    assert any("fireball" in p for p in problems) and any("twice" in p for p in problems)
    for bad in (None, [], "text", {"use_more": "c.mk"}):
        assert parse_reply(bad, ev)[0] == {"use_more": [], "avoid": []} and parse_reply(bad, ev)[1]


def test_lines_read_as_text_laya_advice():
    got = lines({"use_more": [("c.mk", "mid")], "avoid": [("sweep", "close")]})
    assert got == ["use more c.mk at mid range", "avoid sweep up close"]
    assert read(got[0], MOVES).polarity == "soft" and read(got[1], MOVES).polarity == "neg"
    assert read(got[1], MOVES).where == "close"


def test_q1_grade():
    ev = evidence(rows())
    right = {"use_more": [("c.mk", "mid")], "avoid": [("sweep", "close"), ("mp", "close")]}
    assert grade(right, ev)["ok"]
    assert not grade({"use_more": [("hp", "close")], "avoid": [("sweep", "close")]}, ev)["ok"]    # unclear used
    g = grade({"use_more": [("c.mk", "mid")], "avoid": [("mp", "close")]}, ev)                    # missed drain
    assert not g["ok"] and g["missed_drain"] == ("sweep", "close")
    assert not grade({"use_more": [("throw", "close")], "avoid": [("sweep", "close")]}, ev)["ok"]  # too few tries
    assert not grade({"use_more": [], "avoid": [("sweep", "close")]}, ev)["ok"]                   # a good one exists
    assert not grade({"use_more": [("c.mk", "mid")] * 4, "avoid": [("sweep", "close")]}, ev)["ok"]


def test_q2_update_grade_keeps_the_good_and_drops_the_wrong():
    ev = evidence(rows())
    prev = {"use_more": [("c.mk", "mid")], "avoid": [("sweep", "close")]}
    assert grade_update(prev, {"use_more": [("c.mk", "mid")], "avoid": [("sweep", "close"), ("mp", "close")]}, ev)[
        "ok"]
    dropped = grade_update(prev, {"use_more": [], "avoid": [("sweep", "close")]}, ev)
    assert not dropped["ok"] and dropped["dropped_correct"] == [("c.mk", "mid")]
    stale = {"use_more": [("hp", "close")], "avoid": [("sweep", "close")]}                       # hp became unclear
    assert not grade_update(stale, stale, ev)["ok"]
    assert grade_update(stale, {"use_more": [("c.mk", "mid")], "avoid": [("sweep", "close")]}, ev)["ok"]


@pytest.mark.parametrize("ev_rows", [[], [atk("hp", "close", 5)] * 3])
def test_nothing_clear_means_nothing_to_say(ev_rows):
    ev = evidence(ev_rows)
    assert grade({"use_more": [], "avoid": []}, ev)["ok"]
    assert not grade({"use_more": [("hp", "close")], "avoid": []}, ev)["ok"] if ev else True


def test_q2_an_avoided_move_is_not_swapped_out_while_slots_are_free():
    """Seen in the first Ken loop (2026-09-29, game 9): 'avoid sweep up close' (still clearly bad) was swapped for
    another bad move because avoid had only 2 slots; she would sweep up close again. Text laya reads 5 lines."""
    r = random.Random(1)
    ev = evidence(rows() + [atk("c.hp", "close", -9 + r.randint(-3, 3)) for _ in range(40)])
    prev = {"use_more": [("c.mk", "mid")], "avoid": [("sweep", "close"), ("mp", "close")]}
    swapped = {"use_more": [("c.mk", "mid")], "avoid": [("sweep", "close"), ("c.hp", "close")]}
    g = grade_update(prev, swapped, ev)
    assert not g["ok"] and g["dropped_correct"] == [("mp", "close")]
    grown = {"use_more": [("c.mk", "mid")], "avoid": [("sweep", "close"), ("mp", "close"), ("c.hp", "close")]}
    assert grade_update(prev, grown, ev)["ok"]


def test_caps_share_five_lines():
    from sf2.system2.move_coach import MAX_LINES, over_cap
    assert MAX_LINES == 5
    assert not over_cap({"use_more": [("a", "mid")], "avoid": [("b", "close")] * 4})
    assert over_cap({"use_more": [("a", "mid")] * 4, "avoid": []})           # at most 3 use more
    assert over_cap({"use_more": [("a", "mid")] * 2, "avoid": [("b", "close")] * 4})


def test_picks_in_play_show_their_class_now():
    """Seen in the second Ken loop (2026-09-29): 'use more hp up close' became unclear and was kept 2 more updates;
    the picks-in-play list showed only what the pick did since it was added."""
    from sf2.system2.move_coach import messages
    ev = evidence(rows())
    cur = {"use_more": [("hp", "close")], "avoid": [("sweep", "close")]}
    user = messages("chunli", "ken", "sig", cur, {}, ev)[1]["content"]
    assert "use more hp up close: now unclear" in user and "avoid sweep up close: now clearly bad" in user
    assert "must be dropped" in messages("chunli", "ken", "sig")[0]["content"]
