"""M1 / minimal G3 (f): the screen-evidence record carries the exact fields the lesson code reads. The test does not
check hand-named keys in isolation - it RUNS the real accessors (sf2.system2.lessons.condition_evidence / cause,
sf2.system1.advice.opp_doing, sf2.system2.character_prompt.threat / threats / if_you_see) on the rows the adapter
builds, so a missing or mis-shaped field is a failure by construction.

Seen RED: drop any sourced field in sf2.system2.screen_evidence.decision_row (e.g. ``range`` or ``opp_state``) and the
accessor raises / returns the wrong word - confirmed by removing ``opp_state`` (opp_doing then can't reproduce the
reader's "attacking"). Damage is asserted against the health-bar drops the screen record implies.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from sf2.system1.advice import opp_doing                            # noqa: E402
from sf2.system2 import character_prompt as C, lessons as L, screen_evidence as E   # noqa: E402

ME, OPP = "chunli", "honda"


def _dec(k, action, doing, his_air, my_life, his_life, dx, fireball=False, rng="close"):
    return {"i": k, "k": k, "action": action, "situation": [rng, doing, "full", "full"],
            "follows_rule": True, "rule": "soft", "advice_text": "Advice: none.", "lines": [],
            "moment": {"my_life": my_life, "his_life": his_life, "doing": doing, "his_air": his_air,
                       "side": "left", "dx": dx, "fireball": fireball}}


DECS = [_dec(10, "cl.hp", "attacking", False, 176, 176, 36),
        _dec(20, "block_high", "jumping", True, 150, 146, 40, fireball=True, rng="mid")]
REPLAY = {"result": "win", "dealt": 60, "taken": 26, "hp": 34, "my_life_end": 120, "opp_life_end": 100}


def test_damage_comes_from_health_bar_drops_across_the_decision():
    rows, summary = E.round_evidence(0, ME, OPP, DECS, REPLAY)
    r0, r1 = rows
    # raw drops are his/my hp to the NEXT decision (r0 cl.hp: 176->146 = 30; r1 block: 146->100replay = 46), BUT r1
    # is a BLOCK -- it cannot deal damage, so its 46 is the cl.hp hit still draining and is REATTRIBUTED to r0.
    assert (r0["dealt"], r0["taken"]) == (30 + 46, 176 - 150)       # 30 in its window + 46 moved from the block
    assert (r1["dealt"], r1["taken"]) == (0, 150 - 120)             # block's drawn damage reattributed; taken kept
    assert summary["result"] == "win" and summary["hp"] == 34 and summary["source"] == "replay"


def test_rows_feed_the_lesson_accessors():
    rows, _ = E.round_evidence(0, ME, OPP, DECS, REPLAY)
    r0, r1 = rows
    # opp_doing reproduces the reader's word from opp_air / opp_state
    assert opp_doing(r0) == "attacking" and opp_doing(r1) == "jumping"
    # condition_evidence runs (range + opp_doing + action + dealt/taken all present)
    ev = L.condition_evidence(rows, {"kind": "use_more", "move": "cl.hp", "range": "close", "when": "attacking"})
    assert set(ev) >= {"tries", "others", "net", "base", "cls"}
    # cause() reads kind / actual / taken
    assert L.cause(r0) == "stuffed"          # a damaging attack with no screen outcome reads as stuffed
    assert L.cause(r1) == "caught"           # a block that took damage
    # character_prompt threat / threats / if_you_see run on the rows
    assert C.threat(r0) == "hits her from afar" and C.threat(r1) == "jumps in"
    assert isinstance(C.threats(rows), list)
    assert isinstance(C.if_you_see(rows, ["cl.hp", "block_high"]), list)


def test_screen_only_fallback_when_no_replay():
    rows, summary = E.round_evidence(0, ME, OPP, DECS, replay=None)
    assert summary["source"] == "screen"
    # no RAM referee: the result is read off the last drawn health bars (my 150 > his 146 -> win), never "unknown"
    assert summary["result"] == "win"
    assert summary["dealt"] == sum(r["dealt"] for r in rows)        # summed from the per-decision screen drops
