"""BUG (found in the chun4 live test, Fable-grounded): per-decision credit is contaminated by DELAYED HITS.

When she lands an anti-air at decision t, he enters hit-stun and his health BAR drains over the next frames --
which the eye reads in decision t+1's window. So the move she plays at t+1 (often block_high, while he is in
'hit') gets credited with the previous move's damage. Measured in chun4: block_high in (mid,jumping) with his
label 'hit' netted +19.7 -- it wasn't blocking that dealt 20, it was the anti-air landing.

Fix: a decision whose moment.his_label == 'hit' did not cause the damage drawn in its window; reattribute that
window's `dealt` to the nearest EARLIER decision that was not itself in hit-stun. Damage is conserved (only moved).
This corrects credit for BOTH the rule scorer and the value table.
"""
from sf2.system2 import screen_evidence as SE


def _dec(action, his_life, his_label, doing="jumping", rng="mid"):
    air = doing == "jumping"
    return {"action": action, "situation": [rng, doing, "full", "full"],
            "moment": {"my_life": 100, "his_life": his_life, "doing": doing, "his_air": air,
                       "his_label": his_label, "dx": 40, "side": "left"}}


def test_delayed_hit_is_reattributed_to_the_move_that_caused_it():
    # lightning_legs connects (t0); he is in 'hit' at t1 while she blocks; his bar drains 100->60 during t1's window.
    decisions = [
        _dec("lightning_legs", 100, "jump"),      # the anti-air; his bar hasn't drained yet in ITS window
        _dec("block_high", 100, "hit"),           # he is in hit-stun; the 40 hp drains HERE (the bug credits block)
        _dec("s.mk", 60, "stand", doing="standing"),
    ]
    rows, summary = SE.round_evidence(0, "chunli", "ryu", decisions, replay=None)
    la, bl, mk = rows
    assert bl["action"] == "block_high" and bl["dealt"] == 0, "block must NOT be credited with the delayed hit"
    assert la["action"] == "lightning_legs" and la["dealt"] == 40, "the anti-air that caused it gets the credit"
    assert summary["dealt"] == 40, "total damage conserved, only reattributed"


def test_no_reattribution_when_not_in_hitstun():
    # an honest s.mk that drains his bar in its own window keeps its credit
    decisions = [
        _dec("s.mk", 100, "stand", doing="standing"),
        _dec("s.mk", 70, "stand", doing="standing"),
    ]
    rows, _ = SE.round_evidence(0, "chunli", "ryu", decisions, replay=None)
    assert rows[0]["dealt"] == 30 and rows[1]["dealt"] == 0


def test_hitstun_at_round_start_is_left_in_place():
    # no earlier non-hit decision to credit -> leave it (cannot attribute), never crash
    decisions = [_dec("block_high", 100, "hit"), _dec("s.mk", 50, "stand", doing="standing")]
    rows, summary = SE.round_evidence(0, "chunli", "ryu", decisions, replay=None)
    assert rows[0]["dealt"] == 50 and summary["dealt"] == 50       # conserved, not lost
