"""The value-table POLICY driving decisions (sf2/system1/value_table.decider + loop_runner.play_round decide=),
Stage 3a. The decider keys a Moment, restricts to the FOLLOWABLE action set at that range, and `choose` picks;
crediting its own choices makes the table converge. Pure (no emulator) via looptools.make_moment."""
import os
import random
import sys

sys.path.insert(0, os.path.dirname(__file__))
from looptools import make_moment                                   # noqa: E402

from sf2.system1 import value_table as VT                           # noqa: E402
from sf2.system1.advice import available_moves, char_categories, stance_of  # noqa: E402

CLOSE = available_moves(stance_of("stand", "close"), char_categories("chunli"))


def _row(action, net):
    return {"range": "close", "opp_state": "stand", "opp_air": False, "opp_shot": False,
            "his_label": "stand", "action": action, "dealt": max(net, 0), "taken": max(-net, 0)}


def test_decider_picks_a_followable_action_and_keys_the_when():
    decide = VT.decider(VT.blank(), "chunli", random.Random(0))
    d = decide(make_moment(dx=36, doing="standing"))               # dx 36 -> close
    assert d["action"] in CLOSE                                    # only a move she can actually play up close
    assert d["when"] == "close|standing|0"
    assert isinstance(d["explored"], bool) and d["category"]


def test_decider_never_proposes_an_unplayable_move():
    # far range: s.* offered, throws NOT -- the followable set already excludes them
    decide = VT.decider(VT.blank(), "chunli", random.Random(1))
    for _ in range(40):
        d = decide(make_moment(dx=150, doing="standing"))          # far
        assert "throw" not in d["action"]                          # throws are not followable far


def test_table_learns_from_the_deciders_own_choices():
    def net(a):
        return 15.0 if a == "throw_F+hp" else -5.0
    t, rng = VT.blank(), random.Random(0)
    for _ in range(300):
        d = VT.decider(t, "chunli", rng)(make_moment(dx=36, doing="standing"))
        t = VT.credit(t, [_row(d["action"], net(d["action"]))])
    assert VT.mean(t["cells"]["close|standing|0"].get("throw_F+hp")) > 0
    assert VT.decider(t, "chunli", rng, eps0=0.0)(make_moment(dx=36, doing="standing"))["action"] == "throw_F+hp"
