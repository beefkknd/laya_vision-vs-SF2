"""M1 (b)+(c): the advice actually drives the screen runner's move, and a retire signal changes it back.

The decision under test is sf2.system1.loop_runner.two_stage_decide: a faithful follower text laya (tests.looptools)
reads the situation + Qwen's advice out of the prompt and follows it over the unrated two-stage menu.

Seen RED: with the advice line NOT injected into the prompt (e.g. if the runner forgot to pass ``lines`` into
``prompt``), the follower - which reads only the text - sees "Advice: none." and picks the default block, so
test_advice_line_changes_the_move fails. Confirmed by running it with ``lines=[]`` held fixed (asserts below would see
block_high, not throw_F+hp / cl.hp). test_retire changes back only if review() actually retires the registered lesson;
with retirement disabled the move stays cl.hp and the last assert fails.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from looptools import FollowerLaya, make_moment, MENU_MOVES          # noqa: E402

from sf2.system1.action_menu import DEFAULT_MOVE                     # noqa: E402
from sf2.system1.loop_runner import two_stage_decide                 # noqa: E402
from sf2.system2 import lessons as L                                 # noqa: E402


def test_default_is_block_when_no_advice_applies():
    m = make_moment(dx=36, doing="standing")           # up close, nothing to follow
    d = two_stage_decide(FollowerLaya(), "chunli", m, [])
    assert d["action"] == DEFAULT_MOVE and d["category"] == "block" and d["rule"] == "default"
    assert d["follows_rule"] is True                   # following the default IS following the (empty) advice


def test_advice_line_changes_the_move():
    m = make_moment(dx=36, doing="standing")           # up close
    base = two_stage_decide(FollowerLaya(), "chunli", m, [])["action"]
    assert base == DEFAULT_MOVE
    # the owner's "always throw up close", in the two-stage menu's vocabulary (throw -> throw_F+hp)
    d = two_stage_decide(FollowerLaya(), "chunli", m, ["always use throw_F+hp up close"])
    assert d["action"] == "throw_F+hp" and d["category"] == "throw"
    assert d["rule"] == "hard" and d["follows_rule"] and d["rule_answers"] == ["throw_F+hp"]
    assert "always use throw_F+hp up close" in d["advice_text"]      # the advice reached the model's prompt


def _rows(move: str, dealt: int, taken: int, n: int = 25):
    """``n`` decisions of ``move`` up close while he attacks, each dealing / taking the given hp."""
    return [{"action": move, "range": "close", "opp_state": "attack", "opp_air": False,
             "dealt": dealt, "taken": taken} for _ in range(n)]


def test_retire_signal_removes_a_rule_and_the_move_changes_back():
    m = make_moment(dx=36, doing="attacking")          # up close, he attacks
    claim = {"kind": "use_more", "move": "cl.hp", "range": "close", "when": "attacking", "why": "it lands"}
    good = _rows("cl.hp", 30, 0) + _rows("block_high", 0, 20)        # cl.hp clearly better than her other move there
    reg, out = L.propose([], [claim], good, game=0, moves=MENU_MOVES)
    assert [o["state"] for o in out] == ["registered"], out
    line = L.render(claim)
    assert line in L.in_play(reg)
    picked = two_stage_decide(FollowerLaya(), "chunli", m, L.in_play(reg))
    assert picked["action"] == "cl.hp" and picked["follows_rule"]

    bad = _rows("cl.hp", 0, 30) + _rows("block_high", 30, 0)         # now cl.hp is clearly WORSE: evidence flips
    reg = L.review(reg, bad, game=1)
    assert line not in L.in_play(reg)                               # retired
    assert any(r["state"] == "retired" and r["line"] == line for r in reg)
    back = two_stage_decide(FollowerLaya(), "chunli", m, L.in_play(reg))
    assert back["action"] == DEFAULT_MOVE and back["rule"] == "default"
