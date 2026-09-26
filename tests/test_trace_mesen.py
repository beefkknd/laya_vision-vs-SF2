"""The trace replay backend: recorded (or hand-written) RAM rows drive FightEnv without Mesen."""
import pytest

from sf2.env import FightEnv
from sf2.ram import Var
from trace_mesen import TraceMesen, make_trace

MAP = [Var("my_hp", 0x10, 2, True), Var("opp_hp", 0x20, 2, True), Var("my_x", 0x30, 2, False),
       Var("opp_x", 0x40, 2, False), Var("my_y", 0x50, 2, False), Var("opp_y", 0x60, 2, False)]


def _rows(n, **over):
    base = {"my_hp": 176, "opp_hp": 176, "my_x": 200, "opp_x": 384, "my_y": 192, "opp_y": 192}
    return [dict(base, **{k: v(i) if callable(v) else v for k, v in over.items()}) for i in range(n + 1)]


def test_synthetic_trace_round_trips_through_the_env():
    rows = _rows(10, my_x=lambda i: 200 + 3 * i, my_hp=lambda i: 176 - i)
    env = FightEnv(TraceMesen(make_trace(MAP, rows)), MAP, b"")
    env.reset()
    fs = env.run_frames([[]] * 10)
    assert [f.my_x for f in fs] == [200 + 3 * i for i in range(1, 11)]
    assert fs[-1].my_hp == 166 and fs[-1].opp_x == 384  # 2-byte value above 255 survives


def test_signed_values_decode_negative():
    rows = _rows(2, opp_hp=lambda i: [176, 3, -1][i])
    env = FightEnv(TraceMesen(make_trace(MAP, rows)), MAP, b"")
    env.reset()
    assert env.run_frames([[]] * 2)[-1].opp_hp == -1


def test_an_input_the_trace_did_not_record_fails_naming_the_frame():
    rows = _rows(4)
    inputs = [[], ["right"], ["right"], ["right"], ["right"]]
    env = FightEnv(TraceMesen(make_trace(MAP, rows, inputs)), MAP, b"")
    env.reset()
    env.run_frames([["right"]] * 2)
    with pytest.raises(AssertionError, match=r"frame 3: env pressed \['left'\], trace recorded \['right'\]"):
        env.run_frames([["left"]])


def test_running_past_the_end_of_the_trace_fails():
    env = FightEnv(TraceMesen(make_trace(MAP, _rows(3))), MAP, b"")
    env.reset()
    with pytest.raises(AssertionError, match="trace ends at frame 3"):
        env.run_frames([[]] * 4)


def test_load_state_rewinds():
    rows = _rows(3, my_x=lambda i: 200 + i)
    env = FightEnv(TraceMesen(make_trace(MAP, rows)), MAP, b"")
    env.reset()
    env.run_frames([[]] * 3)
    env.reset()
    assert env.f.my_x == 200 and env.run_frames([[]])[-1].my_x == 201
