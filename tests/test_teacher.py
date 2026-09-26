"""The scripted teacher returns a sensible distribution over the action set."""
from sf2.actions import ACTIONS
from sf2.env import Context
from sf2.ram import Fighters
from sf2.teacher import teacher_policy


def fighters(dx, **kw):
    return Fighters(my_hp=176, opp_hp=176, my_x=200, opp_x=200 + dx, my_y=192, opp_y=192, **kw)


def ctx(**kw):
    c = dict(my_air=False, opp_air=False, dx_trend=0, frames_since_hit=1000)
    c.update(kw)
    return Context(**c)


def test_a_distribution_over_the_actions_whoever_she_is():
    for me in ("chunli", "ryu"):
        for dx in (40, 100, 160):
            for c in (ctx(), ctx(my_air=True), ctx(opp_air=True), ctx(frames_since_hit=5)):
                p = teacher_policy(fighters(dx), c, me)
                assert list(p) == ACTIONS and abs(sum(p.values()) - 1) < 1e-9 and min(p.values()) > 0


def top(p):
    return max(p, key=p.get)


def soft(p):
    """A sensible soft target: a clear favourite, but not one-hot."""
    return 0.5 <= max(p.values()) < 0.95 and sorted(p.values())[-2] >= 0.05


def test_rule5_jump_in_from_mid_range_when_he_is_not_attacking():
    p = teacher_policy(fighters(100), ctx(), "chunli")
    assert top(p) == "jump_forward" and soft(p)
    assert top(teacher_policy(fighters(100, opp_state=0x0A), ctx(), "chunli")) != "jump_forward"


def test_in_the_air_kick_only_near_the_top_of_the_jump():
    """On the ROM a jump-in kick pressed near the apex (>= 88 px up) lands 65-75%; pressed on the way up, ~10%."""
    def air(height):
        return Fighters(my_hp=176, opp_hp=176, my_x=200, opp_x=260, my_y=192 - height, opp_y=192)
    for h in (17, 45, 67, 83):
        p = teacher_policy(air(h), ctx(my_air=True), "chunli")
        assert top(p) == "idle" and soft(p)
    for h in (92, 96):
        p = teacher_policy(air(h), ctx(my_air=True), "chunli")
        assert top(p) == "hk" and soft(p)
