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
