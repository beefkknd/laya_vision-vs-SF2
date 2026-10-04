"""n-step / lambda-discounted credit (backlog B1). Diagnostic §0a showed the dense net-hp signal already
telescopes to the round result (r=1.0) but per-decision credit is MYOPIC (r(immediate, 5-step)=0.50): a
setup move's own window looks weak even when it enables a big hit. n-step credit fixes that by crediting
decision t with the discounted sum of net over the next `horizon` decisions -- turning the one-step-reward
bandit into a truncated-return (Q-value) estimator. horizon=0 (default) == the current one-step behaviour."""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from sf2.system1 import value_table as VT  # noqa: E402

WHEN = "close|standing|0"


def _row(action, net):
    return {"range": "close", "opp_state": "stand", "opp_air": False, "opp_shot": False,
            "his_label": "stand", "action": action, "dealt": max(net, 0), "taken": max(-net, 0)}


def test_immediate_credit_under_values_a_setup():
    # a setup that nets 0 now but ENABLES a +40 hit next: one-step credit gives the setup 0 (the myopia)
    t = VT.credit(VT.blank(), [_row("setup", 0), _row("hit", 40)])      # default horizon=0 == today
    assert VT.mean(t["cells"][WHEN]["setup"]) == 0.0
    assert VT.mean(t["cells"][WHEN]["hit"]) == 40.0


def test_nstep_credit_gives_the_setup_its_downstream_payoff():
    t = VT.credit(VT.blank(), [_row("setup", 0), _row("hit", 40)], horizon=5, gamma=1.0)
    assert VT.mean(t["cells"][WHEN]["setup"]) == 40.0                   # now sees the +40 it set up
    assert VT.mean(t["cells"][WHEN]["hit"]) == 40.0


def test_nstep_discounts_the_future_with_gamma():
    t = VT.credit(VT.blank(), [_row("setup", 0), _row("hit", 40)], horizon=5, gamma=0.5)
    assert VT.mean(t["cells"][WHEN]["setup"]) == 20.0                   # 0 + 0.5*40


def test_horizon_is_clamped_to_round_end():
    # last decision has no future -> its n-step return is just its own net
    t = VT.credit(VT.blank(), [_row("a", 5), _row("b", 7)], horizon=5, gamma=1.0)
    assert VT.mean(t["cells"][WHEN]["b"]) == 7.0                        # a=5+7=12, b=7


def test_horizon_zero_is_identical_to_the_default_one_step():
    rows = [_row("a", 10), _row("b", -4), _row("a", 6)]
    assert VT.credit(VT.blank(), rows) == VT.credit(VT.blank(), rows, horizon=0, gamma=1.0)
