"""The value-table core (sf2/system1/value_table.py, Stage 2a): a contextual bandit that credits per-decision
net hp per (when, action) and picks with decaying ε-greedy exploration. Pure, seeded, deterministic."""
import random

from sf2.system1 import value_table as VT


def _row(action, net, rng="mid", doing="stand", fireball=False, his_label="stand"):
    # a decision row shaped like screen_evidence's drows (what credit folds)
    return {"range": rng, "opp_state": doing, "opp_air": doing == "jump", "opp_shot": fireball,
            "his_label": his_label, "action": action, "dealt": max(net, 0), "taken": max(-net, 0)}


WHEN = "mid|standing|0"          # base_key("mid", opp_doing(stand)=="standing", fireball 0)


def test_credit_folds_net_hp_into_when_action_welford():
    t = VT.credit(VT.blank(), [_row("s.mk", 10), _row("s.mk", 20), _row("s.mk", 30)])
    s = t["cells"][WHEN]["s.mk"]
    assert s[0] == 3 and s[1] == 60.0                       # n, sum
    assert VT.mean(s) == 20.0


def test_credit_is_immutable():
    t0 = VT.blank()
    t1 = VT.credit(t0, [_row("s.mk", 10)])
    assert t0["cells"] == {} and VT.count(t1["cells"][WHEN], "s.mk") == 1


def test_row_when_keys_base_by_range_doing_fireball():
    assert VT.row_when(_row("x", 0, rng="close", doing="jump"), {}) == "close|jumping|0"
    assert VT.row_when(_row("x", 0, fireball=True), {}) == "mid|standing|1"
    # a split cell (depth) appends his_label
    assert VT.row_when(_row("x", 0, his_label="hit"), {"mid|standing|0": "his_label"}) == "mid|standing|0|hit"


def test_choose_explores_under_sampled_actions_first():
    t = VT.blank()
    for _ in range(VT.MIN_TRIES + 5):                        # A is well covered; B, C unsampled
        t = VT.credit(t, [_row("A", 10)])
    rng = random.Random(1)
    picks = [VT.choose(t, WHEN, ["A", "B", "C"], rng, eps0=1.0) for _ in range(300)]
    explored = [a for a, ex in picks if ex]
    assert explored and all(a in ("B", "C") for a in explored)   # an EXPLORE pick targets the under-covered arms
    assert any(a == "A" for a, ex in picks if not ex)            # an EXPLOIT pick takes the best (A)


def test_choose_exploits_the_best_mean_when_not_exploring():
    t = VT.blank()
    for _ in range(5):
        t = VT.credit(t, [_row("A", 10)])
        t = VT.credit(t, [_row("B", -5)])
    a, explored = VT.choose(t, WHEN, ["A", "B"], random.Random(0), eps0=0.0)
    assert a == "A" and explored is False


def test_bandit_converges_to_the_best_arm():
    true = {"A": 10.0, "B": 0.0, "C": -10.0}
    t, rng = VT.blank(), random.Random(0)
    for _ in range(300):
        a, _ = VT.choose(t, WHEN, list(true), rng)
        t = VT.credit(t, [_row(a, true[a])])
    cell = t["cells"][WHEN]
    assert max(true, key=lambda a: VT.mean(cell.get(a))) == "A"       # A is the learned best
    assert VT.count(cell, "A") > VT.count(cell, "B") + VT.count(cell, "C")   # exploitation concentrated on A
    assert VT.choose(t, WHEN, list(true), rng, eps0=0.0) == ("A", False)


def test_eps_decays_as_the_cell_fills():
    empty = VT.blank()
    filled = VT.blank()
    for _ in range(200):
        filled = VT.credit(filled, [_row("A", 1)])
    # with a full cell, exploration is rare; with an empty cell it is frequent (sample many draws)
    rng = random.Random(3)
    ex_empty = sum(VT.choose(empty, WHEN, ["A", "B", "C"], rng)[1] for _ in range(400))
    ex_full = sum(VT.choose(filled, WHEN, ["A", "B", "C"], rng)[1] for _ in range(400))
    assert ex_empty > ex_full


# --------------------------------------------------------------------------- 2b: his_label split
def _rows(n, action, net, his_label, doing="stand"):
    return [_row(action, net, doing=doing, his_label=his_label) for _ in range(n)]


def test_cell_splits_by_his_label_when_labels_prefer_different_actions():
    # base (mid,standing): when his_label=='stand' s.mk is best (+10, throw -10); when 'walk' throw is best (+10).
    # The coarse cell would average throw away; the split preserves it. (min_tries=5 to keep the fixture small.)
    t = VT.blank()
    rows = (_rows(6, "s.mk", 10, "stand") + _rows(6, "throw_F+hp", -10, "stand")
            + _rows(6, "throw_F+hp", 10, "walk"))
    t = VT.credit(t, rows, min_tries=5)
    assert t["depth"].get("mid|standing|0") == "his_label"
    assert "mid|standing|0|stand" in t["cells"] and "mid|standing|0|walk" in t["cells"]
    # after the split, a 'walk' decision keys into the split cell and sees throw as best there
    wk = VT.row_when(_row("x", 0, his_label="walk"), t["depth"])
    assert wk == "mid|standing|0|walk"
    assert VT.choose(t, wk, ["s.mk", "throw_F+hp"], random.Random(0), eps0=0.0)[0] == "throw_F+hp"


def test_no_split_when_labels_agree():
    t = VT.credit(VT.blank(), _rows(6, "s.mk", 10, "stand") + _rows(6, "s.mk", 10, "walk"), min_tries=5)
    assert t["depth"] == {}                                  # both labels prefer s.mk -> nothing to split


def test_no_split_when_under_covered():
    # labels differ but neither reaches min_tries -> no confident split
    t = VT.credit(VT.blank(), _rows(3, "s.mk", 10, "stand") + _rows(3, "throw_F+hp", 10, "walk"), min_tries=5)
    assert t["depth"] == {}
