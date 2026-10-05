"""The value-table deciders must be CHARACTER-GENERAL: a decision dict's `category` is read from the PLAYED
character's own menu, not Chun-Li's. Regression (owner 2026-10-05, same species as 9ecca86/screen_evidence): the
`--policy table` and `--policy hybrid` deciders tagged each decision with `action_menu.category_of(action)`, which
only knows Chun-Li's moves -> for a non-Chun-Li character any char-specific move the table chose (ryu's hadoken_hp,
c.mk_xx_shoryuken, ...) crashed with `ValueError: move %r is in no category`, killing the whole table/hybrid run.
The quorum decider was already char-aware (decider._category); these two paths lagged."""
import os
import random
import sys

sys.path.insert(0, os.path.dirname(__file__))
from looptools import make_moment                                   # noqa: E402

from sf2.system1 import value_table as VT                           # noqa: E402


def _row(action, net, rng="far"):
    return {"range": rng, "opp_state": "stand", "opp_air": False, "opp_shot": False,
            "his_label": "stand", "action": action, "dealt": max(net, 0), "taken": max(-net, 0)}


# ------------------------------------------------------------------ --policy table (VT.decider)
def test_table_decider_tags_a_ryu_special_without_crashing():
    # credit a ryu special heavily at far, then exploit (eps0=0) -> the decider MUST pick it and tag its category.
    t = VT.blank()
    for _ in range(VT.MIN_TRIES):
        t = VT.credit(t, [_row("hadoken_hp", 15)])
    d = VT.decider(t, "ryu", random.Random(0), eps0=0.0)(make_moment(dx=150, doing="standing"))
    assert d["action"] == "hadoken_hp"          # the move that used to crash category_of
    assert d["category"] == "special"           # read from ryu's own menu, not Chun-Li's


def test_table_decider_never_crashes_across_ryu_moves_and_ranges():
    # exploration must also survive: over many moments/ranges every returned category is a valid ryu category.
    cats_ryu = {"move", "block", "punch", "kick", "throw", "special", "combo"}
    rng = random.Random(7)
    t = VT.blank()
    for dx in (36, 90, 150):
        for _ in range(60):
            d = VT.decider(t, "ryu", rng)(make_moment(dx=dx, doing="standing"))
            assert d["category"] in cats_ryu
            t = VT.credit(t, [_row(d["action"], 1, "close" if dx < 60 else "mid" if dx < 120 else "far")])


# ------------------------------------------------------------------ --policy hybrid (VT.hybrid_decider)
def _laya(_m):
    return {"action": "block_high", "category": "block", "source": "laya"}


def test_hybrid_override_tags_a_ryu_special_without_crashing():
    # credit a ryu special to the override threshold -> hybrid overrides laya with it and tags its category.
    t = VT.blank()
    for _ in range(VT.OVERRIDE_TRIES):
        t = VT.credit(t, [_row("c.mk_xx_shoryuken", 15, "close")])
    d = VT.hybrid_decider(t, "ryu", random.Random(0), _laya, explore=0.0)(make_moment(dx=36, doing="standing"))
    assert d["action"] == "c.mk_xx_shoryuken" and d["source"] == "table"
    assert d["category"] == "combo"             # ryu combo, read from ryu's own menu


def test_hybrid_table_explore_never_crashes_on_ryu_moves():
    # the table-explore branch (its own category tag) must also survive a char-specific pick.
    cats_ryu = {"move", "block", "punch", "kick", "throw", "special", "combo"}
    decide = VT.hybrid_decider(VT.blank(), "ryu", random.Random(1), _laya, explore=1.0)
    srcs = set()
    for _ in range(60):
        d = decide(make_moment(dx=150, doing="standing"))
        srcs.add(d["source"])
        assert d["category"] in cats_ryu
    assert "table-explore" in srcs              # it really exercised the explore branch


def test_chunli_table_decider_category_unchanged_regression():
    t = VT.blank()
    for _ in range(VT.MIN_TRIES):
        t = VT.credit(t, [_row("throw_F+hp", 15, "close")])
    d = VT.decider(t, "chunli", random.Random(0), eps0=0.0)(make_moment(dx=36, doing="standing"))
    assert d["action"] == "throw_F+hp" and d["category"] == "throw"
