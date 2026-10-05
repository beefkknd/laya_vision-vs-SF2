"""screen_evidence must be CHARACTER-GENERAL: a decision row's `kind` (movement/defense/attack) is read from the
PLAYED character's own categories, not Chun-Li's. Regression (owner 2026-10-05, found training ryu): round_evidence
crashed with `ValueError: move 'hadoken_hp' is in no category` because kind_of called action_menu.category_of, which
only knows Chun-Li's moves -> every ryu/ken/... special, combo or char-specific move killed the evidence pass (and
with it the whole learning run: no table.json was ever saved)."""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from sf2.system2 import screen_evidence as SE                             # noqa: E402


def dec(action, doing="standing"):
    return {"action": action, "k": 10, "situation": ["mid", doing, "half", "half"],
            "moment": {"my_life": 100, "his_life": 100, "doing": doing, "his_air": False,
                       "his_label": "stand", "side": "left", "dx": 40, "fireball": False}}


def test_ryu_special_and_combo_do_not_crash_and_are_attack():
    rows, _ = SE.round_evidence(0, "ryu", "ken", [dec("hadoken_hp"), dec("c.mk_xx_shoryuken"), dec("block_high")])
    kinds = {r["action"]: r["kind"] for r in rows}
    assert kinds["hadoken_hp"] == "attack"          # ryu special -> attack (the move that used to crash)
    assert kinds["c.mk_xx_shoryuken"] == "attack"   # ryu combo -> attack
    assert kinds["block_high"] == "defense"         # block -> defense


def test_ryu_movement_is_movement_kind():
    rows, _ = SE.round_evidence(0, "ryu", "ken", [dec("walk_back")])
    assert rows[0]["kind"] == "movement"


def test_chunli_kinds_unchanged_regression():
    rows, _ = SE.round_evidence(0, "chunli", "ryu", [dec("lightning_legs"), dec("block_low"), dec("walk_forward")])
    k = {r["action"]: r["kind"] for r in rows}
    assert k["lightning_legs"] == "attack" and k["block_low"] == "defense" and k["walk_forward"] == "movement"
