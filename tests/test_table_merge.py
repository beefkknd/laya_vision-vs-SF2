"""Merging independently-collected value tables (parallel data-collection workers) into ONE. Welford stats
[n, sum, sumsq] are additive, so cells (base and split-child) and shadow tallies sum elementwise and depth unions.
Pure: inputs are never mutated."""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from sf2.system1 import value_table as VT  # noqa: E402

WHEN = "close|standing|0"


def _row(action, net):
    return {"range": "close", "opp_state": "stand", "opp_air": False, "opp_shot": False,
            "his_label": "stand", "action": action, "dealt": max(net, 0), "taken": max(-net, 0)}


def test_merge_sums_welford_cells_and_leaves_inputs_unchanged():
    a, b = VT.blank(), VT.blank()
    for _ in range(5):
        a = VT.credit(a, [_row("cl.mk", 10)])
    for _ in range(3):
        b = VT.credit(b, [_row("cl.mk", -4)])
    m = VT.merge([a, b])
    s = m["cells"][WHEN]["cl.mk"]
    assert s[0] == 8                                   # 5 + 3 tries
    assert s[1] == 5 * 10 + 3 * -4                     # summed net
    assert abs(s[2] - (5 * 100 + 3 * 16)) < 1e-9       # summed sumsq
    assert a["cells"][WHEN]["cl.mk"][0] == 5           # input a untouched
    assert b["cells"][WHEN]["cl.mk"][0] == 3           # input b untouched


def test_merge_sums_shadow_tallies():
    a, b = VT.blank(), VT.blank()
    for _ in range(4):
        a = VT.credit(a, [_row("cl.lk", 6)])
    for _ in range(2):
        b = VT.credit(b, [_row("cl.lk", 6)])
    m = VT.merge([a, b])
    assert m["shadow"][WHEN]["stand"]["cl.lk"][0] == 6


def test_merge_unions_splits_and_keeps_child_and_base_cells():
    a = VT.blank()
    a["depth"][WHEN] = "his_label"                      # worker a split this cell
    a["cells"][WHEN + "|jump"] = {"cl.mk": [4, 20.0, 100.0]}
    b = VT.blank()
    for _ in range(3):                                 # worker b never split; its data is in the base cell
        b = VT.credit(b, [_row("cl.mk", 2)])
    m = VT.merge([a, b])
    assert m["depth"].get(WHEN) == "his_label"          # split preserved from a
    assert m["cells"][WHEN + "|jump"]["cl.mk"][0] == 4  # a's child cell kept
    assert m["cells"][WHEN]["cl.mk"][0] == 3            # b's base data kept (fallback recovers it under the split)


def test_merge_of_empty_list_is_blank():
    assert VT.merge([]) == VT.blank()


def test_merge_single_table_is_a_copy_not_the_same_object():
    a = VT.blank()
    a = VT.credit(a, [_row("cl.mk", 7)])
    m = VT.merge([a])
    assert m["cells"][WHEN]["cl.mk"] == [1, 7.0, 49.0]
    m["cells"][WHEN]["cl.mk"][0] = 999                  # mutating the merge must not touch the input
    assert a["cells"][WHEN]["cl.mk"][0] == 1
