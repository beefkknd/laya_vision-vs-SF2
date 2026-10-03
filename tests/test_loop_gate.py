"""M1 (e): the hard gate is CLEAN for the Qwen-in-loop runner's own import closure - it imports no table and no RAM
play path. The runner is also registered in ENTRY_POINTS.

Seen RED: before the reviewed ALLOW entries for advice.opp_doing, the runner's closure reported 2 RAM violations
(sf2/system1/advice.py opp_air / opp_state); confirmed by removing those ALLOW keys (the asserted list is non-empty).
If the runner ever imports a TABLE or RAM module, new violations appear and this fails.
"""
import os

from sf2.config import REPO
from sf2.hard_gate import ENTRY_POINTS, Policy, check

RUNNER = "sf2/system1/loop_runner.py"


def test_runner_is_registered_as_an_entry_point():
    assert RUNNER in ENTRY_POINTS


def test_runner_import_closure_is_clean():
    found = check(REPO, Policy(entry_points=(RUNNER,)))             # extra_roots (sf2/screen) scanned too; must be clean
    assert found == [], "\n".join(str(v) for v in found)


def test_runner_does_not_import_the_table_or_ram_play_paths():
    # a sharper statement of the same: no violation of kind TABLE or RAM anywhere in the runner's reach
    found = check(REPO, Policy(entry_points=(RUNNER,)))
    assert not [v for v in found if v.kind in ("TABLE", "RAM")]
