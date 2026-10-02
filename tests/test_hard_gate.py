"""HARD GATE (owner rule 2026-10-02): the table and RAM only in testing, never in real (play) code.

Fixture repos under tests/fixtures/hard_gate/<case>/ each seed one violation into the same clean skeleton; the clean
case must pass, every seeded case must be caught with the exact rule. The real tree is checked last: known debt today
(xfail strict), so the day it turns clean the suite fails until the marker is removed.
"""
import os
import shutil
import subprocess
import sys

import pytest

from sf2.config import REPO
from sf2.hard_gate import ENTRY_POINTS, Policy, check, reach

FIX = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures", "hard_gate")
FIX_POLICY = Policy(entry_points=("scripts/play.py",))


def rules(case: str, policy: Policy = FIX_POLICY):
    return {(v.path, v.kind, v.rule) for v in check(os.path.join(FIX, case), policy)}


def test_clean_fixture_passes():
    assert check(os.path.join(FIX, "clean"), FIX_POLICY) == []


def test_clean_fixture_reaches_its_modules():
    parent, _ = reach(os.path.join(FIX, "clean"), FIX_POLICY)
    assert {"scripts/play.py", "scripts/_path.py", "sf2/system1/helper.py", "sf2/screen/reader.py",
            "sf2/emu/vs.py", "sf2/data/vs_sweep.py"} <= set(parent)


@pytest.mark.parametrize("case, expected", [
    ("table_import", ("sf2/system1/helper.py", "TABLE", "import:sf2/data/value_oracle.py")),
    ("table_literal", ("scripts/play.py", "TABLE", "table-literal")),
    ("oracle_kwarg", ("scripts/play.py", "TABLE", "table-keyword")),
    ("ram_note", ("scripts/play.py", "RAM", "symbol:note")),
    ("ram_key", ("sf2/system1/loop.py", "RAM", "ram-key:p1_x")),
    ("ram_key", ("sf2/system1/loop.py", "RAM", "ram-attr:rams")),
    ("replay_import", ("scripts/play.py", "TESTING", "import:sf2/system1/screen_replay.py")),
    ("lua_wram", ("mesen/sf2_bridge_screen.lua", "RAM", "lua-work-ram")),
    ("lua_wram", ("mesen/sf2_bridge_screen.lua", "RAM", "lua-mem-read")),
    ("screen_root", ("sf2/screen/hud.py", "RAM", "import:sf2/emu/ram.py")),
])
def test_seeded_violation_is_caught(case, expected):
    assert expected in rules(case)


def test_seeded_ram_note_does_not_flag_the_ram_free_half():
    assert {r for r in rules("ram_note") if r[2] == "symbol:actions"} == set()


def test_clean_turns_red_when_one_line_is_seeded(tmp_path):
    repo = str(tmp_path / "repo")
    shutil.copytree(os.path.join(FIX, "clean"), repo)
    assert check(repo, FIX_POLICY) == []
    with open(os.path.join(repo, "sf2", "system1", "helper.py"), "a") as f:
        f.write("\n\ndef peek(row):\n    return row.get('p2_state')\n")
    assert [(v.kind, v.rule) for v in check(repo, FIX_POLICY)] == [("RAM", "ram-key:p2_state")]


def test_missing_entry_point_fails_closed():
    found = check(os.path.join(FIX, "clean"), Policy(entry_points=("scripts/nope.py",), screen_lua=()))
    assert [(v.kind, v.rule) for v in found] == [("GATE", "missing-root")]


def test_cli_exit_codes():
    cli = os.path.join(REPO, "scripts", "hard_gate.py")
    ok = subprocess.run([sys.executable, cli, "--repo", os.path.join(FIX, "clean"), "--entry", "scripts/play.py"],
                        capture_output=True, text=True)
    bad = subprocess.run([sys.executable, cli, "--repo", os.path.join(FIX, "replay_import"), "--entry",
                          "scripts/play.py"], capture_output=True, text=True)
    assert (ok.returncode, bad.returncode) == (0, 1), ok.stdout + ok.stderr + bad.stdout + bad.stderr
    assert "screen_replay" in bad.stdout


def test_entry_points_include_the_screen_runner():
    assert "scripts/play_screen.py" in ENTRY_POINTS


@pytest.mark.xfail(strict=True, reason="known debt: play_screen uses the table + System1 RAM paths; must turn green "
                                       "when the Qwen loop runner (README M1/G1) replaces it")
def test_real_play_path_is_clean():
    found = check(REPO)
    assert found == [], "\n".join(map(str, found))
