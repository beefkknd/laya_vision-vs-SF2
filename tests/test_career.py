"""Pure helpers of the continuous career driver (scripts/play_career.py)."""
import importlib.util
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
_spec = importlib.util.spec_from_file_location(
    "play_career", os.path.join(os.path.dirname(__file__), "..", "scripts", "play_career.py"))
C = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(C)


def test_ladder_excludes_self_by_default():
    lad = C.ladder_for("chunli", None)
    assert "chunli" not in lad and "ryu" in lad and len(lad) == 7


def test_ladder_explicit_list_in_order():
    assert C.ladder_for("chunli", "honda, guile ,ryu") == ["honda", "guile", "ryu"]


def test_plan_covers_ladder_x_cap_for_each_lap():
    p = C.plan("chunli", ["honda", "guile"], block=4, cap=3, laps=2)
    assert len(p) == 2 * 2 * 3
    assert p[0] == (0, "honda", 0) and p[-1] == (1, "guile", 2)


def test_plan_endless_shows_one_lap():
    p = C.plan("chunli", ["honda"], block=4, cap=2, laps=0)
    assert [x[1:] for x in p] == [("honda", 0), ("honda", 1)]


def test_block_winrate_from_verdict(tmp_path):
    d = tmp_path / "round_00_honda"
    d.mkdir()
    (d / "verdict.json").write_text(json.dumps({"games": [
        {"won": 2, "lost": 0}, {"won": 0, "lost": 2}, {"won": 2, "lost": 1}, {"won": 1, "lost": 1}]}))
    # 2 of 4 games won (the 1-1 game is not a win)
    assert C.block_winrate(str(d)) == 0.5


def test_block_winrate_missing_is_none(tmp_path):
    assert C.block_winrate(str(tmp_path)) is None