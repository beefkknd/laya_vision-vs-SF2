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


# The loss-streak forced change moved INTO the live per-round policy: its behaviour is now tested in
# tests/test_short_memory.py (loss_streak + swap) and the pool in tests/test_explore_pool.py. The career
# driver no longer injects rules between blocks, so trailing_losses / needs_intervention / loss_window /
# pick_forced_rule / EXPLORE_POOL were removed from play_career.py with their tests.
def test_round_results_reads_trace(tmp_path):
    d = tmp_path / "round_00_ryu"
    d.mkdir()
    (d / "trace.jsonl").write_text(
        '{"event":"seed","opp":"ryu"}\n'
        '{"event":"round","result":"loss"}\n'
        '{"event":"round","result":"win"}\n'
        '{"event":"qwen","added":[]}\n')
    assert C.round_results(str(d)) == ["loss", "win"]

def test_resolve_playbook_folder_name_and_json_file(tmp_path):
    import os
    folder, reg = C.resolve_playbook("foo")
    assert folder.endswith(os.path.join("playbooks", "foo"))
    assert reg.endswith(os.path.join("foo", "playbook.json"))
    f = str(tmp_path / "my.json")
    folder2, reg2 = C.resolve_playbook(f)
    assert reg2 == os.path.abspath(f) and folder2 == os.path.dirname(os.path.abspath(f))
