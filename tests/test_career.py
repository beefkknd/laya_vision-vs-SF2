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


# --- the loss-streak forced-change trigger (owner: 3 straight round losses -> change strategy) ---
def test_trailing_losses_counts_only_the_streak():
    assert C.trailing_losses(["win", "loss", "loss", "loss"]) == 3
    assert C.trailing_losses(["loss", "loss", "win"]) == 0
    assert C.trailing_losses([]) == 0


def test_needs_intervention_on_three_straight():
    assert C.needs_intervention(["win", "loss", "loss", "loss"], window=3) is True
    assert C.needs_intervention(["loss", "loss", "win"], window=3) is False      # streak broken by a win
    assert C.needs_intervention(["loss", "loss"], window=3) is False             # not enough rounds yet
    assert C.needs_intervention(["loss", "loss"], window=2) is True              # window is configurable


def test_loss_window_adapts_by_stage():
    # EARLY stage (thin memory, < stage_rules in play): explore aggressively at 2 losses
    assert C.loss_window(0, early=2, late=3, stage_rules=2) == 2
    assert C.loss_window(1, early=2, late=3, stage_rules=2) == 2
    # LATER stage (an established playbook): be patient, 3 losses
    assert C.loss_window(2, early=2, late=3, stage_rules=2) == 3
    assert C.loss_window(5, early=2, late=3, stage_rules=2) == 3


def test_pick_forced_rule_skips_rules_already_in_play():
    first = C.EXPLORE_POOL[0]
    # with the first pool rule already in play, it must pick a DIFFERENT one
    got = C.pick_forced_rule([first], rotate=0)
    assert got and got != first and got in C.EXPLORE_POOL


def test_pick_forced_rule_none_when_pool_exhausted():
    assert C.pick_forced_rule(list(C.EXPLORE_POOL), rotate=0) is None


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
