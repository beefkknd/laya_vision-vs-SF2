"""Stage 3b A/B report core (scripts/ab_table.py): read per-round outcomes from a session, compute learning curves
and a last-N tail comparison with a bootstrap difference CI. Pure, deterministic."""
import importlib.util
import json
import os

_spec = importlib.util.spec_from_file_location(
    "ab_table", os.path.join(os.path.dirname(__file__), "..", "scripts", "ab_table.py"))
AB = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(AB)


def _rounds(seq):  # 'WWLL' + hp per round
    return [{"hp": 20 if c == "W" else -20, "result": "win" if c == "W" else "loss"} for c in seq]


def test_cum_winrate_climbs():
    assert AB.cum_winrate(_rounds("LWWW"))[-1] == 3 / 4
    assert AB.cum_winrate(_rounds("LLLL"))[-1] == 0.0


def test_compare_calls_a_clear_table_win_HELPS():
    table = _rounds("W" * 40)          # table wins its tail
    rules = _rounds("L" * 40)          # rules loses its tail
    c = AB.compare(table, rules, last=40)
    assert c["table"]["winrate"] == 1.0 and c["rules"]["winrate"] == 0.0
    assert c["hp_diff"]["mean"] == 40.0            # +20 vs -20
    assert c["hp_diff"]["verdict"] == "HELPS" and c["hp_diff"]["ci95"][0] > 0
    assert c["winrate_diff"] == 1.0


def test_compare_inconclusive_when_equal():
    both = _rounds("WL" * 20)
    c = AB.compare(both, list(both), last=40)
    assert c["hp_diff"]["verdict"] == "NOT SHOWN"   # identical tails -> CI straddles 0


def test_last_tail_only():
    # early losses, late wins: tail of 10 is all wins even though overall win-rate is lower
    table = _rounds("L" * 30 + "W" * 10)
    c = AB.compare(table, _rounds("L" * 40), last=10)
    assert c["table"]["winrate"] == 1.0 and c["table"]["rounds"] == 40 and c["table"]["tail"] == 10


def test_arm_rounds_reads_round_events(tmp_path):
    d = tmp_path / "round_00_ryu"
    d.mkdir()
    (d / "trace.jsonl").write_text(
        json.dumps({"event": "seed"}) + "\n"
        + json.dumps({"event": "round", "result": "loss", "hp": -30}) + "\n"
        + json.dumps({"event": "round", "result": "win", "hp": 15}) + "\n")
    got = AB.arm_rounds(str(tmp_path))
    assert got == [{"hp": -30, "result": "loss"}, {"hp": 15, "result": "win"}]
