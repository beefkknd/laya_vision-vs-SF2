"""REVIVE: a move dismissed quickly (retired on a small/unlucky sample) comes back when MORE history
shows it is clearly good. Owner 2026-10-03: early churn is aggressive, but good moves shouldn't stay
dead -- judge them over the accumulated history and revive the ones that earn it.
"""
from sf2.system2.lessons import REVIVE_EARLY, review


def _row(action, net):
    return {"range": "mid", "action": action, "dealt": net if net > 0 else 0, "taken": 0 if net > 0 else -net}


def _claim():
    return {"kind": "use_more", "move": "s.mk", "range": None, "when": None, "view": None}


def _retired(claim):
    return {"line": "use more s.mk", "claim": claim, "state": "retired", "since": 0, "evidence": {},
            "why": "dropped early", "qwen_why": ""}


def test_retired_move_revives_with_enough_good_history():
    # s.mk is clearly better than her baseline over plenty of tries -> revived from retired to registered
    rows = [_row("s.mk", 50) for _ in range(REVIVE_EARLY + 2)] + [_row("block_high", 0) for _ in range(REVIVE_EARLY + 2)]
    out = review([_retired(_claim())], rows, game=10, game_hp=None, games_wl=None)
    assert out[0]["state"] == "registered", out[0]["why"]
    assert "reviv" in out[0]["why"].lower()


def test_retired_move_stays_retired_without_enough_tries():
    # good but only a couple of tries -> not enough history to revive yet
    rows = [_row("s.mk", 50) for _ in range(2)] + [_row("block_high", 0) for _ in range(8)]
    out = review([_retired(_claim())], rows, game=10, game_hp=None, games_wl=None)
    assert out[0]["state"] == "retired"


def test_retired_move_stays_retired_when_history_says_its_bad():
    # plenty of tries but the move is NOT better than baseline -> stays dead
    rows = [_row("s.mk", -30) for _ in range(REVIVE_EARLY + 2)] + [_row("block_high", 0) for _ in range(REVIVE_EARLY + 2)]
    out = review([_retired(_claim())], rows, game=10, game_hp=None, games_wl=None)
    assert out[0]["state"] == "retired"
