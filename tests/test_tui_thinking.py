"""BUG: the 'qwen is thinking...' animation disappeared after reflection moved to per-ROUND.

_infer_thinking was written for per-GAME reflection (only the final round of a game is a reflect point),
so with per-round reflection it returns False between rounds and the banner never shows. Reproduce it,
then it must flag thinking whenever the newest round has ended and the run is not finished.
"""
from sf2.eval.tui_model import _infer_thinking


def test_thinking_between_rounds_with_per_round_reflection():
    # games=1, rounds=3. Round 0 just ended, round 1 not started: Qwen reflects AFTER EACH ROUND now,
    # so this gap IS a thinking moment (it was False under the old per-game assumption).
    assert _infer_thinking(1, 3, [(0, 0, "g00_r0")], [{"event": "round", "game": 0, "round": 0}]) is True
    assert _infer_thinking(1, 3, [(0, 1, "g00_r1")], [{"event": "round", "game": 1 - 1, "round": 1}]) is True


def test_not_thinking_before_first_round_ends():
    # the round dir exists but no round event yet (still playing) -> not thinking
    assert _infer_thinking(1, 3, [(0, 0, "g00_r0")], []) is False


def test_not_thinking_when_whole_run_done():
    # final round of the final game ended -> done, not thinking
    assert _infer_thinking(1, 3, [(0, 2, "g00_r2")], [{"event": "round", "game": 0, "round": 2}]) is False
    assert _infer_thinking(2, 2, [(1, 1, "g01_r1")], [{"event": "round", "game": 1, "round": 1}]) is False


def test_thinking_at_game_boundary_still_works():
    # end of game 0's last round, more games to come -> thinking
    assert _infer_thinking(2, 2, [(0, 1, "g00_r1")], [{"event": "round", "game": 0, "round": 1}]) is True
