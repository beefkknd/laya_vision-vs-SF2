"""M1 (close the loop): seed -> play -> screen evidence -> Qwen -> short memory -> next game, end to end, with a MOCK
Qwen and a follower text laya. No network, no emulator. A canned Qwen claim written after game 0 must be in the short
memory by game 1 and must change what text laya plays there.

Seen RED: if the updated registry did not feed the next game's ``lines`` (the loop not closed), every game would play
the seed default and game-1's move would stay block_high; confirmed by pinning ``lines`` to the seed in run_loop (the
game-1 assert then fails).
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from looptools import FollowerLaya, load_driver, make_moment        # noqa: E402

from sf2.system1.loop_runner import two_stage_decide                # noqa: E402

ME, OPP = "chunli", "honda"


class FakeEmu:
    def new_round(self):
        return self


def fake_play(emu, cat_advisor, move_advisor, me, opp, state, state_id, delay, lines, out, reader=None):
    """One decision, up close while he attacks, written in loop_runner's decision-record shape (so screen_evidence
    reads it like a real round). The move is the REAL two_stage_decide, so the advice genuinely drives it."""
    os.makedirs(out, exist_ok=True)
    m = make_moment(dx=36, doing="attacking", my_bar=150 / 176.0, his_bar=150 / 176.0)
    d = two_stage_decide(cat_advisor, move_advisor, me, m, lines)
    rec = dict(d, i=0, k=12, k_prev=8, situation=["close", "attacking", "half", "half"],
               moment={"my_life": 150, "his_life": 150, "doing": "attacking", "his_air": False,
                       "side": "left", "dx": 36, "fireball": False})
    with open(os.path.join(out, "decisions.jsonl"), "w") as f:
        f.write(json.dumps(rec) + "\n")
    return {"decisions": 1}


def fake_score(round_dir):
    return {"result": "loss", "dealt": 20, "taken": 40, "hp": -20, "my_life_end": 100, "opp_life_end": 130}


class MockQwen:
    """Returns a canned ANSWER claim after game 0, nothing after. Records how it was called."""

    def __init__(self):
        self.calls = 0

    def __call__(self, messages, task):
        self.calls += 1
        if self.calls == 1:
            return json.dumps({"answer": {"kind": "always", "move": "cl.hp", "range": "close",
                                          "when": "attacking", "why": "it lands"}, "stop": None})
        return json.dumps({"answer": None, "stop": None})


def _actions(out):
    """Each game's played move, read back from the trace (one decision per round here)."""
    by_game = {}
    with open(os.path.join(out, "trace.jsonl")) as f:
        for line in f:
            e = json.loads(line)
            if e.get("event") == "decision":
                by_game.setdefault(e["game"], e["action"])
    return by_game


def test_qwen_claim_closes_the_loop_and_changes_the_next_game(tmp_path):
    driver = load_driver()
    qwen = MockQwen()
    out = str(tmp_path / "run")
    verdict = driver.run_loop(OPP, FollowerLaya(), FollowerLaya(), qwen, games=2, rounds=1, seed_lines=[], out=out,
                              play_round_fn=fake_play, state=b"x", state_id={"path": "p", "sha256": "0"},
                              emu=FakeEmu(), score_fn=fake_score, seed_rng=1)
    acts = _actions(out)
    assert acts[0] == "block_high", acts                           # game 0: empty seed -> the default
    assert acts[1] == "cl.hp", acts                                # game 1: follows the claim Qwen wrote after game 0
    line = "always cl.hp up close when he attacks"
    assert line in verdict["in_play_end"]
    # the trace records the short-memory diff: the line was ADDED by the game-0 Qwen step
    added = []
    with open(os.path.join(out, "trace.jsonl")) as f:
        for l in f:
            e = json.loads(l)
            if e.get("event") == "qwen":
                added += e["added"]
    assert line in added


def test_seed_lines_play_from_game_zero(tmp_path):
    """With a seed line in force, game 0 already follows it (seed -> short memory -> play)."""
    driver = load_driver()
    out = str(tmp_path / "run2")
    seed_line = {"claim": {"kind": "always", "move": "cl.hp", "range": "close", "when": "attacking", "view": "book"},
                 "line": "always cl.hp up close when he attacks", "state": "verified", "why": "seed",
                 "since": -1, "evidence": {"source": "web"}}
    driver.run_loop(OPP, FollowerLaya(), FollowerLaya(), MockQwen(), games=1, rounds=1, seed_lines=[seed_line],
                    out=out, play_round_fn=fake_play, state=b"x", state_id={"path": "p", "sha256": "0"},
                    emu=FakeEmu(), score_fn=fake_score, seed_rng=1)
    assert _actions(out)[0] == "cl.hp"
