"""END-TO-END DONE gate for the simple early-game policy (docs/plan_simple_learning.md).

Replays the playbooks/chun/round_03_ryu deadlock through the REAL driver (mock Qwen + follower text laya,
no emulator/network): a FULL short memory of trying lines, carried in from a prior block, and she loses
every round. The old code refused every Coach claim ('already 2 claims in test') and the memory stayed
frozen for 12 rounds. The simple policy MUST change the short memory within SWAP_AFTER losing rounds.

Seen RED: against the pre-wiring driver (lessons.propose/review on the live path) the carried trying
lines lock the two test slots forever, so `added` is empty on every reflection and in_play never changes.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from looptools import FollowerLaya, load_driver        # noqa: E402

from sf2.system1.advice import char_menu_moves          # noqa: E402
from sf2.system1.loop_runner import two_stage_decide    # noqa: E402
from sf2.system2 import short_memory as SM              # noqa: E402

ME, OPP = "chunli", "ryu"
MOVES = set(char_menu_moves(ME))


class FakeEmu:
    def new_round(self):
        return self


def fake_play(emu, cat_advisor, move_advisor, me, opp, state, state_id, delay, lines, out, reader=None):
    os.makedirs(out, exist_ok=True)
    from looptools import make_moment
    m = make_moment(dx=36, doing="standing", my_bar=120 / 176.0, his_bar=150 / 176.0)
    d = two_stage_decide(cat_advisor, move_advisor, me, m, lines)
    rec = dict(d, i=0, k=12, k_prev=8, situation=["mid", "standing", "half", "half"],
               moment={"my_life": 120, "his_life": 150, "doing": "standing", "his_air": False,
                       "side": "left", "dx": 36, "fireball": False})
    with open(os.path.join(out, "decisions.jsonl"), "w") as f:
        f.write(json.dumps(rec) + "\n")
    return {"decisions": 1}


def fake_loss(round_dir):
    return {"result": "loss", "dealt": 20, "taken": 60, "hp": -40, "my_life_end": 40, "opp_life_end": 150}


def fake_win(round_dir):
    return {"result": "win", "dealt": 90, "taken": 20, "hp": 70, "my_life_end": 150, "opp_life_end": 40}


class FreshCoachEachRound:
    """Like ryu's Coach in the real trace: proposes a different grounded offensive move every reflection."""
    PICKS = ["s.hk", "lightning_legs", "s.mp", "spinning_bird_kick", "walk_forward"]

    def __init__(self):
        self.n = 0

    def __call__(self, messages, task):
        if task.startswith("scout"):
            return "She blocked most of the round and lost; her offense barely landed."
        move = self.PICKS[self.n % len(self.PICKS)]
        self.n += 1
        return json.dumps({"answer": {"kind": "use_more", "move": move, "range": "mid", "when": "standing",
                                      "why": "a grounded poke she has not tried here"}, "stop": None})


def _full_carried_registry():
    """A FULL (SM.MAX_LINES) short memory carried from a prior block: one kept kit line + aged trying lines that
    fill every slot -- exactly the deadlock shape. Trying moves are disjoint from the Coach's PICKS so its claims
    are fresh admissions, not tone changes."""
    def trying(move, rounds):
        c = {"kind": "use_more", "move": move, "range": None, "when": "standing", "view": None}
        return dict(SM._trying_entry(c), rounds=rounds)
    kit = {"claim": {"kind": "use_more", "move": "s.mk", "range": None, "when": "standing", "view": None},
           "line": "use more s.mk when he stands", "state": "kept", "rounds": 20, "evidence": {}, "why": "kit"}
    fillers = ["walk_back", "throw_F+hp", "throw_F+mp", "c.mk", "c.hk", "s.hp", "c.lp", "j.hk", "cl.mk"]
    reg = [kit] + [trying(m, 10 + i) for i, m in enumerate(fillers)]
    assert len(reg) == SM.MAX_LINES, "carried memory must be FULL to exercise the swap/freeze"
    return reg


def _qwen_added(out):
    added = []
    with open(os.path.join(out, "trace.jsonl")) as f:
        for l in f:
            e = json.loads(l)
            if e.get("event") == "qwen":
                added += e["added"]
    return added


def test_full_memory_losing_streak_changes_within_two_rounds(tmp_path):
    driver = load_driver()
    out = str(tmp_path / "deadlock")
    seed = _full_carried_registry()
    before = SM.in_play(seed)
    verdict = driver.run_loop(OPP, FollowerLaya(), FollowerLaya(), FreshCoachEachRound(), games=1, rounds=4,
                              seed_lines=seed, out=out, play_round_fn=fake_play, state=b"x",
                              state_id={"path": "p", "sha256": "0"}, emu=FakeEmu(), score_fn=fake_loss, seed_rng=1)
    assert _qwen_added(out), "while losing, the short memory MUST gain a fresh line (deadlock broken)"
    assert SM.in_play(verdict["registry_end"]) != before, "the in-play set must actually change under a loss streak"
    assert len(SM.in_play(verdict["registry_end"])) <= SM.MAX_LINES      # never grows past the live cap


def test_full_memory_while_winning_stays_frozen(tmp_path):
    driver = load_driver()
    out = str(tmp_path / "frozen")
    seed = _full_carried_registry()
    before = SM.in_play(seed)
    verdict = driver.run_loop(OPP, FollowerLaya(), FollowerLaya(), FreshCoachEachRound(), games=1, rounds=4,
                              seed_lines=seed, out=out, play_round_fn=fake_play, state=b"x",
                              state_id={"path": "p", "sha256": "0"}, emu=FakeEmu(), score_fn=fake_win, seed_rng=1)
    assert _qwen_added(out) == [], "a FULL memory that is winning must not churn (freeze)"
    assert SM.in_play(verdict["registry_end"]) == before
