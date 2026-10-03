"""Registry CARRYOVER for a multi-round play session (scripts/play_loop_screen.py --carry / --save-registry):
Chun-Li's learned rules must carry from one round into the next so the loop learns ON TOP of them.

Seen RED (how each assertion was confirmed able to fail):
  * round-trip: run first against the pre-change driver (no save_registry/load_registry/starting_registry attrs)
    -> AttributeError. Then against a twin whose save_registry wrote L.in_play(reg) (lines only, not the full entry
    objects) -> load_registry returns a list of STRINGS, loaded != registry_end and the second run's seed event is
    malformed -> the round-trip equality + the 'starting in-play == carried' asserts fail.
  * missing-file fallback: a twin where starting_registry carried unconditionally (ignoring os.path.exists)
    -> load_registry('/no/such/file') raises instead of falling back to the book seed; the test's book-seed
    equality then never runs (errors out), i.e. red.

These drive the whole loop with a MOCK Qwen + a follower text laya (no network, no emulator), like test_loop_wiring.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from looptools import FollowerLaya, load_driver        # noqa: E402

from sf2.system1.loop_runner import two_stage_decide   # noqa: E402
from sf2.system2 import lessons as L                    # noqa: E402

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BOOK = os.path.join(HERE, "lessons", "book.json")
ME, OPP = "chunli", "honda"


class FakeEmu:
    def new_round(self):
        return self


def fake_play(emu, cat_advisor, move_advisor, me, opp, state, state_id, delay, lines, out, reader=None):
    os.makedirs(out, exist_ok=True)
    from looptools import make_moment
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
    """Adds one ANSWER claim after game 0 (so the registry ends with a learned rule), nothing afterwards. Two-stage
    aware: the Scout call (task "scout_*") gets prose; ``calls`` counts only the Coach/single calls."""

    def __init__(self):
        self.calls = 0

    def __call__(self, messages, task):
        if task.startswith("scout"):
            return "She blocked most of the game and lost; her offense barely landed."
        self.calls += 1
        if self.calls == 1:
            return json.dumps({"answer": {"kind": "always", "move": "cl.hp", "range": "close",
                                          "when": "attacking", "why": "it lands"}, "stop": None})
        return json.dumps({"answer": None, "stop": None})


def _seed_event_lines(out):
    with open(os.path.join(out, "trace.jsonl")) as f:
        for line in f:
            e = json.loads(line)
            if e.get("event") == "seed":
                return e["lines"]
    raise AssertionError("no seed event in trace")


def _run(driver, out, seed_lines):
    return driver.run_loop(OPP, FollowerLaya(), FollowerLaya(), MockQwen(), games=2, rounds=1,
                           seed_lines=seed_lines, out=out, play_round_fn=fake_play,
                           state=b"x", state_id={"path": "p", "sha256": "0"},
                           emu=FakeEmu(), score_fn=fake_score, seed_rng=1)


def test_carry_round_trips_the_full_registry(tmp_path):
    """Save the final registry of one round, carry it into the next: the loop STARTS with those exact rules."""
    driver = load_driver()
    first = _run(driver, str(tmp_path / "round0"), seed_lines=[])
    reg_end = first["registry_end"]
    assert reg_end and L.in_play(reg_end), "the first round should have learned at least one in-play rule"

    reg_path = str(tmp_path / "carry.json")
    driver.save_registry(reg_end, reg_path)
    assert os.path.isfile(reg_path)

    loaded = driver.load_registry(reg_path)
    assert loaded == reg_end                                  # the full entry objects round-trip, not just lines

    # starting_registry picks up the carried file (source "carry") even though a book is also given
    reg_start, source = driver.starting_registry(reg_path, OPP, BOOK, ME)
    assert source == "carry"
    assert reg_start == reg_end

    # and the NEXT round actually begins with those rules in play (the loop logs them in its seed event)
    out2 = str(tmp_path / "round1")
    _run(driver, out2, seed_lines=reg_start)
    assert _seed_event_lines(out2) == L.in_play(reg_start)
    assert L.in_play(reg_start)                               # non-empty: the carried rule is in force from game 0


def test_carry_missing_file_falls_back_to_book_seed():
    """An absent --carry file falls back to the normal book seed; no carry at all does the same."""
    driver = load_driver()
    book_seed = driver.seed(OPP, BOOK, ME)
    assert book_seed, "honda has book lines -> a non-empty seed"

    reg_missing, src_missing = driver.starting_registry("/no/such/carry.json", OPP, BOOK, ME)
    assert src_missing == "book"
    assert reg_missing == book_seed

    reg_none, src_none = driver.starting_registry(None, OPP, BOOK, ME)
    assert src_none == "book"
    assert reg_none == book_seed
