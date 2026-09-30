"""Early stop of a test when her rounds tank (sf2.system2.lessons.stop / STOP_DROP), on the real lesson-loop runs.

Calibration: in the no-advice arms nothing changes between games, so every drop there is chance. A test can begin after
any game with >= 2 games before it and is checked after each of its up to TEST_GAMES games; STOP_DROP must be crossed
by chance in < 5% of those checks. Replay: on each run's ledger, how often a claim in test would have been stopped.

Real data: rollouts/ under $SF2_DATA (default: the repo root); skipped where it is absent.
"""
import collections
import glob
import json
import os

import pytest

from sf2.system2 import lessons as L

DATA = os.environ.get("SF2_DATA", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ROOTS = [os.path.join(DATA, "rollouts", "locked", "lesson_loop_v1"), os.path.join(DATA, "rollouts", "qwen_lessons")]
real = pytest.mark.skipif(not os.path.isdir(ROOTS[0]), reason="no lesson-loop runs under %s" % DATA)


def game_hp(path):
    """Mean hp (dealt - taken) per round of each game of one arm."""
    by = collections.defaultdict(list)
    with open(path) as f:
        for x in f:
            if x.strip():
                r = json.loads(x)
                by[r["game"]].append(r["dealt"] - r["taken"])
    return [sum(v) / len(v) for _, v in sorted(by.items())]


def finished(arm):
    return [os.path.join(d, arm) for root in ROOTS for d in sorted(glob.glob(os.path.join(root, "*")))
            if os.path.exists(os.path.join(d, "verdict.json")) and os.path.exists(os.path.join(d, arm, "rounds.jsonl"))]


def chance_checks(hp):
    """Every check a test would get in an arm: begun after game s (>= 2 games before), checked after games s+1..s+T."""
    return [L.stop(hp[:g + 1], s, g) for s in range(1, len(hp) - 1)
            for g in range(s + 1, min(s + L.TEST_GAMES, len(hp) - 1) + 1)]


def test_stop_is_pure_and_needs_two_games_before():
    assert L.stop([0, 0, -100], 0, 2) is None                  # 1 game before
    assert L.stop([0, 0, -100], 1, 2) is not None
    assert L.stop([0, 0, 0], 1, 2) is None
    assert L.stop([0, 0, 0, -300], 1, 2) is None               # only games up to ``game`` count


def test_a_seeded_stop_everything_threshold_is_caught_by_the_calibration():
    """The calibration can fail: at STOP_DROP 0 a flat-noise arm crosses in far more than 5% of checks."""
    hp = [((i * 37) % 23) - 11.0 for i in range(10)]
    fired = [c for c in chance_checks(hp) if c is not None]
    assert len(fired) < 0.05 * len(chance_checks(hp))                  # the real threshold: quiet
    old = L.STOP_DROP
    try:
        L.STOP_DROP = 0
        assert sum(L.stop(hp[:g + 1], s, g) is not None for s in range(1, 8) for g in range(s + 1, s + 3)) > 3
    finally:
        L.STOP_DROP = old


@real
def test_no_advice_arms_cross_stop_drop_by_chance_in_under_5_percent_of_checks():
    arms = [game_hp(os.path.join(a, "rounds.jsonl")) for a in finished("none")]
    checks = [c for hp in arms for c in chance_checks(hp)]
    rate = sum(c is not None for c in checks) / len(checks)
    print("no-advice arms: %d, checks %d, false alarms %.3f" % (len(arms), len(checks), rate))
    assert len(arms) >= 60 and rate < 0.05


@real
def test_replay_on_the_real_ledgers():
    """Each claim that was in test in a finished loop arm: would it have been stopped, and how did it end instead."""
    tested = stopped = 0
    ended = collections.Counter()
    for arm in finished("loop"):
        hp = game_hp(os.path.join(arm, "rounds.jsonl"))
        with open(os.path.join(arm, "ledger.jsonl")) as f:
            led = [json.loads(x) for x in f if x.strip()]
        fired, final = {}, {(r["line"], r["since"]): r["state"] for r in led[-1]["registry"]} if led else {}
        for row in led:                 # a claim in test after game g was in play in game g+1, reviewed after it
            g = row["game"] + 1
            for r in row["registry"]:
                k = (r["line"], r["since"])
                if r["state"] == "testing" and g < len(hp):
                    fired[k] = fired.get(k, False) or L.stop(hp[:g + 1], r["since"], g) is not None
        tested += len(fired)
        stopped += sum(fired.values())
        ended.update(final.get(k, "?") for k, f in fired.items() if f)
    print("claims in test: %d, would be stopped: %d (%.1f%%); they ended %s" % (
        tested, stopped, 100 * stopped / tested, dict(ended)))
    assert tested > 100 and 0 < stopped < tested / 4
