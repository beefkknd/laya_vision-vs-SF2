"""Effective-coverage primitive, test-first: (fire_rate, follows) over a run's decision records.

fire_rate = fraction of decision frames where at least one rule was ROUTED IN (applied) that frame
            (prompt_lines non-empty); a frame with no applicable rule is a non-fire.
follows   = among the FIRED frames only, fraction where text-laya's pick matched the rule
            (the existing follows_rule boolean). Undefined on non-fire frames, so excluded.

These feed BlockStat.fire_rate / .follows for the keep/drop core (sf2.system2.promotion).
"""
from sf2.system2.coverage import coverage


def _d(prompt_lines, follows):
    return {"prompt_lines": prompt_lines, "follows_rule": follows, "lines": ["use more throw up close"]}


def test_empty_is_zero():
    assert coverage([]) == (0.0, 0.0)


def test_fire_rate_counts_frames_with_a_routed_rule():
    rows = [_d(["use more throw up close"], True),   # fired, followed
            _d([], False),                           # no rule applied -> non-fire
            _d(["use more throw up close"], False),  # fired, not followed
            _d(None, True)]                          # None prompt_lines -> non-fire
    fr, fo = coverage(rows)
    assert fr == 0.5, fr                              # 2 of 4 frames fired
    assert fo == 0.5, fo                              # of the 2 fired, 1 followed


def test_follows_ignores_non_fire_frames():
    # a frame that did not fire but has follows_rule=True must NOT inflate follows
    rows = [_d(["r"], True), _d([], True), _d([], True)]
    fr, fo = coverage(rows)
    assert fr == 1 / 3
    assert fo == 1.0, "follows is over FIRED frames only"


def test_all_fire_all_follow():
    rows = [_d(["r"], True) for _ in range(5)]
    assert coverage(rows) == (1.0, 1.0)


def test_follows_zero_when_nothing_fires():
    rows = [_d([], False), _d(None, True)]
    assert coverage(rows) == (0.0, 0.0)
