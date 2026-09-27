"""Saved opening schedules and the paired comparison they make possible (docs/TWO_SYSTEM_PLAN.md §2.4, §2.5)."""
import pytest

from sf2 import openings as O


def test_schedules_are_distinct_and_dev_never_overlaps_eval():
    dev, ev = O.make_schedules(seed=1, n_dev=20, n_eval=40, max_idle=120)
    assert len(dev) == 20 and len(ev) == 40
    assert len(set(dev)) == 20 and len(set(ev)) == 40 and not set(dev) & set(ev)
    assert all(1 <= x <= 120 for x in dev + ev)
    assert (dev, ev) == O.make_schedules(seed=1, n_dev=20, n_eval=40, max_idle=120)  # reproducible


def test_schedule_file_round_trips(tmp_path):
    p = tmp_path / "dev.txt"
    O.save(str(p), [5, 17, 3])
    assert O.load(str(p)) == [5, 17, 3]
    assert O.parse("5,17,3") == [5, 17, 3]
    assert O.parse(str(p)) == [5, 17, 3]


@pytest.mark.parametrize("bad", ["", "5,5", "5,-1", "5,x"])
def test_bad_schedules_are_rejected(bad):
    with pytest.raises(ValueError):
        O.parse(bad)


def test_split_gives_every_opening_to_exactly_one_worker():
    sched = list(range(1, 21))
    chunks = O.split(sched, 3)
    assert sorted(x for c in chunks for x in c) == sched
    assert [len(c) for c in chunks] == [7, 7, 6]
    assert O.split([4, 9], 4) == [[4], [9]]  # never an empty worker


class _Env:
    """Just enough of FightEnv: reset() loads the fight start, run_frames() idles."""

    def __init__(self):
        self.episode, self.jitter, self.frame_no, self.idled = -1, 30, 0, []
        self.frame = "img"

    def reset(self):
        self.episode += 1
        self.frame_no = 0
        return self.frame

    def run_frames(self, frames, capture=True):
        self.idled.append(len(frames))
        self.frame_no += len(frames)


def test_apply_idles_the_scheduled_frames_for_each_match():
    env = _Env()
    O.apply(env, [12, 57])
    assert env.jitter == 0  # the schedule replaces worker jitter
    for want in (12, 57):
        env.reset()
        assert env.idled[-1] == want and env.opening == want and env.frame_no == 0


def test_apply_refuses_to_run_past_the_schedule():
    env = _Env()
    O.apply(env, [12])
    env.reset()
    with pytest.raises(IndexError):
        env.reset()


def _rounds(opening, nets):
    return [{"opening": opening, "round": i, "dmg_for": max(n, 0), "dmg_against": max(-n, 0)}
            for i, n in enumerate(nets)]


def test_paired_difference_by_opening():
    a = _rounds(5, [10, 20]) + _rounds(9, [0]) + _rounds(13, [-30])
    b = _rounds(5, [40, 50]) + _rounds(9, [20]) + _rounds(13, [0])
    r = O.paired(a, b)
    assert r["n"] == 3
    assert r["mean_diff"] == pytest.approx(((45 - 15) + (20 - 0) + (0 + 30)) / 3)
    assert r["ci95"][0] < r["mean_diff"] < r["ci95"][1]
    assert r["unmatched"] == []


def test_paired_reports_openings_only_one_arm_played():
    r = O.paired(_rounds(5, [10]) + _rounds(7, [1]), _rounds(5, [20]) + _rounds(8, [2]))
    assert r["n"] == 1 and r["unmatched"] == [7, 8]


def test_paired_needs_openings_on_every_round():
    with pytest.raises(ValueError, match="opening"):
        O.paired([{"round": 0, "dmg_for": 1, "dmg_against": 0}], _rounds(5, [1]))
