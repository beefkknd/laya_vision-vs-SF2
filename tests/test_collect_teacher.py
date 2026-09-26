"""collect_teacher's choice of action: sampled from the teacher's distribution, or its top choice (--greedy)."""
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))
import collect_teacher  # noqa: E402
from sf2 import actions as A  # noqa: E402

DIST = {a: (0.6 if a == "hk" else 0.4 / (len(A.ACTIONS) - 1)) for a in A.ACTIONS}


def _picks(eps, greedy, n=2000):
    choose = collect_teacher.make_choose(eps, greedy, random.Random(0))
    return [choose(None, None, None, "", DIST) for _ in range(n)]


def test_greedy_without_eps_always_plays_the_teachers_top_choice():
    assert {a for a, _ in _picks(0.0, True)} == {"hk"}


def test_sampling_without_eps_follows_the_distribution():
    share = sum(a == "hk" for a, _ in _picks(0.0, False)) / 2000
    assert 0.55 < share < 0.65


def test_greedy_with_eps_is_random_that_often_and_says_so():
    picks = _picks(0.2, True)
    rand = [a for a, m in picks if m["actor"] == "random"]
    assert 0.17 < len(rand) / len(picks) < 0.23
    assert all(a == "hk" for a, m in picks if m["actor"] == "teacher")
