"""The value-data collector's behaviour policy (docs/plan_laya_vision_value.md): with probability ``explore`` System 1
plays a uniformly random move of its choices (so every move gets tries where laya-vision would never rank it),
otherwise laya-vision's own pick; laya-vision's scores are logged either way, and explore=0 is the old policy."""
import collections

import numpy as np

from sf2.system1.system1 import System1, _close, choices


class FakeAgent:
    """Scores every attack P(hit) 0.9 for 'lk' and 0.1 otherwise; blocks P(blocked) 0.05."""
    def predict(self, state, questions):
        ans = {}
        for a in questions:
            if a in ("block_high", "block_low"):
                p = {"hit": 0.0, "whiff": 0.0, "blocked": 0.05, "none": 0.95, "got_hit": 0.0}
            else:
                h = 0.9 if a == "lk" else 0.1
                p = {"hit": h, "whiff": 1 - h, "blocked": 0.0, "none": 0.0, "got_hit": 0.0}
            ans[a] = {"probabilities": p}
        return {"answers": ans}


def s1(explore, seed=0):
    s = System1(None, "chunli", seed=seed, explore=explore)
    s.agent = FakeAgent()
    return s


IMG = np.zeros((224, 256, 3), np.uint8)


def test_explore_zero_is_the_old_policy():
    d = s1(0.0).decide(IMG, IMG, "note")
    assert d["action"] == "lk" and d["probs"]["lk"] == 0.9
    assert "explored" not in d          # old logs stay byte-identical (locks lesson_loop_v1/v2)


def test_explore_one_is_uniform_over_choices_and_keeps_scores():
    s = s1(1.0)
    picks = collections.Counter()
    for _ in range(3200):
        d = s.decide(IMG, IMG, "note")
        assert d["explored"] is True and d["probs"]["lk"] == 0.9
        picks[d["action"]] += 1
    assert set(picks) == set(choices("chunli"))
    n = len(choices("chunli"))
    assert all(abs(c - 3200 / n) < 0.3 * 3200 / n for c in picks.values())


def test_explore_mix_share_and_determinism():
    s, t = s1(0.5, seed=7), s1(0.5, seed=7)
    xs = [s.decide(IMG, IMG, "n") for _ in range(2000)]
    ys = [t.decide(IMG, IMG, "n") for _ in range(2000)]
    assert [x["action"] for x in xs] == [y["action"] for y in ys]
    share = sum(x["explored"] for x in xs) / 2000
    assert 0.45 < share < 0.55


def test_explore_out_of_range_refused():
    for bad in (-0.1, 1.5):
        try:
            System1(None, "chunli", explore=bad)
        except ValueError:
            continue
        raise AssertionError("explore=%s accepted" % bad)


def test_explored_is_logged():
    r = {"p1_x": 100, "p2_x": 160, "p1_y": 192, "p2_y": 192, "p1_state": 0, "p2_state": 0x0A, "p1_life": 176,
         "p2_life": 176, "p1_react": 0, "p2_react": 0, "timer": 0x99}
    d = dict(s1(1.0).decide(IMG, IMG, "n"), prompt="n")
    e = _close(0, "chunli", "ryu", (r, [r], d, "whiff", 0, None))
    assert e["explored"] is True and e["opp_state"] == "attack"
