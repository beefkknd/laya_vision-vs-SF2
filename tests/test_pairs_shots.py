"""Round 3 (docs/prereg_movement_finetunes.md): the projectile sampling trigger. While a shot slot is on (shot1 =
player 1's projectile, shot2 = player 2's; confirmed from RAM + frames, scripts/probe_shots.py), its flight is sampled
at start / middle / end, only on rows where the projectile is DRAWN (the slot's byte +0x3A bit 0 clear: the game
blinks hadoken / yoga fire; probe 102 / 102 frames), at most SHOT_PER_GAME per (slot, stage) per game; the pair's
images are the captures t - 3 and t + 1 (lag 1)."""
import random

import pytest

from sf2.data import pairs_collect_io as IO
from sf2.data import pairs_shots as S
from sf2.data.movement_collect import RING
from test_pairs_collect import BANDS, image, index_of
from test_pairs_labels import prow


def frow(k, flights, hide_every=0, x1=200, x2=300):
    """A row at stream index k: player 1 at x1, player 2 at x2; each flight (slot, start, end) turns its slot on
    over [start, end]; with hide_every, rows k % hide_every == hide_every - 1 are hidden (blink)."""
    r = dict(prow({"x": x1}, {"x": x2}), timer=k % 256, shot1_hide=0, shot2_hide=0)
    for slot, a, e in flights:
        if a <= k <= e:
            r["shot%d" % slot] = 1
            r["shot%d_x" % slot] = 220 + (k - a)
            if hide_every and k % hide_every == hide_every - 1:
                r["shot%d_hide" % slot] = 1
    return r


def run(flights, n=300, per_game=S.SHOT_PER_GAME, hide_every=0, words=None, chars=("ryu", "ken"), seed=0, x=(200, 300)):
    store = {}

    def save(k, im):
        store[k] = index_of(im)
        return "k%05d.png" % k
    s = S.ShotSampler(0, {1: chars[0], 2: chars[1]}, random.Random(seed), save, BANDS, RING, 0,
                      shot_per_game=per_game)
    for k in range(n):
        for p, w, k0 in (words or []):
            if k0 == k - 1:
                s.press(p, w, k0)
        s.feed(frow(k, flights, hide_every, *x), None if k == 0 else image(k))
    pairs = s.finish()
    return s, pairs, store


def test_visible_needs_the_slot_on_and_the_blink_bit_clear():
    r = frow(10, [(1, 5, 20)])
    assert S.visible(r, 1) and not S.visible(r, 2)
    assert not S.visible(dict(r, shot1_hide=1), 1)
    assert not S.visible(dict(r, shot1_hide=3), 1)
    assert S.visible(dict(r, shot1_hide=2), 1)                 # bit 0 only
    assert not S.visible(dict(r, shot1=0), 1)


def test_one_sample_per_stage_of_each_flight_with_lag_1_images_and_no_movement_pairs():
    s, pairs, store = run([(1, 40, 99)], words=[(1, "hadoken", 30)])
    assert pairs == []                                          # movement per_game 0
    assert sorted(x["flight_stage"] for x in s.shots) == ["end", "middle", "start"]
    for x in s.shots:
        t = x["t"]
        assert x["slot"] == 1 and x["thrower"] == "ryu" and x["other"] == "ken" and x["flight"] == [40, 99]
        assert (x["k_prev"], x["k_now"]) == (t - 3, t + 1)
        assert store[x["k_prev"]] == t - 3 and store[x["k_now"]] == t + 1
        assert x["images"] == ["k%05d.png" % (t - 3), "k%05d.png" % (t + 1)]
        assert x["flight_stage"] == ("start", "middle", "end")[3 * (t - 40) // 60]
        assert x["spawn_word"] == "hadoken" and x["spawn_words"] == {"1": "hadoken", "2": None}
        assert x["side"] == "left" and x["shot_x"] == 220 + t - 40
    assert s.flights == [{"slot": 1, "start": 40, "end": 99, "spawn_words": {"1": "hadoken", "2": None},
                          "thrower": "ryu", "cut_end": False}]


def test_player_2s_slot_is_player_2s_flight_and_side_by_x_at_t():
    s, _, _ = run([(2, 40, 99)], x=(300, 200))
    assert {x["slot"] for x in s.shots} == {2}
    assert all(x["thrower"] == "ken" and x["side"] == "left" for x in s.shots)
    s, _, _ = run([(2, 40, 99)], x=(250, 250))
    assert all(x["side"] is None for x in s.shots)


def test_only_drawn_rows_are_sampled():
    s, _, _ = run([(1, 40, 99)], hide_every=2)                 # every odd row hidden
    assert s.shots and all(x["t"] % 2 == 0 for x in s.shots)
    s, _, _ = run([(1, 40, 99)], hide_every=1)                 # never drawn
    assert s.shots == []


def test_the_per_game_cap_per_slot_and_stage():
    flights = [(1, 20 + 50 * i, 60 + 50 * i) for i in range(5)]
    s, _, _ = run(flights, n=320, per_game=2)
    per = {}
    for x in s.shots:
        per[x["flight_stage"]] = per.get(x["flight_stage"], 0) + 1
    assert per == {"start": 2, "middle": 2, "end": 2}
    assert len(s.flights) == 5


def test_a_flight_running_past_the_game_end_is_cut():
    s, _, _ = run([(2, 250, 400)], n=300)
    assert s.flights[0]["cut_end"] is True and s.flights[0]["end"] == 299
    assert all(x["cut_end"] and x["t"] + 1 <= 299 for x in s.shots)


def test_the_other_slots_shot_at_t_is_recorded():
    s, _, _ = run([(1, 40, 99), (2, 60, 70)])
    both = [x for x in s.shots if 60 <= x["t"] <= 70]
    assert all(x["other_shot"] == 1 for x in both)
    assert all(x["other_shot"] == 0 for x in s.shots if not 60 <= x["t"] <= 70)


def test_collect_pair_writes_shots_and_flights_before_the_commit_and_resumes(tmp_path):
    base = str(tmp_path / "ryu_vs_ken")

    def play(game, sampler):
        for k in range(200):
            sampler.feed(frow(k, [(1, 40, 99)]), None if k == 0 else image(k))
        return {"result": "win", "frames": 200, "moves": []}
    IO.collect_pair(base, play, "ryu", "ken", 2, 0, BANDS, per_game=0, log=lambda *a: None,
                    sampler_cls=S.ShotSampler)
    shots = IO.read_jsonl(base + "/shots.jsonl")
    games = IO.committed(base)
    assert [g["game"] for g in games] == [0, 1] and all(g["shots"] == 3 and len(g["flights"]) == 1 for g in games)
    assert len(shots) == 6 and {x["game"] for x in shots} == {0, 1}
    IO.collect_pair(base, play, "ryu", "ken", 3, 0, BANDS, per_game=0, log=lambda *a: None,
                    sampler_cls=S.ShotSampler)
    assert [g["game"] for g in IO.committed(base)] == [0, 1, 2]
    kept, dropped = S.committed_shots(base)
    assert len(kept) == 9 and dropped == 0


def test_uncommitted_shots_are_dropped(tmp_path):
    base = str(tmp_path / "ryu_vs_ken")
    import os
    os.makedirs(base)
    IO.MIO._append(base + "/shots.jsonl", [{"game": 0, "t": 5}])
    kept, dropped = S.committed_shots(base)
    assert kept == [] and dropped == 1


def test_shot_vars_are_the_blink_bytes_of_both_slots():
    assert [(v.name, v.addr) for v in S.SHOT_VARS] == [("shot1_hide", 0x103A), ("shot2_hide", 0x108A)]


@pytest.mark.parametrize("char, word, ok", [("ryu", "hadoken", True), ("ken", "hadoken", True),
                                            ("guile", "sonic_boom", True), ("dhalsim", "yoga_fire", True),
                                            ("dhalsim", "yoga_flame", False), ("ryu", "shoryuken", False),
                                            ("blanka", "hadoken", False), ("ryu", None, False)])
def test_projectile_words(char, word, ok):
    assert S.is_projectile(char, word) is ok
