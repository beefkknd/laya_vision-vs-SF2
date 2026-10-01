"""The movement-dataset collector (sf2.data.movement_collect; docs/prereg_movement_data.md): a ring buffer of the
displayed frames, his movement answer per frame from RAM, episodes (runs of one answer) sampled at start / middle /
end when they end, per-(opponent, answer) quotas split by whole games, and the bridge proxy that captures every
frame. Synthetic streams: every image carries its capture index in its first pixels, so a pair's frames can be
traced back to the exact stream row."""
import random

import numpy as np
import pytest
from ram_rows import row

from sf2.data import movement as M
from sf2.data import movement_collect as C
from sf2.data.perception import UNKNOWN
from sf2.emu.vs import GROUND_Y

H, W = 224, 256
STATE = {"standing": {}, "crouching": {"state": 0x02}, "jumping": {"state": 0x04, "y": GROUND_Y - 30},
         "attacking": {"state": 0x0A}, "being hit": {"state": 0x0E, "react": 0x0E}, "blocking": {"state": 0x08},
         UNKNOWN: {"state": 0x06}}


def image(k):
    im = np.zeros((H, W, 3), np.uint8)
    im[0, 0] = (k // 256 % 256, k % 256, 7)
    return im


def index_of(im):
    return int(im[0, 0, 0]) * 256 + int(im[0, 0, 1])


def stream(spec):
    """Rows from [(answer, n frames), ...]; his x fixed at 300 (so standing never walks), mine at 200."""
    rows = []
    for ans, n in spec:
        rows += [row({"x": 200}, dict({"x": 300}, **STATE[ans])) for _ in range(n)]
    return rows


class Saved:
    def __init__(self):
        self.images = {}

    def __call__(self, k, im):
        name = "g0000_k%05d.png" % k
        self.images[name] = im
        return name


def run(spec, left=None, ring=C.RING, split="train", seed=0):
    saved = Saved()
    s = C.EpisodeSampler(game=0, split=split, left=dict(left or {a: 10 ** 6 for a in M.ANSWERS}),
                         rng=random.Random(seed), save_image=saved, ring=ring)
    rows = stream(spec)
    for k, r in enumerate(rows):
        s.feed(r, None if k == 0 else image(k))
    return s, s.finish(), saved, rows


# ---- splits and stages -------------------------------------------------------------------------------------------

def test_split_by_whole_games():
    assert [C.split_of_game(g) for g in range(10)] == ["val", "train", "test", "train", "train", "test", "train",
                                                       "train", "test", "train"]
    assert C.split_of_game(25) == "test" and C.split_of_game(30) == "val"


def test_split_quotas_sum_to_the_target_and_clear_the_gates():
    assert sum(C.SPLIT_QUOTA.values()) == C.QUOTA == 1500
    assert C.SPLIT_QUOTA["train"] >= 1000 and C.SPLIT_QUOTA["test"] >= 200


def test_stage_bin_is_the_third_of_the_episode():
    assert [C.stage_bin(p, 9) for p in range(9)] == ["start"] * 3 + ["middle"] * 3 + ["end"] * 3
    assert [C.stage_bin(p, 7) for p in range(7)] == ["start"] * 3 + ["middle"] * 2 + ["end"] * 2
    with pytest.raises(ValueError):
        C.stage_bin(7, 7)


def test_sample_positions_one_per_third_for_six_plus_frames():
    rng = random.Random(1)
    for L in range(6, 40):
        ps = C.sample_positions(100, 100 + L - 1, 0, 10 ** 9, rng)
        assert [C.stage_bin(p - 100, L) for p in ps] == ["start", "middle", "end"]


def test_sample_positions_one_pair_below_six_frames():
    rng = random.Random(2)
    for L in range(1, 6):
        ps = C.sample_positions(50, 50 + L - 1, 0, 10 ** 9, rng)
        assert len(ps) == 1 and 50 <= ps[0] < 50 + L


def test_sample_positions_stay_inside_the_buffered_part():
    rng = random.Random(3)
    for _ in range(200):
        ps = C.sample_positions(10, 400, 300, 400, rng)
        assert len(ps) == 3 and all(300 <= p <= 400 for p in ps)
    assert C.sample_positions(10, 20, 30, 40, rng) == []


# ---- episodes -----------------------------------------------------------------------------------------------------

def test_pairs_are_frames_t_minus_4_and_t_shown_with_lag_one():
    s, pairs, saved, rows = run([("standing", 20), ("crouching", 12), ("attacking", 9), ("standing", 10)])
    assert pairs
    for p in pairs:
        t = p["t"]
        assert (p["k_prev"], p["k_now"]) == (t - 3, t + 1)                  # image k shows row k - LAG
        assert [index_of(saved.images[n]) for n in p["images"]] == [t - 3, t + 1]
        assert p["answer"] == M.movement(rows, t) == M.movement_at(rows, p["k_now"])


def test_each_episode_of_six_plus_frames_gives_start_middle_end():
    s, pairs, _, _ = run([("standing", 20), ("crouching", 12), ("attacking", 9), ("standing", 10)])
    by = {}
    for p in pairs:
        by.setdefault(tuple(p["episode"]), []).append(p)
    assert sorted(by) == [(4, 19), (20, 31), (32, 40), (41, 50)]
    for (s0, e0), ps in by.items():
        assert [p["stage_bin"] for p in ps] == ["start", "middle", "end"]
        for p in ps:
            assert p["length"] == e0 - s0 + 1 and p["pos"] == p["t"] - s0
            assert p["stage"] == round(p["pos"] / p["length"], 4)


def test_short_episodes_give_one_pair():
    _, pairs, _, _ = run([("standing", 20), ("jumping", 3), ("standing", 20)])
    short = [p for p in pairs if p["answer"] == "jumping"]
    assert len(short) == 1 and short[0]["length"] == 3


def test_unknown_episodes_are_not_sampled():
    _, pairs, _, _ = run([("standing", 20), (UNKNOWN, 30), ("standing", 20)])
    assert all(p["answer"] != UNKNOWN for p in pairs)
    assert all(not (24 <= p["t"] < 50) for p in pairs)


def test_the_first_four_rows_start_no_episode():
    _, pairs, _, _ = run([("crouching", 30)])
    assert min(p["t"] for p in pairs) >= 4 and min(p["k_prev"] for p in pairs) >= 1
    assert {tuple(p["episode"]) for p in pairs} == {(4, 29)}          # the episode is counted from row 4


def test_episode_cut_by_the_round_end_is_flagged_and_needs_its_now_frame():
    _, pairs, _, rows = run([("standing", 10), ("crouching", 30)])
    last = [p for p in pairs if p["answer"] == "crouching"]
    assert last and all(p["cut_end"] for p in last)
    assert max(p["k_now"] for p in pairs) <= len(rows) - 1


def test_episode_longer_than_the_buffer_is_sampled_inside_it_and_flagged():
    ring = 40
    _, pairs, _, _ = run([("standing", 10), ("crouching", 200), ("standing", 10)], ring=ring)
    long = [p for p in pairs if p["answer"] == "crouching"]
    assert len(long) == 3 and all(p["long"] for p in long)
    close_k = 210                                     # the first standing row after the episode
    oldest = close_k - ring + 1
    assert all(p["k_prev"] >= oldest for p in long)
    short = [p for p in pairs if p["answer"] == "standing" and not p["cut_end"]]
    assert short and not any(p["long"] for p in short)


def test_quota_stops_saving_an_answer_and_counts_down():
    left = {a: 10 ** 6 for a in M.ANSWERS}
    left["crouching"] = 4
    s, pairs, saved, _ = run([("standing", 10), ("crouching", 12), ("standing", 10)] * 3, left=left)
    assert sum(p["answer"] == "crouching" for p in pairs) == 4
    assert s.left["crouching"] == 0
    assert sum(p["answer"] == "standing" for p in pairs) > 4


def test_images_are_saved_once_and_only_for_kept_pairs():
    left = {a: 0 for a in M.ANSWERS}
    _, pairs, saved, _ = run([("standing", 30), ("crouching", 30)], left=left)
    assert pairs == [] and saved.images == {}


def test_the_ram_stream_keeps_every_row():
    s, _, _, rows = run([("standing", 33), ("crouching", 7)])
    assert s.rows == rows


def test_a_ring_too_small_for_a_pair_is_refused():
    with pytest.raises(ValueError):
        C.EpisodeSampler(game=0, split="train", left={}, rng=random.Random(0), save_image=Saved(), ring=4)


def test_an_unknown_split_is_refused():
    with pytest.raises(ValueError):
        C.EpisodeSampler(game=0, split="heldout", left={}, rng=random.Random(0), save_image=Saved())


def test_a_missing_image_after_row_zero_is_refused():
    s = C.EpisodeSampler(game=0, split="train", left={}, rng=random.Random(0), save_image=Saved())
    s.feed(stream([("standing", 1)])[0], None)
    with pytest.raises(ValueError):
        s.feed(stream([("standing", 1)])[0], None)


def test_sampling_is_deterministic_per_seed():
    a = run([("standing", 40), ("crouching", 40)] * 2, seed=5)[1]
    b = run([("standing", 40), ("crouching", 40)] * 2, seed=5)[1]
    assert a == b


# ---- the bridge proxy ---------------------------------------------------------------------------------------------

class FakeObs:
    def __init__(self, rams, images):
        self.rams, self.images = rams, images


class FakeBridge:
    """Frame f of the whole session has RAM [f] and image(f); a RUN of n frames returns rows f0..f0+n."""
    def __init__(self):
        self.f, self.calls = 0, []

    def load_state(self, state):
        self.f = 1000
        return FakeObs([[self.f]], {})

    def run(self, frames, caps=(), p2=None):
        n = len(frames)
        self.calls.append(sorted(caps))
        rams = [[self.f + i] for i in range(n + 1)]
        imgs = {c: image(self.f + c) for c in caps}
        self.f += n
        return FakeObs(rams, imgs)

    def other(self):
        return "delegated"


def test_proxy_captures_every_frame_and_feeds_the_stream_in_order():
    fb, got = FakeBridge(), []
    p = C.CapturingBridge(fb)
    p.sink = lambda r, im: got.append((r, None if im is None else index_of(im)))
    p.load_state(b"s")
    obs = p.run([[]] * 6, caps=[2, 6])
    assert fb.calls[-1] == [0, 1, 2, 3, 4, 5, 6]
    assert 2 in obs.images and 6 in obs.images
    p.run([[]] * 4, caps=[0, 4])
    assert [r[0] for r, _ in got] == list(range(1000, 1011))            # row 0 of a later RUN is not repeated
    assert got[0][1] is None and [k for _, k in got[1:]] == list(range(1001, 1011))
    assert p.other() == "delegated"


def test_proxy_needs_a_sink():
    p = C.CapturingBridge(FakeBridge())
    p.load_state(b"s")
    with pytest.raises(RuntimeError):
        p.run([[]] * 4)
