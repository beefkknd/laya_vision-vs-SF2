"""The movement-pairs collector (sf2.data.pairs_collect, pairs_collect_io): both fighters' episodes sampled with the
(t - 4, t) pair at lag 1, at most PER_GAME pairs per (slot, key) per game, each row's labels and controller; the
directed play loop over a fake bridge; the fixed game budget and resume."""
import json
import os
import random

import numpy as np
import pytest
from ram_rows import row

from sf2.data import movement_collect as C
from sf2.data import pairs_collect as PC
from sf2.data import pairs_collect_io as IO
from sf2.data import pairs_labels as L
from sf2.data import pairs_moves as PM
from sf2.emu.vs import GROUND_Y, NAMES

BANDS = {"all": 72}
H, W = 224, 256
NAMES_X = NAMES + ["p1_aid", "p1_mclass", "p1_sclass", "p2_aid", "p2_mclass", "p2_sclass"]


def image(k):
    im = np.zeros((H, W, 3), np.uint8)
    im[0, 0] = (k // 256 % 256, k % 256, 7)
    return im


def index_of(im):
    return int(im[0, 0, 0]) * 256 + int(im[0, 0, 1])


P1 = [{}, {"state": 0x02}, {"state": 0x0A, "aid": 3}, {"state": 0x04, "y": GROUND_Y - 30}, {"state": 0x08}]
P2 = [{"state": 0x0E, "react": 2}, {}, {"state": 0x02}, {"state": 0x0A}, {"state": 0x0C}, {"state": 0x08}]


def synth(f):
    """A deterministic two-fighter row for frame f: each fighter steps through its states in 9 / 13 frame blocks."""
    base = {"aid": 0, "mclass": 0, "sclass": 0xFF}
    p1 = dict(base, x=200, **P1[(f // 9) % len(P1)])
    p2 = dict(base, x=300, facing=0x00, **P2[(f // 13) % len(P2)])
    return row(p1, p2)


def feed(sampler, n):
    for k in range(n):
        sampler.feed(synth(k), None if k == 0 else image(k))


def saver(store):
    def save(k, im):
        store[k] = index_of(im)
        return "k%05d.png" % k
    return save


def make(per_game=PC.PER_GAME, ring=C.RING, seed=0, store=None):
    return PC.PairSampler(0, {1: "ryu", 2: "ken"}, random.Random(seed), saver({} if store is None else store),
                          BANDS, ring, per_game)


def test_pairs_for_both_fighters_with_their_labels_and_controllers():
    s = make()
    feed(s, 600)
    pairs = s.finish()
    assert {p["slot"] for p in pairs} == {1, 2}
    for p in pairs:
        assert p["controller"] == {1: "directed", 2: "cpu"}[p["slot"]]
        assert p["char"] == {1: "ryu", 2: "ken"}[p["slot"]] and p["opp"] == {1: "ken", 2: "ryu"}[p["slot"]]
        lab = L.labels(s.rows, p["t"], p["slot"], BANDS)
        assert {k: p[k] for k in lab} == lab
        assert (p["k_prev"], p["k_now"]) == (p["t"] - 4 + C.LAG, p["t"] + C.LAG)


def test_the_saved_images_are_the_captures_named():
    store = {}
    s = make(store=store)
    feed(s, 400)
    for p in s.finish():
        assert p["images"] == ["k%05d.png" % p["k_prev"], "k%05d.png" % p["k_now"]]
        assert store[p["k_prev"]] == p["k_prev"] and store[p["k_now"]] == p["k_now"]


@pytest.mark.parametrize("per_game", [0, 1, 3])
def test_at_most_per_game_pairs_per_slot_and_key(per_game):
    s = make(per_game=per_game)
    feed(s, 2000)
    pairs = s.finish()
    counts = {}
    for p in pairs:
        k = (p["slot"], p["movement"], p["direction"], p["facing"])
        counts[k] = counts.get(k, 0) + 1
    assert all(n <= per_game for n in counts.values())
    if per_game:
        assert max(counts.values()) == per_game          # the synthetic game repeats every key many times


def test_pairs_spread_over_start_middle_end_of_long_episodes():
    s = make(per_game=3)
    feed(s, 200)
    bins = {p["stage_bin"] for p in s.finish() if p["length"] >= C.SHORT}
    assert bins == {"start", "middle", "end"}


def test_unknown_keys_are_never_sampled():
    s = make()
    for k in range(300):
        r = synth(k)
        r["p1_facing"] = 0x13                            # player 1's facing unknown throughout
        s.feed(r, None if k == 0 else image(k))
    assert {p["slot"] for p in s.finish()} == {2}


def test_bad_arguments_and_missing_images_are_refused():
    with pytest.raises(ValueError):
        make(ring=4)
    with pytest.raises(ValueError):
        make(per_game=-1)
    s = make()
    s.feed(synth(0), None)
    with pytest.raises(ValueError):
        s.feed(synth(1), None)


# ---- the play loop over a fake bridge ------------------------------------------------------------------------------

class Bridge:
    """Rows: player 1 can act except while 'busy' after an attack press; the round ends at END frames."""
    END = 900

    def __init__(self):
        self.f, self.busy = 0, 0

    def load_state(self, state):
        self.f, self.busy = 0, 0

    def _row(self):
        r = synth(0)
        r.update(p1_state=0x0A if self.busy else 0, p1_x=200, p2_x=300, p1_facing=0x40,
                 result=1 if self.f >= self.END else 0)
        return [r[n] for n in NAMES_X]

    def run(self, frames):
        from sf2.emu.mesen import Obs
        rams = [self._row()]
        for b in frames:
            self.f += 1
            if any(x in b for x in ("y", "x", "l", "b", "a", "r")):
                self.busy = 10
            elif self.busy:
                self.busy -= 1
            rams.append(self._row())
        return Obs(rams)


class Stream:
    def __init__(self, inner):
        self.inner, self.rows = inner, []

    def load_state(self, s):
        self.inner.load_state(s)

    def run(self, frames):
        obs = self.inner.run(frames)
        self.rows += obs.rams if not self.rows else obs.rams[1:]
        return obs


def test_play_directed_presses_each_word_in_cycle_order_until_the_round_ends():
    words = list(PM.moves("ryu"))
    br = Stream(Bridge())
    cyc = PM.Cycle(words, random.Random(5))
    expect = PM.Cycle(words, random.Random(5))
    out = PC.play_directed(br, NAMES_X, "ryu", cyc, b"", random.Random(0), lambda: len(br.rows))
    assert out["result"] == "win"
    assert [m[0] for m in out["moves"]] == [expect.next() for _ in out["moves"]]
    ks = [(m[1], m[2]) for m in out["moves"]]
    assert all(a < b for a, b in ks) and all(ks[i][1] == ks[i + 1][0] for i in range(len(ks) - 1))
    assert len(out["moves"]) > len(words)              # the cycle starts again
    assert out["cycle_rounds"] >= 2


# ---- the files and the budget ---------------------------------------------------------------------------------------

def fake_play(n=300):
    def play(game, sampler):
        for k in range(n):
            sampler.feed(dict(synth(k), timer=k % 256), None if k == 0 else image(k))
        return {"result": "loss", "frames": n, "moves": [["hp", 0, 20], ["crouch", 20, 40]]}
    return play


def test_collect_pair_spends_the_budget_and_resumes_without_reusing_games(tmp_path):
    base = str(tmp_path / "ryu_vs_ken")
    rec = IO.collect_pair(base, fake_play(), "ryu", "ken", 2, 0, BANDS, log=lambda *a: None)
    assert rec["games"] == 2 and rec["reason"] == "budget spent"
    games = IO.committed(base)
    assert [g["game"] for g in games] == [0, 1]
    assert games[0]["moves"][0][:3] == ["hp", 0, 20] and games[0]["moves"][0][3] in ("done", "missed", "interrupted")
    assert games[0]["pair"] == ["ryu", "ken"]
    # a crashed game 2 left an image: the next game is 3, and the budget of 3 is spent with it
    os.makedirs(os.path.join(base, "images"), exist_ok=True)
    open(os.path.join(base, "images", "g0002_k00010.png"), "w").close()
    IO.collect_pair(base, fake_play(), "ryu", "ken", 3, 0, BANDS, log=lambda *a: None)
    assert [g["game"] for g in IO.committed(base)] == [0, 1, 3]
    assert json.load(open(os.path.join(base, "stop.json")))["games"] == 3
    kept, dropped = IO.committed_pairs(base)
    assert kept and dropped == 0


def test_the_memory_cap_stops_the_pair(tmp_path):
    with pytest.raises(IO.MemoryCapExceeded):
        IO.collect_pair(str(tmp_path / "a"), fake_play(), "ryu", "ken", 3, 0, BANDS, mem_cap_gb=1.0,
                        rss_gb=lambda: 2.0, log=lambda *a: None)
    assert len(IO.committed(str(tmp_path / "a"))) == 1


# ---- Plan B: 2P versus, both controllers ours ----------------------------------------------------------------------

BUTTONS_PHYS = ("y", "x", "l", "b", "a", "r")


class Bridge2:
    """Two controllers: each player is busy (attack state) for 10 frames after its own button press; player 2 is on
    the right facing left; every frame's inputs per controller are recorded. No CPU: nothing moves on its own."""
    END = 1200

    def __init__(self):
        self.load_state(b"")

    def load_state(self, state):
        self.f, self.busy, self.inputs = 0, {1: 0, 2: 0}, {1: [], 2: []}

    def _row(self):
        r = synth(0)
        r.update(p1_state=0x0A if self.busy[1] else 0, p2_state=0x0A if self.busy[2] else 0, p1_x=200, p2_x=300,
                 p1_facing=0x40, p2_facing=0x00, p1_y=GROUND_Y, p2_y=GROUND_Y,
                 result=1 if self.f >= self.END else 0)
        return [r[n] for n in NAMES_X]

    def run(self, frames, p2=None):
        from sf2.emu.mesen import Obs
        if p2 is None or len(p2) != len(frames):
            raise AssertionError("both controllers must be driven every frame")
        rams = [self._row()]
        for b1, b2 in zip(frames, p2):
            self.f += 1
            for p, b in ((1, b1), (2, b2)):
                self.inputs[p].append(list(b))
                if any(x in b for x in BUTTONS_PHYS):
                    self.busy[p] = 10
                elif self.busy[p]:
                    self.busy[p] -= 1
            rams.append(self._row())
        return Obs(rams)


class Stream2(Stream):
    def run(self, frames, p2=None):
        obs = self.inner.run(frames, p2=p2)
        self.rows += obs.rams if not self.rows else obs.rams[1:]
        return obs


def _play_both(seed=5):
    inner = Bridge2()
    br = Stream2(inner)
    chars = {1: "ryu", 2: "zangief"}
    cycles = {p: PM.Cycle(list(PM.moves(chars[p])), random.Random("%d:%d" % (seed, p))) for p in (1, 2)}
    out = PC.play_both(br, NAMES_X, chars, cycles, b"", random.Random(0), lambda: len(br.rows))
    return inner, br, chars, out


def test_play_both_drives_each_side_through_its_own_cycle():
    inner, br, chars, out = _play_both()
    assert out["result"] == "win"
    for p in (1, 2):
        mine = [m for m in out["moves"] if m[3] == p]
        expect = PM.Cycle(list(PM.moves(chars[p])), random.Random("5:%d" % p))
        assert [m[0] for m in mine] == [expect.next() for _ in mine]
        assert len(mine) > len(PM.moves(chars[p]))           # its cycle starts again
        ks = [(m[1], m[2]) for m in mine]
        assert all(a < b for a, b in ks) and all(ks[i][1] == ks[i + 1][0] for i in range(len(ks) - 1))
        assert out["cycle_rounds"][p] >= 2


def test_play_both_executes_each_word_in_full_on_its_own_controller_and_nothing_else():
    inner, br, chars, out = _play_both()
    first_k = min(m[1] for m in out["moves"])
    rows = [dict(zip(NAMES_X, r)) for r in br.rows]
    for p in (1, 2):
        mine = [m for m in out["moves"] if m[3] == p]
        sent = inner.inputs[p]
        for i, (w, k0, k1, _) in enumerate(mine[:-1]):
            want = PC.press_frames(chars[p], w, rows[k0], p)
            got = sent[k0:k0 + len(want)]
            assert got == want, (p, w)
            # after the word's input and until its next word: nothing pressed (no CPU, no filler)
            assert all(f == [] for f in sent[k0 + len(want):k1]), (p, w)
        assert all(f == [] for f in sent[:first_k])       # the random idle start


def test_press_frames_resolve_forward_per_player():
    r = dict(synth(0), p1_x=200, p2_x=300, p1_facing=0x40, p2_facing=0x00)
    assert PC.press_frames("ryu", "forward", r, 1)[0] == ["right"]
    assert PC.press_frames("ryu", "forward", r, 2)[0] == ["left"]
    assert PC.press_frames("ryu", "block_high", r, 2)[0] == ["right"]      # back = away from player 1
    swapped = dict(r, p1_x=320)                                            # crossed over: x decides walks
    assert PC.press_frames("ryu", "forward", swapped, 2)[0] == ["right"]
    assert PC.can_act(dict(r, p2_state=0x0A), 1) and not PC.can_act(dict(r, p2_state=0x0A), 2)


def test_executed_reads_the_slot_it_is_asked_about():
    base = dict(synth(0), p1_state=0, p2_state=0, p1_aid=0, p2_aid=0, result=0)
    rows = [base, dict(base, p2_state=0x0A, p2_aid=4), base]
    assert PM.executed("ryu", "hp", rows, 2)["status"] == "done"
    assert PM.executed("ryu", "hp", rows, 1)["status"] != "done"
    assert PM.as_p1(rows[1], 2)["p1_aid"] == 4 and PM.as_p1(rows[1], 2)["p2_aid"] == 0
    with pytest.raises(ValueError):
        PM.as_p1(rows[1], 3)


def test_vs_sampler_labels_both_players_as_ours():
    s = PC.PairSampler(0, {1: "ryu", 2: "ken"}, random.Random(0), saver({}), BANDS, controllers=PC.VS_SLOTS)
    feed(s, 600)
    pairs = s.finish()
    assert {(p["slot"], p["controller"]) for p in pairs} == {(1, "p1"), (2, "p2")}
    with pytest.raises(ValueError):
        PC.PairSampler(0, {1: "ryu", 2: "ken"}, random.Random(0), saver({}), BANDS, controllers={1: "p1"})


def test_vs_move_log_has_both_slots_with_their_status(tmp_path):
    def play(game, sampler):
        for k in range(300):
            sampler.feed(dict(synth(k), timer=k % 256), None if k == 0 else image(k))
        return {"result": "win", "frames": 300, "moves": [["hp", 0, 20, 1], ["crouch", 27, 35, 2]]}
    base = str(tmp_path / "ryu_vs_ken")
    IO.collect_pair(base, play, "ryu", "ken", 1, 0, BANDS, log=lambda *a: None, controllers=PC.VS_SLOTS)
    g = IO.committed(base)[0]
    assert [m[4] for m in g["moves"]] == [1, 2]
    # synth: in frames 27-35 player 2 crouches (state 0x02) while player 1 jumps: done only when read as player 2
    assert g["moves"][1][3] == "done"
    rows = [dict(synth(k), timer=k % 256) for k in range(300)]
    assert PM.executed("ken", "crouch", rows[27:36], 1)["status"] != "done"
    assert {p["controller"] for p in IO.committed_pairs(base)[0]} == {"p1", "p2"}


def test_cursor_plans_cover_the_swap_with_a_parked_player_2():
    from sf2.emu.vs import CURSOR_START, cursor_plans
    from sf2.vocab import IDS
    plans = cursor_plans(IDS["ken"], IDS["ryu"])             # 1P wants 2P's start, 2P wants 1P's start
    assert plans[:2] == [[(1, IDS["ken"]), (2, IDS["ryu"])], [(2, IDS["ryu"]), (1, IDS["ken"])]]
    parked = [p for p in plans if len(p) == 3]
    assert parked and all(p[0][0] == 2 and p[0][1] not in (IDS["ken"], IDS["ryu"], *CURSOR_START.values())
                          and p[1:] == [(1, IDS["ken"]), (2, IDS["ryu"])] for p in parked)
    for a in range(8):
        for b in range(8):
            if a != b:
                assert all(p[-2:] in ([(1, a), (2, b)], [(2, b), (1, a)]) for p in cursor_plans(a, b))
