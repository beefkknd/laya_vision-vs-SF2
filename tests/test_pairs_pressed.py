"""Attack vs special from the move WE pressed (docs/prereg_movement_pairs.md, "Owner after round 1"): the pressed word
per row from the move log (a word started on row k0 owns rows k0 + 1 .. k1), its kind from the hardcoded list
(normal / crouching normal / jump attack / throw -> attack, the character's special -> special), confirmed by RAM (the
fighter in an attack state); otherwise the RAM rule, counted as a fallback. The collector records it, the builder
relabels every pair from RAM + the log (round 1 too), the gate re-derives it independently."""
import json
import os
import random

import pytest

from sf2.data import movement_collect_io as MIO
from sf2.data import pairs_collect as PC
from sf2.data import pairs_collect_io as IO
from sf2.data import pairs_data as D
from sf2.data import pairs_gate as G
from sf2.data import pairs_labels as L
from sf2.data import pairs_moves as PM
from sf2.emu.vs import GROUND_Y
from test_pairs_collect import BANDS, Bridge2, NAMES_X, Stream2, image, saver, synth
from test_pairs_labels import prow


# ---- the pressed word per row ---------------------------------------------------------------------------------------

def test_pressed_words_own_rows_after_their_start_up_to_the_next_start():
    moves = [["hp", 3, 7, "done", 1], ["forward", 7, 9, "done", 1], ["clothesline", 2, 6, "done", 2]]
    w = PM.pressed_words(moves, 10)
    assert w[1] == [None] * 4 + ["hp"] * 4 + ["forward"] * 2
    assert w[2] == [None] * 3 + ["clothesline"] * 4 + [None] * 3
    # the old P1-vs-CPU log: [word, k0, k1(, status)] is player 1's
    assert PM.pressed_words([["lp", 0, 2], ["mp", 2, 3, "done"]], 4)[1] == [None, "lp", "lp", "mp"]


def test_pressed_words_refuse_overlaps_and_rows_out_of_range():
    with pytest.raises(ValueError):
        PM.pressed_words([["hp", 0, 5, "done", 1], ["lp", 3, 8, "done", 1]], 10)
    with pytest.raises(ValueError):
        PM.pressed_words([["hp", 0, 12, "done", 1]], 10)


@pytest.mark.parametrize("word, want", [("hp", "attack"), ("c.mk", "attack"), ("sweep", "attack"),
                                        ("jump_forward_hk", "attack"), ("throw", "attack"),
                                        ("clothesline", "special"), ("spinning_piledriver", "special"),
                                        ("forward", None), ("block_high", None), ("jump", None), (None, None)])
def test_pressed_class_from_the_hardcoded_list(word, want):
    assert PM.pressed_class("zangief", word) == want


# ---- the label --------------------------------------------------------------------------------------------------

def _at(over):
    return [prow(over)] * 5


@pytest.mark.parametrize("over, pressed, want, source", [
    ({"state": 0x0A, "mclass": 0x0A}, "special", "special", "pressed"),     # Zangief's clothesline (round 1 zero)
    ({"state": 0x0A, "mclass": 0x08}, "attack", "attack", "pressed"),       # RAM would say special
    ({"state": 0x0C}, "attack", "attack", "pressed"),
    ({"state": 0x04, "y": GROUND_Y - 30, "aid": 3}, "attack", "attack", "pressed"),   # jump attack
    ({"state": 0x04, "y": GROUND_Y - 30, "aid": 3}, "special", "special", "pressed"),  # an air special
    ({"state": 0x0A, "mclass": 0x08}, None, "special", "fallback"),           # nothing of ours: the RAM rule
    ({"state": 0x0A}, None, "attack", "fallback"),
    ({"state": 0x00}, "special", "stand", "ram"),                              # pressed, not confirmed by RAM
    ({"state": 0x0E, "react": 2}, "attack", "hit", "ram"),
])
def test_movement_from_the_pressed_move_confirmed_by_ram(over, pressed, want, source):
    rows = _at(over)
    assert L.movement_pressed(rows, 4, 1, pressed) == (want, source)
    assert L.labels(rows, 4, 1, BANDS, pressed)["movement"] == want
    assert G.independent_labels(rows, 4, 1, BANDS, pressed)["movement"] == want


def test_labels_without_pressed_info_are_the_ram_rule():
    rows = _at({"state": 0x0A, "mclass": 0x0A})
    assert L.labels(rows, 4, 1, BANDS)["movement"] == "attack"


# ---- the collector records it ------------------------------------------------------------------------------------

def test_play_both_announces_each_word_with_its_start_row():
    inner, seen = Bridge2(), []
    br = Stream2(inner)
    chars = {1: "ryu", 2: "zangief"}
    cycles = {p: PM.Cycle(list(PM.moves(chars[p])), random.Random(p)) for p in (1, 2)}
    out = PC.play_both(br, NAMES_X, chars, cycles, b"", random.Random(0), lambda: len(br.rows),
                       on_word=lambda p, w, k0: seen.append([w, k0, p]))
    assert sorted(seen, key=lambda m: (m[1], m[2])) == [[m[0], m[1], m[3]] for m in out["moves"]]


def test_the_sampler_labels_with_the_pressed_move_and_records_it():
    chars = {1: "zangief", 2: "ken"}
    s = PC.PairSampler(0, chars, random.Random(0), saver({}), BANDS, controllers=PC.VS_SLOTS)
    rows = []
    for k in range(400):
        r = dict(synth(k), p1_char=6, p2_char=4)
        rows.append(r)
        s.feed(r, None if k == 0 else image(k))
        if k % 40 == 0:                                   # a word starts on the last row fed, as in play_both
            s.press(1, "clothesline" if k % 80 == 0 else "hp", k)
            s.press(2, "hadoken" if k % 80 else "lk", k)
    with pytest.raises(ValueError):
        s.press(1, "hp", 3)
    pairs = s.finish()
    assert pairs
    for p in pairs:
        cls = PM.pressed_class(chars[p["slot"]], p["pressed"])
        want = L.labels(rows, p["t"], p["slot"], BANDS, cls)
        assert {k: p[k] for k in want} == want
        assert p["pressed_class"] == cls and p["mv_source"] == L.movement_pressed(rows, p["t"], p["slot"], cls)[1]
    assert any(p["movement"] == "special" and p["mv_source"] == "pressed" for p in pairs)


# ---- the builder relabels from RAM + the log; the gate re-derives -------------------------------------------------

def _stale_collection(root, games=2):
    """A Plan B collection written WITHOUT pressed info (as round 1): every attack-state row of player 1 is under a
    special word in the log, so the relabel turns RAM-rule "attack" into "special"."""
    def play(game, sampler):
        for k in range(500):
            sampler.feed(dict(synth(k), timer=k % 256), None if k == 0 else image(k))
        special = sorted(PM.specials_of(sampler.chars[1]))[0]
        moves = [[special, k0, min(k0 + 50, 499), 1] for k0 in range(0, 500, 50)]
        moves += [["lk", k0, k0 + 100, 2] for k0 in range(0, 400, 100)]
        return {"result": "win", "frames": 500, "moves": moves}
    for a, b in (("zangief", "ken"), ("ken", "zangief")):
        IO.collect_pair(os.path.join(root, IO.pair_name(a, b)), play, a, b, games, 0, BANDS, per_game=3,
                        log=lambda *x: None, controllers=PC.VS_SLOTS)


def test_the_builder_relabels_every_pair_from_ram_and_the_move_log(tmp_path):
    root, out = str(tmp_path / "col"), str(tmp_path / "data")
    _stale_collection(root)
    stale = [p for n in os.listdir(root) for p in IO.committed_pairs(os.path.join(root, n))[0]]
    assert any(p["slot"] == 1 and p["movement"] == "attack" for p in stale)
    meta = D.build(root, out, {"train": 40, "test": 20}, bands=BANDS)
    rows = [r for f in ("train", "test") for r in MIO.read_jsonl(os.path.join(out, "movement", f + ".jsonl"))]
    assert rows
    assert not [r for r in rows if r["slot"] == 1 and r["movement"] == "attack"]       # all under clothesline
    assert any(r["slot"] == 1 and r["movement"] == "special" and r["mv_source"] == "pressed" for r in rows)
    assert all(r["movement"] != "special" for r in rows if r["slot"] == 2 and r["mv_source"] == "pressed")
    for r in rows:
        ram = MIO.read_ram(os.path.join(root, r["pair_name"], "ram", "g%04d.json.gz" % r["game"]))
        cls = PM.pressed_class(r["char"], r["pressed"])
        assert r["movement"] == L.labels(ram, r["t"], r["slot"], BANDS, cls)["movement"]
    # the per-game cap still holds after the relabel
    per = {}
    for r in rows:
        k = (r["pair_name"], r["game"], r["slot"]) + D.cell(r)
        per[k] = per.get(k, 0) + 1
    assert max(per.values()) <= PC.PER_GAME
    assert sum(meta["mv_sources"][c].get("pressed", 0) for c in meta["mv_sources"]) > 0
    rep = G.run_gates(out, BANDS, sample=50, min_disc=0)
    assert rep["gates"]["labels"]["pass"] and rep["gates"]["caps"]["pass"], json.dumps(rep["gates"])[:1500]


def test_a_wrong_pressed_word_fails_the_label_gate(tmp_path):
    root, out = str(tmp_path / "col"), str(tmp_path / "data")
    _stale_collection(root)
    D.build(root, out, {"train": 40, "test": 20}, bands=BANDS)
    path = os.path.join(out, "movement", "train.jsonl")
    rows = MIO.read_jsonl(path)
    i = next(i for i, r in enumerate(rows) if r["slot"] == 1 and r["pressed"])
    rows[i] = dict(rows[i], pressed="hp")
    with open(path, "w") as f:
        f.writelines(json.dumps(r) + "\n" for r in rows)
    assert not G.run_gates(out, BANDS, sample=50, min_disc=0)["gates"]["labels"]["pass"]


def test_the_live_record_equals_the_one_rebuilt_from_the_move_log():
    """The relabel of round 1 rests on this: the per-row pressed word the sampler records live (play_both's on_word)
    is exactly the one pressed_words rebuilds from the logged moves."""
    chars = {1: "zangief", 2: "dhalsim"}
    s = PC.PairSampler(0, chars, random.Random(0), saver({}), BANDS, controllers=PC.VS_SLOTS)

    class Feed(Stream2):
        def run(self, frames, p2=None):
            obs = self.inner.run(frames, p2=p2)
            new = obs.rams if not self.rows else obs.rams[1:]
            for r in new:
                s.feed(dict(zip(NAMES_X, r)), None if not self.rows and r is obs.rams[0] else image(len(s.rows)))
            self.rows += new
            return obs
    br = Feed(Bridge2())
    cycles = {p: PM.Cycle(list(PM.moves(chars[p])), random.Random(p)) for p in (1, 2)}
    out = PC.play_both(br, NAMES_X, chars, cycles, b"", random.Random(0), lambda: len(s.rows), on_word=s.press)
    rebuilt = PM.pressed_words(IO.move_log(chars, out["moves"], s.rows), len(s.rows))
    assert rebuilt[1] == s.pressed[1] and rebuilt[2] == s.pressed[2]
    assert any(w for w in s.pressed[2])


def test_the_gates_independent_pressed_word_agrees_with_pressed_words_on_every_row():
    rng = random.Random(1)
    for _ in range(50):
        moves, k = [], {1: rng.randrange(5), 2: rng.randrange(5)}
        for _ in range(12):
            p = rng.choice((1, 2))
            k1 = k[p] + rng.randrange(1, 9)
            if k1 < 80:
                moves.append([rng.choice(["hp", "forward", "hadoken"]), k[p], k1, "done", p])
                k[p] = k1
        want = PM.pressed_words(moves, 80)
        for p in (1, 2):
            assert [G.independent_pressed(moves, t, p) for t in range(80)] == want[p]
    assert G.independent_pressed([["hp", 3, 7, "done", 1], ["lk", 7, 9, "done", 1]], 7, 1) == "hp"
    assert G.independent_pressed([["hp", 3, 7, "done", 1], ["lk", 7, 9, "done", 1]], 3, 1) is None
