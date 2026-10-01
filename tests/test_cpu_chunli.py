"""CPU Chun-Li games (docs/prereg_movement_data.md, "Addition: CPU Chun-Li games"): another character is player 1,
CPU Chun-Li is player 2. Only Chun-Li's episodes are sampled, the stop rule counts only new Chun-Li (code, split),
every game must really hold (player-1 character, Chun-Li) in RAM, every pair and row carries its provenance, the
builder (sf2.data.action_data_cpu) keeps only Chun-Li's rows asked about "Chun-Li (him)", and the gate checks labels,
completeness and provenance against the collection's run.json. Synthetic streams from the real collector loop."""
import json
import os
import random
from collections import Counter

import numpy as np
import pytest
from ram_rows import row

from sf2.data import action_codes as A
from sf2.data import action_collect as C
from sf2.data import action_collect_io as IO
from sf2.data import action_data_cpu as DC
from sf2.data import action_gate as G
from sf2.data import cpu_chunli as K
from sf2.emu.vs import GROUND_Y
from sf2.vocab import IDS

NONE = {"aid": 0, "mclass": 0xFF, "sclass": 0xFF}
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TH = os.path.join(ROOT, "lessons", "perception_thresholds_v2.json")
P1 = "ken"
ACTORS = {1: P1, 2: "chunli"}
EXPECT = {1: IDS[P1], 2: IDS["chunli"]}


def r(p1=None, p2=None, p1_char=IDS[P1], **extra):
    return row(dict(NONE, char=p1_char, **(p1 or {})), dict(NONE, char=IDS["chunli"], **(p2 or {})), **extra)


def image(k):
    im = np.zeros((224, 256, 3), np.uint8)
    im[0, 0] = (k // 256 % 256, k % 256, 7)
    return im


def script(game, p1_code=9, p1_char=IDS[P1]):
    """Him (player 1): stand, an attack (``p1_code``), stand. Chun-Li (player 2): walk in, a jump with an attack
    (ID 25), a cut attack (unknown), crouch."""
    rows = []
    for i in range(12):
        rows.append(r({"x": 200}, {"x": 300 - 2 * i}, p1_char))
    for i in range(9):
        rows.append(r({"x": 200, "state": 0x0A, "aid": p1_code if i >= 2 else 0},
                      {"x": 276, "state": 0x04, "y": GROUND_Y - 10 - i, "aid": 25 if i >= 3 else 0}, p1_char))
    for i in range(5):
        rows.append(r({"x": 200}, {"x": 276, "state": 0x0A}, p1_char))
    for i in range(10):
        rows.append(r({"x": 200}, {"x": 230 + game % 3, "state": 0x02}, p1_char))
    return rows


def player(fn=script):
    def play(game, sampler):
        for k, x in enumerate(fn(game)):
            sampler.feed(x, None if k == 0 else image(k))
        return {}
    return play


def collect(base, fn=script, **kw):
    args = dict(seed=0, cap=10, min_games=100, patience=50, log=lambda *a: None, sample_actors={"chunli"},
                provenance=K.provenance(P1), expect_ids=EXPECT)
    args.update(kw)
    return IO.collect_opponent(base, player(fn), P1, ACTORS, **args)


# ---- provenance ---------------------------------------------------------------------------------------------------

def test_provenance_fields_and_values():
    p = K.provenance("guile")
    assert p == {"collection": "mv4", "chunli_slot": "p2", "controller": "cpu", "p1_char": "guile",
                 "note": "me=guile"}
    assert set(p) == set(K.PROVENANCE_KEYS)


@pytest.mark.parametrize("bad", ["chunli", "nobody", ""])
def test_provenance_refuses_bad_player_one(bad):
    with pytest.raises(ValueError):
        K.provenance(bad)


def test_provenance_problems_names_each_wrong_or_missing_field():
    want = K.provenance("ken")
    assert K.provenance_problems(dict(want, x=1), want) == []
    for k in K.PROVENANCE_KEYS:
        assert K.provenance_problems(dict(want, **{k: "other"}), want), k
        assert K.provenance_problems({x: v for x, v in want.items() if x != k}, want), k


def test_run_settings_hold_the_provenance_constants():
    s = K.run_settings(["ken", "ryu"], model="m", explore=0.5, seed=0, cap=150, min_games=40, patience=15)
    assert (s["collection"], s["chunli_slot"], s["controller"], s["cpu"]) == ("mv4", "p2", "cpu", "chunli")
    assert s["p1_chars"] == ["ken", "ryu"] and s["explore"] == 0.5
    assert K.expected_provenance(s, "ryu") == K.provenance("ryu")
    with pytest.raises(ValueError):
        K.expected_provenance(s, "guile")          # not a player-1 character of this run


# ---- P2 Chun-Li selection: every game's RAM must show (player-1 char, Chun-Li) -----------------------------------

def test_char_problems():
    good = script(0)
    assert IO.char_problems(good, EXPECT) == []
    assert IO.char_problems([], EXPECT)
    wrong_p2 = [dict(x, p2_char=IDS["ken"]) for x in good]
    assert any("player 2" in m for m in IO.char_problems(wrong_p2, EXPECT))
    wrong_p1 = good[:5] + [dict(good[5], p1_char=IDS["ryu"])] + good[6:]
    assert any("player 1" in m for m in IO.char_problems(wrong_p1, EXPECT))


def test_a_game_with_the_wrong_characters_is_not_committed(tmp_path):
    base = str(tmp_path / P1)
    with pytest.raises(IO.WrongCharacters):
        collect(base, fn=lambda g: script(g, p1_char=IDS["ryu"]) if g == 2 else script(g))
    games = [g["game"] for g in IO.read_games(base)]
    assert games == [0, 1]
    assert {p["game"] for p in IO.read_jsonl(os.path.join(base, "pairs.jsonl"))} <= {0, 1}
    assert json.load(open(os.path.join(base, "stop.json")))["reason"] == "wrong characters"


# ---- only Chun-Li's episodes are sampled; the stop rule counts only her ------------------------------------------

def run_sampler(rows, sample_actors):
    s = C.ActionSampler(1, "train", ACTORS, Counter(), 10 ** 6, random.Random(0), lambda k, im: "k%d" % k,
                        sample_actors=sample_actors)
    for k, x in enumerate(rows):
        s.feed(x, None if k == 0 else image(k))
    return s, s.finish()


def test_sampler_samples_only_the_named_actor_but_observes_both():
    s, pairs = run_sampler(script(1), {"chunli"})
    assert pairs and {p["actor"] for p in pairs} == {"chunli"} and {p["player"] for p in pairs} == {2}
    assert s.observed[(P1, 9)] >= 1 and s.observed[("chunli", 25)] >= 1
    _, both = run_sampler(script(1), None)
    assert {p["actor"] for p in both} == {P1, "chunli"}


def test_sampler_refuses_an_unknown_sample_actor():
    with pytest.raises(ValueError):
        run_sampler(script(1), {"ryu"})


def test_stop_rule_counts_only_new_chunli_codes(tmp_path):
    # player 1 brings a NEW code every game; Chun-Li always the same: the stop must not wait for player 1
    rec = collect(str(tmp_path / P1), fn=lambda g: script(g, p1_code=1 + g % 50), cap=100, min_games=5, patience=3)
    assert rec["reason"] == "no new action" and rec["games"] == 6 and rec["last_new_game"] == 2


def test_novelty_ignores_other_actors():
    prog = IO.Progress()
    pairs = [{"actor": P1, "code": 40, "split": "train", "stg": 1}]
    after = IO.advance(prog, 1, pairs, actors={"chunli"})
    assert after.since_new == 1 and after.last_new_game is None and not after.seen
    after = IO.advance(prog, 1, pairs + [{"actor": "chunli", "code": 25, "split": "train", "stg": 1}],
                       actors={"chunli"})
    assert after.since_new == 0 and after.seen == {("chunli", 25, "train")}


def test_resume_keeps_the_chunli_only_novelty(tmp_path):
    base = str(tmp_path / P1)
    collect(base, fn=lambda g: script(g, p1_code=1 + g % 50), cap=4)
    prog = IO.load_progress(base, actors={"chunli"})
    assert prog.played == 4 and all(a == "chunli" for a, _, _ in prog.seen)


def test_pairs_and_games_carry_the_provenance(tmp_path):
    base = str(tmp_path / P1)
    rec = collect(base, cap=3)
    pairs, _ = IO.committed_pairs(base)
    assert pairs and all(K.provenance_problems(p, K.provenance(P1)) == [] for p in pairs)
    assert all(K.provenance_problems(g, K.provenance(P1)) == [] for g in IO.read_games(base))
    assert rec["provenance"] == K.provenance(P1) and rec["sample_actors"] == ["chunli"]


def test_collect_refuses_a_provenance_for_another_player_one(tmp_path):
    with pytest.raises(ValueError):
        collect(str(tmp_path / P1), provenance=K.provenance("ryu"))


# ---- the builder and the gate --------------------------------------------------------------------------------------

@pytest.fixture
def built(tmp_path):
    root = str(tmp_path / "mv4")
    collect(os.path.join(root, P1))
    K.write_run(root, K.run_settings([P1], model="m", explore=0.5, seed=0, cap=10, min_games=100, patience=50))
    out = str(tmp_path / "act_cpu")
    res = DC.build(root, out, thresholds=TH)
    return root, out, res


def rows_of(out, p1=P1):
    rows = []
    for f in ("train", "val", "test_real"):
        rows += [json.loads(x) for x in open(os.path.join(out, p1, f + ".jsonl"))]
    return rows


def test_build_keeps_only_chunli_rows_asked_about_him(built):
    _, out, res = built
    assert res["problems"] == []
    rows = rows_of(out)
    assert rows and {(x["actor"], x["player"]) for x in rows} == {("chunli", 2)}
    assert {x["question"]["instructions"] for x in rows} == {"What move is Chun-Li (him) doing?",
                                                            "Which part of Chun-Li's (him) move is this?"}
    assert {x["state_text"] for x in rows} == {"me=%s" % P1}
    assert any(x["answer"] == "act25" for x in rows)
    by_dec = Counter(x["decision"] for x in rows)
    assert set(by_dec.values()) == {2}                   # act + stage per pair
    for x in rows:
        assert x["label_full"] == A.label("chunli", x["code"], x["stg"])
        assert list(x["question"]["criteria"])[x["label"]] == x["answer"]


def test_build_rows_carry_provenance_and_split_by_game(built):
    _, out, _ = built
    for f, split in (("train", "train"), ("val", "val"), ("test_real", "test")):
        for line in open(os.path.join(out, P1, f + ".jsonl")):
            x = json.loads(line)
            assert x["split"] == split == C.split_of_game(x["game"])
            assert K.provenance_problems(x, K.provenance(P1)) == []
    meta = json.load(open(os.path.join(out, "build.json")))
    assert meta["layout"] == "cpu_chunli" and meta["opps"] == [P1] and meta["players"] == [2]


def test_gate_passes_on_a_clean_build(built):
    _, out, _ = built
    rep = G.run_gates(out, TH, check_train=False)
    for g in ("labels", "provenance", "disk"):
        assert rep["gates"][g]["pass"], (g, rep["gates"][g])
    assert rep["gates"]["labels"]["mismatches"] == 0
    cov = rep["coverage"]
    assert "chunli act25" in cov["table"] and not any(k.startswith(P1) for k in cov["table"])


def _tamper(path, fn):
    rows = [json.loads(x) for x in open(path)]
    with open(path, "w") as f:
        f.write("".join(json.dumps(x) + "\n" for x in fn(rows)))


@pytest.mark.parametrize("fn", [
    lambda rs: [dict(x, answer="stg1", label=0) if x["key"] == "stage" and x["answer"] == "stg3" else x for x in rs],
    lambda rs: [x for x in rs if x["key"] != "stage"],
    lambda rs: [dict(x, code=x["code"] + 1) for x in rs],
    lambda rs: [dict(x, actor=P1) for x in rs],
    lambda rs: rs + [dict(x, player=1, actor=P1, id=x["id"] + "-p1") for x in rs][:1],
])
def test_gate_labels_catch_a_wrong_row(built, fn):
    _, out, _ = built
    _tamper(os.path.join(out, P1, "train.jsonl"), fn)
    assert not G.run_gates(out, TH, check_train=False)["gates"]["labels"]["pass"]


@pytest.mark.parametrize("key,value", [("collection", "mv3"), ("chunli_slot", "p1"), ("controller", "policy"),
                                       ("p1_char", "ryu"), ("note", "me=chunli"), ("state_text", "me=chunli")])
def test_gate_provenance_catches_a_wrong_field(built, key, value):
    _, out, _ = built
    _tamper(os.path.join(out, P1, "val.jsonl"), lambda rs: [dict(x, **{key: value}) if i == 0 else x
                                                            for i, x in enumerate(rs)])
    assert not G.run_gates(out, TH, check_train=False)["gates"]["provenance"]["pass"]


def test_gate_provenance_catches_a_missing_field(built):
    _, out, _ = built
    _tamper(os.path.join(out, P1, "test_real.jsonl"),
            lambda rs: [{k: v for k, v in x.items() if k != "controller"} if i == 0 else x for i, x in enumerate(rs)])
    assert not G.run_gates(out, TH, check_train=False)["gates"]["provenance"]["pass"]


def test_gate_provenance_is_checked_against_the_run_settings(built):
    root, out, _ = built
    run = json.load(open(os.path.join(root, "run.json")))
    with open(os.path.join(root, "run.json"), "w") as f:
        json.dump(dict(run, controller="policy"), f)
    assert not G.run_gates(out, TH, check_train=False)["gates"]["provenance"]["pass"]


def test_gate_provenance_fails_without_run_json(built):
    root, out, _ = built
    os.remove(os.path.join(root, "run.json"))
    assert not G.run_gates(out, TH, check_train=False)["gates"]["provenance"]["pass"]


def test_builder_refuses_pairs_without_provenance_or_not_chunli(tmp_path, built):
    root, _, _ = built
    p = os.path.join(root, P1, "pairs.jsonl")
    _tamper(p, lambda rs: [dict(x, controller="policy") if i == 0 else x for i, x in enumerate(rs)])
    res = DC.build(root, str(tmp_path / "again1"), thresholds=TH)
    assert any("provenance" in x for x in res["problems"])
    _tamper(p, lambda rs: [dict(x, controller="cpu", actor=P1, player=1) if i == 0 else x for i, x in enumerate(rs)])
    res = DC.build(root, str(tmp_path / "again2"), thresholds=TH)
    assert any("not a Chun-Li pair" in x for x in res["problems"])


def test_builder_refuses_an_existing_out(built):
    root, out, _ = built
    with pytest.raises(SystemExit):
        DC.build(root, out, thresholds=TH)


def test_compare_codes_lists_new_codes_and_the_watched_ids():
    cmp = K.compare_codes({"chunli act25": {"train": 3, "val": 0, "test": 1, "episodes_seen": 4},
                           "chunli act02": {"train": 2, "val": 0, "test": 0, "episodes_seen": 2}},
                          mv3_codes=[2, 70])
    assert cmp["new_vs_mv3"] == ["act25"] and cmp["watched"]["act25"]["train"] == 3
    assert cmp["watched"]["act23"] is None and set(cmp["watched"]) == {"act%02d" % i for i in K.WATCHED}


# ---- the HUD clock read by its digit colours (the China stage's teal sky fools the blue mask) ----------------------

def _clock_images(tmp_path, digit_a, digit_b, sky_a, sky_b):
    from PIL import Image
    from sf2.data import movement_gate as MG
    names = []
    for i, (digit, sky) in enumerate(((digit_a, sky_a), (digit_b, sky_b))):
        im = np.zeros((256, 256, 3), np.uint8)
        im[MG.CLOCK] = (0, 189, 189) if sky else (66, 66, 66)        # teal sky or grey behind the clock
        y0, x0 = MG.CLOCK[0].start, MG.CLOCK[1].start
        im[y0 + 2:y0 + 10, x0 + 4 + digit:x0 + 8 + digit] = (255, 156, 107)   # an orange digit stroke
        im[y0 + 10, x0 + 4 + digit] = (24, 66, 173)                              # its blue outline
        Image.fromarray(im).save(str(tmp_path / ("c%d.png" % i)))
        names.append("c%d.png" % i)
    return str(tmp_path), names




@pytest.mark.parametrize("digit_b,sky_b,changed", [(0, False, False), (6, False, True), (0, True, False),
                                                    (6, True, True)])
def test_digits_changed(tmp_path, digit_b, sky_b, changed):
    from sf2.data import movement_gate as MG
    frames, (a, b) = _clock_images(tmp_path, 0, digit_b, False, sky_b)
    assert MG.digits_changed(frames, a, b) is changed


def test_blue_mask_is_fooled_by_the_sky(tmp_path):
    from sf2.data import movement_gate as MG
    frames, (a, b) = _clock_images(tmp_path, 0, 0, False, True)
    assert MG._clock_changed(frames, a, b) and not MG.digits_changed(frames, a, b)


def test_cpu_gate_reads_the_clock_by_digits_and_reports_the_blue_mask(built):
    _, out, _ = built
    rep = G.run_gates(out, TH, check_train=False)
    assert rep["gates"]["alignment"]["clock"] == "digits"
    assert rep["alignment_blue_mask"]["clock"] == "blue" and rep["alignment_blue_mask"]["report_only"]


def test_resume_does_not_count_another_actors_pairs_as_new(tmp_path):
    base = str(tmp_path / P1)
    collect(base, cap=4)
    with open(os.path.join(base, "pairs.jsonl"), "a") as f:      # a player-1 pair in a committed game
        f.write(json.dumps({"game": 1, "split": "train", "actor": P1, "code": 40, "stg": 1}) + "\n")
    assert (P1, 40, "train") not in IO.load_progress(base, actors={"chunli"}).seen
    assert (P1, 40, "train") in IO.load_progress(base).seen


def test_builder_refuses_a_collector_label_the_ram_does_not_give(tmp_path, built):
    root, _, _ = built
    _tamper(os.path.join(root, P1, "pairs.jsonl"),
            lambda rs: [dict(x, stg=1 + x["stg"] % 3) if i == 0 else x for i, x in enumerate(rs)])
    res = DC.build(root, str(tmp_path / "again"), thresholds=TH)
    assert any("collector label" in x for x in res["problems"])
