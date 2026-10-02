"""docs/eye_questions_v1.md, "Datasets v1": the four aligned datasets (sf2.data.eye_data), their shortcut check
(sf2.data.eye_shortcut) and gates (sf2.data.eye_gate), on a small 2P collection: Ryu vs Ken both ways, player 1 throws a
blinking hadoken (throwing pose before and after the spawn) and jumps, player 2 cycles stand / attack / special /
crouch and every other 40 rows stands close; games 0-1 have no blink bytes (as the real games < 10)."""
import collections
import json
import os

import pytest

from sf2.data import eye_data as E
from sf2.data import eye_gate as G
from sf2.data import eye_pool as P
from sf2.data import eye_shortcut as SC
from sf2.data import pairs_collect as PC
from sf2.data import pairs_collect_io as IO
from sf2.data import pairs_shots as S
from sf2.data import pairs_train as T
from sf2.emu.vs import GROUND_Y
from test_pairs_collect import BANDS, image
from test_pairs_labels import prow

N, GAMES, OLD = 240, 14, 2
WORDS = [(1, "hadoken", 33), (1, "forward", 60), (2, "hadoken", 99), (2, "back", 125)]
FLIGHTS = [(1, 45, 90), (2, 110, 140)]


def game_row(k, game):
    left, right = (200, 300) if game % 2 == 0 else (300, 200)
    p1 = {"x": left}
    if 34 <= k <= 55:
        p1["state"] = 0x0C                                 # the throwing pose, before and after the spawn at 45
    if 112 <= k <= 135:
        p1.update(state=0x04, y=GROUND_Y - 40)             # a jump in a flight (shot samples): take-off 112, landing 136
    p2 = dict({"x": right}, **[{}, {"state": 0x0A}, {"state": 0x0C}, {"state": 0x02}][(k % 40) // 10])
    if (k // 40) % 2 == 1:
        p2["x"] = left + (50 if right > left else -50)     # every other 40 rows close (gap 50 <= 72)
    r = dict(prow(p1, p2), timer=k % 256)
    if game >= OLD:
        r.update(shot1_hide=0, shot2_hide=0)
    for s, a, e in FLIGHTS:
        if a <= k <= e:
            r["shot%d" % s], r["shot%d_x" % s] = 1, 220 + (k - a) * (1 if s == 1 else -1)
            if game >= OLD and k % 4 == 3:
                r["shot%d_hide" % s] = 1
    return r


def drawn_image(k, shown):
    im = image(k)
    if any(shown["shot%d" % s] and not shown.get("shot%d_hide" % s, 0) & 1 for s in (1, 2)):
        im[120:140, 100:130] = (60, 120, 255)
    return im


def play(game, sampler):
    moves = []
    for i, (p, w, k0) in enumerate(WORDS):
        nxt = [k for q, _, k in WORDS[i + 1:] if q == p]
        moves.append([w, k0, nxt[0] if nxt else N - 1, p])
    for k in range(N):
        for p, w, k0 in WORDS:
            if k0 == k - 1:
                sampler.press(p, w, k0)
        sampler.feed(game_row(k, game), None if k == 0 else drawn_image(k, game_row(k - 1, game)))
    return {"result": "win", "frames": N, "moves": moves}


@pytest.fixture(scope="module")
def root(tmp_path_factory):
    r = str(tmp_path_factory.mktemp("eye") / "col")
    for a, b in (("ryu", "ken"), ("ken", "ryu")):
        for games, cls in ((OLD, PC.PairSampler), (GAMES, S.ShotSampler)):     # the old games: no shot sampler
            IO.collect_pair(os.path.join(r, IO.pair_name(a, b)), play, a, b, games, 0, BANDS, per_game=3,
                            log=lambda *x: None, controllers=PC.VS_SLOTS, sampler_cls=cls)
    return r


@pytest.fixture(scope="module")
def pool(root):
    return P.pool(root, BANDS["all"], workers=1)


@pytest.fixture(scope="module")
def built(root, pool, tmp_path_factory):
    tmp = tmp_path_factory.mktemp("eyeb")
    out = {}
    for q in E.Q:
        out[q] = str(tmp / E.out_of(q))
        E.build(pool, q, out[q], root, cap=None if q != "q1" else -1)
    return out


def _rows(out):
    return G.rows_of(out)


def test_the_question_texts_and_answers_are_the_docs():
    assert E.question("q1")["instructions"] == "Is there a fireball on the screen?"
    assert E.question("q3", "left")["instructions"] == "What is the fighter on the left doing?"
    assert E.question("q4", "right")["instructions"] == "Is the fighter on the right on the ground or in the air?"
    assert E.question("q5")["instructions"] == "Are the two fighters close or far?"
    assert [tuple(E.question(q, s)["criteria"]) for q, s in (("q1", None), ("q3", "left"), ("q4", "left"),
                                                              ("q5", None))] == [
        ("yes", "no"), ("moving", "attack", "special"), ("ground", "air"), ("close", "far")]
    assert "q2" not in E.Q
    with pytest.raises(ValueError):
        E.question("q3")
    with pytest.raises(ValueError):
        E.question("q1", "left")


@pytest.mark.parametrize("g, want", [(0, "g0-9"), (9, "g0-9"), (10, "g10-15"), (16, "g16-23"), (31, "g24-31"),
                                     (40, "g32+")])
def test_game_ranges(g, want):
    assert E.game_range(g) == want


def _c(i, stratum, answer, tier=2):
    return {"id": "r%03d" % i, "stratum": stratum, "answer": answer, "tier": tier}


def test_select_takes_equal_answers_per_stratum_with_the_cap_and_the_tiers_first():
    cands = [_c(i, "A", "yes") for i in range(5)] + [_c(10 + i, "A", "no") for i in range(3)]
    cands += [_c(20 + i, "B", "yes") for i in range(4)] + [_c(30, "B", "no", tier=0)] + [
        _c(31 + i, "B", "no") for i in range(5)]
    cands += [_c(40 + i, "C", "yes") for i in range(4)]                  # no "no" at all: nothing
    got = E.select(cands, ("yes", "no"), cap=None, seed=0)
    n = collections.Counter((c["stratum"], c["answer"]) for c in got)
    assert n == {("A", "yes"): 3, ("A", "no"): 3, ("B", "yes"): 4, ("B", "no"): 4}
    capped = E.select(cands, ("yes", "no"), cap=1, seed=0)
    assert collections.Counter((c["stratum"], c["answer"]) for c in capped) == {
        ("A", "yes"): 1, ("A", "no"): 1, ("B", "yes"): 1, ("B", "no"): 1}
    assert [c["id"] for c in capped if c["stratum"] == "B" and c["answer"] == "no"] == ["r030"]
    assert E.select(cands, ("yes", "no"), None, 0) == got and E.select(cands, ("yes", "no"), None, 1) != got


def test_q1_yes_and_no_from_the_same_strata_only_games_with_the_blink_byte(built, pool):
    rows = _rows(built["q1"])
    facts = {(f["pair_name"], f["game"], f["t"]): f for f in pool}
    assert {r["answer"] for r in rows} == {"yes", "no"} and all(r["game"] >= OLD for r in rows)
    for r in rows:
        f = facts[(r["pair_name"], r["game"], r["t"])]
        assert r["answer"] == f["fire"] == r["_dir"] and r["label"] == ["yes", "no"].index(r["answer"])
        assert r["stratum"].split("|")[:3] == [r["split"], r["pair_name"], E.game_range(r["game"])]
    per = collections.defaultdict(collections.Counter)
    for r in rows:
        per[r["stratum"]][r["answer"]] += 1
    assert all(c["yes"] == c["no"] > 0 for c in per.values())
    meta = json.load(open(os.path.join(built["q1"], "build.json")))
    assert meta["problems"] == [] and meta["dropped"]["no_blink_byte"] > 0
    assert sum(meta["extras"]["hard_negatives"].values()) > 0 and meta["extras"]["hard_tags"].get("pose", 0) > 0


def test_q1_matches_the_throwing_pose_so_pose_does_not_give_the_answer(built):
    rows = _rows(built["q1"])
    pose = collections.Counter((r["answer"], "projectile" in r["poses"]) for r in rows)
    assert pose[("yes", True)] == pose[("no", True)] > 0


def test_q3_rows_in_one_episode_by_side_equal_per_stratum(built, pool):
    rows = _rows(built["q3"])
    facts = {(f["pair_name"], f["game"], f["t"]): f for f in pool}
    assert {r["answer"] for r in rows} == {"moving", "attack", "special"}
    for r in rows:
        me = facts[(r["pair_name"], r["game"], r["t"])]["fighters"][r["slot"]]
        assert me["in_episode"] and me["act"] == r["answer"] and me["side"] == r["side"]
        assert r["question"]["instructions"] == "What is the fighter on the %s doing?" % r["side"]
    assert {r["side"] for r in rows} == {"left", "right"}


def test_q4_includes_take_off_and_landing(built):
    meta = json.load(open(os.path.join(built["q4"], "build.json")))
    assert sum(sum(v.values()) for v in meta["extras"]["take_off"].values()) > 0
    assert meta["problems"] == []


def test_q5_close_and_far_by_the_gap(built):
    for r in _rows(built["q5"]):
        assert (r["gap"] <= BANDS["all"]) == (r["answer"] == "close")


def test_one_split_table_for_every_dataset(built):
    rep = G.split_check({q: _rows(d) for q, d in built.items()})
    assert rep["pass"], rep
    assert all(r["split"] == T.split3(r["pair_name"], r["game"]) for d in built.values() for r in _rows(d))


def test_build_refuses_an_existing_out(root, pool, built):
    with pytest.raises(FileExistsError, match="never overwritten"):
        E.build(pool, "q5", built["q5"], root)


def test_equal_x_has_no_side_and_is_dropped_from_q3_q4_q5():
    from test_eye_pool import rows_with
    rows = rows_with(20, p2=lambda k: {"x": 200} if k >= 10 else {})          # both at x 200 from row 10
    facts = [dict(f, pair_name="ryu_vs_ken", pair=["ryu", "ken"], game=3, k_prev=f["t"] - 3, k_now=f["t"] + 1,
                  images=["a", "b"]) for f in P.game_facts(rows, [], {1: "ryu", 2: "ken"}, [8, 15], 72)]
    for q in ("q3", "q4", "q5"):
        cands, dropped = E.candidates(facts, q)
        assert all(c["t"] == 8 for c in cands) and dropped["equal_x"] >= 1, q


def test_problems_catch_an_unequal_stratum_and_a_wrong_label(built, tmp_path):
    import shutil
    out = str(tmp_path / "q5")
    shutil.copytree(built["q5"], out, symlinks=True)
    path = os.path.join(out, "far", "train.jsonl")
    lines = open(path).read().splitlines()
    open(path, "w").write("\n".join(lines[1:]) + "\n")
    assert any("unequal" in p for p in E.problems(out, "q5"))
    r = json.loads(lines[1])
    r["label"] = 0
    open(path, "w").write("\n".join([lines[0], json.dumps(r)] + lines[2:]) + "\n")
    assert any("label" in p for p in E.problems(out, "q5"))


# ---- the shortcut check -------------------------------------------------------------------------------------------

def _srows(leak, n=480):
    """Every (pair, other, answer) combination equally often in each block of 16 rows; blocks split 1 in 3 to test.
    leak: the answer follows "other"."""
    out = []
    for i in range(n):
        feat = {"pair": "p%d" % (i % 4), "game": (i // 16) % 7, "other": "ab"[(i // 8) % 2]}
        ans = ("yes" if feat["other"] == "a" else "no") if leak else ("yes", "no")[(i // 4) % 2]
        out.append({"shortcut": feat, "answer": ans, "split": "test" if (i // 16) % 3 == 0 else "train"})
    return out


def test_shortcut_check_fails_a_metadata_leak_and_passes_balanced_rows():
    bad = SC.check(_srows(True), ("yes", "no"))
    assert not bad["pass"] and bad["best_score"] > 0.9 and bad["over_chance"] > SC.MARGIN
    ok = SC.check(_srows(False), ("yes", "no"))
    assert ok["pass"] and ok["best_score"] <= 0.5 + SC.MARGIN


def test_balanced_accuracy():
    assert SC.balanced_accuracy(["a", "a", "b", "b"], ["a", "a", "a", "a"], ("a", "b")) == 0.5
    assert SC.balanced_accuracy(["a", "b", "b", "b"], ["a", "b", "b", "a"], ("a", "b")) == pytest.approx(5 / 6)


def test_the_built_datasets_pass_the_shortcut_check(built):
    for q, d in built.items():
        res = SC.check(_rows(d), G.ANSWERS[q])
        assert res["pass"], (q, res)


# ---- the gates ----------------------------------------------------------------------------------------------------

def test_the_label_episode_split_drawn_gates_pass_on_a_clean_build(built, root):
    store, band = G.Store(root), BANDS["all"]
    for q, d in built.items():
        lab = G.label_check(q, d, _rows(d), store, band)
        assert lab["pass"], (q, lab)
    assert G.episode_check(_rows(built["q3"]), store, band)["pass"]
    dr = G.drawn_check(built["q1"], _rows(built["q1"]))
    assert dr["pass"] and dr["hadoken_yes"] > 0 and dr["ryu_ken_no"] > 0, dr


@pytest.mark.parametrize("q, field, value", [("q1", "answer", "yes"), ("q3", "side", None), ("q4", "t", None),
                                             ("q5", "answer", None)])
def test_the_label_gate_catches_a_tampered_row(built, root, q, field, value):
    rows = _rows(built[q])
    r = dict(rows[0])
    if field == "side":
        r["side"] = "right" if r["side"] == "left" else "left"
    elif field == "t":
        r["t"] += 1
    elif value is None:
        r["answer"] = [a for a in G.ANSWERS[q] if a != r["answer"]][0]
    else:
        r = next(dict(x, answer="yes") for x in rows if x["answer"] == "no")
    res = G.label_check(q, built[q], [r] + rows[1:], G.Store(root), BANDS["all"])
    assert not res["pass"] and res["mismatches"] >= 1


def test_the_split_gate_catches_a_match_in_two_splits(built):
    rows = _rows(built["q5"])
    moved = [dict(rows[0], _file="test" if rows[0]["_file"] != "test" else "train")] + rows[1:]
    assert not G.split_check({"q5": moved})["pass"]


def test_the_drawn_gate_fails_when_a_yes_frame_shows_nothing(built):
    rows = _rows(built["q1"])
    bad = [dict(r, images=[r["images"][0], r["images"][0]]) if r["answer"] == "yes" else r for r in rows]
    assert not G.drawn_check(built["q1"], bad, need=1.0)["pass"]


def test_the_episode_gate_catches_a_pair_across_two_movements(built, root):
    rows = _rows(built["q3"])
    r = dict(rows[0], t=36 if rows[0]["slot"] == 1 else 10)       # rows 32..36 / 6..10 change movement
    assert not G.episode_check([r], G.Store(root), BANDS["all"])["pass"]


def test_fire_at_independent_rule(root):
    ram = G.Store(root).rows("ryu_vs_ken", OLD)
    moves = G.Store(root).moves("ryu_vs_ken", OLD)
    assert G.fire_at(ram, moves, 52, ["ryu", "ken"]) == "yes"
    assert G.fire_at(ram, moves, 51, ["ryu", "ken"]) is None          # rows 47 and 51 both blinked off
    assert G.fire_at(ram, moves, 46, ["ryu", "ken"]) == "yes"         # row 46 drawn, row 42 before the spawn
    assert G.fire_at(ram, moves, 40, ["ryu", "ken"]) == "no"          # windup: no slot on in rows 36 / 40
    assert G.fire_at(ram, moves, 92, ["ryu", "ken"]) == "yes"         # drawn in row 88, gone at 92 (flight ends 90)
    old = G.Store(root).rows("ryu_vs_ken", 0)
    assert G.fire_at(old, moves, 52, ["ryu", "ken"]) is None


def test_contact_sheets(built, tmp_path):
    for q, d in built.items():
        p = G.contact_sheet(q, d, str(tmp_path / ("%s.png" % q)), per_band=2)
        assert os.path.getsize(p) > 0


def test_lookup_is_the_majority_per_feature_value_and_the_first_answer_when_unseen():
    train = [({"f": "a"}, "no"), ({"f": "a"}, "no"), ({"f": "a"}, "yes"), ({"f": "b"}, "yes")]
    test = [({"f": "a"}, "no"), ({"f": "b"}, "yes"), ({"f": "c"}, "no")]
    assert SC.lookup(train, test, ["f"], ("yes", "no")) == ["no", "yes", "yes"]


def test_logreg_learns_a_feature_that_gives_the_answer():
    train = [({"f": "ab"[i % 2], "g": i % 3}, ("yes", "no")[i % 2]) for i in range(60)]
    test = [({"f": "ab"[i % 2], "g": 5}, ("yes", "no")[i % 2]) for i in range(10)]
    assert SC.logreg(train, test, ("yes", "no")) == [a for _, a in test]


def test_the_split_gate_catches_a_whole_match_in_the_wrong_split(built):
    rows = _rows(built["q5"])
    m = (rows[0]["pair_name"], rows[0]["game"])
    wrong = "train" if rows[0]["_file"] != "train" else "test"
    moved = [dict(r, _file=wrong, split=wrong) if (r["pair_name"], r["game"]) == m else r for r in rows]
    assert not G.split_check({"q5": moved})["pass"]


def test_fire_at_a_fireball_off_the_screen_is_no_row(root):
    st = G.Store(root)
    ram, moves = [dict(r) for r in st.rows("ryu_vs_ken", OLD)], st.moves("ryu_vs_ken", OLD)
    ram[48]["shot1_x"] = ram[52]["shot1_x"] = 500                  # past the screen's right edge in both frames
    assert G.fire_at(ram, moves, 52, ["ryu", "ken"]) is None


def test_the_drawn_gate_checks_the_frame_ram_names(built):
    rows = _rows(built["q1"])
    only_n = [r for r in rows if r["answer"] == "yes" and r["fire_frames"] == ["n"]]
    assert only_n
    swapped = [dict(r, images=r["images"][::-1]) if r in only_n else r for r in rows]   # the blue frame now n-4
    assert not G.drawn_check(built["q1"], swapped, need=1.0)["pass"]
