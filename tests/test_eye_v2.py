"""docs/eye_questions_v1.md, "Questions v2 - relabel": q3 v2 (attack / special attack / block / walk / jump / stand /
hit) and q4 v2 (high / normal / low), relabelled from the same frame pool (sf2.data.eye_v2, eye_pool, eye_data) and
gated (sf2.data.eye_gate) on a small 2P collection written here: player 1 stands, player 2 cycles through every
behaviour in blocks of 10 rows (stand, walk, crouch, standing attack, crouching attack, special, standing guard,
crouching guard, crouching block stun, hit, knocked down on the ground, jump, jump attack, thrown in the air)."""
import collections
import json
import os

import pytest

from sf2.data import eye_data as E
from sf2.data import eye_gate as G
from sf2.data import eye_pool as P
from sf2.data import eye_shortcut as SC
from sf2.data import eye_v2 as V
from sf2.data import movement_collect_io as MIO
from sf2.data import pairs_train as T
from sf2.emu.vs import GROUND_Y
from test_pairs_labels import prow

AIR_Y = GROUND_Y - 40
BLOCK = 10
# (name, player 2's RAM overrides, pressed word, q3 v2 answer, q4 v2 answer)
SCRIPT = [
    ("stand", {}, "idle", "stand", "normal"),
    ("walk", {"state": 0x00}, "forward", "walk", "normal"),
    ("crouch", {"state": 0x02}, "crouch", "stand", "low"),
    ("attack", {"state": 0x0A, "mclass": 0x00, "aid": 3}, "lp", "attack", "normal"),
    ("crouch_attack", {"state": 0x0A, "mclass": 0x02, "aid": 4}, "c.mp", "attack", "low"),
    ("special", {"state": 0x0C, "mclass": 0x02}, "hadoken", "special attack", "normal"),
    ("guard", {"state": 0x08, "sub": 0x00}, "block_high", "block", "normal"),
    ("crouch_guard", {"state": 0x08, "sub": 0x04}, "block_low", "block", "low"),
    ("crouch_blockstun", {"state": 0x0E, "sub": 0x02, "react": 0x08}, "block_low", "block", "low"),
    ("hit", {"state": 0x0E, "sub": 0x02, "react": 0x02}, None, "hit", "normal"),
    ("down", {"state": 0x14}, None, "hit", "normal"),
    ("jump", {"state": 0x04, "y": AIR_Y}, "jump", "jump", "high"),
    ("jump_attack", {"state": 0x04, "y": AIR_Y, "aid": 7}, "jump_hk", "attack", "high"),
    ("thrown_air", {"state": 0x14, "y": AIR_Y}, None, "hit", "high"),
    ("jump_squat", {"state": 0x04}, "jump_forward", "jump", "normal"),     # the jump state on the ground
]
N = BLOCK * len(SCRIPT) * 2 + 8
GAMES = 12
LATE = 32                     # a game past the v1 pool (games 0-31): must never enter the pool


def p2_at(k):
    i = (k // BLOCK) % len(SCRIPT)
    over = dict(SCRIPT[i][1])
    x = 330
    if SCRIPT[i][0] == "walk":
        x = 300 + 3 * (k % BLOCK)
    return dict({"x": x}, **over), i


def game_rows(game):
    left, right = (150, 0) if game % 2 == 0 else (450, 1)
    out = []
    for k in range(N):
        p2, _ = p2_at(k)
        if right:
            p2 = dict(p2, x=600 - p2["x"])          # mirrored: player 2 on the left of player 1
        out.append(dict(prow({"x": left}, p2), timer=k % 256, shot1_hide=0, shot2_hide=0))
    return out


def moves_of():
    out = []
    for b in range(N // BLOCK):
        w = SCRIPT[b % len(SCRIPT)][2]
        if w is not None:
            out.append([w, b * BLOCK - 1 if b else 0, min(N - 1, b * BLOCK + BLOCK - 1), "done", 2])
    return out


def write_collection(r):
    from PIL import Image
    for a, b in (("ryu", "ken"), ("ken", "ryu")):
        base = os.path.join(r, "%s_vs_%s" % (a, b))
        os.makedirs(os.path.join(base, "images"))
        games = []
        for g in list(range(GAMES)) + [LATE]:
            MIO.write_ram(os.path.join(base, "ram"), g, game_rows(g))
            for k in range(N):
                Image.new("RGB", (16, 16), (k % 256, g, 9)).save(os.path.join(base, "images", "g%04d_k%05d.png" % (g, k)))
            games.append({"game": g, "moves": moves_of(), "result": "win"})
        with open(os.path.join(base, "games.jsonl"), "w") as f:
            f.write("".join(json.dumps(x) + "\n" for x in games))


@pytest.fixture(scope="module")
def root(tmp_path_factory):
    r = str(tmp_path_factory.mktemp("eyev2") / "col")
    write_collection(r)
    return r


@pytest.fixture(scope="module")
def pool(root):
    return P.pool(root, 72, workers=1, max_game=31)


@pytest.fixture(scope="module")
def built(root, pool, tmp_path_factory):
    tmp = tmp_path_factory.mktemp("eyev2b")
    out = {}
    for q in E.V2:
        out[q] = str(tmp / E.out_of(q))
        E.build(pool, q, out[q], root)
    return out


def _rows(out):
    return G.rows_of(out)


# ---- the labels (pure) ------------------------------------------------------------------------------------------

def _seq(over, n=6, x=330):
    return [prow({"x": 150}, dict({"x": x}, **over)) for _ in range(n)]


@pytest.mark.parametrize("name, over, word, act, pos", SCRIPT)
def test_each_behaviour_has_the_docs_answers(name, over, word, act, pos):
    rows = _seq(over)
    if name == "walk":
        rows = [prow({"x": 150}, {"x": 300 + 3 * k}) for k in range(6)]
    cls = "attack" if word in ("lp", "c.mp", "jump_hk") else "special" if word == "hadoken" else None
    assert V.act2(rows, 5, 2, cls) == act, name
    assert V.position(rows, 5, 2) == pos, name


def test_the_answer_sets_and_texts():
    assert V.ACT2 == ("attack", "special attack", "block", "walk", "jump", "stand", "hit")
    assert V.POS == ("high", "normal", "low")
    assert E.question("q3v2", "left")["instructions"] == "What is the fighter on the left doing?"
    assert tuple(E.question("q3v2", "right")["criteria"]) == V.ACT2
    assert E.question("q4v2", "right")["instructions"] == "Is the fighter on the right high, normal or low?"
    assert tuple(E.question("q4v2", "left")["criteria"]) == V.POS
    assert E.out_of("q3v2") == "test_data_eye_q3v2_act" and E.out_of("q4v2") == "test_data_eye_q4v2_pos"
    assert V.dir_of("special attack") == "special_attack" and V.dir_of("stand") == "stand"


def test_the_pressed_word_decides_attack_vs_special_attack_when_ram_shows_an_attack():
    rows = _seq({"state": 0x0C})
    assert V.act2(rows, 5, 2, "attack") == "attack"
    assert V.act2(rows, 5, 2, None) == "special attack"            # no attack word pressed: the RAM rule
    rows = _seq({"state": 0x0A, "mclass": 0x08})                  # the CPU's specials run in 0x0A with class 8
    assert V.act2(rows, 5, 2, None) == "special attack"
    assert V.act2(_seq({"state": 0x0A}), 5, 2, "special") == "special attack"


def test_crouching_is_stand_for_the_action_and_low_for_the_position():
    rows = _seq({"state": 0x02, "sub": 0x02})                      # rising out of a crouch: still state 0x02
    assert V.act2(rows, 5, 2) == "stand" and V.position(rows, 5, 2) == "low"
    assert V.low_kind(rows[5], 2) == "crouch"


@pytest.mark.parametrize("over, kind", [
    ({"state": 0x02}, "crouch"), ({"state": 0x0A, "mclass": 0x02}, "crouch_attack"),
    ({"state": 0x08, "sub": 0x04}, "crouch_block"), ({"state": 0x08, "sub": 0x06}, "crouch_block"),
    ({"state": 0x0E, "react": 0x08}, "crouch_block"),
    ({"state": 0x0A, "mclass": 0x00}, None), ({"state": 0x0A, "mclass": 0x06}, None),
    ({"state": 0x08, "sub": 0x00}, None), ({"state": 0x08, "sub": 0x02}, None), ({"state": 0x0E, "react": 0x06}, None),
    ({"state": 0x0E, "react": 0x02}, None), ({"state": 0x0C, "mclass": 0x02}, None), ({"state": 0x00}, None),
])
def test_low_kind_from_the_ram_fields(over, kind):
    assert V.low_kind(_seq(over, 1)[0], 2) == kind


def test_the_air_wins_over_crouching_fields():
    assert V.position(_seq({"state": 0x0A, "mclass": 0x02, "y": AIR_Y}), 5, 2) == "high"
    assert V.position(_seq({"state": 0x04}), 5, 2) == "normal"        # a jump state on the ground (take-off squat)


def test_unknown_movement_is_no_action_and_out_of_range_no_position():
    assert V.act2(_seq({"state": 0x06}), 5, 2) is None                # turning: unknown
    assert V.position(_seq({}), 9, 2) is None


def test_the_episode_is_one_v2_action_over_rows_t_minus_4_to_t():
    def ep(first, second, pressed=None, t=9):
        rows = _seq(first, 7) + _seq(second, 3)                      # rows 5, 6 first; rows 7, 8, 9 second
        return V.act2_in_episode(rows, t, 2, pressed or [None] * 10)
    assert ep({"state": 0x02}, {})                                    # crouch then stand: both "stand"
    assert ep({"state": 0x14, "y": AIR_Y}, {"state": 0x14})          # thrown in the air then down: both "hit"
    assert ep({"state": 0x0E, "react": 0x08}, {"state": 0x08, "sub": 0x00})    # crouch block stun, standing guard
    assert not ep({}, {"state": 0x0A})                                # stand then attack
    assert not ep({"state": 0x0C}, {"state": 0x0C}, [None] * 7 + ["attack"] * 3)   # the pressed class changes it
    assert ep({"state": 0x0C}, {"state": 0x0C}, [None] * 10)
    assert not ep({"state": 0x0C}, {"state": 0x0C}, t=3)             # t - 4 < 0
    assert not ep({"state": 0x06}, {"state": 0x06})                   # unknown


# ---- the pool and the builds ------------------------------------------------------------------------------------

def test_the_pool_keeps_games_0_to_31_only(root, pool):
    assert {f["game"] for f in pool} == set(range(GAMES))
    assert LATE in {f["game"] for f in P.pool(root, 72, workers=1)}


def test_the_pool_fighters_carry_the_v2_facts(pool):
    f = next(x for x in pool if x["t"] == 4 * BLOCK + 9)                  # player 2 in a crouching attack
    me = f["fighters"][2]
    assert me["act2"] == "attack" and me["act2_in_episode"] and me["pos"] == "low" and me["low_kind"] == "crouch_attack"
    assert f["fighters"][1]["pos"] == "normal" and f["fighters"][1]["act2"] == "stand"


def test_both_builds_every_answer_equal_per_stratum_in_answer_dirs(built, pool):
    facts = {(f["pair_name"], f["game"], f["t"]): f for f in pool}
    for q, field, answers in (("q3v2", "act2", V.ACT2), ("q4v2", "pos", V.POS)):
        rows = _rows(built[q])
        assert {r["answer"] for r in rows} == set(answers), q
        assert sorted(d for d in os.listdir(built[q]) if d != "frames" and os.path.isdir(os.path.join(built[q], d))) \
            == sorted(V.dir_of(a) for a in answers)
        per = collections.defaultdict(collections.Counter)
        for r in rows:
            me = facts[(r["pair_name"], r["game"], r["t"])]["fighters"][r["slot"]]
            assert me[field] == r["answer"] and r["_dir"] == V.dir_of(r["answer"]) and me["side"] == r["side"]
            assert r["label"] == list(answers).index(r["answer"])
            assert r["question"]["instructions"] == E.Q[q]["text"] % r["side"]
            assert r["stratum"].split("|")[:5] == [r["split"], r["pair_name"], str(r["game"]), r["side"],
                                                   facts[(r["pair_name"], r["game"], r["t"])]["fighters"][3 - r["slot"]]["mv10"]]
            if q == "q3v2":
                assert me["act2_in_episode"]
            per[r["stratum"]][r["answer"]] += 1
        assert per and all(len({c[a] for a in answers}) == 1 for c in per.values()), q
        meta = json.load(open(os.path.join(built[q], "build.json")))
        assert meta["problems"] == [] and meta["dirs"] == [V.dir_of(a) for a in answers]


def test_q3v2_keeps_crouching_kinds_in_their_answers(built):
    meta = json.load(open(os.path.join(built["q3v2"], "build.json")))
    ex = meta["extras"]
    assert ex["by_kind"]["attack"].get("crouch_attack", 0) > 0 and ex["by_kind"]["attack"].get("jump_attack", 0) > 0
    assert ex["by_kind"]["stand"].get("crouch", 0) > 0 and ex["by_kind"]["block"].get("crouch_block", 0) > 0
    assert ex["by_kind"]["hit"].get("down", 0) > 0


def test_q4v2_low_has_every_crouching_kind(built):
    meta = json.load(open(os.path.join(built["q4v2"], "build.json")))
    low = meta["extras"]["low_kinds"]
    assert all(low.get(k, 0) > 0 for k in ("crouch", "crouch_attack", "crouch_block")), low


def test_the_v2_builds_pass_the_shortcut_check(built):
    for q, d in built.items():
        res = SC.check(_rows(d), G.ANSWERS[q])
        assert res["pass"], (q, res)


def test_the_split_is_the_one_table(built):
    rep = G.split_check({q: _rows(d) for q, d in built.items()})
    assert rep["pass"], rep
    assert all(r["split"] == T.split3(r["pair_name"], r["game"]) for d in built.values() for r in _rows(d))


def test_problems_catch_a_row_in_the_wrong_dir(built, tmp_path):
    import shutil
    out = str(tmp_path / "q4v2")
    shutil.copytree(built["q4v2"], out, symlinks=True)
    src, dst = os.path.join(out, "low", "train.jsonl"), os.path.join(out, "high", "train.jsonl")
    lines = open(src).read().splitlines()
    open(src, "w").write("\n".join(lines[1:]) + "\n")
    open(dst, "a").write(lines[0] + "\n")
    assert E.problems(out, "q4v2")


# ---- the gates --------------------------------------------------------------------------------------------------

def test_the_gates_pass_on_the_clean_v2_builds(built, root):
    store = G.Store(root)
    for q, d in built.items():
        lab = G.label_check(q, d, _rows(d), store, 72)
        assert lab["pass"] and lab["checked"] > 0, (q, lab)
    ep = G.episode_check(_rows(built["q3v2"]), store, 72, q="q3v2")
    assert ep["pass"] and ep["checked"] > 0, ep
    sf = G.second_fact_check(_rows(built["q4v2"]), store)
    assert sf["pass"] and sf["crouch_attack"]["n"] > 0 and sf["crouch_block"]["n"] > 0, sf


@pytest.mark.parametrize("q", ["q3v2", "q4v2"])
def test_the_label_gate_catches_a_wrong_answer(built, root, q):
    rows = _rows(built[q])
    for i in range(min(len(rows), 30)):
        r = rows[i]
        other = [a for a in G.ANSWERS[q] if a != r["answer"]][0]
        bad = dict(r, answer=other, label=list(G.ANSWERS[q]).index(other), _dir=V.dir_of(other))
        res = G.label_check(q, built[q], [bad], G.Store(root), 72)
        assert not res["pass"], (q, r["id"])


@pytest.mark.parametrize("name, want", [("crouch", "low"), ("crouch_attack", "low"), ("crouch_guard", "low"),
                                        ("crouch_blockstun", "low"), ("guard", "normal"), ("attack", "normal"),
                                        ("special", "normal"), ("jump", "high"), ("down", "normal")])
def test_the_gates_independent_position(root, name, want):
    ram = G.Store(root).rows("ryu_vs_ken", 0)
    i = [s[0] for s in SCRIPT].index(name)
    assert G.position_at(ram, i * BLOCK + 5, 2) == want


@pytest.mark.parametrize("name", [s[0] for s in SCRIPT])
def test_the_gates_independent_action(root, name):
    st = G.Store(root)
    ram, moves = st.rows("ryu_vs_ken", 0), st.moves("ryu_vs_ken", 0)
    i = [s[0] for s in SCRIPT].index(name)
    assert G.act2_at(ram, moves, i * BLOCK + 9, 2, "ken", 72) == SCRIPT[i][3], name


def test_the_episode_gate_catches_a_pair_across_two_actions(built, root):
    r = dict(_rows(built["q3v2"])[0], slot=2, t=3 * BLOCK + 1)        # rows 27..31: crouch then attack
    assert not G.episode_check([r], G.Store(root), 72, q="q3v2")["pass"]


def test_the_second_fact_gate_fails_when_the_pressed_word_disagrees(built, root):
    rows = _rows(built["q4v2"])
    assert any(r["answer"] == "low" and r.get("low_kind") == "crouch_attack" for r in rows)
    clean, st = G.Store(root), G.Store(root)
    for pair in ("ryu_vs_ken", "ken_vs_ryu"):                       # every crouching normal logged as a standing lp
        st.logs[pair] = {g: [["lp" if m[0] == "c.mp" else m[0]] + list(m[1:]) for m in clean.moves(pair, g)]
                         for g in range(GAMES)}
    assert not G.second_fact_check(rows, st)["pass"]


def test_contact_sheets_for_v2(built, tmp_path):
    for q, d in built.items():
        p = G.contact_sheet(q, d, str(tmp_path / ("%s.png" % q)), per_band=2)
        assert os.path.getsize(p) > 0
        names = [b[0] for b in G.bands_of(q, _rows(d))]
        assert all(a in names for a in G.ANSWERS[q])


def test_the_pool_uses_the_pressed_word_for_special_attack():
    rows = [prow({"x": 150}, {"x": 330, "state": 0x0A}) for _ in range(12)]     # RAM: an attack state, class 0
    moves = [["hadoken", 0, 11, "done", 2]]                                      # we pressed the special
    f = P.game_facts(rows, moves, {1: "ryu", 2: "ken"}, [10], 72)[0]
    assert f["fighters"][2]["act2"] == "special attack" and f["fighters"][2]["act2_in_episode"]


def test_the_label_gate_catches_a_row_in_the_wrong_dir(built, root):
    rows = _rows(built["q3v2"])
    r = next(x for x in rows if x["answer"] == "special attack")
    assert G.label_check("q3v2", built["q3v2"], [r], G.Store(root), 72)["pass"]
    assert not G.label_check("q3v2", built["q3v2"], [dict(r, _dir="attack")], G.Store(root), 72)["pass"]


# ---- the owner's amendment: q3v2b (no game in the matching), q4v2b (no jump state on the ground) ---------------

def test_the_pool_marks_the_jump_state_on_the_ground(pool):
    i = [x[0] for x in SCRIPT].index("jump_squat")
    f = next(x for x in pool if x["t"] == i * BLOCK + 5)
    assert f["fighters"][2]["jump_ground"] and not f["fighters"][1]["jump_ground"]
    j = [x[0] for x in SCRIPT].index("jump")
    assert not next(x for x in pool if x["t"] == j * BLOCK + 5)["fighters"][2]["jump_ground"]   # in the air


def test_q3v2b_matches_without_the_game(built, pool):
    facts = {(f["pair_name"], f["game"], f["t"]): f for f in pool}
    rows = _rows(built["q3v2b"])
    assert {r["answer"] for r in rows} == set(V.ACT2)
    for r in rows:
        other = facts[(r["pair_name"], r["game"], r["t"])]["fighters"][3 - r["slot"]]["mv10"]
        assert r["stratum"].split("|") == [r["split"], r["pair_name"], r["side"], other]
    assert len(rows) >= len(_rows(built["q3v2"]))
    assert json.load(open(os.path.join(built["q3v2b"], "build.json")))["cap_per_stratum"] is None


def test_q4v2b_drops_the_jump_state_on_the_ground_only(built, pool):
    facts = {(f["pair_name"], f["game"], f["t"]): f for f in pool}
    rows = _rows(built["q4v2b"])
    assert rows and not any(facts[(r["pair_name"], r["game"], r["t"])]["fighters"][r["slot"]]["jump_ground"]
                            for r in rows)
    assert any(r["answer"] == "high" and r["kind"] == "jump" for r in rows)          # jumps in the air stay
    meta = json.load(open(os.path.join(built["q4v2b"], "build.json")))
    assert meta["dropped"]["jump_state_on_ground"] > 0
    old = _rows(built["q4v2"])
    assert any(facts[(r["pair_name"], r["game"], r["t"])]["fighters"][r["slot"]]["jump_ground"] for r in old)
    assert all(r["stratum"].split("|")[2] == str(r["game"]) for r in rows)          # q4v2b keeps the game


def test_the_gates_pass_on_the_amended_builds_and_catch_a_jump_squat_row(built, root):
    store = G.Store(root)
    for q in ("q3v2b", "q4v2b"):
        lab = G.label_check(q, built[q], _rows(built[q]), store, 72)
        assert lab["pass"] and lab["checked"] > 0, (q, lab)
    assert G.episode_check(_rows(built["q3v2b"]), store, 72, q="q3v2b")["pass"]
    assert G.second_fact_check(_rows(built["q4v2b"]), store)["pass"]
    squat = [r for r in _rows(built["q4v2"]) if r.get("kind") == "jump" and r["answer"] == "normal"]
    assert squat
    assert G.label_check("q4v2", built["q4v2"], squat, store, 72)["pass"]
    assert not G.label_check("q4v2b", built["q4v2"], squat, store, 72)["pass"]


def test_the_gate_counts_a_walk_through_the_other_fighters_x_as_walk():
    rows = [prow({"x": 300}, {"x": 285 + 3 * k}) for k in range(8)]           # player 2 walks across x 300
    rows[5]["p2_x"] = 300                                                      # same x as player 1: no direction
    assert G.act2_at(rows, [], 5, 2, "ken", 72) == "walk"
    assert V.act2(rows, 5, 2) == "walk"
