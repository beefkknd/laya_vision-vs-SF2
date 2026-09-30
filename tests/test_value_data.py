"""The value dataset builder (sf2/data/value_data.py -> test_data_v2/): the all8 rows in note v2, a value row next to
every live row, the new live rows from the collection logs, Chun-Li vs Guile held out, and the gates. Tiny fixtures
in tmp_path only (never the real data)."""
import json
import os
import subprocess
import sys

import numpy as np
import pytest
from PIL import Image

from sf2.data import value_data as V
from sf2.data.value import VALUE_BUCKETS, value_bucket, value_question
from sf2.data.vs_sweep import OUTCOMES, outcome_question

V1 = "me=%s dist=%s side=%s dx=%+d my_bar=full opp_bar=full opp_airborne=0 opp_crouch=0"
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def png(path, w=256, h=224, color=(10, 20, 30)):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    img = np.zeros((h, w, 3), np.uint8)
    img[:, : w // 3] = color            # left third coloured: a mirror is visibly different
    Image.fromarray(img).save(path)


def write_jsonl(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def entry(game, frame, action="c.lk", opp="ryu", me="chunli", side="left", gap=45, dealt=0, taken=0,
          opp_state="stand", actual="hit", explored=False, prompt=None):
    dx = gap if side == "left" else -gap
    return {"game": game, "frame": frame, "clock": 99, "me": me, "opp": opp, "side": side, "gap": gap,
            "range": "close", "my_life": 176, "opp_life": 176, "my_state": "stand", "opp_state": opp_state,
            "opp_air": False, "action": action, "kind": "attack", "actual": actual, "dealt": dealt, "taken": taken,
            "frames": 30, "explored": explored, "scores": {},
            "prompt": prompt if prompt is not None else V1 % (me, "close", side, dx),
            "images": ["g%02d_%05d_prev.png" % (game, frame), "g%02d_%05d_now.png" % (game, frame)]}


def write_log(repo, sub, char, entries, images=True):
    d = os.path.join(repo, "rollouts", "lv_value", sub, char)
    write_jsonl(os.path.join(d, "actions.jsonl"), entries)
    if images:
        for e in entries:
            for p in e["images"]:
                png(os.path.join(d, "images", p), color=(e["game"] * 7 % 255, e["frame"] % 255, 40))


def old_row(rid, kind, action, split="train", side="left", mirrored=False, **extra):
    """An all8 row as test_data/<char>/*.jsonl has it (v1 note); its frames are named after the id's position."""
    key = rid.split("-")[1]
    pre = "frames/mirror_" if mirrored else "frames/"
    dx = 40 if side == "left" else -40
    r = {"id": rid, "char": "chunli", "opp": "ryu", "side": side, "dx": dx, "facing": "right", "range": "close",
         "gap": 40, "posture": "stand", "action": action, "kind": kind, "buttons": [],
         "images": [pre + key + "_prev.png", pre + key + "_now.png"],
         "state_text": V1 % ("chunli", "close", side, dx), "split": split, "mirrored": mirrored, "source": "real",
         "outcome": "hit", "damage": 0, "damage_taken": 0, "question": outcome_question(action), "label": 0}
    r.update(extra)
    return r


def live_old(game, frame, action, dealt, taken, mirrored=False, split="train", log="live_dhalsim"):
    key = "live_%s_g%02d_f%05d" % (log, game, frame)
    rid = "chunli-%s-%s%s" % (key, action, "-m" if mirrored else "")
    side = "right" if mirrored else "left"
    return old_row(rid, "live", action, split=split, side=side, mirrored=mirrored, opp="dhalsim",
                   posture="live", game=game, source="live:rollouts/" + log, damage=dealt, damage_taken=taken,
                   images=[("frames/mirror_" if mirrored else "frames/") + key + s for s in ("_prev.png", "_now.png")])


def make_old(repo, char="chunli", live_log=True, rows=None):
    """test_data/<char>/ with a still-dummy attack, a movement, two block rows (ground and jump-in probes), and two
    live rows (+ mirror) joined to rollouts/live_dhalsim/<char>/actions.jsonl."""
    train = rows if rows is not None else [
        old_row("chunli-left_close_0_stand-lp", "attack", "lp"),
        old_row("chunli-left_close_0_stand-lp-m", "attack", "lp", side="right", mirrored=True),
        old_row("chunli-left_close_0_stand-forward", "movement", "forward", outcome="none"),
        old_row("chunli-left_close_0_def_shk-block_high", "defense", "block_high", probe="s.hk"),
        old_row("chunli-left_close_0_def_jumpin-block_high", "defense", "block_high", probe="jump_in"),
        live_old(0, 0, "hk", 0, 19),
        live_old(0, 0, "hk", 0, 19, mirrored=True),
        live_old(1, 64, "throw", 32, 0),
    ]
    test_left = [live_old(2, 128, "sweep", 20, 5, split="test")]
    test_right = [old_row("chunli-right_close_2_stand-lp", "attack", "lp", split="test", side="right")]
    d = os.path.join(repo, "test_data", char)
    write_jsonl(os.path.join(d, "train.jsonl"), train)
    write_jsonl(os.path.join(d, "test_real_left.jsonl"), test_left)
    write_jsonl(os.path.join(d, "test_real_right.jsonl"), test_right)
    for r in train + test_left + test_right:
        for p in r["images"]:
            png(os.path.join(d, p))
    if live_log:
        log = [entry(0, 0, "hk", opp="dhalsim", taken=19, opp_state="attack"),
               entry(1, 64, "throw", opp="dhalsim", dealt=32, opp_state="stand"),
               entry(2, 128, "sweep", opp="dhalsim", dealt=20, taken=5, opp_state="special")]
        write_jsonl(os.path.join(repo, "rollouts", "live_dhalsim", char, "actions.jsonl"), log)
    return d


def make_new(repo):
    """Chun-Li vs Ryu (games 0-3: game 2 is a test game) and vs Guile (held out), one dx == 0 decision."""
    write_log(repo, "ryu", "chunli", [
        entry(0, 0, "c.lk", dealt=10, taken=0, opp_state="attack"),
        entry(0, 32, "throw", dealt=40, taken=0),
        entry(1, 0, "block_high", dealt=0, taken=35, opp_state="special", side="right"),
        entry(2, 0, "sweep", dealt=5, taken=20),                     # test game
        entry(3, 0, "hk", gap=0),                                   # crossing: dropped
    ])
    write_log(repo, "guile", "chunli", [entry(0, 0, "c.lk", opp="guile", dealt=0, taken=0),
                                        entry(2, 0, "hk", opp="guile", dealt=0, taken=31)])


def build(repo, out=None, **kw):
    return V.build(test_data=os.path.join(repo, "test_data"), lv_root=os.path.join(repo, "rollouts", "lv_value"),
                   out=out or os.path.join(repo, "test_data_v2"), repo=repo, **dict({"min_value_train": 0}, **kw))


def load(repo, name, char="chunli", out="test_data_v2"):
    with open(os.path.join(repo, out, char, name + ".jsonl")) as f:
        return [json.loads(line) for line in f]


def by_id(rows):
    return {r["id"]: r for r in rows}


@pytest.fixture
def repo(tmp_path):
    make_old(str(tmp_path))
    make_new(str(tmp_path))
    return str(tmp_path)


# --- units ---

def test_with_attacking_appends_one_field():
    t = V1 % ("ryu", "mid", "left", 65)
    assert V.with_attacking(t, 1) == t + " opp_attacking=1"
    with pytest.raises(ValueError):
        V.with_attacking(t + " opp_attacking=0", 1)        # never twice
    with pytest.raises(ValueError):
        V.with_attacking(t + "\nmemory: x", 1)               # a prompt with memory is not a note


def test_attacking_from_the_logged_state():
    assert [V.attacking(s) for s in ("attack", "special", "stand", "jump", "hit_stun", "guard")] == [1, 1, 0, 0, 0, 0]


def test_parse_live_id_real_and_mirrored():
    assert V.parse_live_id("chunli-live_live_dhalsim_g07_f01234-c.mk") == {
        "char": "chunli", "log": "live_dhalsim", "game": 7, "frame": 1234, "action": "c.mk", "mirrored": False}
    m = V.parse_live_id("ryu-live_live_dhalsim_g00_f00000-block_high-m")
    assert m["action"] == "block_high" and m["mirrored"] and m["game"] == 0
    assert V.parse_live_id("ryu-left_close_0_stand-lp") is None


def test_value_row_label_and_question():
    r = live_old(0, 0, "hk", 0, 19)
    v = V.value_row(r)
    assert v["id"] == r["id"] + "-value" and v["question"] == value_question("hk")
    assert v["label"] == list(VALUE_BUCKETS).index("loss") and v["state_text"] == r["state_text"]
    assert v["images"] == r["images"] and r["question"] == outcome_question("hk")      # the source is not touched


def test_split_of_game():
    assert [V.split_of_game(g) for g in (0, 2, 5, 8, 12, 15, 18, 19)] == [
        "train", "test", "test", "test", "test", "test", "test", "train"]


# --- the existing rows ---

def test_old_rows_get_note_v2(repo):
    assert build(repo)["problems"] == []
    t = by_id(load(repo, "train"))
    assert t["chunli-left_close_0_stand-lp"]["state_text"].endswith(" opp_attacking=0")
    assert t["chunli-left_close_0_stand-forward"]["state_text"].endswith(" opp_attacking=0")
    assert t["chunli-left_close_0_def_shk-block_high"]["state_text"].endswith(" opp_attacking=1")    # mid-attack
    assert t["chunli-left_close_0_def_jumpin-block_high"]["state_text"].endswith(" opp_attacking=0")  # still a jump
    assert t["chunli-live_live_dhalsim_g00_f00000-hk"]["state_text"].endswith(" opp_attacking=1")
    assert t["chunli-live_live_dhalsim_g00_f00000-hk-m"]["state_text"].endswith(" opp_attacking=1")
    assert t["chunli-live_live_dhalsim_g01_f00064-throw"]["state_text"].endswith(" opp_attacking=0")
    left = by_id(load(repo, "test_real_left"))
    assert left["chunli-live_live_dhalsim_g02_f00128-sweep"]["state_text"].endswith(" opp_attacking=1")   # special


def test_old_rows_otherwise_unchanged_and_images_resolve(repo):
    build(repo)
    old = {json.loads(line)["id"]: json.loads(line) for line in open(os.path.join(repo, "test_data/chunli/train.jsonl"))}
    for r in load(repo, "train"):
        if r["id"] in old:
            o = old[r["id"]]
            assert {k: v for k, v in r.items() if k not in ("state_text", "images")} == \
                {k: v for k, v in o.items() if k not in ("state_text", "images")}
            assert r["images"] == ["../../test_data/chunli/" + p for p in o["images"]]
            # laya's loader joins <root>/<name> with the path (laya.vlm_train.jsonl_example)
            assert all(os.path.exists(os.path.join(repo, "test_data_v2", "chunli", p)) for p in r["images"])


def test_every_live_row_has_a_value_row_and_still_rows_none(repo):
    build(repo)
    rows = load(repo, "train") + load(repo, "test_real_left") + load(repo, "test_real_right")
    ids = {r["id"] for r in rows}
    for r in rows:
        if r["kind"] == "live" and r.get("task") != "value":
            assert r["id"] + "-value" in ids
        if r["kind"] != "live":
            assert r.get("task") != "value" and r["id"] + "-value" not in ids
    t = by_id(load(repo, "train"))
    assert t["chunli-live_live_dhalsim_g01_f00064-throw-value"]["label"] == list(VALUE_BUCKETS).index("big_gain")


def test_invariant_value_label_is_the_bucket_of_its_source_net(repo):
    build(repo)
    for name in ("train", "test_real_left", "test_real_right", "test_heldout_guile"):
        rows = load(repo, name)
        src = by_id(rows)
        for v in (r for r in rows if r.get("task") == "value"):
            s = src[v["id"][: -len("-value")]]
            assert v["label"] == list(VALUE_BUCKETS).index(value_bucket(s["damage"] - s["damage_taken"]))
            assert v["question"] == value_question(s["action"]) and v["state_text"] == s["state_text"]
            if "dealt" in s:
                assert (s["dealt"], s["taken"]) == (s["damage"], s["damage_taken"])


# --- the new rows ---

def test_new_rows_split_mirror_and_frames(repo):
    assert build(repo)["problems"] == []
    t = by_id(load(repo, "train"))
    rid = "chunli-live_lv_ryu_g00_f00000-c.lk"
    r = t[rid]
    assert r["state_text"] == V1 % ("chunli", "close", "left", 45) + " opp_attacking=1"
    assert r["outcome"] == "hit" and r["label"] == OUTCOMES.index("hit") and r["split"] == "train"
    assert (r["opp"], r["opp_state"], r["explored"], r["dealt"], r["taken"], r["game"], r["frame"]) == (
        "ryu", "attack", False, 10, 0, 0, 0)
    assert r["move_kind"] == "attack" and r["kind"] == "live"
    m = t[rid + "-m"]
    assert m["side"] == "right" and "dx=-45" in m["state_text"] and m["state_text"].endswith(" opp_attacking=1")
    assert rid + "-value" in t and rid + "-m-value" in t
    base = os.path.join(repo, "test_data_v2", "chunli")
    now = np.asarray(Image.open(os.path.join(base, r["images"][1])))
    mir = np.asarray(Image.open(os.path.join(base, m["images"][1])))
    assert now.shape == (256, 256, 3) and not now[:62].any()       # model_frame: HUD blank, padded to 256
    assert np.array_equal(mir, now[:, ::-1])
    left = by_id(load(repo, "test_real_left"))
    assert "chunli-live_lv_ryu_g02_f00000-sweep" in left and "chunli-live_lv_ryu_g02_f00000-sweep-value" in left
    assert not any(i.endswith("-m") for i in left)                  # test rows are real, never mirrored
    right = by_id(load(repo, "test_real_right"))
    assert "chunli-live_lv_ryu_g01_f00000-block_high" not in right   # game 1 trains
    assert "chunli-live_lv_ryu_g01_f00000-block_high" in t


def test_dx_zero_dropped(repo):
    s = build(repo)
    rows = load(repo, "train") + load(repo, "test_real_left") + load(repo, "test_real_right")
    assert not [r for r in rows if "g03_f00000" in r["id"]]
    assert s["counts"]["chunli"]["dropped_dx0"] == 1


def test_val_split_keeps_mirrored_twins_together(repo):
    from sf2.data.train_data import position
    build(repo)
    t = by_id(load(repo, "train"))
    rid = "chunli-live_lv_ryu_g00_f00000-c.lk"
    keys = {position({"dataset": "chunli", "state": {"images": t[i]["images"]}})
            for i in (rid, rid + "-m", rid + "-value", rid + "-m-value")}
    assert len(keys) == 1 and next(iter(keys))[1].startswith("live_")


# --- the hold-out ---

def test_chunli_vs_guile_only_in_the_heldout_file(repo):
    assert build(repo)["problems"] == []
    held = load(repo, "test_heldout_guile")
    assert {r["id"] for r in held} == {"chunli-live_lv_guile_g00_f00000-c.lk", "chunli-live_lv_guile_g00_f00000-c.lk-value",
                                       "chunli-live_lv_guile_g02_f00000-hk", "chunli-live_lv_guile_g02_f00000-hk-value"}
    assert all(r["opp"] == "guile" and not r["mirrored"] for r in held)
    for name in ("train", "test_real_left", "test_real_right"):
        assert not [r for r in load(repo, name) if r["opp"] == "guile"]
    assert not [f for f in os.listdir(os.path.join(repo, "test_data_v2/chunli/frames")) if "mirror_live_lv_guile" in f]


def test_gate_refuses_a_guile_row_in_train():
    leak = dict(live_old(0, 0, "hk", 0, 0), opp="guile", state_text=V1 % ("chunli", "close", "left", 40) + " opp_attacking=0")
    probs = V.row_problems("chunli", {"train": [leak], "test_real_left": [], "test_real_right": []})
    assert any("guile" in p for p in probs)
    ok = V.row_problems("ryu", {"train": [dict(leak, char="ryu")], "test_real_left": [], "test_real_right": []})
    assert not any("guile" in p for p in ok)                    # only Chun-Li's hold-out


def test_gate_refuses_a_note_without_exactly_one_trailing_field():
    good = dict(live_old(0, 0, "hk", 0, 0), state_text=V1 % ("chunli", "close", "left", 40) + " opp_attacking=0")
    assert V.row_problems("chunli", {"train": [good]}) == []
    for bad in (V1 % ("chunli", "close", "left", 40),
                V1 % ("chunli", "close", "left", 40) + " opp_attacking=0 opp_attacking=1",
                "opp_attacking=1 " + V1 % ("chunli", "close", "left", 40)):
        assert V.row_problems("chunli", {"train": [dict(good, state_text=bad)]})


def test_gate_refuses_duplicate_ids():
    good = dict(live_old(0, 0, "hk", 0, 0), state_text=V1 % ("chunli", "close", "left", 40) + " opp_attacking=0")
    assert any("duplicate" in p for p in V.row_problems("chunli", {"train": [good], "test_real_left": [good]}))


# --- FAIL paths ---

def test_missing_image_is_a_problem(repo):
    os.remove(os.path.join(repo, "rollouts/lv_value/ryu/chunli/images/g00_00032_now.png"))
    probs = build(repo)["problems"]
    assert any("missing" in p and "g00_00032" in p for p in probs)


def test_missing_old_frame_is_a_problem(repo):
    os.remove(os.path.join(repo, "test_data/chunli/frames/left_close_0_stand_now.png"))
    assert any("missing" in p and "left_close_0_stand_now" in p for p in build(repo)["problems"])


def test_unjoinable_live_row_is_a_problem_never_guessed(tmp_path):
    repo = str(tmp_path)
    make_old(repo, rows=[live_old(0, 0, "hk", 0, 19), live_old(9, 999, "hk", 0, 0)])
    s = build(repo)
    assert any("g09_f00999" in p for p in s["problems"])
    assert not [r for r in load(repo, "train") if "g09_f00999" in r["id"]]     # dropped, not guessed


def test_ambiguous_join_is_a_problem(tmp_path):
    repo = str(tmp_path)
    make_old(repo, rows=[live_old(0, 0, "hk", 0, 19)])
    log = os.path.join(repo, "rollouts/live_dhalsim/chunli/actions.jsonl")
    with open(log, "a") as f:                                      # two decisions logged at one game + frame
        f.write(json.dumps(entry(0, 0, "hk", opp="dhalsim", taken=19, opp_state="stand")) + "\n")
    assert any("2 log entries" in p for p in build(repo)["problems"])


def test_live_row_disagreeing_with_its_log_is_a_problem(tmp_path):
    repo = str(tmp_path)
    make_old(repo, rows=[live_old(0, 0, "hk", 0, 7)])       # the log says taken 19
    assert any("g00_f00000" in p for p in build(repo)["problems"])


def test_prompt_with_memory_is_a_problem(tmp_path):
    repo = str(tmp_path)
    make_old(repo)
    write_log(repo, "ryu", "chunli", [entry(0, 0, prompt=V1 % ("chunli", "close", "left", 45) + "\nmemory: x")])
    assert any("memory" in p for p in build(repo)["problems"])


def test_too_few_value_rows_per_move_is_a_problem(repo):
    probs = build(repo, min_value_train=1)["problems"]
    assert any("chunli" in p and "spinning_bird_kick" in p for p in probs)
    assert not any("c.lk" in p and "value training rows" in p for p in probs)   # c.lk has one


# --- edges ---

def test_missing_log_root_and_empty_opponent_folder(tmp_path):
    repo = str(tmp_path)
    make_old(repo)
    s = build(repo)                                                  # no rollouts/lv_value at all
    assert s["problems"] == [] and load(repo, "test_heldout_guile") == []
    os.makedirs(os.path.join(repo, "rollouts/lv_value/honda/chunli"))  # an opponent folder with no log yet
    os.makedirs(os.path.join(repo, "rollouts/lv_value/ken"))
    assert build(repo, out=os.path.join(repo, "v2b"))["problems"] == []


def test_partial_last_line_is_skipped_and_reported(repo):
    with open(os.path.join(repo, "rollouts/lv_value/ryu/chunli/actions.jsonl"), "a") as f:
        f.write('{"game": 4, "frame"')                            # a line still being written
    s = build(repo)
    assert s["problems"] == [] and s["counts"]["chunli"]["partial_lines"] == 1


def test_refuses_to_build_over_an_existing_output(repo):
    build(repo)
    assert any("exists" in p for p in build(repo)["problems"])


def test_deterministic(repo):
    build(repo, out=os.path.join(repo, "a", "v2"))
    build(repo, out=os.path.join(repo, "b", "v2"))
    for d, _, files in os.walk(os.path.join(repo, "a", "v2")):
        for f in files:
            p = os.path.join(d, f)
            q = p.replace(os.path.join(repo, "a"), os.path.join(repo, "b"))
            assert open(p, "rb").read() == open(q, "rb").read(), p
    rows = load(repo, "train", out="a/v2")
    assert [r["id"] for r in rows] == sorted(r["id"] for r in rows)


def test_script_exits_1_on_a_problem(repo):
    os.remove(os.path.join(repo, "rollouts/lv_value/ryu/chunli/images/g00_00032_now.png"))
    args = [sys.executable, os.path.join(REPO, "scripts", "build_value_data.py"), "--test-data",
            os.path.join(repo, "test_data"), "--lv-root", os.path.join(repo, "rollouts/lv_value"), "--repo", repo,
            "--out", os.path.join(repo, "v2"), "--min-value-train", "0"]
    p = subprocess.run(args, capture_output=True, text=True)
    assert p.returncode == 1 and "PROBLEM" in p.stdout


# --- the live cap (every character the same number of new live training decisions) ---

def test_cap_is_the_smallest_positive_count():
    assert V.cap_of({"chunli": 10, "ryu": 4, "ken": 0}) == 4
    assert V.cap_of({"chunli": 10}) == 10                     # one character: its own count, nothing moves
    assert V.cap_of({}) is None and V.cap_of({"ryu": 0}) is None


def test_keep_games_round_robin_whole_games_at_or_under_the_cap():
    counts = {("ryu", 0): 3, ("ryu", 1): 3, ("ken", 0): 2, ("ken", 1): 2, ("ken", 3): 2}
    # game 0 of each opponent (ken 2 + ryu 3), then game 1 of each: ken 2 -> 7; ryu's 3 would pass the cap: stop
    assert V.keep_games(counts, 7) == {("ken", 0), ("ryu", 0), ("ken", 1)}
    assert V.keep_games(counts, 12) == set(counts)
    assert V.keep_games(counts, 1) == set()


def as_char(rows, char):
    return [json.loads(json.dumps(r).replace("chunli", char)) for r in rows]


def test_a_character_above_the_cap_moves_its_later_games_to_test_extra(repo):
    # Ryu has 2 new training decisions (games 0 and 1 vs Ken): the cap is 2; Chun-Li keeps game 0 vs Ryu (2) and
    # game 1 (one more would pass the cap) goes, real only, to test_extra
    write_log(repo, "p_vs_ryu", "ryu", [entry(0, 0, "hk", opp="ken", me="ryu"), entry(1, 0, "hk", opp="ken", me="ryu")])
    s = build(repo)
    assert s["problems"] == []
    assert s["cap"] == 2
    assert s["counts"]["chunli"]["new_train_decisions"] == 2 and s["counts"]["chunli"]["extra_decisions"] == 1
    assert s["counts"]["ryu"]["new_train_decisions"] == 2 and s["counts"]["ryu"]["extra_decisions"] == 0
    train = by_id(load(repo, "train"))
    extra = by_id(load(repo, "test_extra"))
    moved = "chunli-live_lv_ryu_g01_f00000-block_high"
    assert moved not in train and moved + "-m" not in train
    assert set(extra) == {moved, moved + "-value"} and not any(r["mirrored"] for r in extra.values())
    assert "chunli-live_lv_ryu_g00_f00032-throw" in train                    # the kept game, mirrored too
    assert "chunli-live_lv_ryu_g00_f00032-throw-m" in train
    assert "chunli-live_lv_ryu_g02_f00000-sweep" in by_id(load(repo, "test_real_left"))     # test games untouched
    assert len(load(repo, "test_heldout_guile")) == 4                         # the hold-out untouched
    assert load(repo, "test_extra", char="ryu") == []


def test_one_character_with_new_data_keeps_everything(repo):
    s = build(repo)
    assert s["cap"] == 3 and s["counts"]["chunli"]["extra_decisions"] == 0
    assert load(repo, "test_extra") == []


def test_balanced_characters_pass_the_training_share_check(tmp_path):
    from sf2.data.train_data import coverage_problems, load_data
    repo = str(tmp_path)
    d = make_old(repo)
    make_new(repo)
    for name in ("train", "test_real_left", "test_real_right"):      # Ryu: the same all8 rows as Chun-Li's
        rows = [json.loads(line) for line in open(os.path.join(d, name + ".jsonl"))]
        write_jsonl(os.path.join(repo, "test_data/ryu", name + ".jsonl"), as_char(rows, "ryu"))
        for r in rows:
            for p in r["images"]:
                png(os.path.join(repo, "test_data/ryu", p))
    write_jsonl(os.path.join(repo, "rollouts/live_dhalsim/ryu/actions.jsonl"),
                as_char([json.loads(line) for line in open(os.path.join(repo, "rollouts/live_dhalsim/chunli/actions.jsonl"))],
                        "ryu"))
    # Ryu vs Ken: 2 training decisions over 2 games -> the cap is 2, Chun-Li (3) keeps game 0 (2)
    write_log(repo, "p_vs_ryu", "ryu", [entry(g, 0, "hk", opp="ken", me="ryu") for g in (0, 1)])
    s = build(repo)
    assert s["problems"] == [] and s["cap"] == 2
    dirs = [os.path.join(repo, "test_data_v2", c) for c in ("chunli", "ryu")]
    train, val = load_data(dirs)
    probs = coverage_problems(train, val, dirs)
    # the train share check (val is a 5% sample of positions of different sizes: too few in a tiny fixture)
    assert not [p for p in probs if p.startswith("train rows unequal")], probs
    n = {c: sum(e["dataset"] == c for e in train + val) for c in ("chunli", "ryu")}
    assert n["chunli"] == n["ryu"]
