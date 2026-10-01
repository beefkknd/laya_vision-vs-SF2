"""The U arm's dataset (sf2.data.u_data, scripts/build_u_data.py; docs/prereg_u_perception.md): per decision the two
frames with the HUD visible (frames v3, no mirroring), note v3 "me=<char>", the perception questions labelled from the
stored RAM rows (thresholds given as an argument), question 8 per move (forward excluded) with the soft target of the
decision's RAM cell; "unknown" -> no row; split by whole games (2,5,8 test, 0 val, rest train; Chun-Li vs Guile held
out). A small synthetic collection is built end to end."""
import json
import os

import numpy as np
import pytest

from ram_rows import row
from sf2.data import u_data as U
from sf2.data.perception import Q8_ANSWERS, QUESTIONS
from sf2.system1.system1 import choices
from sf2.vocab import FIGHTERS

TH = {"lag": 1, "throw_max": {"chunli": 47, "all": 40}, "poke_max": {"chunli": 87, "all": 90}, "mid_max": 120,
      "trend_eps": 2, "k": 6, "walls": [53, 459], "corner_d": 14, "near_px": 120}
CHUNLI = 5
ME = "chunli"
CELL = ["chunli", "mid", 0, 0]
TARGETS = {"answers": list(Q8_ANSWERS), "targets": [
    {"cell": CELL, "move": "lk", "p": {"likely works": 0.7, "may work": 0.2, "likely fails": 0.1}},
    {"cell": CELL, "move": "throw", "p": {"likely works": 0.1, "may work": 0.3, "likely fails": 0.6}},
    {"cell": CELL, "move": "forward", "p": {"likely works": 1.0, "may work": 0.0, "likely fails": 0.0}},
    {"cell": ["chunli", "close", 0, 0], "move": "mp", "p": {"likely works": 1.0, "may work": 0.0, "likely fails": 0.0}}]}


def ram_rec(game, frame, n=60, ahead=28, gap=100, p1=None, p2=None):
    rows = [row(dict({"char": CHUNLI}, **(p1 or {})), dict({"x": 200 + gap}, **(p2 or {}))) for _ in range(n + 1 + ahead)]
    names = list(rows[0])
    return {"game": game, "frame": frame, "n": n, "names": names, "rows": [[r[k] for k in names] for r in rows]}


def entry(game, frame, opp="ryu", gap=100, action="lk", rng="mid", opp_state="stand", air=False):
    return {"game": game, "frame": frame, "me": ME, "opp": opp, "gap": gap, "range": rng, "my_life": 176,
            "opp_life": 176, "opp_state": opp_state, "opp_air": air, "action": action, "explored": True,
            "images": ["g%02d_%05d_prev.png" % (game, frame), "g%02d_%05d_now.png" % (game, frame)]}


def raw_png(path):
    from sf2.data.dataset import save_png
    img = np.zeros((224, 256, 3), np.uint8)
    img[:62] = 200                        # the HUD rows: must stay visible
    img[62:] = 50
    save_png(img, path)


def collection(root, logs):
    """logs: {(sub, char): [(entry, ram record)]}"""
    for (sub, char), decs in logs.items():
        d = os.path.join(root, sub, char)
        os.makedirs(os.path.join(d, "images"))
        with open(os.path.join(d, "actions.jsonl"), "w") as f:
            f.write("".join(json.dumps(e) + "\n" for e, _ in decs))
        with open(os.path.join(d, "ram.jsonl"), "w") as f:
            f.write("".join(json.dumps(r) + "\n" for _, r in decs if r is not None))
        for e, _ in decs:
            for p in e["images"]:
                raw_png(os.path.join(d, "images", p))


def files(tmp_path, th=TH):
    t, q = str(tmp_path / "th.json"), str(tmp_path / "q8.json")
    json.dump(th, open(t, "w"))
    json.dump(TARGETS, open(q, "w"))
    return t, q


def read(path):
    return [json.loads(x) for x in open(path)] if os.path.exists(path) else []


# ---- frames v3 and note v3 ------------------------------------------------------------------------------------------

def test_hud_frame_pads_and_keeps_the_hud():
    img = np.full((224, 256, 3), 9, np.uint8)
    out = U.hud_frame(img)
    assert out.shape == (256, 256, 3) and (out[:224] == 9).all() and (out[224:] == 0).all()
    assert (U.hud_frame(out) == out).all()                     # a stored frame passes unchanged


@pytest.mark.parametrize("shape", [(224, 255, 3), (257, 256, 3), (224, 256)])
def test_hud_frame_refuses_other_shapes(shape):
    with pytest.raises(ValueError):
        U.hud_frame(np.zeros(shape, np.uint8))


def test_note_v3_is_the_character_only():
    assert U.eye_note("chunli") == "me=chunli" and U.FRAMES_VERSION == 3 and U.NOTE_VERSION == 3
    with pytest.raises(ValueError):
        U.eye_note("akuma")


# ---- questions ------------------------------------------------------------------------------------------------------

def test_every_question_is_a_choice_whose_criteria_are_the_answers():
    qs = U.questions(ME)
    moves = [m for m in choices(ME) if m != "forward"]
    assert list(qs) == list(U.PERCEPTION) + ["q8:" + m for m in moves]
    for k in U.PERCEPTION:
        assert qs[k]["type"] == "choice" and tuple(qs[k]["criteria"]) == QUESTIONS[k]
    for m in moves:
        assert tuple(qs["q8:" + m]["criteria"]) == Q8_ANSWERS and m in qs["q8:" + m]["instructions"]
    with pytest.raises(ValueError):
        U.q8_question("forward")


def test_questions_are_general_short_and_consistent():
    qs = U.questions(ME)
    text = json.dumps(qs).lower()
    assert not [c for c in FIGHTERS if c != ME and c in text]          # no opponent names
    assert all(len(q["instructions"]) <= 70 and q["instructions"].endswith("?") for q in qs.values())
    assert len({q["instructions"] for q in qs.values()}) == len(qs)
    assert all(len(d) <= 45 for q in qs.values() for d in q["criteria"].values())
    assert U.questions(ME) == U.questions(ME)                         # byte-identical at train and play time


# ---- split -----------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("opp,game,want", [("ryu", 2, "test"), ("ryu", 15, "test"), ("ryu", 28, "test"),
                                           ("ryu", 0, "val"), ("ryu", 30, "val"), ("ryu", 1, "train"),
                                           ("ryu", 9, "train"), ("guile", 1, "heldout"), ("guile", 2, "heldout"),
                                           ("guile", 0, "heldout")])
def test_split_by_whole_games(opp, game, want):
    assert U.split_of(ME, opp, game) == want


def test_guile_is_held_out_for_chunli_only():
    assert U.split_of("ryu", "guile", 1) == "train"


# ---- one decision ------------------------------------------------------------------------------------------------------

def targets():
    return {tuple(t["cell"]): {t2["move"]: t2["p"] for t2 in TARGETS["targets"] if t2["cell"] == t["cell"]}
            for t in TARGETS["targets"]}


def test_decision_rows_labels_and_soft_q8():
    rows, rec = U.decision_rows(ME, "ryu", entry(1, 64), ram_rec(1, 64), TH, targets())
    by = {r["key"]: r for r in rows}
    assert by["range"]["answer"] == "mid" and by["phase"]["answer"] == "neutral"
    assert by["my_bar"]["label"] == QUESTIONS["my_bar"].index("full")
    assert by["q8:lk"]["target"] == [0.7, 0.2, 0.1] and by["q8:lk"]["answer"] == "likely works"
    assert by["q8:throw"]["label"] == Q8_ANSWERS.index("likely fails") and not by["q8:throw"]["confident"]
    assert "q8:forward" not in by and "q8:mp" not in by                 # no target in this cell: masked
    assert all(r["state_text"] == "me=chunli" and r["split"] == "train" for r in rows)
    assert all(r["images"] == ["frames/ryu_g001_f00064_prev.png", "frames/ryu_g001_f00064_now.png"] for r in rows)
    assert rec["cell"] == ("chunli", "mid", 0, 0)
    assert all("action" not in r for r in rows)          # train_data's stage-1 coverage reads "action" rows


def test_unknown_labels_get_no_row():
    rows, rec = U.decision_rows(ME, "ryu", entry(1, 64), ram_rec(1, 64, n=3), TH, targets())
    assert rec["labels"]["trend"] == "unknown" and "trend" not in {r["key"] for r in rows}


def test_q8_target_follows_the_logged_cell_not_the_labels():
    rows, _ = U.decision_rows(ME, "ryu", entry(1, 64, rng="close"), ram_rec(1, 64), TH, targets())
    assert {r["key"] for r in rows if r["task"] == "q8"} == {"q8:mp"}


def test_thresholds_are_an_argument():
    th = dict(TH, throw_max={"all": 150})
    rows, _ = U.decision_rows(ME, "ryu", entry(1, 64), ram_rec(1, 64), th, targets())
    assert {r["key"]: r["answer"] for r in rows}["range"] == "throw"


def test_join_checks_the_decision_row():
    assert U.join_problem(entry(1, 64), ram_rec(1, 64)) is None
    assert U.join_problem(entry(1, 64, gap=90), ram_rec(1, 64))
    assert U.join_problem(entry(1, 68), ram_rec(1, 64))
    ko = dict(entry(1, 64), my_life=0)
    assert U.join_problem(ko, ram_rec(1, 64, p1={"life": 255})) is None      # a KO wraps the byte; the entry logs 0


# ---- build end to end ------------------------------------------------------------------------------------------------

def world(tmp_path, extra=None):
    root = str(tmp_path / "u")
    logs = {("ryu", ME): [(entry(g, 64), ram_rec(g, 64)) for g in (0, 1, 2, 10)],
            ("guile", ME): [(entry(g, 64, opp="guile"), ram_rec(g, 64)) for g in (1, 2)]}
    logs.update(extra or {})
    collection(root, logs)
    return root


def test_build_splits_frames_and_stats(tmp_path):
    t, q = files(tmp_path)
    out = str(tmp_path / "out")
    res = U.build(world(tmp_path), out, t, q)
    assert res["problems"] == []
    base = os.path.join(out, ME)
    games = {f: sorted({(r["sub"], r["game"]) for r in read(os.path.join(base, f + ".jsonl"))})
             for f in ("train", "val", "test_real", "test_heldout_guile")}
    assert games == {"train": [("ryu", 1)], "val": [("ryu", 0), ("ryu", 10)], "test_real": [("ryu", 2)],
                     "test_heldout_guile": [("guile", 1), ("guile", 2)]}
    from sf2.data.build import _load
    img = _load(os.path.join(base, "frames", "ryu_g001_f00064_now.png"))
    assert img.shape == (256, 256, 3) and (img[:62] == 200).all() and (img[62:224] == 50).all()
    assert (img[224:] == 0).all()
    assert not any(p.startswith("mirror") for p in os.listdir(os.path.join(base, "frames")))
    st = json.load(open(os.path.join(base, "stats.json")))
    assert st["decisions"] == 6 and st["decisions_by_split"] == {"train": 1, "val": 2, "test": 1, "heldout": 2}
    assert st["files"]["train"]["questions"]["q8"] == {"rows": 2, "labels": {"likely fails": 1, "likely works": 1}}
    assert st["files"]["val"]["questions"]["range"] == {"rows": 2, "labels": {"mid": 2}}
    assert st["unknown"]["trend"] == 0 and st["mapping"]["decisions"] == 6
    meta = json.load(open(os.path.join(out, "build.json")))
    assert meta["frames_version"] == 3 and len(meta["thresholds_sha256"]) == 64


def test_build_refuses_an_existing_out(tmp_path):
    t, q = files(tmp_path)
    out = tmp_path / "out"
    os.makedirs(out / "x")
    assert U.build(world(tmp_path), str(out), t, q)["problems"]


def test_a_decision_without_its_ram_record_is_a_problem(tmp_path):
    t, q = files(tmp_path)
    root = world(tmp_path, {("ken", ME): [(entry(1, 64, opp="ken"), ram_rec(1, 64)), (entry(3, 64, opp="ken"), None)]})
    res = U.build(root, str(tmp_path / "out"), t, q)
    assert any("ken/chunli g3 f64: 0 ram records" in p for p in res["problems"])


def test_a_ram_record_without_its_entry_is_a_problem(tmp_path):
    t, q = files(tmp_path)
    root = world(tmp_path)
    with open(os.path.join(root, "ryu", ME, "ram.jsonl"), "a") as f:
        f.write(json.dumps(ram_rec(7, 64)) + "\n")
    res = U.build(root, str(tmp_path / "out"), t, q)
    assert any("ram record without an action entry" in p for p in res["problems"])


def test_a_partial_last_line_is_counted_not_a_problem(tmp_path):
    t, q = files(tmp_path)
    root = world(tmp_path)
    with open(os.path.join(root, "ryu", ME, "actions.jsonl"), "a") as f:
        f.write('{"game": 11, "fra')
    res = U.build(root, str(tmp_path / "out"), t, q)
    assert res["problems"] == [] and res["counts"][ME]["partial_lines"] == 1


def test_row_gates_catch_a_note_with_ram_and_a_game_in_two_files():
    good, _ = U.decision_rows(ME, "ryu", entry(1, 64), ram_rec(1, 64), TH, targets())
    bad = [dict(good[0], state_text="me=chunli dist=mid")]
    assert U.row_problems(ME, {"train": bad})
    assert U.row_problems(ME, {"train": good[:1], "val": [dict(good[1], id="x")]})
    assert U.row_problems(ME, {"train": good}) == []


# ---- step numbers -------------------------------------------------------------------------------------------------

def test_training_steps_match_train_py_and_the_budget():
    assert U.training_steps(800_000) == 200_000 and U.training_steps(3) == 1
    assert U.step_budget(4.5, 135) == 36_450


# ---- the validation cap (opt-in): whole decisions of the val games, chosen by hash ---------------------------------

def test_val_cap_keeps_whole_decisions_by_hash_and_the_rest_apart(tmp_path):
    t, q = files(tmp_path)
    out = str(tmp_path / "out")
    res = U.build(world(tmp_path), out, t, q, val_decisions=1)
    assert res["problems"] == []
    base = os.path.join(out, ME)
    val, rest = read(os.path.join(base, "val.jsonl")), read(os.path.join(base, "val_rest.jsonl"))
    assert len({r["decision"] for r in val}) == 1 and len({r["decision"] for r in rest}) == 1
    want = min(["chunli-u_ryu_g000_f00064", "chunli-u_ryu_g010_f00064"], key=U.val_rank)
    assert {r["decision"] for r in val} == {want}
    assert json.load(open(os.path.join(base, "stats.json")))["val_cap"] == {"decisions": 2, "kept": 1, "rest": 1}
    full, _ = U.decision_rows(ME, "ryu", entry(0, 64), ram_rec(0, 64), TH, targets())
    assert len(val) == len(rest) == len(full)                   # every question of a decision stays together


def test_no_val_cap_by_default(tmp_path):
    t, q = files(tmp_path)
    out = str(tmp_path / "out")
    U.build(world(tmp_path), out, t, q)
    assert not os.path.exists(os.path.join(out, ME, "val_rest.jsonl"))
    assert len({r["decision"] for r in read(os.path.join(out, ME, "val.jsonl"))}) == 2


def test_a_decision_whose_ram_row_disagrees_is_a_problem_in_the_build(tmp_path):
    t, q = files(tmp_path)
    root = world(tmp_path, {("ken", ME): [(entry(1, 64, opp="ken", gap=90), ram_rec(1, 64))]})
    res = U.build(root, str(tmp_path / "out"), t, q)
    assert any(p.startswith("ken/chunli g1 f64: ram (game, frame, gap") for p in res["problems"])
    assert not os.path.exists(os.path.join(str(tmp_path / "out"), ME, "frames", "ken_g001_f00064_now.png"))
