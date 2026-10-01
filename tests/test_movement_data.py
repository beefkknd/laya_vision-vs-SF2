"""The movement dataset (sf2.data.movement_data, scripts/build_movement_data.py; docs/prereg_movement.md): one row
per decision of test_data_u, the SAME frames (symlinked) and note, the SAME split per decision (read from
test_data_u's files: test, held-out Guile, train; the val games capped by u_data.val_rank), "unknown" masked, and
train.jsonl class-balanced by deterministic repetition per character (val and test natural). A small synthetic
test_data_u + U collection is built end to end, and train.py's own data gates are run on the result."""
import collections
import json
import os

import pytest
from ram_rows import row

from sf2.data import movement as M
from sf2.data import movement_data as D
from sf2.data import u_data as U
from sf2.emu.vs import GROUND_Y

SUB = "p_vs_x"
# (state overrides at the displayed row t, his x at t - 4) per class; x at t is 300, mine 200
CLASS_ROWS = {
    "standing": ({}, 300), "walking toward me": ({}, 306), "walking away": ({}, 294),
    "crouching": ({"state": 0x02}, 300), "jumping": ({"state": 0x04, "y": GROUND_Y - 30}, 300),
    "attacking": ({"state": 0x0A}, 300), "being hit": ({"state": 0x0E, "react": 0x0E}, 300),
    "blocking": ({"state": 0x08}, 300)}


def ram_rec(game, frame, cls, n=60, ahead=20):
    over, prev_x = CLASS_ROWS[cls] if cls != "unknown" else ({"state": 0x06}, 300)
    rows = [row({"x": 200}, {"x": 300}) for _ in range(n + 1 + ahead)]
    t = n - 1
    rows[t] = row({"x": 200}, dict({"x": 300}, **over))
    rows[t - 4] = row({"x": 200}, {"x": prev_x})
    names = list(rows[0])
    return {"game": game, "frame": frame, "n": n, "names": names, "rows": [[r[k] for k in names] for r in rows]}


def action(char, opp, game, frame):
    return {"game": game, "frame": frame, "me": char, "opp": opp, "gap": 100, "my_life": 176, "opp_life": 176,
            "action": "lk", "images": ["a.png", "b.png"]}


U_FILE = {"train": "train", "val": "val", "test": "test_real", "heldout": "test_heldout_guile"}


def u_rows(char, opp, game, frame, split):
    did = U.decision_id(char, SUB, game, frame)
    common = {"decision": did, "char": char, "opp": opp, "sub": SUB, "game": game, "frame": frame, "split": split,
              "images": U.frame_names(SUB, game, frame), "state_text": U.eye_note(char), "perception": True}
    return [dict(common, id=did + "-range", key="range"), dict(common, id=did + "-air", key="air")]


def decisions(char):
    """(opp, game, frame, class, u file): every class in train (standing x3), 3 val-game decisions, test, held out."""
    out = []
    f = 64
    for i, cls in enumerate(M.ANSWERS + ("standing", "standing", "unknown")):
        out.append(("ryu", (1, 3, 4, 6, 7, 9)[i % 6], f, cls, "train"))
        f += 64
    out += [("ryu", 0, f, "attacking", "val"), ("ryu", 10, f + 64, "standing", "val_rest"),
            ("ryu", 20, f + 128, "blocking", "val_rest"), ("ryu", 2, f + 192, "jumping", "test_real"),
            ("ryu", 12, f + 256, "standing", "test_real")]
    if char == "chunli":
        out += [("guile", 41, f + 320, "crouching", "test_heldout_guile"),
                ("guile", 42, f + 384, "being hit", "test_heldout_guile")]
    return out


def world(tmp_path, chars=("chunli", "ryu"), drop_ram=None, wrong_file=None):
    root, u = str(tmp_path / "rollouts"), str(tmp_path / "test_data_u")
    for char in chars:
        logd = os.path.join(root, SUB, char)
        os.makedirs(logd)
        base = os.path.join(u, char)
        os.makedirs(os.path.join(base, "frames"))
        files = collections.defaultdict(list)
        acts, rams = [], []
        for opp, game, frame, cls, uf in decisions(char):
            split = {"train": "train", "val": "val", "val_rest": "val", "test_real": "test",
                     "test_heldout_guile": "heldout"}[uf]
            if wrong_file and (char, game, frame) == wrong_file[0]:
                uf = wrong_file[1]
            files[uf] += u_rows(char, opp, game, frame, split)
            acts.append(action(char, opp, game, frame))
            if drop_ram != (char, game, frame):
                rams.append(ram_rec(game, frame, cls))
            for p in U.frame_names(SUB, game, frame):
                open(os.path.join(base, p), "wb").close()
        for name, rows in files.items():
            with open(os.path.join(base, name + ".jsonl"), "w") as fh:
                fh.write("".join(json.dumps(r) + "\n" for r in rows))
        with open(os.path.join(logd, "actions.jsonl"), "w") as fh:
            fh.write("".join(json.dumps(a) + "\n" for a in acts))
        with open(os.path.join(logd, "ram.jsonl"), "w") as fh:
            fh.write("".join(json.dumps(r) + "\n" for r in rams))
    return root, u


def read(path):
    return [json.loads(x) for x in open(path)] if os.path.exists(path) else []


def built(tmp_path, val_decisions=2, **kw):
    root, u = world(tmp_path, **kw)
    out = str(tmp_path / "test_data_mv")
    return D.build(root, u, out, val_decisions=val_decisions, workers=1), u, out


# ---- end to end ------------------------------------------------------------------------------------------------------

def test_build_is_clean_and_keeps_test_datas_split_per_decision(tmp_path):
    res, u, out = built(tmp_path)
    assert res["problems"] == []
    for char in ("chunli", "ryu"):
        base = os.path.join(out, char)
        got = {f: sorted({r["decision"] for r in read(os.path.join(base, f + ".jsonl"))})
               for f in ("train", "val", "val_rest", "test_real", "test_heldout_guile")}
        want = {f: sorted({r["decision"] for r in read(os.path.join(u, char, f + ".jsonl"))})
                for f in ("train", "test_real", "test_heldout_guile")}
        unknown = U.decision_id(char, SUB, 7, 64 * 11)                 # the masked train decision
        assert got["train"] == [d for d in want["train"] if d != unknown]
        assert got["test_real"] == want["test_real"] and got["test_heldout_guile"] == want["test_heldout_guile"]
        val_games = {r["decision"] for f in ("val", "val_rest") for r in read(os.path.join(u, char, f + ".jsonl"))}
        assert set(got["val"]) | set(got["val_rest"]) == val_games and not set(got["val"]) & set(got["val_rest"])
        assert got["val"] == sorted(sorted(val_games, key=U.val_rank)[:2])          # the cap: by u_data.val_rank
    assert read(os.path.join(out, "ryu", "test_heldout_guile.jsonl")) == []
    assert not os.path.exists(os.path.join(out, "ryu", "test_heldout_guile.jsonl"))


def test_rows_ask_the_one_question_with_the_same_frames_and_note(tmp_path):
    res, u, out = built(tmp_path)
    base = os.path.join(out, "chunli")
    assert os.path.islink(os.path.join(base, "frames"))
    assert os.path.realpath(os.path.join(base, "frames")) == os.path.realpath(os.path.join(u, "chunli", "frames"))
    urows = {r["decision"]: r for f in ("test_real",) for r in read(os.path.join(u, "chunli", f + ".jsonl"))}
    for r in read(os.path.join(base, "test_real.jsonl")):
        assert r["images"] == urows[r["decision"]]["images"] and r["state_text"] == "me=chunli"
        assert r["question"] == M.movement_question() and r["perception"] is True
        assert list(r["question"]["criteria"])[r["label"]] == r["answer"]
        assert all(os.path.exists(os.path.join(base, p)) for p in r["images"])


def test_labels_come_from_the_ram_rows(tmp_path):
    res, u, out = built(tmp_path)
    got = {r["decision"]: r["answer"] for f in ("train", "val", "val_rest", "test_real", "test_heldout_guile")
           for r in read(os.path.join(out, "chunli", f + ".jsonl"))}
    for opp, game, frame, cls, _ in decisions("chunli"):
        did = U.decision_id("chunli", SUB, game, frame)
        if cls == "unknown":
            assert did not in got
        else:
            assert got[did] == cls


def test_train_is_class_balanced_by_deterministic_repetition(tmp_path):
    res, u, out = built(tmp_path)
    rows = read(os.path.join(out, "chunli", "train.jsonl"))
    counts = collections.Counter(r["answer"] for r in rows)
    assert counts == {a: 3 for a in M.ANSWERS}                       # standing has 3 decisions: every class x3
    assert len({r["id"] for r in rows}) == len(rows)
    natural = [r for r in rows if r["copy"] == 0]
    assert len(natural) == 10 and collections.Counter(r["answer"] for r in natural)["standing"] == 3
    res2, _, out2 = built(tmp_path / "again")
    assert open(os.path.join(out, "chunli", "train.jsonl")).read() == \
        open(os.path.join(out2, "chunli", "train.jsonl")).read()


def test_val_and_test_stay_natural(tmp_path):
    res, u, out = built(tmp_path)
    for f in ("val", "val_rest", "test_real", "test_heldout_guile"):
        assert all(r["copy"] == 0 for r in read(os.path.join(out, "chunli", f + ".jsonl")))


def test_stats_count_classes_per_split_and_opponent(tmp_path):
    res, u, out = built(tmp_path)
    st = json.load(open(os.path.join(out, "chunli", "stats.json")))
    assert st["unknown"] == 1 and st["decisions"] == 18 and st["labelled"] == 17
    assert st["natural"]["train"]["standing"] == 3 and st["balanced_train"]["standing"] == 3
    assert st["natural"]["test_heldout_guile"] == {"crouching": 1, "being hit": 1}
    assert st["by_opponent"]["test_heldout_guile"]["guile"] == {"crouching": 1, "being hit": 1}
    assert res["counts"]["chunli"]["train_rows"] == 24


def test_trainers_data_gates_pass_on_the_layout(tmp_path):
    from sf2.data.train_data import (
        checkpoint_tags,
        coverage_problems,
        load_data,
        sampling_problems,
    )
    res, u, out = built(tmp_path)
    dirs = [os.path.join(out, c) for c in ("chunli", "ryu")]
    train, val = load_data(dirs)
    assert len(train) == 48 and len(val) == 4
    assert coverage_problems(train, val, dirs, share_check=False) == []
    assert sampling_problems(train, val, dirs, min_train=24, min_val=2) == []
    assert checkpoint_tags(train + val) == {"note_version": 3, "value_questions": False, "perception": True}


# ---- problems ---------------------------------------------------------------------------------------------------------

def test_a_decision_without_its_ram_record_is_a_problem(tmp_path):
    res, _, _ = built(tmp_path, drop_ram=("chunli", 2, 64 * 12 + 192))
    assert any("ram" in p for p in res["problems"])


def test_a_decision_in_a_file_its_game_does_not_belong_to_is_a_problem(tmp_path):
    res, _, _ = built(tmp_path, wrong_file=(("ryu", 1, 64), "test_real"))
    assert any("split" in p for p in res["problems"])


def test_a_class_missing_from_a_characters_train_is_a_problem(tmp_path):
    root, u = world(tmp_path, chars=("ryu",))
    path = os.path.join(u, "ryu", "train.jsonl")
    blocking = U.decision_id("ryu", SUB, (1, 3, 4, 6, 7, 9)[7 % 6], 64 * 8)
    rows = [r for r in read(path) if r["decision"] != blocking]
    open(path, "w").write("".join(json.dumps(r) + "\n" for r in rows))
    res = D.build(root, u, str(tmp_path / "mv"), val_decisions=2, workers=1)
    assert any("blocking" in p and "train" in p for p in res["problems"])


def test_build_refuses_an_existing_out(tmp_path):
    root, u = world(tmp_path)
    out = tmp_path / "mv"
    out.mkdir()
    (out / "x").write_text("")
    with pytest.raises(SystemExit):
        D.build(root, u, str(out), val_decisions=2, workers=1)


# ---- the step budget -----------------------------------------------------------------------------------------------

@pytest.mark.parametrize("rows", [1, 7, 8, 1000, 123457, 797253])
def test_epochs_for_steps_give_exactly_the_steps_train_py_runs(rows):
    e = D.epochs_for_steps(10000, rows, 8)
    assert max(1, int(float(e) * rows / 8)) == 10000


def test_the_script_builds_and_prints_the_epochs(tmp_path, capsys):
    import sys
    scripts = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts")
    if scripts not in sys.path:
        sys.path.insert(0, scripts)
    import build_movement_data as B
    root, u = world(tmp_path)
    out = str(tmp_path / "mv")
    assert B.main(["--root", root, "--u-data", u, "--out", out, "--val-decisions", "2", "--workers", "2"]) == 0
    s = json.load(open(os.path.join(out, "summary.json")))
    assert s["train_rows"] == 48 and s["natural"]["train"]["standing"] == 6
    assert max(1, int(float(s["epochs_for_steps"]) * 48 / 8)) == 10000
    assert "train (balanced)" in capsys.readouterr().out


def test_the_val_cap_keeps_the_first_by_val_rank_not_by_id(tmp_path):
    res, u, out = built(tmp_path, val_decisions=1)
    assert res["problems"] == []
    pool = [U.decision_id("ryu", SUB, g, f) for _, g, f, _, uf in decisions("ryu") if uf in ("val", "val_rest")]
    by_rank, by_id = min(pool, key=U.val_rank), min(pool)
    assert by_rank != by_id                                          # the fixture tells the two orders apart
    assert [r["decision"] for r in read(os.path.join(out, "ryu", "val.jsonl"))] == [by_rank]
    assert sorted(r["decision"] for r in read(os.path.join(out, "ryu", "val_rest.jsonl"))) == \
        sorted(set(pool) - {by_rank})
