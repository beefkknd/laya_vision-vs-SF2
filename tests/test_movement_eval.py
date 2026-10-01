"""The movement eval (sf2.data.movement_eval, scripts/eval_movement.py; docs/prereg_movement.md "Judged against RAM"):
per-answer recall / precision, confusion, overall and BALANCED accuracy; the majority baseline (the training split's
most common answer); round 1's eye mapped onto the eight answers ("before"); the "trainable" verdict."""
import json
import os
import sys

import pytest

from sf2.data import movement as M
from sf2.data import movement_eval as E

A = M.ANSWERS


def rows_of(truths, char="chunli", opp="ryu", game=1):
    return [{"decision": "d%d" % i, "char": char, "opp": opp, "sub": "s", "game": game + i % 3, "answer": t}
            for i, t in enumerate(truths)]


# ---- metrics --------------------------------------------------------------------------------------------------------

def test_recall_precision_confusion_and_balanced_accuracy():
    truths = ["standing"] * 6 + ["attacking"] * 2
    preds = ["standing"] * 5 + ["attacking"] + ["attacking", "standing"]
    m = E.metrics(truths, preds)
    assert m["n"] == 8 and m["accuracy"] == pytest.approx(6 / 8)
    assert m["recall"]["standing"] == pytest.approx(5 / 6) and m["recall"]["attacking"] == pytest.approx(0.5)
    assert m["precision"]["standing"] == pytest.approx(5 / 6) and m["precision"]["attacking"] == pytest.approx(0.5)
    assert m["balanced_accuracy"] == pytest.approx((5 / 6 + 0.5) / 2)       # over the answers present in the truth
    assert m["confusion"]["standing"] == {"standing": 5, "attacking": 1}
    assert m["recall"]["jumping"] is None and m["precision"]["jumping"] is None
    assert m["support"]["standing"] == 6


def test_balanced_accuracy_is_not_plain_accuracy():
    truths = ["standing"] * 9 + ["jumping"]
    m = E.metrics(truths, ["standing"] * 10)
    assert m["accuracy"] == pytest.approx(0.9) and m["balanced_accuracy"] == pytest.approx(0.5)


def test_predicted_answer_is_the_argmax_ties_to_the_first_answer():
    assert E.predicted({a: 0.1 for a in A}) == "standing"
    assert E.predicted(dict({a: 0.1 for a in A}, blocking=0.3)) == "blocking"


def test_majority_is_the_training_splits_most_common_answer_and_scores_one_over_k():
    assert E.majority({"standing": 5, "attacking": 9}) == "attacking"
    assert E.majority({"standing": 9, "attacking": 9}) == "standing"           # tie: answer order
    truths = list(A) * 3
    m = E.metrics(truths, ["attacking"] * len(truths))
    assert m["balanced_accuracy"] == pytest.approx(1 / 8) and m["recall"]["attacking"] == 1.0


# ---- before: round 1's answers mapped -------------------------------------------------------------------------------

@pytest.mark.parametrize("truth,word,hit", [
    ("standing", "neutral", True), ("walking toward me", "neutral", True), ("walking away", "neutral", True),
    ("crouching", "neutral", False), ("jumping", "neutral", False), ("attacking", "neutral", False),
    ("attacking", "attacking", True), ("attacking", "recovering after a miss", True),
    ("blocking", "blocking", True), ("being hit", "being hit", True), ("being hit", "blocking", False),
    ("standing", "attacking", False), ("crouching", "attacking", False), ("jumping", "being hit", False)])
def test_before_mapping(truth, word, hit):
    assert E.before_hit(truth, word) is hit


def test_before_metrics_recall_per_truth_and_raw_confusion():
    truths = ["standing", "walking away", "attacking", "attacking", "crouching"]
    words = ["neutral", "neutral", "neutral", "recovering after a miss", "neutral"]
    m = E.before_metrics(truths, words)
    assert m["accuracy"] == pytest.approx(3 / 5)
    assert m["recall"] == {"standing": 1.0, "walking toward me": None, "walking away": 1.0, "crouching": 0.0,
                           "jumping": None, "attacking": 0.5, "being hit": None, "blocking": None}
    assert m["balanced_accuracy"] == pytest.approx((1 + 1 + 0.5 + 0) / 4)
    assert m["confusion"]["attacking"] == {"neutral": 1, "recovering after a miss": 1}
    assert set(E.BEFORE_MAP) == {"neutral", "attacking", "recovering after a miss", "blocking", "being hit"}


# ---- breakdowns and the verdict -----------------------------------------------------------------------------------------

def test_breakdown_by_character_and_opponent():
    rows = rows_of(["standing", "attacking"], char="ken", opp="ryu") + rows_of(["jumping"], char="ryu", opp="guile")
    preds = ["standing", "standing", "jumping"]
    b = E.breakdown(rows, preds, "char")
    assert b["ken"]["n"] == 2 and b["ken"]["accuracy"] == 0.5 and b["ryu"]["accuracy"] == 1.0
    assert set(E.breakdown(rows, preds, "opp")) == {"ryu", "guile"}


def test_bootstrap_lower_bound_is_by_game_and_seeded():
    truths = list(A) * 20
    rows = [dict(r, game=i // 8) for i, r in enumerate(rows_of(truths))]
    perfect = E.balanced_lower_bound(rows, truths, resamples=200)
    assert perfect == pytest.approx(1.0)
    noisy = [t if i % 3 else "standing" for i, t in enumerate(truths)]
    lb1 = E.balanced_lower_bound(rows, noisy, resamples=200)
    assert lb1 == E.balanced_lower_bound(rows, noisy, resamples=200) and lb1 < E.metrics(truths, noisy)[
        "balanced_accuracy"]


def file_result(bal, lb, att, maj_bal=0.125):
    return {"model": {"balanced_accuracy": bal, "recall": {"attacking": att}}, "balanced_lower_bound": lb,
            "majority": {"balanced_accuracy": maj_bal}}


@pytest.mark.parametrize("res,ok", [
    (file_result(0.6, 0.5, 0.7), True),
    (file_result(0.6, 0.5, 0.49), False),                 # attacking recall below 0.5
    (file_result(0.6, 0.12, 0.7), False),                 # not clearly above chance: lower bound <= 1/8
    (file_result(0.3, 0.2, 0.7, maj_bal=0.25), False),    # lower bound not above the majority's balanced accuracy
    (file_result(0.6, 0.5, None), False),                 # no attacking decisions: cannot say
])
def test_trainable_verdict_per_file(res, ok):
    assert E.file_verdict(res)["trainable"] is ok


def test_overall_verdict_needs_both_test_files():
    good, bad = file_result(0.6, 0.5, 0.7), file_result(0.6, 0.5, 0.2)
    assert E.verdict({"test_real": good, "test_heldout_guile": good})["trainable"] is True
    assert E.verdict({"test_real": good, "test_heldout_guile": bad})["trainable"] is False
    assert E.verdict({"test_real": good})["trainable"] is False


# ---- the script, with a fake agent ---------------------------------------------------------------------------------------

class FakeAgent:
    def __init__(self, pick):
        self.cfg = {"perception": True, "note_version": 3}
        self.pick, self.seen = pick, []

    def predict(self, state, questions):
        self.seen.append((state, dict(questions)))
        out = {}
        for k, q in questions.items():
            p = {a: 0.0 for a in q["criteria"]}
            p[self.pick(k, list(q["criteria"]))] = 1.0
            out[k] = {"probabilities": p}
        return {"answers": out}


def test_script_asks_the_question_scores_before_and_writes_json(tmp_path, monkeypatch):
    from sf2.data import movement_data as D
    from tests.test_movement_data import world
    root, u = world(tmp_path)
    data = str(tmp_path / "mv")
    D.build(root, u, data, val_decisions=2, workers=1)
    for c in ("chunli", "ryu"):                                     # frames: real pngs for the loader
        import numpy as np

        from sf2.data.dataset import save_png
        fr = os.path.join(u, c, "frames")
        for f in os.listdir(fr):
            save_png(np.zeros((256, 256, 3), np.uint8), os.path.join(fr, f))
    model, before = FakeAgent(lambda k, opts: "attacking"), FakeAgent(lambda k, opts: "neutral")
    scripts = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts")
    if scripts not in sys.path:
        sys.path.insert(0, scripts)
    import eval_movement as EV
    monkeypatch.setattr(EV, "load", lambda path, device: {"m": model, "b": before}[path])
    out = str(tmp_path / "ev.json")
    assert EV.main(["--model", "m", "--before", "b", "--data", data, "--out", out]) == 0
    doc = json.load(open(out))
    tr = doc["files"]["test_real"]
    assert tr["n"] == 4 and tr["model"]["recall"]["attacking"] is None
    assert tr["model"]["recall"]["jumping"] == 0.0 and tr["majority"]["answer"] == "standing"
    assert tr["before"]["recall"]["standing"] == 1.0 and tr["before"]["recall"]["jumping"] == 0.0
    assert set(tr["by_char"]) == {"chunli", "ryu"} and set(doc["files"]["test_heldout_guile"]["by_opp"]) == {"guile"}
    assert doc["verdict"]["trainable"] is False
    st, qs = model.seen[0]
    assert qs == {M.KEY: M.movement_question()} and st["context"] in ("me=chunli", "me=ryu")
    assert len(st["images"]) == 2
    from sf2.data.u_data import perception_question
    assert before.seen[0][1] == {"phase": perception_question("phase")}


def test_script_decisions_limit_takes_the_first_n_by_val_rank(tmp_path, monkeypatch):
    from sf2.data import movement_data as D
    from tests.test_movement_data import world
    root, u = world(tmp_path)
    data = str(tmp_path / "mv")
    D.build(root, u, data, val_decisions=2, workers=1)
    import numpy as np

    from sf2.data.dataset import save_png
    for c in ("chunli", "ryu"):
        fr = os.path.join(u, c, "frames")
        for f in os.listdir(fr):
            save_png(np.zeros((256, 256, 3), np.uint8), os.path.join(fr, f))
    scripts = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts")
    if scripts not in sys.path:
        sys.path.insert(0, scripts)
    import eval_movement as EV
    model = FakeAgent(lambda k, opts: "standing")
    monkeypatch.setattr(EV, "load", lambda path, device: model)
    out = str(tmp_path / "ev.json")
    assert EV.main(["--model", "m", "--before", "none", "--data", data, "--out", out, "--decisions", "1"]) == 0
    doc = json.load(open(out))
    assert doc["files"]["test_real"]["n"] == 2 and doc["decisions_limit"] == 1          # 1 per character
    assert doc["files"]["test_real"]["before"] is None
