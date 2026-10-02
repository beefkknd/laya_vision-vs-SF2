"""scripts/train.py --balance question (docs/eye_questions_v1.md, "Training pre-registration", eye_all): every
QUESTION (a --data dir's parent folder) gets an equal share of training draws, and the answer dirs within a question
are drawn equally; the default (--balance rows) and --balance sampling are unchanged."""
import importlib.util
import os
import sys
from types import SimpleNamespace

import pytest

from sf2.data import train_data as TD

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EYE = ["q1/yes", "q1/no", "q3/moving", "q3/attack", "q3/special", "q4/ground", "q4/air", "q5/close", "q5/far"]
SIZES = {"yes": 571, "no": 571, "moving": 3575, "attack": 3575, "special": 3575, "ground": 9160, "air": 9160,
         "close": 4144, "far": 4144}


def _train_module():
    if os.path.join(ROOT, "scripts") not in sys.path:
        sys.path.insert(0, os.path.join(ROOT, "scripts"))
    spec = importlib.util.spec_from_file_location("train_script", os.path.join(ROOT, "scripts", "train.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _rows(name, n, split):
    return [{"dataset": name, "split": split, "id": "%s-%s-%d" % (name, split, i)} for i in range(n)]


def _eye(train_sizes=SIZES, val_n=135):
    train = [r for n, k in train_sizes.items() for r in _rows(n, k, "train")]
    val = [r for n in train_sizes for r in _rows(n, val_n, "val")]
    return train, val


def test_question_weights_quarter_each_question_answers_equal():
    w = TD.question_weights(EYE)
    total = sum(w.values())
    q = {"q1": ("yes", "no"), "q3": ("moving", "attack", "special"), "q4": ("ground", "air"), "q5": ("close", "far")}
    for answers in q.values():
        assert sum(w[a] for a in answers) / total == pytest.approx(0.25)
        assert len({w[a] for a in answers}) == 1
    assert w["moving"] / total == pytest.approx(0.25 / 3)
    assert w["yes"] / total == pytest.approx(0.125)


def test_laya_mix_with_question_weights_matches_dir_sizes_ignored():
    train, _ = _eye()
    shares = TD.sampling_shares(train, TD.question_weights(EYE))
    assert shares["yes"] == pytest.approx(0.125) and shares["air"] == pytest.approx(0.125)
    assert shares["special"] == pytest.approx(1 / 12)
    assert sum(shares.values()) == pytest.approx(1.0)


def test_question_problems_pass_on_the_eye_layout():
    train, val = _eye()
    assert TD.sampling_problems(train, val, EYE, 500, 100, by_question=True) == []


def test_plain_sampling_would_give_q3_three_ninths():
    # the reason for the option: per-dir equal draws give q3 (three answer dirs) 1/3 of the draws, not 1/4
    train, _ = _eye()
    shares = TD.sampling_shares(train)
    assert sum(shares[a] for a in ("moving", "attack", "special")) == pytest.approx(1 / 3)


def test_colliding_answer_names_refused():
    with pytest.raises(ValueError):
        TD.question_weights(["q1/yes", "q2/yes"])


def test_question_problems_catch_an_unequal_mix(monkeypatch):
    train, val = _eye()
    bad = dict(TD.question_weights(EYE), moving=1.0)
    monkeypatch.setattr(TD, "question_weights", lambda dirs: bad)
    problems = TD.sampling_problems(train, val, EYE, 500, 100, by_question=True)
    assert any("question" in p for p in problems) and any("within q3" in p for p in problems)


def test_question_problems_keep_the_row_minimums():
    train, val = _eye()
    assert any("yes has 571 train rows < 1000" in p for p in TD.sampling_problems(train, val, EYE, by_question=True))


def test_train_flag_passes_weights_and_check():
    mod = _train_module()
    args = mod.parse_args(["--data", "x", "--out", "y"])
    assert args.balance == "rows" and mod.mix_weights(args) is TD.MIX_WEIGHTS is None
    args = mod.parse_args(["--data", "x", "--out", "y", "--balance", "sampling"])
    assert mod.mix_weights(args) is None
    args = mod.parse_args(["--out", "y", "--balance", "question"] + sum([["--data", d] for d in EYE], []))
    assert mod.mix_weights(args) == TD.question_weights(EYE)


def test_train_data_problems_by_question(monkeypatch):
    mod = _train_module()
    seen = {}

    def fake(train, val, dirs, min_train=0, min_val=0, by_question=False):
        seen["call"] = (min_train, min_val, by_question)
        return []
    monkeypatch.setattr(mod, "sampling_problems", fake)
    def cov(*a, **k):
        seen["share_check"] = k.get("share_check")
        return []
    monkeypatch.setattr(mod, "coverage_problems", cov)
    args = SimpleNamespace(balance="question", data=EYE, min_sampled_train=500, min_sampled_val=100)
    mod.data_problems([], [], args)
    assert seen["call"] == (500, 100, True) and seen["share_check"] is False
