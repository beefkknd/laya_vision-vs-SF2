"""The mechanical shortcut check of docs/eye_questions_v1.md rule 5 (scripts/gate_eye_data.py, gate "shortcut"):
can a trivial predictor that sees ONLY metadata - never the frames - answer the question?

Features per row (row["shortcut"], written by sf2.data.eye_data; finer than the strata it was matched on):
  q1  match pair, game index, the pose of player 1 and of player 2 ("projectile" = the throwing pose, else the
      grid movement)
  q3 / q4  match pair, side asked, game index, the other fighter's grid movement
  q5  match pair, game index, the left and the right fighter's grid movement
Predictors, fit on the train split only, scored on the test split by balanced accuracy (mean per-answer recall):
  lookup:<feature>   the majority answer of that feature's value in train (unseen value / tie: the first answer)
  lookup:all-but-game  the same over the tuple of every feature except the game index
  logreg            multinomial logistic regression on the one-hot of every feature (numpy, L2, full batch)
The check FAILS when the best predictor beats chance (1 / number of answers) by more than MARGIN.
"""
import collections
from typing import Dict, List, Sequence, Tuple

import numpy as np

MARGIN = 0.05
STEPS, LR, L2 = 400, 0.5, 1e-3


def balanced_accuracy(truth: Sequence[str], pred: Sequence[str], answers: Sequence[str]) -> float:
    rec = []
    for a in answers:
        idx = [i for i, t in enumerate(truth) if t == a]
        if idx:
            rec.append(sum(pred[i] == a for i in idx) / len(idx))
    return sum(rec) / len(rec) if rec else 0.0


def _key(feat: Dict, names: Sequence[str]) -> Tuple:
    return tuple(feat[n] for n in names)


def lookup(train: List[Tuple[Dict, str]], test: List[Tuple[Dict, str]], names: Sequence[str],
           answers: Sequence[str]) -> List[str]:
    votes: Dict[Tuple, collections.Counter] = collections.defaultdict(collections.Counter)
    for feat, a in train:
        votes[_key(feat, names)][a] += 1

    def best(c: collections.Counter) -> str:
        top = max(c[a] for a in answers)
        return next(a for a in answers if c[a] == top)
    return [best(votes.get(_key(feat, names), collections.Counter())) for feat, _ in test]


def _onehot(rows: List[Dict], vocab: Dict[Tuple[str, str], int]) -> np.ndarray:
    x = np.zeros((len(rows), len(vocab) + 1))
    x[:, -1] = 1.0
    for i, feat in enumerate(rows):
        for k, v in feat.items():
            j = vocab.get((k, str(v)))
            if j is not None:
                x[i, j] = 1.0
    return x


def logreg(train: List[Tuple[Dict, str]], test: List[Tuple[Dict, str]], answers: Sequence[str]) -> List[str]:
    vocab: Dict[Tuple[str, str], int] = {}
    for feat, _ in train:
        for k, v in sorted(feat.items()):
            vocab.setdefault((k, str(v)), len(vocab))
    x = _onehot([f for f, _ in train], vocab)
    y = np.zeros((len(train), len(answers)))
    y[np.arange(len(train)), [list(answers).index(a) for _, a in train]] = 1.0
    w = np.zeros((x.shape[1], len(answers)))
    for _ in range(STEPS):
        z = x @ w
        p = np.exp(z - z.max(axis=1, keepdims=True))
        p /= p.sum(axis=1, keepdims=True)
        w -= LR * (x.T @ (p - y) / len(train) + L2 * w)
    scores = _onehot([f for f, _ in test], vocab) @ w
    return [answers[i] for i in scores.argmax(axis=1)]


def check(rows: Sequence[Dict], answers: Sequence[str], margin: float = MARGIN) -> Dict:
    """rows: dataset rows with "shortcut", "answer", "split". The verdict and every predictor's balanced accuracy."""
    train = [(r["shortcut"], r["answer"]) for r in rows if r["split"] == "train"]
    test = [(r["shortcut"], r["answer"]) for r in rows if r["split"] == "test"]
    if not train or not test:
        return {"pass": False, "error": "no train or no test rows"}
    names = sorted(train[0][0])
    truth = [a for _, a in test]
    scores = {}
    for n in names:
        scores["lookup:%s" % n] = balanced_accuracy(truth, lookup(train, test, [n], answers), answers)
    rest = [n for n in names if n != "game"]
    scores["lookup:all-but-game"] = balanced_accuracy(truth, lookup(train, test, rest, answers), answers)
    scores["logreg"] = balanced_accuracy(truth, logreg(train, test, answers), answers)
    chance = 1.0 / len(answers)
    best = max(scores, key=scores.get)
    return {"pass": scores[best] - chance <= margin, "chance": round(chance, 4), "margin": margin,
            "best": best, "best_score": round(scores[best], 4), "over_chance": round(scores[best] - chance, 4),
            "scores": {k: round(v, 4) for k, v in sorted(scores.items())}, "features": names,
            "train": len(train), "test": len(test)}
