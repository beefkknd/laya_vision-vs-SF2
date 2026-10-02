"""Scoring the four movement fine-tunes on their held-out test matches (docs/prereg_movement_finetunes.md, "Judged on
the held-out test matches"; scripts/eval_mv.py). Pure functions, any answer set.

- metrics: per-answer recall (and precision), the confusion (truth -> answer), accuracy, and BALANCED accuracy = the
  mean recall over the answers present in the truth.
- baselines: always the most common TRAINING answer (natural counts, a tie to the earlier answer) and chance (1/K);
  both score balanced accuracy 1/K when every answer is present, plain accuracy differs.
- balanced_lower_bound: the 2.5% quantile of the balanced accuracy over 1,000 cluster-bootstrap resamples of whole
  matches ((pair_name, game)), seeded.
- learned: that lower bound above BOTH the majority baseline's and chance's balanced accuracy.
- breakdown: the same metrics per character and per facing (the row's own facing label); per screen side when
  the rows were asked by side (round 2); per flight stage for the fireball rows (round 3).
- collapsed: rows and answers mapped to a coarser answer set (round 3: mv2_move's ten movements as act's three).
"""
import collections
from typing import Dict, List, Optional, Sequence

import numpy as np

RESAMPLES = 1000
SEED = 0
Q = 0.025


def _ratio(num: int, den: int) -> Optional[float]:
    return num / den if den else None


def metrics(truths: Sequence[str], preds: Sequence[str], answers: Sequence[str]) -> Dict:
    if len(truths) != len(preds):
        raise ValueError("%d truths, %d answers" % (len(truths), len(preds)))
    bad = (set(truths) | set(preds)) - set(answers)
    if bad:
        raise ValueError("not answers of this question: %s" % sorted(bad))
    conf = {a: collections.Counter() for a in answers}
    for t, p in zip(truths, preds):
        conf[t][p] += 1
    support = {a: sum(conf[a].values()) for a in answers}
    answered = collections.Counter(preds)
    recall = {a: _ratio(conf[a][a], support[a]) for a in answers}
    present = [v for v in recall.values() if v is not None]
    return {"n": len(truths), "accuracy": _ratio(sum(conf[a][a] for a in answers), len(truths)),
            "balanced_accuracy": sum(present) / len(present) if present else None, "recall": recall,
            "precision": {a: _ratio(conf[a][a], answered[a]) for a in answers}, "support": support,
            "confusion": {a: dict(conf[a]) for a in answers if conf[a]}}


def majority(train_answers: Sequence[str], answers: Sequence[str]) -> str:
    n = collections.Counter(train_answers)
    return max(answers, key=lambda a: (n[a], -list(answers).index(a)))


def balanced_lower_bound(rows: Sequence[Dict], preds: Sequence[str], answers: Sequence[str],
                         resamples: int = RESAMPLES, seed: int = SEED, q: float = Q) -> Optional[float]:
    if not rows:
        return None
    games: Dict[tuple, List[int]] = collections.defaultdict(list)
    for i, r in enumerate(rows):
        games[(r["pair_name"], r["game"])].append(i)
    keys = sorted(games)
    idx = {a: i for i, a in enumerate(answers)}
    tab = np.zeros((len(keys), len(answers), 2))
    for g, k in enumerate(keys):
        for i in games[k]:
            t = rows[i]["answer"]
            tab[g, idx[t], 0] += 1
            tab[g, idx[t], 1] += preds[i] == t
    picks = np.random.default_rng(seed).integers(0, len(keys), (resamples, len(keys)))
    sums = tab[picks].sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        rec = sums[:, :, 1] / sums[:, :, 0]
    return float(np.quantile(np.nanmean(rec, axis=1), q))


def breakdown(rows: Sequence[Dict], preds: Sequence[str], key: str, answers: Sequence[str]) -> Dict[str, Dict]:
    groups: Dict[str, List[int]] = collections.defaultdict(list)
    for i, r in enumerate(rows):
        groups[r[key]].append(i)
    out = {}
    for g, ix in sorted(groups.items()):
        m = metrics([rows[i]["answer"] for i in ix], [preds[i] for i in ix], answers)
        out[g] = {k: m[k] for k in ("n", "accuracy", "balanced_accuracy", "recall")}
    return out


def evaluate(rows: Sequence[Dict], preds: Sequence[str], answers: Sequence[str], train_answers: Sequence[str],
             resamples: int = RESAMPLES, seed: int = SEED) -> Dict:
    truths = [r["answer"] for r in rows]
    maj = majority(train_answers, answers)
    model = metrics(truths, preds, answers)
    base_maj = dict(metrics(truths, [maj] * len(rows), answers), answer=maj)
    chance = 1.0 / len(answers)
    lb = balanced_lower_bound(rows, preds, answers, resamples, seed)
    floor = max(chance, base_maj["balanced_accuracy"] or 0.0)
    res = {"n": len(rows), "matches": len({(r["pair_name"], r["game"]) for r in rows}), "answers": list(answers),
           "model": model, "balanced_lower_bound": lb, "resamples": resamples, "majority": base_maj,
           "chance": {"balanced_accuracy": chance, "accuracy": chance},
           "learned": lb is not None and lb > floor, "floor": floor,
           "by_char": breakdown(rows, preds, "char", answers), "by_facing": breakdown(rows, preds, "facing", answers)}
    if rows and all("side" in r for r in rows):          # round 2: asked by screen side
        res["by_side"] = breakdown(rows, preds, "side", answers)
    if rows and all("flight_stage" in r for r in rows):  # round 3: the fireball's flight stage
        res["by_flight_stage"] = breakdown(rows, preds, "flight_stage", answers)
    return res


def collapsed(rows: Sequence[Dict], preds: Sequence[str], mapping: Dict[str, str]):
    """(rows with answer mapped and the original kept as movement10, preds mapped): e.g. round 2's ten movements
    scored as round 3's act answers (moving / attack / special)."""
    return ([dict(r, answer=mapping[r["answer"]], movement10=r["answer"]) for r in rows],
            [mapping[p] for p in preds])
