"""Scoring the movement question (docs/prereg_movement.md "Judged against RAM"), pure functions used by
scripts/eval_movement.py.

- metrics: per answer recall (of the decisions whose truth it is) and precision (of those it was answered), the
  confusion (truth -> answer), accuracy, and BALANCED accuracy = the mean recall over the answers present in the truth.
- majority: the most common answer of the TRAINING split (natural counts; a tie goes to the earlier answer), answered
  always; its balanced accuracy is 1 / (answers present), 1/8 when all are.
- before: round 1's eye (runs/u_eye/best) asked its own question 2, "What is he doing?" (neutral / attacking /
  recovering after a miss / blocking / being hit), mapped onto the eight answers by BEFORE_MAP: neutral -> standing
  or walking (a hit if the truth is standing, walking toward me or walking away; never for crouching or jumping,
  which round 1 had no word for), recovering after a miss -> attacking, the others to themselves. Its recall per
  truth answer is the share of hits; accuracy and balanced accuracy as above; the confusion is truth -> its raw word.
- verdict ("trainable", per test file): balanced accuracy CLEARLY above chance (1/8) and above the majority
  baseline's balanced accuracy, read as: the 95% cluster-bootstrap lower bound of the balanced accuracy (decisions
  resampled by whole game, BOOT_RESAMPLES, seeded) above both; and attacking recall >= 0.5. Overall: both
  test_real and test_heldout_guile trainable.
"""
import collections
from typing import Dict, List, Optional, Sequence

import numpy as np

from .movement import ANSWERS

CHANCE = 1.0 / len(ANSWERS)
ATTACK_RECALL_MIN = 0.5
BOOT_RESAMPLES = 1000
BOOT_SEED = 0
BOOT_Q = 0.025
VERDICT_FILES = ("test_real", "test_heldout_guile")
STANDING_OR_WALKING = ("standing", "walking toward me", "walking away")
BEFORE_MAP: Dict[str, tuple] = {"neutral": STANDING_OR_WALKING, "attacking": ("attacking",),
                                "recovering after a miss": ("attacking",), "blocking": ("blocking",),
                                "being hit": ("being hit",)}


def predicted(probs: Dict[str, float], order: Sequence[str] = ANSWERS) -> str:
    """The most likely answer; a tie goes to the first in ``order``."""
    return max(order, key=lambda a: (probs.get(a, 0.0), -list(order).index(a)))


def _ratio(num: int, den: int) -> Optional[float]:
    return num / den if den else None


def _balanced(recall: Dict[str, Optional[float]]) -> Optional[float]:
    vals = [v for v in recall.values() if v is not None]
    return sum(vals) / len(vals) if vals else None


def metrics(truths: Sequence[str], preds: Sequence[str]) -> Dict:
    if len(truths) != len(preds):
        raise ValueError("%d truths, %d answers" % (len(truths), len(preds)))
    conf: Dict[str, collections.Counter] = {a: collections.Counter() for a in ANSWERS}
    for t, p in zip(truths, preds):
        conf[t][p] += 1
    support = {a: sum(conf[a].values()) for a in ANSWERS}
    answered = collections.Counter(preds)
    recall = {a: _ratio(conf[a][a], support[a]) for a in ANSWERS}
    precision = {a: _ratio(conf[a][a], answered[a]) for a in ANSWERS}
    hits = sum(conf[a][a] for a in ANSWERS)
    return {"n": len(truths), "accuracy": _ratio(hits, len(truths)), "balanced_accuracy": _balanced(recall),
            "recall": recall, "precision": precision, "support": support,
            "confusion": {a: dict(conf[a]) for a in ANSWERS if conf[a]}}


def majority(train_counts: Dict[str, int]) -> str:
    return max(ANSWERS, key=lambda a: (train_counts.get(a, 0), -ANSWERS.index(a)))


def before_hit(truth: str, word: str) -> bool:
    return truth in BEFORE_MAP[word]


def before_metrics(truths: Sequence[str], words: Sequence[str]) -> Dict:
    by: Dict[str, List[bool]] = {a: [] for a in ANSWERS}
    conf: Dict[str, collections.Counter] = {a: collections.Counter() for a in ANSWERS}
    for t, w in zip(truths, words):
        by[t].append(before_hit(t, w))
        conf[t][w] += 1
    recall = {a: _ratio(sum(v), len(v)) for a, v in by.items()}
    hits = sum(sum(v) for v in by.values())
    return {"n": len(truths), "accuracy": _ratio(hits, len(truths)), "balanced_accuracy": _balanced(recall),
            "recall": recall, "mapping": {k: list(v) for k, v in BEFORE_MAP.items()},
            "confusion": {a: dict(conf[a]) for a in ANSWERS if conf[a]}}


def breakdown(rows: Sequence[Dict], preds: Sequence[str], key: str) -> Dict[str, Dict]:
    """metrics per value of rows[i][key] (character or opponent), recall per answer kept, confusion dropped."""
    groups: Dict[str, List[int]] = collections.defaultdict(list)
    for i, r in enumerate(rows):
        groups[r[key]].append(i)
    out = {}
    for g, idx in sorted(groups.items()):
        m = metrics([rows[i]["answer"] for i in idx], [preds[i] for i in idx])
        out[g] = {k: m[k] for k in ("n", "accuracy", "balanced_accuracy", "recall", "support")}
    return out


def balanced_lower_bound(rows: Sequence[Dict], preds: Sequence[str], resamples: int = BOOT_RESAMPLES,
                         seed: int = BOOT_SEED, q: float = BOOT_Q) -> Optional[float]:
    """The ``q`` quantile of the balanced accuracy over a cluster bootstrap: whole games ((char, sub, game))
    resampled with replacement."""
    if not rows:
        return None
    games: Dict[tuple, List[int]] = collections.defaultdict(list)
    for i, r in enumerate(rows):
        games[(r["char"], r["sub"], r["game"])].append(i)
    keys = sorted(games)
    idx = {a: i for i, a in enumerate(ANSWERS)}
    # per game: (truth, hit) counts as a len(ANSWERS) x 2 table, so a resample is a sum of tables
    tab = np.zeros((len(keys), len(ANSWERS), 2))
    for g, k in enumerate(keys):
        for i in games[k]:
            t = rows[i]["answer"]
            tab[g, idx[t], 0] += 1
            tab[g, idx[t], 1] += preds[i] == t
    rng = np.random.default_rng(seed)
    picks = rng.integers(0, len(keys), (resamples, len(keys)))
    sums = tab[picks].sum(axis=1)                    # resamples x answers x 2
    with np.errstate(invalid="ignore", divide="ignore"):
        rec = sums[:, :, 1] / sums[:, :, 0]
    bal = np.nanmean(rec, axis=1)
    return float(np.quantile(bal, q))


def file_verdict(res: Dict) -> Dict:
    bal = res["model"]["balanced_accuracy"]
    lb = res["balanced_lower_bound"]
    att = res["model"]["recall"].get("attacking")
    maj = res["majority"]["balanced_accuracy"]
    floor = max(CHANCE, maj if maj is not None else CHANCE)
    checks = {"balanced_lb_above_chance_and_majority": lb is not None and lb > floor,
              "attacking_recall_ge_0.5": att is not None and att >= ATTACK_RECALL_MIN}
    return {"trainable": all(checks.values()), "checks": checks, "balanced_accuracy": bal, "balanced_lb": lb,
            "floor": floor, "attacking_recall": att}


def verdict(files: Dict[str, Dict]) -> Dict:
    per = {f: file_verdict(files[f]) for f in VERDICT_FILES if f in files}
    return {"trainable": len(per) == len(VERDICT_FILES) and all(v["trainable"] for v in per.values()),
            "files": per}
