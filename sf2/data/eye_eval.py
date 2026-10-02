"""Scoring the eye fine-tunes (docs/eye_questions_v1.md, "Training pre-registration"; scripts/eval_eye.py). Pure.

- summary: balanced accuracy, its 2.5% lower bound (1,000 resamples by whole match, sf2.data.mv_eval), per-answer
  recall, vs always the most common TRAINING answer and vs chance; "learned" = lower bound above both; per side and
  per character when the question names a side (q3, q4); q1 also per hard tag of its "no" rows and per frame of its
  "yes" rows (drawn in n-4 only / n only / both).
- weighted_accuracy: the recall per answer weighted to real play (REAL_PLAY: the prereg's shares from the old U
  collection's RAM; q3's "attacking" 28.5% split between attack and special in the pool's natural ratio, the
  dataset's build.json "candidates"; q5 has no real-play share and gets none).
- keep_combined: the mechanical keep rule - eye_all is kept only if every question's balanced accuracy is within
  MARGIN of its separate run on the same test set.
"""
import collections
from typing import Dict, List, Optional, Sequence

from . import mv_eval as E

MARGIN = 0.02
REAL_PLAY = {"q1": {"yes": 0.035, "no": 0.965}, "q4": {"air": 0.28, "ground": 0.72}}
Q3_MOVING, Q3_ATTACKING = 0.715, 0.285


def real_play_weights(question: str, candidates: Optional[Dict[str, int]] = None) -> Optional[Dict[str, float]]:
    if question == "q3":
        if not candidates or not (candidates.get("attack", 0) + candidates.get("special", 0)):
            raise ValueError("q3 needs the pool's attack / special candidate counts")
        a = candidates["attack"] / (candidates["attack"] + candidates["special"])
        return {"moving": Q3_MOVING, "attack": Q3_ATTACKING * a, "special": Q3_ATTACKING * (1 - a)}
    return REAL_PLAY.get(question)


def weighted_accuracy(recall: Dict[str, Optional[float]], weights: Optional[Dict[str, float]]) -> Optional[float]:
    if not weights:
        return None
    if abs(sum(weights.values()) - 1.0) > 1e-9:
        raise ValueError("real-play weights do not sum to 1: %s" % weights)
    if any(recall.get(a) is None for a in weights):
        return None
    return sum(w * recall[a] for a, w in weights.items())


def _fire_frames(r: Dict) -> str:
    f = set(r.get("fire_frames") or [])
    return "both" if f == {"n-4", "n"} else "n-4 only" if f == {"n-4"} else "n only" if f == {"n"} else "none"


def _groups(rows: Sequence[Dict], preds: Sequence[str], answers: Sequence[str], keyf) -> Dict[str, Dict]:
    ix: Dict[str, List[int]] = collections.defaultdict(list)
    for i, r in enumerate(rows):
        for k in keyf(r):
            ix[k].append(i)
    out = {}
    for k, ii in sorted(ix.items()):
        m = E.metrics([rows[i]["answer"] for i in ii], [preds[i] for i in ii], answers)
        out[k] = {"n": m["n"], "accuracy": m["accuracy"], "balanced_accuracy": m["balanced_accuracy"],
                  "recall": m["recall"]}
    return out


def summary(rows: Sequence[Dict], preds: Sequence[str], answers: Sequence[str], train_answers: Sequence[str],
            question: str, candidates: Optional[Dict[str, int]] = None, resamples: int = E.RESAMPLES,
            seed: int = E.SEED) -> Dict:
    truths = [r["answer"] for r in rows]
    model = E.metrics(truths, preds, answers)
    maj = E.majority(train_answers, answers)
    base_maj = dict(E.metrics(truths, [maj] * len(rows), answers), answer=maj)
    chance = 1.0 / len(answers)
    lb = E.balanced_lower_bound(rows, preds, answers, resamples, seed)
    floor = max(chance, base_maj["balanced_accuracy"] or 0.0)
    weights = real_play_weights(question, candidates)
    res = {"question": question, "n": len(rows), "matches": len({(r["pair_name"], r["game"]) for r in rows}),
           "answers": list(answers), "model": model, "balanced_lower_bound": lb, "majority": base_maj,
           "chance": chance, "floor": floor, "learned": lb is not None and lb > floor,
           "real_play_weights": weights, "weighted_accuracy": weighted_accuracy(model["recall"], weights)}
    if rows and all("side" in r for r in rows):
        res["by_side"] = _groups(rows, preds, answers, lambda r: [r["side"]])
        res["by_char"] = _groups(rows, preds, answers, lambda r: [r["char"]])
    if question == "q1":
        res["by_no_tag"] = _groups(rows, preds, answers,
                                   lambda r: (r.get("hard") or ["plain"]) if r["answer"] == "no" else [])
        res["by_yes_frames"] = _groups(rows, preds, answers,
                                       lambda r: [_fire_frames(r)] if r["answer"] == "yes" else [])
    return res


def keep_combined(separate: Dict[str, float], combined: Dict[str, float], margin: float = MARGIN) -> Dict:
    """Keep eye_all iff for every question: combined balanced accuracy >= separate - margin (same test set)."""
    if set(separate) != set(combined) or not separate:
        raise ValueError("questions differ: separate %s, combined %s" % (sorted(separate), sorted(combined)))
    per = {q: {"separate": separate[q], "combined": combined[q], "diff": combined[q] - separate[q],
               "within": combined[q] >= separate[q] - margin - 1e-12} for q in sorted(separate)}
    return {"margin": margin, "keep": all(v["within"] for v in per.values()), "per_question": per,
            "separate_adapters_for": [q for q, v in per.items() if not v["within"]]}
