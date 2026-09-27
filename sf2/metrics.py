"""Training-progress metrics that say *where* the student agrees with the label, not just how often.

Validation is held out by whole rounds (``holdout_round_ids``): frames a few ticks apart are near-duplicates, so a
random frame split puts the neighbours of every val frame in the training set and overstates accuracy.

``slice_metrics`` breaks one eval into situations: the labelled move, time into the round, how much life is left,
distance, opponent airborne, rounds where a hit is coming, and dataset; each with accuracy against the label, the
probability the model gave the labelled move, soft cross-entropy, and ``t_of_pred``: the label distribution's
probability of the model's own top move (how acceptable the move it would play is). ``train.py`` appends one entry per eval to
``runs/<x>/eval_slices.jsonl``; ``scripts/report.py`` compares runs and steps.
"""
import hashlib
import math
import re
from collections import Counter, defaultdict
from typing import Dict, List, Sequence, Set

from . import actions as A
from .config import ROUND_LIFE

FPS = 60
TIME_BINS = ((10.0, "early"), (25.0, "mid"), (math.inf, "late"))          # seconds into the round
HP_BINS = ((1 / 3, "endgame"), (2 / 3, "midgame"), (math.inf, "opening"))  # lower of the two life bars, as a fraction


def round_key(rec: Dict):
    return (rec.get("dataset", ""), rec["meta"].get("episode", rec.get("episode")), rec["meta"]["round"])


def holdout_round_ids(recs: Sequence[Dict], every: int = 10) -> Set[str]:
    """Ids of the records in every ``every``-th round (by a stable hash), so no round straddles train and val."""
    def held(key):
        return int(hashlib.sha1(repr(key).encode()).hexdigest(), 16) % every == 0

    return {r["id"] for r in recs if held(round_key(r))}


def annotate(recs: Sequence[Dict]) -> None:
    """``meta["t_round"]``: seconds since the round's first recorded frame (``frame`` counts from match start)."""
    start = {}
    for r in recs:
        k = round_key(r)
        start[k] = min(start.get(k, r["meta"]["frame"]), r["meta"]["frame"])
    for r in recs:
        r["meta"]["t_round"] = (r["meta"]["frame"] - start[round_key(r)]) / FPS


def _bin(x, bins):
    return next(name for edge, name in bins if x < edge)


def _field(text: str, key: str, default="?"):
    m = re.search(r"\b%s=(\S+)" % key, text or "")
    return m.group(1) if m else default


def situation(rec: Dict) -> Dict[str, str]:
    m, text = rec["meta"], rec.get("state_text", "")
    hp = min(m.get("my_hp", ROUND_LIFE), m.get("opp_hp", ROUND_LIFE)) / ROUND_LIFE
    rnd = m.get("round", 0)
    return {
        "by_move": A.ACTIONS[rec["label"]],
        "by_time": _bin(m.get("t_round", 0.0), TIME_BINS),
        "by_hp": _bin(hp, HP_BINS),
        "by_dist": _field(text, "dist"),
        "by_opp_air": "air" if re.search(r"\bopp=\S+ jump", text or "") or _field(text, "opp_airborne") == "1"
                      else "ground",
        "by_round": "r%d" % (rnd + 1) if rnd < 2 else "r3+",
        "by_danger": "hit_next" if m.get("hot") else "safe",
        "by_dataset": rec.get("dataset", "?"),
    }


def _softmax(z):
    mx = max(z)
    e = [math.exp(v - mx) for v in z]
    s = sum(e)
    return [v / s for v in e]


def slice_metrics(recs: Sequence[Dict], logits: Sequence[Sequence[float]], targets: Sequence[Sequence[float]]) -> Dict:
    """Per-situation accuracy / p(labelled move) / soft cross-entropy, a label->model confusion, the model's move mix."""
    acc: Dict[str, Dict[str, List]] = defaultdict(lambda: defaultdict(lambda: [0, 0.0, 0.0, 0.0, 0.0]))
    confusion: Dict[str, Counter] = defaultdict(Counter)
    pred_mix = Counter()
    for rec, z, t in zip(recs, logits, targets):
        p = _softmax([float(v) for v in z])
        pred = max(range(len(p)), key=p.__getitem__)
        label = rec["label"]
        stats = (pred == label, p[label], -sum(float(tj) * math.log(max(pj, 1e-12)) for tj, pj in zip(t, p)),
                 float(t[pred]) / max(1e-12, sum(float(tj) for tj in t)))
        cells = [("all", "all")] + list(situation(rec).items())
        for dim, val in cells:
            c = acc[dim][val]
            c[0] += 1
            for i, v in enumerate(stats):
                c[i + 1] += float(v)
        confusion[A.ACTIONS[label]][A.ACTIONS[pred]] += 1
        pred_mix[A.ACTIONS[pred]] += 1

    def cell(c):
        n = c[0]
        return {"n": n, "acc": c[1] / n, "p_label": c[2] / n, "soft_xent": c[3] / n, "t_of_pred": c[4] / n}

    out = {dim: {val: cell(c) for val, c in sorted(vals.items())} for dim, vals in acc.items()}
    out["all"] = out["all"]["all"]
    out["confusion"] = {k: dict(v) for k, v in sorted(confusion.items())}
    n = sum(pred_mix.values())
    out["pred_mix"] = {k: v / n for k, v in pred_mix.most_common()}
    return out


def summary(s: Dict) -> str:
    """One log line: accuracy per labelled move, per time bin and per life bin."""
    def part(dim, order=None):
        vals = s.get(dim, {})
        keys = order or sorted(vals, key=lambda k: -vals[k]["n"])
        return " ".join("%s %.2f" % (k, vals[k]["acc"]) for k in keys if k in vals)

    return "moves: %s | time: %s | hp: %s" % (part("by_move"), part("by_time", ["early", "mid", "late"]),
                                              part("by_hp", ["opening", "midgame", "endgame"]))
