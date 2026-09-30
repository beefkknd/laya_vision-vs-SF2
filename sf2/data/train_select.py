"""How scripts/train.py keeps its best checkpoint and stops early, and what it logs per eval.

``select="acc"`` (the default, the first runs' rule): keep and stop on pooled validation accuracy (higher is better).
``select="nll"`` (opt-in, the value fine-tune's second run, docs/reviews/2026-09-30_dr_fable_lv_value.md B.1): on the
validation NLL over all rows (lower is better); accuracy saturates at "always even" on the value rows while NLL keeps
falling. Every eval also records, per dataset (character), the value (score-question) cross-entropy next to its
prior's (laya.vlm_train.metrics_from's ``xent`` / ``prior_xent``): a value head equal to the class prior shows there.
"""
import math
import os
from typing import Callable, Dict, Optional, Tuple

SELECTS = {"acc": (max, -1.0), "nll": (min, math.inf)}


class EarlyStop(Exception):
    pass


def start_best(select: str) -> Dict:
    if select not in SELECTS:
        raise ValueError("select must be one of %s, got %r" % (sorted(SELECTS), select))
    return {select: SELECTS[select][1], "step": None, "bad": 0}


def metric(m: Dict, select: str) -> float:
    return float(m["all"][select])


def update_best(best: Dict, m: Dict, step: int, select: str) -> Tuple[Dict, bool]:
    """A new best record (never the old one mutated) and whether this eval improved on it (strictly)."""
    value = metric(m, select)
    better = value > best[select] if SELECTS[select][0] is max else value < best[select]
    if better:
        return {select: value, "step": step, "bad": 0}, True
    return dict(best, bad=best["bad"] + 1), False


def value_xent(m: Dict) -> Dict[str, Dict]:
    """Per dataset (not "all"), the value questions' cross-entropy and its prior's, where metrics_from gives them."""
    return {k: {"n_score": v.get("n_score"), "xent": v["xent"], "prior_xent": v.get("prior_xent")}
            for k, v in m.items() if k != "all" and "xent" in v}


def format_value_xent(vx: Dict[str, Dict]) -> str:
    def one(k, v):
        prior = "%.3f" % v["prior_xent"] if v["prior_xent"] is not None else "-"
        return "%s %.3f (prior %s)" % (k, v["xent"], prior)
    return "value xent: " + ", ".join(one(k, v) for k, v in sorted(vx.items()))


def out_problem(out: str, guard: bool) -> Optional[str]:
    """--resume-guard: a run never starts over an existing --out."""
    if guard and os.path.exists(out):
        return "%s exists: refusing to overwrite a run (pick a new --out, or --no-resume-guard)" % out
    return None


class Selection:
    """train.py's eval_fn: ``evaluate(step)`` -> metrics_from's dict; ``save()`` writes <out>/best. Keeps ``best``
    and ``hist`` (the train_log.json entries); raises EarlyStop after ``patience`` evals without improvement."""

    def __init__(self, evaluate: Callable[[int], Dict], save: Callable[[], None], select: str, patience: int):
        self.evaluate, self.save, self.select, self.patience = evaluate, save, select, patience
        self.best, self.hist = start_best(select), []

    def __call__(self, step: int) -> bool:
        m = self.evaluate(step)
        vx = value_xent(m)
        self.hist = self.hist + [{"step": step, **m, "value_xent": vx}]
        if vx:
            print("  " + format_value_xent(vx), flush=True)
        self.best, improved = update_best(self.best, m, step, self.select)
        if improved:
            self.save()
        elif self.best["bad"] >= self.patience:
            raise EarlyStop()
        return True
