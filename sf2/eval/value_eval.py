"""Offline gates of the laya-vision value fine-tune (docs/prereg_lv_value.md): calibration of the predicted expected
net against the real net, and how often a move reaches the top 3 of a ranking."""
from typing import Callable, Dict, List, Mapping, Sequence, Tuple


def calibration(pairs: Sequence[Tuple[float, float]], k: int = 5) -> List[Dict]:
    """(predicted, real) pairs -> ``k`` equal-count bins by predicted value: mean predicted, mean real, n."""
    if len(pairs) < k * 5:
        raise ValueError("%d pairs are too few for %d bins" % (len(pairs), k))
    ordered = sorted(pairs, key=lambda pr: pr[0])
    bins = [ordered[i * len(ordered) // k:(i + 1) * len(ordered) // k] for i in range(k)]
    return [{"mean_pred": sum(p for p, _ in b) / len(b), "mean_real": sum(r for _, r in b) / len(b), "n": len(b)}
            for b in bins]


def calibration_gate(rows: List[Dict], min_spread: float) -> Dict:
    """Monotone mean real net over the bins, and the top bin above the bottom one by at least ``min_spread``."""
    real = [r["mean_real"] for r in rows]
    monotone = all(a <= b for a, b in zip(real, real[1:]))
    spread = real[-1] - real[0]
    return {"monotone": monotone, "spread": spread, "pass": monotone and spread >= min_spread}


def _top3(values: Mapping[str, float]) -> List[str]:
    return [m for m, _ in sorted(values.items(), key=lambda kv: (-kv[1], kv[0]))[:3]]


def top3_share(decisions: Sequence[Dict], move: str, where: Callable[[Dict], bool], key: str = "values") -> Dict:
    """Share of the decisions matching ``where`` whose ranking ``key`` puts ``move`` in its top 3 (ties broken by
    name, so a tie never favours the move under test unless its name sorts first)."""
    ds = [d for d in decisions if where(d)]
    if not ds:
        return {"share": None, "n": 0}
    return {"share": sum(move in _top3(d[key]) for d in ds) / len(ds), "n": len(ds)}
