"""A/B statistics (scripts/ab_memory.py). Score per round = hit points (dealt - taken). Each arm is paired round by
round with "none" (same savestate, same seed: the same start delays). Rounds of one opponent are not independent
evidence about advice in general, so pooling treats the OPPONENT as the unit: the pooled mean is the mean of the
per-opponent means, its 95% interval a two-level bootstrap (resample opponents, then rounds within each).

A verdict needs every pair to line up and every job to have finished ("NO VERDICT" otherwise), and at least MIN_OPPS
opponents with MIN_ROUNDS paired rounds each ("TOO FEW" otherwise: a smoke run is not evidence).
"""
import os
import random
from typing import Dict, List, Sequence, Tuple

REPS = 4000
BOOT_SEED = 0
MIN_OPPS = 2
MIN_ROUNDS = 10


def hp(r: Dict) -> int:
    return r["dealt"] - r["taken"]


def paired(arm: Sequence[Dict], base: Sequence[Dict]) -> List[int]:
    """Round-by-round hit-point difference, arm - base. Unequal lengths (a crashed or cut run) are an error, never
    silently truncated."""
    if len(arm) != len(base):
        raise ValueError("%d rounds vs %d in the control: pairs do not line up" % (len(arm), len(base)))
    return [hp(a) - hp(b) for a, b in zip(arm, base)]


def ci(d: Sequence[float]) -> Tuple[float, float, float]:
    """(mean, lo, hi): a normal 95% interval over rounds; nan bounds with fewer than 2 rounds."""
    n = len(d)
    if n < 2:
        return (sum(d) / max(1, n), float("nan"), float("nan"))
    m = sum(d) / n
    sd = (sum((v - m) ** 2 for v in d) / (n - 1)) ** 0.5
    return m, m - 1.96 * sd / n ** 0.5, m + 1.96 * sd / n ** 0.5


def slope(ys: Sequence[float]) -> float:
    """Least-squares change per round (0 with fewer than 2 rounds)."""
    n = len(ys)
    if n < 2:
        return 0.0
    mx, my = (n - 1) / 2, sum(ys) / n
    return sum((i - mx) * (y - my) for i, y in enumerate(ys)) / sum((i - mx) ** 2 for i in range(n))


def verdict(lo: float, hi: float) -> str:
    return "HELPS" if lo > 0 else "HURTS" if hi < 0 else "NOT SHOWN"


def pooled(by_opp: Dict[str, Sequence[float]], reps: int = REPS, seed: int = BOOT_SEED) -> Dict:
    """Pooled over opponents, the opponent as the unit (see the module doc)."""
    opps = [o for o, d in sorted(by_opp.items()) if d]
    means = [sum(by_opp[o]) / len(by_opp[o]) for o in opps]
    if not opps:
        return {"opponents": 0, "paired_rounds": 0, "verdict": "NO VERDICT", "why": "no paired rounds"}
    rng, boot = random.Random(seed), []
    for _ in range(reps):
        pick = [by_opp[rng.choice(opps)] for _ in opps]
        boot.append(sum(sum(rng.choice(d) for _ in d) / len(d) for d in pick) / len(pick))
    boot.sort()
    lo, hi = boot[int(0.025 * reps)], boot[int(0.975 * reps) - 1]
    out = {"opponents": len(opps), "paired_rounds": sum(len(by_opp[o]) for o in opps),
           "mean": sum(means) / len(means), "ci95": [lo, hi], "verdict": verdict(lo, hi)}
    if len(opps) < MIN_OPPS or min(len(by_opp[o]) for o in opps) < MIN_ROUNDS:
        out.update(verdict="TOO FEW", why="a verdict needs %d+ opponents with %d+ paired rounds each" % (
            MIN_OPPS, MIN_ROUNDS))
    return out


def summarize(data: Dict[str, Dict[str, List[Dict]]], arms: Sequence[str],
              failed: Sequence[Tuple[str, str]] = ()) -> Dict:
    """``data``: {opp: {arm: rounds}}. Per opponent and arm: rounds, won, hit points per round, the paired difference
    to "none" with its interval; pooled verdicts per arm, and qwen - code_short when both ran."""
    out = {"per_opp": {}, "pooled": {}, "failed": [list(k) for k in failed]}
    diffs = {a: {} for a in arms if a != "none"}
    broken = {a: [] for a in arms}
    for opp, got in data.items():
        row = {}
        for arm in arms:
            rs = got.get(arm) or []
            if not rs:
                continue
            row[arm] = {"rounds": len(rs), "won": sum(r["result"] == "win" for r in rs),
                        "hp": sum(map(hp, rs)) / len(rs)}
            if arm == "none":
                continue
            try:
                d = paired(rs, got.get("none") or [])
            except ValueError as e:
                row[arm]["error"] = str(e)
                broken[arm].append(opp)
                continue
            row[arm]["vs_none"] = ci(d)
            diffs[arm][opp] = d
        out["per_opp"][opp] = row
    for arm, by_opp in diffs.items():
        p = dict(pooled(by_opp), missing=sorted(o for o in data if o not in by_opp and o not in broken[arm]))
        why = ["%s %s" % (o, a) for (o, a) in failed if a in (arm, "none")] + broken[arm]
        if why:
            p = dict(p, verdict="NO VERDICT", why="unfinished or unpaired: " + ", ".join(sorted(set(why))))
        out["pooled"][arm] = p
    if "qwen" in diffs and "code_short" in diffs:           # proof 3, static part: qwen vs code on the same rounds
        by_opp = {}
        for opp, got in data.items():
            try:
                by_opp[opp] = paired(got.get("qwen") or [], got.get("code_short") or [])
            except ValueError:
                pass
        p = pooled(by_opp)
        if p["verdict"] not in ("NO VERDICT", "TOO FEW"):
            p["verdict"] = "WORSE" if p["ci95"][1] < 0 else "NOT WORSE"
        if out["pooled"]["qwen"]["verdict"] == "NO VERDICT" or out["pooled"]["code_short"]["verdict"] == "NO VERDICT":
            p = dict(p, verdict="NO VERDICT")
        out["pooled"]["qwen_minus_code_short"] = p
    return out


def load_runs(roots: Sequence[str], opps: Sequence[str], arms: Sequence[str],
              strict: bool = True) -> Dict[str, Dict[str, List[Dict]]]:
    """{opp: {arm: rounds}} from one or more A/B run folders (<root>/<opp>_<arm>/rounds.jsonl), each run's rounds
    appended in the same order for every arm so the pairs still line up. An arm that played a different number of
    rounds than "none" in any run is an error (a crashed or cut job) when ``strict``; with one run, ``strict=False``
    leaves it to ``summarize``, which marks that arm unpaired (NO VERDICT)."""
    from ..data.dataset import read

    data: Dict[str, Dict[str, List[Dict]]] = {o: {a: [] for a in arms} for o in opps}
    for root in roots:
        for o in opps:
            got = {a: read(os.path.join(root, "%s_%s" % (o, a), "rounds.jsonl"), missing_ok=True) for a in arms}
            for a in arms:
                if strict and got[a] and got.get("none") is not None and len(got[a]) != len(got["none"]):
                    raise ValueError("%s: %s %s played %d rounds, none %d" % (root, o, a, len(got[a]),
                                                                            len(got["none"])))
                data[o][a] += got[a]
    return data
