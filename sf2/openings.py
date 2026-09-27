"""Saved opening schedules, and the paired comparison they make possible (docs/TWO_SYSTEM_PLAN.md §2.4, §2.5).

An *opening* is how many idle frames pass after the fight-start savestate loads before the first decision; it
decides the CPU's fight. A schedule is a saved list of distinct openings. Every arm plays the same schedule, so a
match is identified by its opening, whichever worker played it, and two arms are compared opening by opening:

    python scripts/parallel.py --workers 4 play_student ... --openings openings/dev.txt
    python scripts/paired.py rollouts/<control> rollouts/<arm>

``dev`` openings are for writing and debugging rules; ``eval`` openings are only for verdicts.
"""
import math
import os
import random
from collections import defaultdict
from typing import Dict, List, Sequence


def make_schedules(seed: int, n_dev: int, n_eval: int, max_idle: int):
    """Two disjoint random schedules of distinct idle counts in 1..max_idle."""
    picks = random.Random(seed).sample(range(1, max_idle + 1), n_dev + n_eval)
    return picks[:n_dev], picks[n_dev:]


def save(path: str, sched: Sequence[int]) -> None:
    with open(path, "w") as f:
        f.write("\n".join(map(str, sched)) + "\n")


def load(path: str) -> List[int]:
    with open(path) as f:
        return parse(",".join(line.strip() for line in f if line.strip() and not line.startswith("#")))


def parse(text: str) -> List[int]:
    """``"5,17,3"`` or a schedule file's path -> [5, 17, 3]. Distinct whole numbers >= 0."""
    if text and os.path.exists(text):
        return load(text)
    try:
        sched = [int(x) for x in text.split(",") if x.strip()]
    except ValueError:
        raise ValueError("an opening schedule is idle-frame counts separated by commas, got %r" % text) from None
    if not sched or any(x < 0 for x in sched) or len(set(sched)) != len(sched):
        raise ValueError("an opening schedule needs distinct idle-frame counts >= 0, got %r" % text)
    return sched


def split(sched: Sequence[int], workers: int) -> List[List[int]]:
    """Contiguous chunks, one per worker, sizes differing by at most 1; no empty chunk."""
    n = min(workers, len(sched))
    out, start = [], 0
    for i in range(n):
        size = len(sched) // n + (1 if i < len(sched) % n else 0)
        out.append(list(sched[start:start + size]))
        start += size
    return out


def apply(env, sched: Sequence[int]) -> None:
    """Make ``env.reset()`` idle ``sched[k]`` frames before match k, and record it as ``env.opening``."""
    env.jitter = 0  # the schedule replaces the worker jitter
    reset = env.reset

    def reset_with_opening():
        reset()
        n = sched[env.episode]  # IndexError past the end: a run plays its schedule, no more
        if n:
            env.run_frames([[]] * n)
        env.opening = n
        env.frame_no = 0
        return env.frame

    env.reset = reset_with_opening


# --- the paired comparison -----------------------------------------------------------------------------------------

_T95 = {1: 12.71, 2: 4.30, 3: 3.18, 4: 2.78, 5: 2.57, 6: 2.45, 7: 2.36, 8: 2.31, 9: 2.26, 10: 2.23, 12: 2.18,
        15: 2.13, 20: 2.09, 25: 2.06, 30: 2.04, 40: 2.02, 60: 2.00}


def _t95(df: int) -> float:
    return _T95[max(k for k in _T95 if k <= df)] if df < 120 else 1.98


def per_opening(rounds: Sequence[Dict]) -> Dict[int, float]:
    """Net damage per round (dealt - taken) for each opening's match."""
    nets = defaultdict(list)
    for r in rounds:
        if r.get("opening") is None:
            raise ValueError("every round needs its opening (play with --openings)")
        nets[r["opening"]].append(r["dmg_for"] - r["dmg_against"])
    return {k: sum(v) / len(v) for k, v in nets.items()}


def paired(control: Sequence[Dict], arm: Sequence[Dict]) -> Dict:
    """Arm minus control, opening by opening: mean difference and its 95% t interval."""
    a, b = per_opening(control), per_opening(arm)
    common = sorted(set(a) & set(b))
    diffs = [b[k] - a[k] for k in common]
    n = len(diffs)
    mean = sum(diffs) / n if n else float("nan")
    sd = math.sqrt(sum((d - mean) ** 2 for d in diffs) / (n - 1)) if n > 1 else float("nan")
    half = _t95(n - 1) * sd / math.sqrt(n) if n > 1 else float("nan")
    return {"n": n, "mean_diff": mean, "sd_diff": sd, "ci95": (mean - half, mean + half),
            "control_mean": sum(a[k] for k in common) / n if n else float("nan"),
            "arm_mean": sum(b[k] for k in common) / n if n else float("nan"),
            "unmatched": sorted(set(a) ^ set(b))}
