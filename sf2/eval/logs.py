"""Which logs are play data. Test runs never feed a coach or System 2 (they replay the same savestates with the same
seed as the A/B test, so learning from them would leak the test into the lesson):

    rollouts/ab/, rollouts/notebook/,     A/B, notebook and Qwen move tests
      rollouts/qwen_moves/
    a run marked "fresh" in run.json      learn_loop --fresh (demo sessions with their own memory_runs/<name>)
    a run marked "test" in run.json       any other test run
    a run.json that cannot be read        not trusted as play data

A log dir holds actions.jsonl; each action loaded here is tagged ``log`` = its dir relative to the root.
"""
import glob
import json
import os
from typing import Dict, Iterable, List, Tuple

from ..data.dataset import read

ROOT = "rollouts"
TEST_DIRS = ("ab", "notebook", "qwen_moves", "qwen_lessons")
RUN_FILE = "run.json"


def mark_run(out: str, **meta) -> None:
    """Record how a run was made (memory root, fresh name, seed), written whole or not at all."""
    path = os.path.join(out, RUN_FILE)
    with open(path + ".tmp", "w") as f:
        json.dump(meta, f)
    os.replace(path + ".tmp", path)


def is_test(d: str, root: str = ROOT) -> bool:
    if os.path.relpath(d, root).split(os.sep)[0] in TEST_DIRS:
        return True
    path = os.path.join(d, RUN_FILE)
    if not os.path.exists(path):
        return False
    try:
        with open(path) as f:
            meta = json.load(f)
    except (OSError, ValueError):
        return True
    return not isinstance(meta, dict) or bool(meta.get("fresh")) or bool(meta.get("test"))


def play_dirs(root: str = ROOT) -> List[str]:
    """Every log dir under ``root`` with actions.jsonl that is play data, sorted."""
    out = []
    for d, _, files in os.walk(root):
        if "actions.jsonl" in files and not is_test(d, root):
            out.append(d)
    return sorted(out)


def load_actions(d: str, root: str = ROOT) -> List[Dict]:
    tag = os.path.relpath(d, root)
    return [dict(a, log=tag) for a in read(os.path.join(d, "actions.jsonl"))]


def load_rounds(d: str) -> List[Dict]:
    """Per-round summaries: rounds.jsonl, or games.jsonl in the older logs (games_v*: one line per round)."""
    path = os.path.join(d, "rounds.jsonl")
    return read(path if os.path.exists(path) else os.path.join(d, "games.jsonl"), missing_ok=True)


def sources(rows: Iterable[Dict]) -> List[str]:
    return sorted({a.get("log", "?") for a in rows})


def history(me: str, root: str = ROOT) -> Dict[str, Tuple[List[Dict], List[Dict]]]:
    """Every earlier play log of ``me`` that System 2 learns from (the games_v* sweeps and the learning sessions; test
    runs excluded), by opponent: {opp: (actions, rounds)}, each action tagged with its log dir."""
    by_opp: Dict[str, Tuple[List[Dict], List[Dict]]] = {}
    dirs = sorted(glob.glob(os.path.join(root, "games_v*", me))) + sorted(glob.glob(os.path.join(root, "learn", me, "*")))
    for d in dirs:
        if not os.path.exists(os.path.join(d, "actions.jsonl")) or is_test(d, root):
            continue
        acts, rounds = load_actions(d, root), load_rounds(d)
        opps = {a["opp"] for a in acts}
        for o in opps:
            a0, r0 = by_opp.setdefault(o, ([], []))
            a0 += [a for a in acts if a["opp"] == o]
            r0 += [r for r in rounds if r.get("opp", next(iter(opps))) == o]
    return by_opp
