"""Which logs are play data. Test runs never feed a coach or System 2 (they replay the same savestates with the same
seed as the A/B test, so learning from them would leak the test into the lesson):

    rollouts/ab/, rollouts/notebook/      A/B and notebook tests
    a run marked "fresh" in run.json      learn_loop --fresh (demo sessions with their own memory_runs/<name>)
    a run.json that cannot be read        not trusted as play data

A log dir holds actions.jsonl; each action loaded here is tagged ``log`` = its dir relative to the root.
"""
import json
import os
from typing import Dict, Iterable, List

from ..dataset import read

ROOT = "rollouts"
TEST_DIRS = ("ab", "notebook")
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
    return not isinstance(meta, dict) or bool(meta.get("fresh"))


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
