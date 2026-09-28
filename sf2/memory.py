"""Two memories for System 1, both written by System 2 (Qwen, sf2/system2.py) from the game logs.

    memory/playbook/<me>.json          long term: what this character's own moves do, by range (kept across sessions)
    memory/short/<me>_vs_<opp>.json    short term: lessons against this opponent (starts empty; a new opponent gets
                                       a new file, so the memory resets on opponent change)

Each file: {"me", "opp" (short only), "source" (the logs it was built from), "lessons": [lesson, ...]}; a lesson is
{"text", "kind", "action", "range", "evidence": {"tries", "count", "rate", "refs": ["g<game>f<frame>", ...]}}:
``count`` of the ``tries`` are what the lesson claims (landed, got punished, the habit), ``refs`` point at them.

The short memory reaches System 1 through text laya (sf2/advisor.py, sf2/advice.py), which picks the move following
its lessons. Without an advisor, ``prompt_text`` appends the lesson texts to laya-vision's RAM note instead (the old
path: laya-vision was never trained to read them).
"""
import json
import os
from typing import Dict, List, Optional

ROOT = "memory"
KINDS = ("use_more", "avoid", "opponent_habit", "counter")
MAX_PROMPT_LESSONS = 5


def short_path(me: str, opp: str, root: str = ROOT) -> str:
    return os.path.join(root, "short", "%s_vs_%s.json" % (me, opp))


def playbook_path(me: str, root: str = ROOT) -> str:
    return os.path.join(root, "playbook", "%s.json" % me)


def check(mem: Dict, actions: List[str]) -> List[str]:
    """Problems with a memory file (empty = OK): the shape above, known kinds, real moves, evidence that adds up."""
    problems = []
    for i, les in enumerate(mem.get("lessons", [])):
        where = "lesson %d" % i
        if not isinstance(les.get("text"), str) or not les["text"]:
            problems.append("%s: no text" % where)
        if les.get("kind") not in KINDS:
            problems.append("%s: kind %r not in %s" % (where, les.get("kind"), KINDS))
        if les.get("action") is not None and les["action"] not in actions:
            problems.append("%s: %r is not one of this character's actions" % (where, les["action"]))
        ev = les.get("evidence") or {}
        if not (isinstance(ev.get("tries"), int) and ev["tries"] > 0 and 0 <= ev.get("count", -1) <= ev["tries"]
                and (ev.get("refs") or ev["count"] == 0)):
            problems.append("%s: evidence missing or inconsistent: %s" % (where, ev))
    return problems


def load(path: str, actions: List[str]) -> Optional[Dict]:
    """The memory at ``path`` (None if there is none yet); a malformed file stops the run."""
    if not os.path.exists(path):
        return None
    with open(path) as f:
        mem = json.load(f)
    problems = check(mem, actions)
    if problems:
        raise SystemExit("%s is malformed:\n  %s" % (path, "\n  ".join(problems)))
    return mem


def prompt_text(note: str, short: Optional[Dict]) -> str:
    """laya's context: the RAM note, then the short memory's lessons (at most MAX_PROMPT_LESSONS)."""
    lessons = (short or {}).get("lessons", [])[:MAX_PROMPT_LESSONS]
    if not lessons:
        return note
    return note + "\nmemory vs %s: %s" % (short["opp"], "; ".join(les["text"] for les in lessons))
