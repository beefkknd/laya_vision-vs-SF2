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
import tempfile
from typing import Dict, List, Optional, Tuple

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


def save(path: str, mem: Dict) -> None:
    """Write ``mem`` whole or not at all (a temp file in the same folder, then a rename): a reader never sees half."""
    folder = os.path.dirname(path) or "."
    os.makedirs(folder, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=folder, prefix=".", suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(mem, f, indent=1)
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise


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


def try_load(path: str, actions: List[str]) -> Tuple[Optional[Dict], Optional[str]]:
    """(memory, None), (None, None) when there is no file, or (None, why) when it cannot be used; never raises."""
    if not os.path.exists(path):
        return None, None
    try:
        with open(path) as f:
            mem = json.load(f)
    except ValueError as e:
        return None, "not JSON (%s)" % e
    except OSError as e:
        return None, "unreadable (%s)" % e
    if not isinstance(mem, dict) or not isinstance(mem.get("lessons", []), list):
        return None, "not a memory: %.60r" % (mem,)
    odd = [i for i, les in enumerate(mem.get("lessons", []))
           if not isinstance(les, dict) or not isinstance(les.get("evidence", {}), dict)]
    if odd:
        return None, "lessons %s are not lesson objects" % odd
    try:
        problems = check(mem, actions)
    except (AttributeError, TypeError) as e:           # a field of the wrong type deeper down
        return None, "malformed (%s)" % e
    return (None, "; ".join(problems)) if problems else (mem, None)


def _stamp(path: str) -> Optional[float]:
    return os.path.getmtime(path) if os.path.exists(path) else None


class OutsideWatch:
    """Spots a short memory changed from outside the loop (the brain panel's buttons). Checked only while System 2
    is idle, since System 2's own write also changes the file; a file that cannot be used keeps the memory in play."""
    KEEP = object()

    def __init__(self):
        self._seen: Dict[str, Optional[float]] = {}

    def seen(self, opp: str, path: str, stamp: Optional[float] = None) -> None:
        """The loop has the file for ``opp`` as it was at ``stamp`` (default: as it is now). For System 2's new
        version pass the stamp of System 2's own save, so a panel edit made after it is still seen as new."""
        self._seen[opp] = _stamp(path) if stamp is None else stamp

    def check(self, opp: str, path: str, busy: bool, actions: List[str]):
        """None (nothing new), or (memory, message): the new memory (None if the file is gone) or KEEP."""
        stamp = _stamp(path)
        if busy or opp not in self._seen or stamp == self._seen[opp]:
            return None
        self._seen[opp] = stamp
        mem, why = try_load(path, actions)
        if why:
            return self.KEEP, "short memory vs %s changed outside the loop but %s: kept the memory in play" % (opp, why)
        return mem, "short memory vs %s changed outside the loop: %d lessons in play from this round" % (
            opp, len((mem or {}).get("lessons", [])))


def prompt_text(note: str, short: Optional[Dict]) -> str:
    """laya's context: the RAM note, then the short memory's lessons (at most MAX_PROMPT_LESSONS)."""
    lessons = (short or {}).get("lessons", [])[:MAX_PROMPT_LESSONS]
    if not lessons:
        return note
    return note + "\nmemory vs %s: %s" % (short["opp"], "; ".join(les["text"] for les in lessons))
