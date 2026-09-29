"""The demo page's buttons (scripts/brain_panel.py): "cheat" copies the seed's playbook / short memory into a --fresh
run's memory folder (memory_runs/<name>), "clean" blanks it. Pressed before the game, a button is saved in
out/live/pending.json and the next learn_loop --fresh run applies it before its first fight.
"""
import json
import os
import time
from typing import Dict, List, Optional

from .memory import save
from .vocab import IDS

SEED = "memory_seeds/video"
PENDING = "out/live/pending.json"
RUNS = "memory_runs"
WHAT = ("playbook", "short")
DO = ("cheat", "clean")


def _valid(what: str, do: str, opp: Optional[str]) -> None:
    """Only known words reach a file path (the request comes from a web page)."""
    if not all(isinstance(x, str) for x in (what, do)) or not (opp is None or isinstance(opp, str)) or \
            what not in WHAT or do not in DO or (what == "short" and opp not in IDS):
        raise ValueError("not a button: what=%r do=%r opp=%r" % (what, do, opp))


def _rel(me: str, what: str, opp: Optional[str]) -> str:
    return os.path.join("playbook", "%s.json" % me) if what == "playbook" else os.path.join(
        "short", "%s_vs_%s.json" % (me, opp))


def _inside_a_run(root: str, dest: str) -> None:
    """``root`` is one folder inside memory_runs/ (links resolved) and ``dest`` is inside it: never memory/."""
    runs, real = os.path.realpath(RUNS), os.path.realpath(root)
    if os.path.dirname(real) != runs or not os.path.realpath(dest).startswith(real + os.sep):
        raise ValueError("only a --fresh run's memory (%s/<name>), not %r" % (RUNS, root))


def apply(root: str, me: str, what: str, do: str, opp: Optional[str], seed: str = SEED) -> str:
    """Cheat or clean one memory in ``root`` now; returns what was done, for the log."""
    _valid(what, do, opp)
    if me not in IDS:
        raise ValueError("not a character: %r" % (me,))
    dest = os.path.join(root, _rel(me, what, opp))
    _inside_a_run(root, dest)
    if do == "cheat":
        with open(os.path.join(seed, _rel(me, what, opp))) as f:
            mem = json.load(f)
    else:
        mem = {"me": me, "opp": opp, "lessons": []} if what == "short" else {"me": me, "lessons": []}
    save(dest, mem)
    return "brain panel: %s %s (by hand)" % ("cheated" if do == "cheat" else "cleaned",
                                            "the playbook" if what == "playbook" else "the short memory vs %s" % opp)


def pending() -> Dict:
    """The buttons saved for the next game ({} when none, or when the file cannot be read)."""
    try:
        with open(PENDING) as f:
            p = json.load(f)
    except (OSError, ValueError):
        return {}
    return p if isinstance(p, dict) else {}


def add_pending(what: str, do: str, opp: Optional[str], me: str = "chunli", seed: str = SEED) -> str:
    """Save a button for the next game (a cheat only if the seed has that memory)."""
    _valid(what, do, opp)
    if do == "cheat" and not os.path.exists(os.path.join(seed, _rel(me, what, opp))):
        raise ValueError("no seed %s for this button in %s" % (_rel(me, what, opp), seed))
    p = pending()
    p["playbook" if what == "playbook" else "short:%s" % opp] = {"what": what, "do": do, "opp": opp,
                                                                 "at": time.strftime("%H:%M:%S")}
    save(PENDING, p)
    return "ready for the next game: %s %s" % (do, "the playbook" if what == "playbook" else
                                                "the short memory vs %s" % opp)


def apply_pending(root: str, me: str, seed: str = SEED) -> List[str]:
    """At the start of a --fresh run: apply the saved buttons, then forget them. A button that cannot be applied is
    reported, never a reason not to start."""
    done = []
    try:
        for x in pending().values():
            try:
                done.append(apply(root, me, x["what"], x["do"], x.get("opp"), seed) + " before the game")
            except (OSError, ValueError, KeyError, TypeError, AttributeError) as e:
                done.append("brain panel: could not apply %r: %s" % (x, e))
    finally:
        if os.path.exists(PENDING):
            os.remove(PENDING)
    return done


def preview(me: str, seed: str = SEED) -> Dict:
    """What the saved buttons will put in play, for the page while no game runs."""
    out = {"playbook": None, "short": {}}
    for x in pending().values():
        lessons = []
        path = os.path.join(seed, _rel(me, x["what"], x.get("opp")))
        if x["do"] == "cheat" and os.path.exists(path):
            with open(path) as f:
                lessons = [les.get("text") for les in json.load(f).get("lessons", [])]
        if x["what"] == "playbook":
            out["playbook"] = lessons
        else:
            out["short"][x["opp"]] = lessons
    return out
