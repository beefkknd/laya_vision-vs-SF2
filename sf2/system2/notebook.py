"""The notebook: System 2's long-term knowledge, kept like a novice's notes, and the plan for the next round.

    {"me": "chunli", "self": [...], "opponents": {"ryu": [...]}, "questions": [...]}
      self       what I am good and bad at (my moves, my defence), from every round
      opponents  per opponent: what he does, what works on him, what he punishes
      questions  what I do not know yet and want to find out
    plan         the next round's lines for text laya (the short memory): notebook + this round, "try Z" experiments

Qwen writes it; code only checks it: ``check`` (shape, sizes), ``merged`` (a change budget per round, trimmed: the book
moves slowly), ``clean_plan`` (plan lines text laya can read, at most ``tries_allowed`` experiments, more when losing).
"""
import math
import re
from typing import Dict, List, Optional, Sequence, Tuple

from ..advice import FORWARD, read

MAX_LINES = 10          # per section
MAX_CHARS = 90          # a notebook line
MAX_PLAN = 5            # plan lines (text laya reads at most 5)
MAX_PLAN_CHARS = 60
MAX_ADD = 4             # notebook lines added per round (lets an empty book fill up in a few rounds)
MAX_REMOVE = 2          # notebook lines removed per round (the book cannot be rewritten in one round)
SECTIONS = ("self", "questions")


def empty(me: str) -> Dict:
    return {"me": me, "self": [], "opponents": {}, "questions": []}


def lines_of(nb: Dict, opp: str) -> List[str]:
    return list(nb.get("self", [])) + list(nb.get("opponents", {}).get(opp, [])) + list(nb.get("questions", []))


def _norm(s: str) -> str:
    return re.sub(r"\W+", " ", s.lower()).strip()


def changes(old: Dict, new: Dict, opp: str) -> Tuple[int, int]:
    """(lines added, lines removed) across the sections this round could change (self, this opponent, questions)."""
    added = removed = 0
    for get in (lambda nb: nb.get("self", []), lambda nb: nb.get("opponents", {}).get(opp, []),
                lambda nb: nb.get("questions", [])):
        a, b = {_norm(x) for x in get(old)}, {_norm(x) for x in get(new)}
        added, removed = added + len(b - a), removed + len(a - b)
    return added, removed


def edits(old: Dict, new: Dict, opp: str) -> int:
    return sum(changes(old, new, opp))


def tries_allowed(results: Sequence[str]) -> int:
    """Experiments allowed next round: more when losing. ``results``: recent round results, newest last. Of the last
    4 rounds: no loss -> 0, up to half lost -> 1, more -> 2 (no rounds yet -> 1)."""
    recent = list(results)[-4:]
    if not recent:
        return 1
    return math.ceil(2 * sum(r != "win" for r in recent) / len(recent))


def is_try(line: str) -> bool:
    return line.lower().startswith("try ")


def check(old: Dict, reply: Dict, opp: str, moves: Sequence[str], allowed: int) -> Dict[str, List[str]]:
    """Problems with Qwen's reply, by part: {"notebook": [...], "plan": [...]} (empty lists = accept that part).
    Checks shape and sizes only; the change budget is applied by ``merged`` (it trims, it does not reject) and the
    plan lines by ``clean_plan``."""
    nb, plan = reply.get("notebook"), reply.get("plan")
    bad = {"notebook": [], "plan": []}
    if not isinstance(nb, dict) or not all(isinstance(nb.get(k), list) for k in ("self", "opponent", "questions")):
        bad["notebook"].append("notebook must have lists self, opponent, questions")
    else:
        for k in ("self", "opponent", "questions"):
            if len(nb[k]) > MAX_LINES:
                bad["notebook"].append("%s has %d lines > %d" % (k, len(nb[k]), MAX_LINES))
            bad["notebook"] += ["%s line too long: %r" % (k, x) for x in nb[k]
                                if not isinstance(x, str) or len(x) > MAX_CHARS]
        # the change budget is not a reason to reject: ``merged`` trims the update to it
    if not isinstance(plan, list) or not plan:
        bad["plan"].append("no plan")
    return bad


def clean_plan(plan, moves: Sequence[str], allowed: int) -> Tuple[List[str], List[str]]:
    """(the plan lines text laya can follow, why the others were dropped): plain lines naming one of my moves, the
    first ``allowed`` experiments, at most MAX_PLAN lines."""
    keep, dropped, n_try = [], [], 0
    for line in plan if isinstance(plan, list) else []:
        if not isinstance(line, str) or len(line) > MAX_PLAN_CHARS or re.search(r"\d", line):
            dropped.append("not plain (<= %d chars, no digits): %r" % (MAX_PLAN_CHARS, line))
        elif read(line, list(moves) + [FORWARD]).move is None:
            dropped.append("names none of my moves, or two ranges at once: %r" % line)
        elif is_try(line) and n_try >= allowed:
            dropped.append("experiment over this round's %d: %r" % (allowed, line))
        elif len(keep) >= MAX_PLAN:
            dropped.append("over %d lines: %r" % (MAX_PLAN, line))
        else:
            n_try += is_try(line)
            keep.append(line)
    return keep, dropped


def _trim(old: List[str], new: List[str], budget: List[int]) -> List[str]:
    """``new`` within the round's budget [adds left, removals left]: lines Qwen dropped beyond the removal budget
    stay, new lines beyond the add budget wait for a later round (in Qwen's order)."""
    olds = {_norm(x) for x in old}
    news = {_norm(x) for x in new}
    out = []
    for x in old:                                   # keep, or remove while the budget lasts
        if _norm(x) in news or budget[1] <= 0:
            out.append(x)
        else:
            budget[1] -= 1
    for x in new:                                   # add while the budget lasts
        if _norm(x) not in olds and budget[0] > 0 and len(out) < MAX_LINES:
            out.append(x)
            olds.add(_norm(x))
            budget[0] -= 1
    return out


def merged(old: Dict, reply: Dict, opp: str) -> Dict:
    """The notebook after Qwen's reply, within the round's change budget (MAX_ADD / MAX_REMOVE over self, this
    opponent and questions; only those can change)."""
    nb, budget = reply["notebook"], [MAX_ADD, MAX_REMOVE]
    return {"me": old["me"], "self": _trim(old.get("self", []), nb["self"], budget),
            "opponents": dict(old.get("opponents", {}),
                              **{opp: _trim(old.get("opponents", {}).get(opp, []), nb["opponent"], budget)}),
            "questions": _trim(old.get("questions", []), nb["questions"], budget)}


def apply_reply(nb: Dict, plan: List[str], reply, opp: str, moves: Sequence[str],
                allowed: int) -> Tuple[Dict, List[str], Dict]:
    """(notebook, plan, what happened) after Qwen's reply. A part that fails its checks is not taken; a reply that is
    not a JSON object changes nothing; a plan with no usable line keeps last round's plan."""
    if not isinstance(reply, dict):
        return nb, plan, {"error": "reply is not a JSON object: %.80r" % (reply,)}
    bad = check(nb, reply, opp, moves, allowed)
    new_nb = merged(nb, reply, opp) if not bad["notebook"] else nb
    keep, dropped = clean_plan(reply.get("plan"), moves, allowed)
    return new_nb, keep or plan, {"reflection": reply.get("reflection"), "rejected": dict(bad, plan=dropped),
                                  "edits": edits(nb, new_nb, opp)}


def shortlist_memory(plan: Sequence[str], me: str, opp: str) -> Optional[Dict]:
    """The plan as the short-memory shape System 1 reads (sf2.system2.memory: lessons with a text)."""
    return {"me": me, "opp": opp, "lessons": [{"text": x} for x in plan]} if plan else None
