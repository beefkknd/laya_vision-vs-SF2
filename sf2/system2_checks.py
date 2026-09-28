"""Mechanical checks on Qwen's RAW reply to a System 2 prompt (before sf2.system2.vet filters anything): every rule
is a yes/no a program computes. Used by scripts/test_system2_prompts.py (live Qwen) and tests/ (canned replies).

    reply_problems(...)   the rules every reply must meet: JSON with lessons, at most the limit, known kind / claim /
                          range / move, laya-sized text (<= MAX_TEXT, no numbers), no move twice, no move both
                          use_more and avoid, and every lesson backed by the log (the runtime's own evidence rule)
"""
from typing import Dict, List, Optional

from .memory import KINDS
from .system2 import evidence_for, supported
from .system2_prompts import CLAIMS, MAX_TEXT, RANGES
from .vs_sweep import actions


def lessons_of(reply) -> Optional[List[Dict]]:
    if not isinstance(reply, dict) or not isinstance(reply.get("lessons"), list):
        return None
    return [x for x in reply["lessons"] if isinstance(x, dict)]


def reply_problems(me: str, reply, limit: int, acts: List[Dict], fallback: Optional[List[Dict]]) -> List[str]:
    """Every rule the raw reply breaks (empty: none). ``acts``: the log the lessons must rest on (this opponent's
    games), ``fallback``: all games (own-move claims fall back to them, as at runtime)."""
    les = lessons_of(reply)
    if les is None:
        return ["no {\"lessons\": [...]} object"]
    out = []
    if not les:
        out.append("no lessons")
    if len(les) > limit:
        out.append("%d lessons > limit %d" % (len(les), limit))
    moves, seen, uses, avoids = set(actions(me)), set(), set(), set()
    for i, x in enumerate(les):
        text = x.get("text") if isinstance(x.get("text"), str) else ""
        tag = "lesson %d %r" % (i, text)
        if x.get("kind") not in KINDS or x.get("claim") not in CLAIMS or x.get("range") not in RANGES + (None,):
            out.append("%s: unknown kind / claim / range (%s, %s, %s)" % (tag, x.get("kind"), x.get("claim"),
                                                                          x.get("range")))
            continue
        if x.get("action") is not None and x["action"] not in moves:
            out.append("%s: %r is not one of %s's moves" % (tag, x["action"], me))
            continue
        if not text or len(text) > MAX_TEXT:
            out.append("%s: %d characters (1-%d)" % (tag, len(text), MAX_TEXT))
        if any(c.isdigit() for c in text):
            out.append("%s: numbers in the text" % tag)
        key = (x["action"], x["kind"]) if x["action"] else (x["claim"], x["range"])
        if key in seen:
            out.append("%s: repeats %s" % (tag, key))
        seen.add(key)
        (uses if x["kind"] == "use_more" else avoids if x["kind"] == "avoid" else set()).add(x["action"])
        why = supported(x, evidence_for(x, acts, fallback))
        if why:
            out.append("%s: not backed by the log (%s)" % (tag, why))
    for m in sorted(uses & avoids - {None}):
        out.append("%s is both use_more and avoid" % m)
    return out
