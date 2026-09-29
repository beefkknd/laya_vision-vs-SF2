"""A coach with no LLM: advice written by code from a character's logged rounds, to test whether advice itself helps
(scripts/ab_memory.py). Per (move, range) it counts net hit points per try (damage dealt - taken, from each attack
to the next decision), then writes plain lines text laya can follow:

    short     from rounds against THIS opponent only        (does a short memory work?)
    playbook  from rounds against every OTHER opponent      (does general knowledge of myself carry over?)

Best moves by net hit points -> "use more <move> <range>"; worst -> "avoid <move> <range>".
"""
import collections
from typing import Dict, Iterable, List, Optional, Tuple

from .advice import RANGE_WORDS
from .eval.logs import ROOT, load_actions, play_dirs, sources

MIN_TRIES = 20          # a (move, range) needs this many logged tries to be judged
N_USE, N_AVOID = 3, 2   # lines of each kind (text laya reads at most 5)


def attacks(me: str, root: str = ROOT) -> List[Dict]:
    """``me``'s attacks in every play log (sf2.eval.logs: test runs excluded), each tagged with its log dir."""
    return [a for d in play_dirs(root) for a in load_actions(d, root)
            if a.get("me") == me and a.get("kind") == "attack"]


def table(rows: Iterable[Dict]) -> Dict[Tuple[str, str], Dict]:
    """(move, range) -> tries, net hit points per try, hit rate."""
    acc = collections.defaultdict(lambda: [0, 0, 0])
    for a in rows:
        t = acc[(a["action"], a["range"])]
        t[0] += 1
        t[1] += a["dealt"] - a["taken"]
        t[2] += a["actual"] == "hit"
    return {k: {"tries": n, "net": s / n, "hit": h / n} for k, (n, s, h) in acc.items()}


def lines(stats: Dict[Tuple[str, str], Dict], min_tries: int = MIN_TRIES) -> List[str]:
    """Best moves to use more, worst to avoid, each at its range; one line per move (its best / worst range)."""
    judged = [(k, v) for k, v in stats.items() if v["tries"] >= min_tries]
    good = [(k, v) for k, v in sorted(judged, key=lambda kv: -kv[1]["net"]) if v["net"] > 0]
    bad = [(k, v) for k, v in sorted(judged, key=lambda kv: kv[1]["net"]) if v["net"] < 0]
    out, used = [], set()
    for (move, rng), _ in good:
        if move not in used and len(out) < N_USE:
            out.append("use more %s %s" % (move, RANGE_WORDS[rng]))
            used.add(move)
    n = len(out)
    for (move, rng), _ in bad:
        if move not in used and len(out) < n + N_AVOID:
            out.append("avoid %s %s" % (move, RANGE_WORDS[rng]))
            used.add(move)
    return out


def memory(me: str, opp: str, kind: str, rows: Optional[List[Dict]] = None) -> Dict:
    """A short-memory file (sf2.memory shape) written by code: ``kind`` "short" (this opponent) or "playbook"
    (every other opponent)."""
    rows = attacks(me) if rows is None else rows
    pick = [a for a in rows if (a["opp"] == opp) == (kind == "short")]
    stats = table(pick)
    return {"me": me, "opp": opp, "source": "code_coach:%s" % kind, "logs": sources(pick),
            "lessons": [{"text": t} for t in lines(stats)],
            "stats": {"%s@%s" % k: v for k, v in sorted(stats.items(), key=lambda kv: -kv[1]["tries"])}}


def lesson_file(me: str, opp: Optional[str], kind: str, rows: Optional[List[Dict]] = None) -> Dict:
    """``memory`` in sf2.memory's full format (kind, action, range, claim, evidence with refs into the logs), so it
    passes the same checks as System 2's own memory and can seed a run (scripts/seed_memory.py). ``kind`` "all" =
    every opponent (a playbook)."""
    rows = attacks(me) if rows is None else rows
    pick = rows if kind == "all" else [a for a in rows if (a["opp"] == opp) == (kind == "short")]
    lessons = []
    for text in lines(table(pick)):
        use = text.startswith("use more")
        move, rng = text.split()[2 if use else 1], next(r for r, w in RANGE_WORDS.items() if text.endswith(w))
        mine = [a for a in pick if a["action"] == move and a["range"] == rng]
        hit = [a for a in mine if (a["actual"] == "hit") if use] or [a for a in mine if a["i_was_hit"] and not use]
        lessons.append({"text": text, "kind": "use_more" if use else "avoid", "action": move, "range": rng,
                        "claim": "lands" if use else "punished",
                        "evidence": {"tries": len(mine), "count": len(hit), "rate": round(len(hit) / len(mine), 2),
                                     "refs": ["%s/g%df%d" % (a.get("log", "?"), a.get("game", 0), a.get("frame", 0))
                                              for a in hit[:5]], "scope": "code_coach:%s" % kind}})
    out = {"me": me, "source": ["code_coach:%s" % kind], "lessons": lessons}
    if opp and kind != "all":
        out["opp"] = opp
    return out
