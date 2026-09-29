"""Qwen picks which of her moves to use more and which to avoid, from evidence code computes; code writes the advice
lines text laya reads, and code grades the picks (scripts/qwen_moves.py).

    evidence   per (move, range) she tried: tries, net hit points per try (damage dealt - taken, to the next
               decision), its 95% interval, the total, and a class: good (interval above 0), bad (below 0),
               unclear, or few (fewer than MIN_TRIES tries)
    signal     the evidence as text for Qwen, rows in a given order; level "table" (numbers only) or "classes"
               (numbers + the class in words)
    messages   the prompt: the signal and, game after game, the picks in play with what they did since
    parse_reply  Qwen's JSON -> picks {"use_more": [(move, range)], "avoid": [...]} + problems
    lines      picks -> advice lines in text laya's grammar (sf2.system1.advice)
    grade      Q1: every "use more" is good, every "avoid" bad, the biggest drain avoided, a good move used if one
               exists, at most MAX_LINES lines (MAX_USE of them "use more")
    grade_update  Q2: ``grade`` on the new picks, and no pick that is still right was dropped unless all MAX_LINES
               slots are full (first Ken loop: a correct "avoid" was swapped out while slots were free)
"""
import json
import random
from typing import Dict, List, Optional, Sequence, Tuple

from ..vocab import RANGE_WORDS

MIN_TRIES = 20
MAX_LINES = 5          # text laya reads at most 5 advice lines
MAX_USE = 3            # "use more" lines; "avoid" gets every slot they leave
KINDS = ("use_more", "avoid")
RIGHT = {"use_more": "good", "avoid": "bad"}
CLASS_WORDS = {"good": "clearly good", "bad": "clearly bad", "unclear": "unclear", "few": "too few tries to judge"}

Pick = Tuple[str, str]
Picks = Dict[str, List[Pick]]


def classify(n: int, lo: float, hi: float) -> str:
    if n < MIN_TRIES:
        return "few"
    return "good" if lo > 0 else "bad" if hi < 0 else "unclear"


def evidence(rows: Sequence[Dict]) -> List[Dict]:
    """Per (move, range) of her attacks in ``rows``, sorted by total (worst first)."""
    nets: Dict[Pick, List[int]] = {}
    for a in rows:
        if a.get("kind", "attack") == "attack":
            nets.setdefault((a["action"], a["range"]), []).append(a["dealt"] - a["taken"])
    out = []
    for (move, rng), xs in nets.items():
        n, m = len(xs), sum(xs) / len(xs)
        sd = (sum((x - m) ** 2 for x in xs) / (n - 1)) ** 0.5 if n > 1 else 0.0
        half = 1.96 * sd / n ** 0.5 if n > 1 else float("inf")
        out.append({"move": move, "range": rng, "tries": n, "net": m, "lo": m - half, "hi": m + half,
                    "total": sum(xs), "cls": classify(n, m - half, m + half)})
    return sorted(out, key=lambda e: e["total"])


def _row(e: Dict, level: str) -> str:
    s = "- move %s, range %s: %d tries, net %+.1f hit points per try, total %+d" % (
        e["move"], e["range"], e["tries"], e["net"], e["total"])
    if level == "classes":
        s += "; %s" % (CLASS_WORDS[e["cls"]] if e["cls"] == "few" else
                       "95%% range %+.1f to %+.1f: %s" % (e["lo"], e["hi"], CLASS_WORDS[e["cls"]]))
    return s


def signal(ev: Sequence[Dict], level: str, rng: random.Random) -> str:
    rows = [_row(e, level) for e in ev]
    rng.shuffle(rows)
    return "\n".join(rows)


SYSTEM = """You coach {me} in Street Fighter II against {opp}. You read what her attacks did and choose which to use
more and which to avoid. A move is judged at one range: close, mid or far. "Net hit points per try" is the damage she
dealt minus the damage she took, from the attack until her next decision; positive is good for her.

Rules:
- At most {max_lines} picks in all (she can only follow {max_lines} lines).
- "use_more": at most {max_use} (move, range) whose net is clearly above 0 with enough tries.
- "avoid": (move, range) whose net is clearly below 0 with enough tries, in the slots "use_more" leaves; if there are
  more than fit, the ones that cost her the most in total first.
- Only moves and ranges listed below. Say nothing about a move whose evidence is unclear or thin.
- Picks already in play: keep each one whose evidence is still clear in the same direction. A pick whose evidence is
  now unclear or too thin must be dropped at this update. Otherwise drop one only when all {max_lines} slots are full
  and a clearly stronger one needs the room. A move she was told
  to avoid shows few new tries and a total that stops growing: that is the advice working, not a reason to drop it.

Answer with JSON only:
{{"use_more": [{{"move": "...", "range": "close|mid|far"}}], "avoid": [{{"move": "...", "range": "close|mid|far"}}],
"why": "one short sentence"}}"""


def messages(me: str, opp: str, sig: str, current: Optional[Picks] = None,
             since: Optional[Dict[Pick, Dict]] = None, ev: Optional[Sequence[Dict]] = None) -> List[Dict]:
    user = "Her attacks against %s, per move and range:\n%s" % (opp, sig)
    cls = {(e["move"], e["range"]): e["cls"] for e in ev or []}
    if current is not None:
        rows = []
        for kind in KINDS:
            for p in current.get(kind, []):
                s = (since or {}).get(p)
                rows.append("- %s %s %s: now %s; %s" % (
                    kind.replace("_", " "), p[0], RANGE_WORDS[p[1]], CLASS_WORDS.get(cls.get(p), "not in the evidence"),
                    "used %d times since it was picked, net %+.1f per try" % (s["tries"], s["net"])
                    if s and s["tries"] else "not used since it was picked"))
        user += "\n\nPicks in play now:\n%s" % ("\n".join(rows) if rows else "(none yet)")
    return [{"role": "system", "content": SYSTEM.format(me=me, opp=opp, max_use=MAX_USE, max_lines=MAX_LINES)},
            {"role": "user", "content": user}]


def parse_reply(reply, ev: Sequence[Dict]) -> Tuple[Picks, List[str]]:
    picks: Picks = {k: [] for k in KINDS}
    if not isinstance(reply, dict):
        return picks, ["reply is not a JSON object: %.60r" % (reply,)]
    known = {(e["move"], e["range"]) for e in ev}
    problems = []
    for kind in KINDS:
        items = reply.get(kind, [])
        if not isinstance(items, list):
            problems.append("%s is not a list" % kind)
            continue
        for it in items:
            p = (it.get("move"), it.get("range")) if isinstance(it, dict) else None
            if p not in known:
                problems.append("%s: %r is not a move and range she tried" % (kind, it))
            elif p in picks[kind]:
                problems.append("%s: %s %s named twice" % (kind, *p))
            else:
                picks[kind].append(p)
    return picks, problems


def lines(picks: Picks) -> List[str]:
    return (["use more %s %s" % (m, RANGE_WORDS[r]) for m, r in picks.get("use_more", [])] +
            ["avoid %s %s" % (m, RANGE_WORDS[r]) for m, r in picks.get("avoid", [])])


def over_cap(picks: Picks) -> bool:
    n_use, n_avoid = len(picks.get("use_more", [])), len(picks.get("avoid", []))
    return n_use > MAX_USE or n_use + n_avoid > MAX_LINES


def grade(picks: Picks, ev: Sequence[Dict]) -> Dict:
    cls = {(e["move"], e["range"]): e["cls"] for e in ev}
    bad = [(e["move"], e["range"]) for e in ev if e["cls"] == "bad"]             # worst total first
    wrong = {k: [p for p in picks.get(k, []) if cls.get(p) != RIGHT[k]] for k in KINDS}
    drain = bad[0] if bad else None
    out = {"wrong_use": wrong["use_more"], "wrong_avoid": wrong["avoid"],
           "missed_drain": drain if drain and drain not in picks.get("avoid", []) else None,
           "missed_good": any(c == "good" for c in cls.values()) and not picks.get("use_more"),
           "over_cap": over_cap(picks)}
    out["ok"] = not (out["wrong_use"] or out["wrong_avoid"] or out["missed_drain"] or out["missed_good"]
                     or out["over_cap"])
    return out


def grade_update(prev: Picks, new: Picks, ev: Sequence[Dict]) -> Dict:
    out = grade(new, ev)
    cls = {(e["move"], e["range"]): e["cls"] for e in ev}
    full = sum(len(new.get(k, [])) for k in KINDS) >= MAX_LINES
    out["dropped_correct"] = [p for k in KINDS for p in prev.get(k, [])
                              if cls.get(p) == RIGHT[k] and p not in new.get(k, []) and not full]
    out["ok"] = out["ok"] and not out["dropped_correct"]
    return out


def picks_json(picks: Picks) -> str:
    return json.dumps({k: [list(p) for p in v] for k, v in picks.items()})
