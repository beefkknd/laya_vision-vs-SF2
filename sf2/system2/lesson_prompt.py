"""Qwen's one prompt in the lesson loop: after each game, propose at most MAX_CLAIMS new claims worth learning.
Qwen sees what she already knows (the registry with each verdict), her moves overall (the classes code computed) and
the last game by move, range and what he was doing; code verifies what it proposes (sf2.system2.lessons)."""
import collections
import random
from typing import Dict, List, Sequence, Tuple

from ..system1.advice import opp_doing
from ..vocab import RANGE_WORDS
from . import lessons as L
from .move_coach import evidence, signal

MAX_CLAIMS = 2
SITUATION_MIN = 3      # a (move, range, his state) needs this many tries in the last game to be shown
SITUATION_ROWS = 12

SYSTEM = """You coach {me} in Street Fighter II against {opp}. After each game you propose what she should learn next.
A claim is one line of advice: "use more" or "avoid" a move, at one range (close, mid, far) or anywhere, and
optionally only when he does something (jumping, crouching, attacking, standing, stunned). Code then checks each claim
on her rounds in exactly that situation: an "avoid" is kept if the move clearly loses her hit points there; a "use
more" is tried for a few games and kept if it clearly gains. "Net hit points per try" = damage she dealt minus damage
she took, from the attack until her next decision.

Propose at most {n} claims that are NOT already registered, being tested, rejected, or contradicting a registered
lesson. Good claims are ones the data suggests but has not settled yet, for example a move that works only when he
does something, or a move that costs her in one situation. Use only moves she has tried.

Answer with JSON only:
{{"claims": [{{"kind": "use_more|avoid", "move": "...", "range": "close|mid|far|null", "when": "jumping|crouching|attacking|standing|stunned|null", "why": "one short sentence"}}]}}"""


def by_situation(rows: Sequence[Dict], min_tries: int = SITUATION_MIN) -> List[str]:
    """The last game per (move, range, what he was doing), biggest totals first."""
    acc = collections.defaultdict(list)
    for a in rows:
        if a.get("kind", "attack") == "attack":
            acc[(a["action"], a["range"], opp_doing(a))].append(a["dealt"] - a["taken"])
    out = sorted(((k, xs) for k, xs in acc.items() if len(xs) >= min_tries), key=lambda kv: -abs(sum(kv[1])))
    return ["- %s %s %s: %d tries, net %+.1f per try, total %+d" % (
        m, RANGE_WORDS[r], L.WHEN_WORDS[d], len(xs), sum(xs) / len(xs), sum(xs)) for (m, r, d), xs in out][:SITUATION_ROWS]


def _registry(reg: L.Registry) -> str:
    if not reg:
        return "(nothing yet)"
    return "\n".join("- %s: %s (%s)" % (r["state"], r["line"], r["why"]) for r in reg)


def messages(me: str, opp: str, reg: L.Registry, rows: Sequence[Dict], last_game: Sequence[Dict]) -> List[Dict]:
    user = ("What she knows about %s so far:\n%s\n\nHer moves overall (every game so far), per move and range:\n%s"
            "\n\nThe last game, per move, range and what he was doing:\n%s" % (
                opp, _registry(reg), signal(evidence(rows), "classes", random.Random(0)),
                "\n".join(by_situation(last_game)) or "(no move used 3 times in one situation)"))
    return [{"role": "system", "content": SYSTEM.format(me=me, opp=opp, n=MAX_CLAIMS)},
            {"role": "user", "content": user}]


def _cond(v):
    return None if v in (None, "", "null", "none", "None", "anywhere", "any") else v


def parse_claims(reply) -> Tuple[List[Dict], List[str]]:
    if not isinstance(reply, dict) or not isinstance(reply.get("claims"), list):
        return [], ["reply is not {\"claims\": [...]}: %.80r" % (reply,)]
    claims, problems = [], []
    for it in reply["claims"]:
        if not isinstance(it, dict):
            problems.append("not a claim: %.60r" % (it,))
            continue
        claims.append({"kind": it.get("kind"), "move": it.get("move"), "range": _cond(it.get("range")),
                       "when": _cond(it.get("when")), "why": it.get("why", "")})
    if len(claims) > MAX_CLAIMS:
        problems.append("%d claims: only the first %d taken" % (len(claims), MAX_CLAIMS))
        claims = claims[:MAX_CLAIMS]
    return claims, problems
