"""Qwen's one prompt in the lesson loop: after each game, one claim from each of two views.

    attack view    her attacks in the last game per move, range and what he was doing: net per decision vs her
                   average in that situation, best first (what worked, what did not)
    defense view   where the damage she took came from: traded / punished / stuffed / caught (sf2.system2.lessons.cause)
                   after which move, where; and what she chose when he attacked
    record         the last game and all games so far: rounds won and lost, damage dealt and taken per round

Qwen answers {"attack": claim or null, "defense": claim or null}; code verifies (sf2.system2.lessons).

Her moves for lessons leave out those the verifier refuses (sf2.system2.lessons.UNFOLLOWABLE: forward, 2026-09-30);
the views still show what she did. ``forward_lessons=True``: the prompt before that, byte for byte.
"""
import collections
from typing import Dict, List, Optional, Sequence, Tuple

from ..system1.advice import opp_doing
from ..vocab import RANGE_WORDS, RANGES
from . import lessons as L
from . import track_record as T

VIEWS = ("attack", "defense", "what_if")
DEFENSIVE = ("block_high", "block_low", "back", "jump_back", "crouch")
STREAK = 3             # games won (or lost) in a row, with no lesson changing, before Qwen is asked "what if?"
MIN_SHOWN = 3          # a situation needs this many decisions in the last game to be shown
ROWS_SHOWN = 10

SYSTEM = """You coach {me} in Street Fighter II against {opp}. After each game you look at it from two sides and
propose one new lesson from each:
- ATTACK: what her attacks did. Which move, where and when, gains her hit points, or loses them?
- DEFENSE: where the damage she took came from (traded: both hit; punished: she whiffed or was blocked; stuffed: her
  attack did not land; caught: hit while walking or blocking). What should she do instead, or stop doing?

A lesson is one line of advice: "use more", "always" or "avoid" one of her moves, at one range (close, mid, far) or
anywhere, and optionally only when he does something (jumping, crouching, attacking, standing, stunned). "always" is
for a move she should do every time in that situation even when it looks unlikely to work (for example a block).
Code checks each lesson against what she does now in exactly that situation: kept when the move is clearly better
(use more / always) or clearly worse (avoid) than her average there; "use more" and "always" lessons are tried for a
few games first. "Net" = damage she dealt minus damage she took, per decision.

Her moves: {moves}. Defensive ones: {defensive}. A move she has never used can still be proposed as "use more" or
"always" (it is tried in play); "avoid" needs a move she has used.

When she loses most rounds, the defense lesson matters most; when she wins, the attack lesson. Do not repeat a lesson
that is registered, being tested, rejected or refused, and do not contradict a registered one (an exception for a
narrower situation is fine: "avoid X when he jumps" next to "use more X").

Answer with JSON only (null for a side with nothing worth proposing):
{{"attack": {{"kind": "use_more|always|avoid", "move": "...", "range": "close|mid|far|null", "when": "jumping|crouching|attacking|standing|stunned|null", "why": "one short sentence"}},
 "defense": {{"kind": "use_more|always|avoid", "move": "...", "range": "close|mid|far|null", "when": "jumping|crouching|attacking|standing|stunned|null", "why": "one short sentence"}}}}"""


def _where(move: str, rng: str, doing: str) -> str:
    return "%s %s %s" % (move, RANGE_WORDS[rng], L.WHEN_WORDS[doing])


def attack_view(rows: Sequence[Dict]) -> List[str]:
    """Her attacks per (move, range, what he was doing), best total first, each against her average there."""
    by = collections.defaultdict(list)
    for a in rows:
        by[(a["action"], a["range"], opp_doing(a))].append(a)
    out = []
    for (move, rng, doing), xs in by.items():
        if len(xs) < MIN_SHOWN or xs[0].get("kind", "attack") != "attack":
            continue
        others = [a["dealt"] - a["taken"] for k, ys in by.items() if k[1:] == (rng, doing) and k[0] != move for a in ys]
        nets = [a["dealt"] - a["taken"] for a in xs]
        out.append((sum(nets), "- %s: %d tries, net %+.1f per decision vs her %s there, total %+d" % (
            _where(move, rng, doing), len(nets), sum(nets) / len(nets),
            "%+.1f" % (sum(others) / len(others)) if others else "- (nothing else)", sum(nets))))
    return [t for _, t in sorted(out, key=lambda x: -x[0])][:ROWS_SHOWN]


COMPARE = {"better": "better than her other moves there", "worse": "worse than her other moves there",
           "unclear": "no clearer than her other moves there", "few": "too few to compare"}


def defense_view(rows: Sequence[Dict], all_rows: Optional[Sequence[Dict]] = None,
                 moves: Sequence[str] = ()) -> List[str]:
    """The damage she took by cause and situation, most first, each move compared with her other moves there (over
    ``all_rows``); what she chose when he attacked; the defensive moves she never used."""
    all_rows = rows if all_rows is None else all_rows
    hurt = collections.defaultdict(lambda: [0, 0])
    for a in rows:
        c = L.cause(a)
        if c:
            h = hurt[(c, a["action"], a["range"], opp_doing(a))]
            h[0] += 1
            h[1] += a["taken"]
    out = []
    for (c, m, r, d), (n, t) in sorted(hurt.items(), key=lambda kv: -kv[1][1])[:ROWS_SHOWN]:
        ev = L.condition_evidence(all_rows, {"kind": "use_more", "move": m, "range": r, "when": d})
        out.append("- %s %s %s: %d times, %d damage; %s there nets %+.1f per decision vs her %+.1f with other moves: "
                   "%s" % (c, "while" if c == "caught" else "after", _where(m, r, d), n, t, m, ev["net"], ev["base"],
                           COMPARE[ev["cls"]]))
    chose = collections.defaultdict(lambda: [0, 0])
    for a in rows:
        if a.get("opp_attacked"):
            chose[a["action"]][0] += 1
            chose[a["action"]][1] += a["taken"]
    if chose:
        out.append("- when he attacked, she chose: %s" % ", ".join(
            "%s %d (took %.1f each)" % (m, n, t / n) for m, (n, t) in sorted(chose.items(), key=lambda kv: -kv[1][0])))
    used = {a["action"] for a in all_rows}
    never = [m for m in DEFENSIVE if m in moves and m not in used]
    if never:
        out.append("- never used: %s (laya-vision rates them unlikely to work, so a \"use more\" lesson will not make "
                   "her try one; an \"always\" lesson will)" % ", ".join(never))
    return out


def streak(games: Sequence[Dict], changed: Sequence[bool], n: int = STREAK) -> Optional[str]:
    """"losing" / "winning" when she lost (won) each of the last ``n`` games and no lesson changed in them."""
    if len(games) < n or any(changed[-n:]):
        return None
    last = games[-n:]
    if all(g["lost"] > g["won"] for g in last):
        return "losing"
    if all(g["won"] > g["lost"] for g in last):
        return "winning"
    return None


WHAT_IF = {
    "losing": "WHAT IF: she lost each of the last %d games and her lessons have not changed. Time to try something new "
              "to break it: propose one \"what_if\" lesson (use more or always) with a move she has rarely or never used, "
              "in a situation where she keeps losing. It is tried for a few games.",
    "winning": "WHAT IF: she won each of the last %d games and her lessons have not changed. Check what makes the "
               "difference: propose one \"what_if\" lesson (use more or always) with a different option in a situation "
               "where she wins, to confirm whether her current way is really the best there."}
WHAT_IF_JSON = (',\n "what_if": {{"kind": "use_more|always", "move": "...", "range": "close|mid|far|null", '
                '"when": "jumping|crouching|attacking|standing|stunned|null", "why": "one short sentence"}}')


def record(all_rounds: Sequence[Dict], last: Sequence[Dict]) -> str:
    if not all_rounds:
        return "No game played yet in this session: the views below are from her earlier games against him."
    def one(rs):
        n = max(1, len(rs))
        return (sum(r["result"] == "win" for r in rs), sum(r["result"] != "win" for r in rs),
                sum(r["dealt"] for r in rs) / n, sum(r["taken"] for r in rs) / n)
    w, lo, d, t = one(last)
    aw, alo, ad, at = one(all_rounds)
    return ("Last game: %d round%s won, %d lost (dealt %.0f, took %.0f per round). So far: %d won, %d lost (dealt %.0f, "
            "took %.0f per round)." % (w, "" if w == 1 else "s", lo, d, t, aw, alo, ad, at))


def overall(rows: Sequence[Dict]) -> List[str]:
    """Every (move, range) with enough decisions, against her average at that range (all games)."""
    out = []
    for move in sorted({a["action"] for a in rows}):
        for rng in RANGES:
            ev = L.condition_evidence(rows, {"kind": "use_more", "move": move, "range": rng, "when": None})
            if ev["cls"] != "few":
                out.append((ev["diff"], "- %s %s: %d tries, net %+.1f vs her %+.1f there: %s" % (
                    move, RANGE_WORDS[rng], ev["tries"], ev["net"], ev["base"],
                    {"better": "clearly better", "worse": "clearly worse", "unclear": "unclear"}[ev["cls"]])))
    return [t for _, t in sorted(out, key=lambda x: -x[0])]


def _registry(reg: L.Registry) -> str:
    if not reg:
        return "(nothing yet)"
    note = ["", L.VERIFIED_NOTE] if any(r["state"] == "verified" for r in reg) else []
    return "\n".join(["- %s: %s (%s)" % (r["state"], r["line"], r["why"]) for r in reg] + note)


def lesson_moves(moves: Sequence[str], forward_lessons: bool = False) -> List[str]:
    """The moves offered for lessons: without those System 1 cannot follow (``forward_lessons``: the old list)."""
    return list(moves) if forward_lessons else [m for m in moves if m not in L.UNFOLLOWABLE]


def messages(me: str, opp: str, reg: L.Registry, rows: Sequence[Dict], last: Sequence[Dict],
             all_rounds: Sequence[Dict], last_rounds: Sequence[Dict], moves: Sequence[str] = (),
             refused: Sequence[Dict] = (), stable: Optional[str] = None,
             track: Optional[Dict] = None, forward_lessons: bool = False) -> List[Dict]:
    moves = lesson_moves(list(moves) or sorted({a["action"] for a in rows}), forward_lessons)
    parts = [
        record(all_rounds, last_rounds),
        "What she knows about %s so far:\n%s" % (opp, _registry(reg)),
        "ATTACK - her attacks in the last game:\n%s" % ("\n".join(attack_view(last)) or "(none shown)"),
        "DEFENSE - the damage she took in the last game:\n%s" % ("\n".join(defense_view(last, rows, moves)) or
                                                                  "(none)"),
        "All games so far, per move and range, against her average at that range:\n%s" % (
            "\n".join(overall(rows)) or "(not enough yet)")]
    if track is not None:
        parts.append(T.prompt_part(opp, track))
    if refused:
        parts.append("Refused last time (do not propose again):\n%s" % "\n".join(
            "- %s: %s" % (L.render(o["claim"]) if _renderable(o["claim"]) else o["claim"], o["why"]) for o in refused))
    if stable:
        parts.append(WHAT_IF[stable] % STREAK)
    system = SYSTEM.format(me=me, opp=opp, moves=", ".join(moves),
                           defensive=", ".join(m for m in DEFENSIVE if m in moves) or "none")
    if stable:
        system = system[:-1] + WHAT_IF_JSON.replace("{{", "{").replace("}}", "}") + "}"      # inside the object
    return [{"role": "system", "content": system}, {"role": "user", "content": "\n\n".join(parts)}]


def _renderable(c) -> bool:
    return isinstance(c, dict) and c.get("kind") in L.KINDS and isinstance(c.get("move"), str) \
        and c.get("range") in (None, "close", "mid", "far") and c.get("when") in (None,) + tuple(L.WHEN_WORDS)


def _cond(v):
    return None if v in (None, "", "null", "none", "None", "anywhere", "any") else v


def _when(v):
    """A "when" as the grammar key; Qwen sometimes copies a lesson line's words ("jumps", "when he attacks"), round 4."""
    v = _cond(v)
    if isinstance(v, str):
        for key, words in L.WHEN_WORDS.items():                  # "when he jumps" / "he jumps" / "jumps"
            if v.strip().lower() in (words, words[5:], words.split(" ", 2)[2]):
                return key
    return v


def parse_claims(reply) -> Tuple[List[Dict], List[str]]:
    if not isinstance(reply, dict) or not any(v in reply for v in VIEWS):
        return [], ["reply is not {\"attack\": ..., \"defense\": ...}: %.80r" % (reply,)]
    claims, problems = [], []
    for view in VIEWS:
        it: Optional[Dict] = reply.get(view)
        if it is None:
            continue
        if not isinstance(it, dict):
            problems.append("%s: not a claim: %.60r" % (view, it))
            continue
        claims.append({"view": view, "kind": it.get("kind"), "move": it.get("move"), "range": _cond(it.get("range")),
                       "when": _when(it.get("when")), "why": it.get("why", "")})
    return claims, problems
