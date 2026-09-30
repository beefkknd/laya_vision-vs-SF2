"""Qwen's character prompt in the lesson loop: one opponent, one moment. After each game:

    his threats    what he does that hurts her (jumps in / attacks on the ground / hits her from afar), at which range
                   of hers, and after which of her moves: damage and share, all games against him
    if you see X   per situation of his (what he is doing when she decides, and the range), her answers ranked: net per
                   decision vs her other moves there (works / fails / unclear), and the moves she never tried there
    record         rounds won and lost, damage per round (sf2.system2.lesson_prompt.record)

Qwen answers {"answer": claim, "stop": claim} (+ "what_if" when stuck); every claim names what he is doing ("when").
Code verifies exactly as in the two-view prompt (sf2.system2.lessons).

The gap (docs/qwen_learning.md): text laya reads only "when he jumps / crouches / attacks / stands / is stunned" (not
retrained, owner 2026-09-29), so Qwen is told how to say his moves in those words. New runs log his move per decision
(``opp_move``, sf2/system1/opp_moves.py): "his threats" name it then (a fireball, an uppercut, ...); older play data
lacks it and keeps the coarse wording (jumps in / attacks on the ground / hits her from afar), byte for byte.
"""
import collections
from typing import Dict, List, Optional, Sequence, Tuple

from ..system1.advice import opp_doing
from ..vocab import RANGE_WORDS
from . import lessons as L
from . import track_record as T
from .lesson_prompt import STREAK, WHAT_IF, _registry, _renderable, _when, record

VIEWS = ("answer", "stop", "what_if")
MIN_SITUATION = 10     # decisions in a situation of his before it is shown
SITUATIONS = 5         # his situations shown, the most damaging first
THREATS = 8

SYSTEM = """You coach {me} in Street Fighter II against {opp}, and only against {opp}. Think like a player studying
this one opponent: which of his moves beat her, and "if I see him do X, what works and what does not?"

You are shown his threats (what he does that hurts her, where) and, for each of his situations, her answers ranked
against her other moves there. Propose two lessons for the situation that costs her most, or the one where the fix is
clearest:
- ANSWER: what she should do when she sees it ("use more" or "always" one of her moves).
- STOP: what she should stop doing there ("avoid" one of her moves).

A lesson is one line: "use more", "always" or "avoid" one of her moves, at one range (close, mid, far) or anywhere,
and ALWAYS with what he is doing: jumping, crouching, attacking, standing or stunned. "always" is for a move she should
do every time there even when it looks unlikely to work (for example a block).

His moves can only be named in those words. His specials look like any attack, so say them by where they happen:
a fireball is "when he attacks far away" (or at mid range), an uppercut or a jump-in kick close by is "when he jumps"
or "when he attacks up close"; a sweep or a charge is "when he attacks" at its range.

Code checks each lesson against what she does now in exactly that situation: kept when the move is clearly better
(use more / always) or clearly worse (avoid) than her average there; "use more" and "always" lessons are tried for a
few games first. "Net" = damage she dealt minus damage she took, per decision.

Her moves: {moves}. A move she has never used there can still be proposed as "use more" or "always" (it is tried in
play); "avoid" needs a move she has used. Do not repeat a lesson that is registered, being tested, rejected or refused,
and do not contradict a registered one (a narrower exception is fine).

Answer with JSON only (null for a side with nothing worth proposing):
{{"answer": {{"kind": "use_more|always", "move": "...", "range": "close|mid|far|null", "when": "jumping|crouching|attacking|standing|stunned", "why": "one short sentence"}},
 "stop": {{"kind": "avoid", "move": "...", "range": "close|mid|far|null", "when": "jumping|crouching|attacking|standing|stunned", "why": "one short sentence"}}}}"""
WHAT_IF_JSON = (',\n "what_if": {"kind": "use_more|always", "move": "...", "range": "close|mid|far|null", '
                '"when": "jumping|crouching|attacking|standing|stunned", "why": "one short sentence"}')


# His move as the threat, when the log names it (opp_move); "normal" and "none" keep the coarse wording.
MOVE_THREATS = {"fireball": "hits with a fireball", "uppercut": "hits with an uppercut",
                "hurricane": "hits with a hurricane kick", "slap": "hits with the hundred hand slap",
                "throw": "throws her", "jump_attack": "jumps in"}
MOVE_NOTE = ("(His moves are named from the game's memory. A lesson still says them in its own words, by what he is "
             "doing when she decides and the range: a fireball is \"when he attacks\" far away or at mid range; an "
             "uppercut, a hurricane kick or a throw is \"when he attacks\" or \"when he jumps\" up close.)")


def threat(a: Dict) -> Optional[str]:
    """What of his hurt her after this decision of hers (None when she took nothing): his move when it is logged
    (``opp_move``), else what he was doing (jumps in, attacks on the ground, hits her from afar)."""
    if not a.get("taken"):
        return None
    if a.get("opp_move") in MOVE_THREATS:
        return MOVE_THREATS[a["opp_move"]]
    reaction = a.get("opp_reaction") or []
    if a.get("opp_air") or "jump" in reaction:
        return "jumps in"
    if "attack" in reaction or "special" in reaction:
        return "attacks on the ground"
    return "hits her from afar"              # no attack of his in the window: a fireball already on the way


def threats(rows: Sequence[Dict]) -> List[str]:
    """His threats by damage, with the share of all the damage she took and her moves they caught most."""
    hurt: Dict[Tuple[str, str], List] = collections.defaultdict(lambda: [0, 0, collections.Counter()])
    for a in rows:
        t = threat(a)
        if t:
            h = hurt[(t, a["range"])]
            h[0] += 1
            h[1] += a["taken"]
            h[2][a["action"]] += a["taken"]
    total = sum(h[1] for h in hurt.values()) or 1
    out = []
    for (t, rng), (n, dmg, by) in sorted(hurt.items(), key=lambda kv: -kv[1][1])[:THREATS]:
        out.append("- he %s %s: %d times, %d damage (%d%%); she was doing %s" % (
            t, RANGE_WORDS[rng], n, dmg, round(100 * dmg / total),
            ", ".join("%s (%d)" % (m, d) for m, d in by.most_common(3))))
    return out


VERDICT = {"better": "works", "worse": "fails", "unclear": "unclear", "few": "too few to tell"}


def _answers(rows: Sequence[Dict], rng: str, doing: str, moves: Sequence[str]) -> List[str]:
    used = collections.Counter(a["action"] for a in rows)
    ranked = []
    for m in used:
        ev = L.condition_evidence(rows, {"kind": "use_more", "move": m, "range": rng, "when": doing})
        ranked.append((ev["diff"] if ev["cls"] != "few" else float("-inf"), "- %s: %d tries, net %+.1f vs her %+.1f "
                       "with other moves: %s" % (m, ev["tries"], ev["net"], ev["base"], VERDICT[ev["cls"]])))
    out = [t for _, t in sorted(ranked, key=lambda x: -x[0])]
    never = [m for m in moves if m not in used]
    if never:
        out.append("- never tried there: %s" % ", ".join(never))
    return out


def term(doing: str, rng: str) -> str:
    """The fighting-game word for her decision in this situation of his (docs/qwen_learning.md 0g)."""
    if doing == "jumping":
        return "anti-air"
    if doing == "attacking":
        return "his zoning (fireball)" if rng == "far" else "his attack: block it or beat it"
    if doing == "stunned":
        return "punish: he cannot act"
    if doing == "crouching":
        return "he crouches: low pokes, or charging"
    return "footsies" if rng in ("mid", "far") else "up close in neutral: throw range"


def if_you_see(rows: Sequence[Dict], moves: Sequence[str] = (), terms: bool = False) -> List[str]:
    """One block per situation of his (what he is doing, range), the most damage to her first; ``terms``: each named
    the way players do (anti-air, footsies, ...)."""
    by = collections.defaultdict(list)
    for a in rows:
        by[(a["range"], opp_doing(a))].append(a)
    shown = sorted(((k, xs) for k, xs in by.items() if len(xs) >= MIN_SITUATION),
                   key=lambda kx: -sum(a["taken"] for a in kx[1]))[:SITUATIONS]
    blocks = []
    for (rng, doing), xs in shown:
        head = "%s, %s%s (%d decisions, she takes %.1f and deals %.1f per decision):" % (
            RANGE_WORDS[rng][0].upper() + RANGE_WORDS[rng][1:], L.WHEN_WORDS[doing],
            " - " + term(doing, rng) if terms else "", len(xs),
            sum(a["taken"] for a in xs) / len(xs), sum(a["dealt"] for a in xs) / len(xs))
        blocks.append("\n".join([head] + _answers(xs, rng, doing, moves)))
    return blocks


def messages(me: str, opp: str, reg: L.Registry, rows: Sequence[Dict], last: Sequence[Dict],
             all_rounds: Sequence[Dict], last_rounds: Sequence[Dict], moves: Sequence[str] = (),
             refused: Sequence[Dict] = (), stable: Optional[str] = None,
             track: Optional[Dict] = None, fgc: bool = False) -> List[Dict]:
    """Same signature as sf2.system2.lesson_prompt.messages; ``rows`` = all her decisions against him so far.
    ``fgc``: situations named the way players do, and a primer on the game and this opponent (``messages_fgc``)."""
    moves = list(moves) or sorted({a["action"] for a in rows})
    parts = [
        record(all_rounds, last_rounds),
        "What she knows about %s so far:\n%s" % (opp, _registry(reg)),
        "HIS THREATS - what he did that hurt her, all games:\n%s" % ("\n".join(threats(rows)) or "(none)")
        + ("\n" + MOVE_NOTE if any(a.get("opp_move") in MOVE_THREATS for a in list(rows) + list(last)) else ""),
        "HIS THREATS - the last game:\n%s" % ("\n".join(threats(last)) or "(none)"),
        "IF YOU SEE - her answers in each of his situations, all games (the most damage to her first):\n\n%s" % (
            "\n\n".join(if_you_see(rows, moves, terms=fgc)) or "(not enough yet)")]
    if track is not None:
        parts.append(T.prompt_part(opp, track))
    if refused:
        parts.append("Refused last time (do not propose again):\n%s" % "\n".join(
            "- %s: %s" % (L.render(o["claim"]) if _renderable(o["claim"]) else o["claim"], o["why"]) for o in refused))
    if stable:
        parts.append(WHAT_IF[stable] % STREAK)
    system = SYSTEM.format(me=me, opp=opp, moves=", ".join(moves))
    if fgc:
        head, tail = system.split("\n\nAnswer with JSON only", 1)
        system = "%s\n\n%s\n\nAnswer with JSON only%s" % (head, primer(opp), tail)
    if stable:
        system = system[:-1] + WHAT_IF_JSON + "}"                     # inside the object
    return [{"role": "system", "content": system}, {"role": "user", "content": "\n\n".join(parts)}]


# What players write about World Warrior (arcade sources, docs/qwen_learning.md 0g): hypotheses for Qwen to test, not
# facts - code keeps only what her own games show.
PRIMER = """What players say about World Warrior (from the arcade version; the SNES port may differ, and code checks every
lesson against her own games, so treat these as ideas to test):
- Chun-Li's strengths are her normals, walk speed and long throw. Her specials are slow and unsafe in this version:
  Spinning Bird Kick loses to a crouching medium punch, Lightning Legs recovers slowly.
- Her anti-airs are standing mk, hk and hp from mid range; c.mk catches jumps from far away. c.mk and standing mk are
  her pokes in footsies.
- No reversals, no throw escape, no super meter; only blocked specials chip.
- The CPU reacts to the move she commits to (it reads it the moment it starts), but not every time, and it gets more
  aggressive as the round clock runs down.
- Words: anti-air = hitting him out of a jump; footsies = the mid-range game of pokes; whiff punish = hitting him while
  he recovers from a missed attack; zoning = keeping her out with fireballs; keep-away = block and stay out of reach."""
OPPONENT = {
    "ken": "Ken: Shoryuken beats jump-ins and is very unsafe if blocked; Hurricane Kick does not combo - block the "
           "first hit, then punish before he lands; Hadoken is slow to start and recover. A known CPU pattern: "
           "fireballs, then a fierce punch into Shoryuken.",
    "ryu": "Ryu: Shoryuken beats jump-ins and is very unsafe if blocked; Hurricane Kick does not combo - block the "
           "first hit, then punish before he lands; Hadoken is slow to start and recover. A known CPU pattern: "
           "fireballs, then a fierce punch into Shoryuken.",
    "honda": "E. Honda: the Sumo Headbutt (a flying charge, also his anti-air) is punishable when blocked, and a rapid "
             "standing jab beats it; the Hundred Hand Slap is strong at mid range. Players advise keep-away: take a "
             "lead, then stay out of reach; one throw from him starts a lot of damage.",
}


def primer(opp: str) -> str:
    return PRIMER + ("\n" + OPPONENT[opp] if opp in OPPONENT else "")


def messages_fgc(*args, **kw) -> List[Dict]:
    return messages(*args, fgc=True, **kw)


def _cond(v):
    return None if v in (None, "", "null", "none", "None", "anywhere", "any") else v


def parse_claims(reply) -> Tuple[List[Dict], List[str]]:
    if not isinstance(reply, dict) or not any(v in reply for v in VIEWS):
        return [], ["reply is not {\"answer\": ..., \"stop\": ...}: %.80r" % (reply,)]
    claims, problems = [], []
    for view in VIEWS:
        it = reply.get(view)
        if it is None:
            continue
        if not isinstance(it, dict):
            problems.append("%s: not a claim: %.60r" % (view, it))
            continue
        when = _when(it.get("when"))
        if when is None:
            problems.append("%s: a character lesson names what he is doing (when): %.60r" % (view, it))
            continue
        claims.append({"view": view, "kind": it.get("kind"), "move": it.get("move"), "range": _cond(it.get("range")),
                       "when": when, "why": it.get("why", "")})
    return claims, problems
