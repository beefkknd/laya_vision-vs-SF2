"""Qwen's character prompt in the lesson loop: one opponent, one moment. After each game:

    his threats    what he does that hurts her (jumps in / attacks on the ground / hits her from afar), at which range
                   of hers, and after which of her moves: damage and share, all games against him
    if you see X   per situation of his (what he is doing when she decides, and the range), her answers ranked: net per
                   decision vs her other moves there (works / fails / unclear), and the moves she never tried there
    record         rounds won and lost, damage per round (sf2.system2.lesson_prompt.record)

Qwen answers {"answer": claim, "stop": claim} (+ "what_if" when stuck); every claim names what he is doing ("when").
Code verifies exactly as in the two-view prompt (sf2.system2.lessons). Her moves for lessons (the list and "never
tried there") leave out forward, which the verifier refuses (2026-09-30); ``forward_lessons=True``: the old prompt.

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
from .lesson_prompt import DEFENSIVE, STREAK, WHAT_IF, _registry, _renderable, _when, lesson_moves, record

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
             track: Optional[Dict] = None, fgc: bool = False, forward_lessons: bool = False) -> List[Dict]:
    """Same signature as sf2.system2.lesson_prompt.messages; ``rows`` = all her decisions against him so far.
    ``fgc``: situations named the way players do, and a primer on the game and this opponent (``messages_fgc``).
    ``forward_lessons``: forward still offered for lessons (the prompt before 2026-09-30)."""
    moves = lesson_moves(list(moves) or sorted({a["action"] for a in rows}), forward_lessons)
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


# ===================================================================== two-stage Qwen (System 2): Scout + Coach
# The single prompt above (SYSTEM / messages) did two jobs at once (observe + strategize) and defaulted to the safe
# one - it kept adding blocks and turtled (docs/plan_two_stage_qwen.md). The split: Stage 1 SCOUT summarizes the game
# (grounded, temperature 0); Stage 2 COACH changes the rules (creative, higher temperature) in one of two modes decided
# from the trend. The DIGEST below is computed in CODE from the evidence (faithful to the rows, gradable), and the
# Scout's prose is attached to it as a narrative; the Coach consumes the digest. parse_claims is unchanged, so a
# proposed rule stays followable and verified exactly as before.

DOMINANT = 3           # her most-used moves named in the digest
DIGEST_THREATS = 4     # his threats carried into the digest
RECENT_GAMES = 3       # games of win/loss the verdict reads


def _kind_of(a: Dict) -> str:
    return a.get("kind", "attack")


def digest_facts(me: str, opp: str, reg: L.Registry, rows: Sequence[Dict], last: Sequence[Dict],
                 all_rounds: Sequence[Dict], last_rounds: Sequence[Dict], game_hp: Sequence[float],
                 games_wl: Sequence[Dict], moves: Sequence[str] = ()) -> Dict:
    """The per-game DIGEST, computed in code from the evidence so it is faithful to the rows (the Scout is graded on
    matching it). ``last`` = this game's decisions, ``rows`` = all games so far. Fields: her dominant actions, the
    offense she actually used and what landed, total dealt/taken this game, his top threats (all games), how each
    in-play rule tracked (fired N, net up/down, still good?), a winning/losing/stable verdict from the recent games,
    and every move she has ALREADY TRIED (so the Coach does not repeat it)."""
    game_rows = list(last) or list(rows)
    actions = collections.Counter(a["action"] for a in game_rows)
    attacks = [a for a in game_rows if _kind_of(a) == "attack"]
    offense_used = sorted({a["action"] for a in attacks})
    offense_landed = sorted({a["action"] for a in attacks if a.get("dealt")})
    dealt = sum(a.get("dealt", 0) for a in game_rows)
    taken = sum(a.get("taken", 0) for a in game_rows)
    in_play = set(L.in_play(reg))
    rule_tracking = []
    for r in reg:
        if r["line"] not in in_play:
            continue
        ev = L.condition_evidence(rows, r["claim"])
        rule_tracking.append({"line": r["line"], "state": r["state"], "fired": ev["tries"],
                              "net": round(ev["net"], 1), "tracked": "up" if ev["net"] >= 0 else "down",
                              "good": ev["cls"] == L.RIGHT[r["claim"]["kind"]]})
    recent = list(games_wl)[-RECENT_GAMES:]
    won = sum(int(g.get("won", 0)) for g in recent)
    lost = sum(int(g.get("lost", 0)) for g in recent)
    hp_recent = list(game_hp)[-RECENT_GAMES:]
    verdict = "losing" if lost > won else "winning" if won > lost else (
        "losing" if hp_recent and sum(hp_recent) < 0 else "winning" if hp_recent and sum(hp_recent) > 0 else "stable")
    cells = collections.Counter((a["range"], opp_doing(a)) for a in game_rows if a.get("range"))
    n_cells = sum(cells.values()) or 1
    his_cells = [["%s/%s" % (r, d), n, round(100 * n / n_cells)] for (r, d), n in cells.most_common(6)]
    return {"opp": opp, "dominant": [[m, n] for m, n in actions.most_common(DOMINANT)],
            "offense_used": offense_used, "offense_landed": offense_landed, "his_cells": his_cells,
            "dealt": dealt, "taken": taken, "threats": threats(rows)[:DIGEST_THREATS],
            "rule_tracking": rule_tracking, "verdict": verdict,
            "won_recent": won, "lost_recent": lost, "hp_recent": round(sum(hp_recent) / len(hp_recent), 1) if hp_recent
            else 0.0, "already_tried": sorted({a["action"] for a in rows})}


def _render_digest(d: Dict) -> str:
    """The digest as compact lines for a prompt (the Scout faithfully, the Coach to act on)."""
    rules = "\n".join("  - %s [%s]: fired %d, net %+.1f (%s), %s" % (
        t["line"], t["state"], t["fired"], t["net"], t["tracked"],
        "still works" if t["good"] else "no longer clearly working") for t in d["rule_tracking"]) or "  (none)"
    return "\n".join([
        "verdict: %s (last games won %d / lost %d, %+.1f hp per round)" % (
            d["verdict"], d["won_recent"], d["lost_recent"], d["hp_recent"]),
        "dealt this game: %d; taken: %d" % (d["dealt"], d["taken"]),
        "HIS COMMON SITUATIONS (range/state, count, %% of decisions - key each rule to one of these so it FIRES): %s" % (
            ", ".join("%s x%d (%d%%)" % (c, n, p) for c, n, p in d.get("his_cells", [])) or "(none)"),
        "her dominant moves: %s" % (", ".join("%s x%d" % (m, n) for m, n in d["dominant"]) or "(none)"),
        "offense she used: %s" % (", ".join(d["offense_used"]) or "(none)"),
        "offense that landed: %s" % (", ".join(d["offense_landed"]) or "(none)"),
        "moves already tried (do not treat as new): %s" % (", ".join(d["already_tried"]) or "(none)"),
        "how the current rules tracked:\n%s" % rules])


SCOUT_SYSTEM = """You are the SCOUT for {me} in Street Fighter II against {opp}. Your ONE job is to summarize the game
that just happened - faithfully, with no strategy and no advice. Do not propose rule changes; a coach does that next.

Report, in a few short sentences: her dominant actions, which offense she actually used and whether it landed, the
damage she dealt and took, {opp}'s key patterns and threats, how the current rules fired and whether they tracked
damage up or down, and whether she is winning or losing. Stick to what the measured facts below show - add nothing
they do not support."""


def summary_messages(me: str, opp: str, reg: L.Registry, rows: Sequence[Dict], last: Sequence[Dict],
                     all_rounds: Sequence[Dict], last_rounds: Sequence[Dict], game_hp: Sequence[float],
                     games_wl: Sequence[Dict], moves: Sequence[str] = ()) -> List[Dict]:
    """Stage 1 SCOUT prompt (temperature 0): the evidence plus the code-computed digest, asking for a faithful prose
    summary. ``reg`` and ``moves`` are added to the plan's bare signature so the rule-tracking and move substrate are
    available; the rest matches the plan's summary_messages(me, opp, rows, last, all_rounds, last_rounds, game_hp,
    games_wl)."""
    moves = lesson_moves(list(moves) or sorted({a["action"] for a in rows}))
    digest = digest_facts(me, opp, reg, rows, last, all_rounds, last_rounds, game_hp, games_wl, moves)
    parts = [
        record(all_rounds, last_rounds),
        "HIS THREATS - what he did that hurt her, all games:\n%s" % ("\n".join(threats(rows)) or "(none)"),
        "IF YOU SEE - her answers in each of his situations, all games:\n\n%s" % (
            "\n\n".join(if_you_see(rows, moves)) or "(not enough yet)"),
        "The measured facts of THIS game (summarize these faithfully, add nothing):\n%s" % _render_digest(digest)]
    return [{"role": "system", "content": SCOUT_SYSTEM.format(me=me, opp=opp)},
            {"role": "user", "content": "\n\n".join(parts)}]


# The Coach's answer grammar and JSON are the SAME as the single prompt's, so parse_claims is unchanged.
_GRAMMAR = """A lesson is one line: "use more", "always" or "avoid" one of her moves, at one range (close, mid, far) or
anywhere, and ALWAYS with what he is doing: jumping, crouching, attacking, standing or stunned. His specials look like
any attack, so say them by where they happen: a fireball is "when he attacks far away" or at mid range; an uppercut or
a jump-in kick close by is "when he jumps" or "when he attacks up close".

Her moves: {moves}. A move she has never used there can still be proposed as "use more" or "always" (it is tried in
play); "avoid" needs a move she has used. Do not repeat or contradict a rule already in play (a narrower exception is
fine)."""
# A plain string (joined, not .format()ted), so its braces are single, exactly the object parse_claims reads.
_ANSWER_JSON = """Answer with JSON only (null for a side with nothing worth proposing):
{"answer": {"kind": "use_more|always", "move": "...", "range": "close|mid|far|null", "when": "jumping|crouching|attacking|standing|stunned", "why": "one short sentence"},
 "stop": {"kind": "avoid", "move": "...", "range": "close|mid|far|null", "when": "jumping|crouching|attacking|standing|stunned", "why": "one short sentence"}}"""

STRATEGIZE_HEAD = """You are the COACH for {me} in Street Fighter II against {opp}. A scout has already summarized the
last game: her dominant moves, the offense she actually used, the damage she dealt and took, his threats, how the
rules in play tracked, and whether she is winning or losing. Your job is to CHANGE her rules - not to re-summarize.
Propose at most two lessons: an ANSWER (what she should do) and a STOP (what she should stop doing).
Every rule MUST key to one of HIS COMMON SITUATIONS the scout lists (a range+state that actually OCCURS, so the rule
FIRES often) and name a GROUNDED STANDING move - a standing normal (s.*), a special, a throw, or movement. A crouch
normal (c.*) or a jump attack (j./jf.) voids to block and never fires (she is not reliably crouching or airborne on
his cue)."""
CONSOLIDATE = """She is WINNING or STABLE. Consolidate what works: keep and sharpen the rules that are tracking good
outcomes, and STOP ("avoid") a move that only loses her hit points. Do not pile on new defense - sharpen, do not
sprawl."""
ESCALATE = """She is LOSING, and she is losing by turtling - blocking and backing off. You are FORBIDDEN from proposing
a block or any other defensive move ({defensive}) as the ANSWER; "block more" is not an answer here.

Give her OFFENSE that applies in COMMON situations - the ones that happen every round - so the rule FIRES OFTEN. A rule
keyed on a rare or narrow moment almost never fires and she falls back to block. Prefer a condition that is range-only
or a common state of his (he attacks, he stands), NOT a rare one. Good, broad shapes:
- a baseline poke or approach at mid range to control space and close the distance (range-only is fine),
- MOVEMENT to fix her spacing - she can now reliably walk: "walk_forward" to CLOSE IN on a passive opponent (e.g.
  walk_forward when he stands), "walk_back" to make space when he attacks. Movement is a valid GROUNDED answer; use it
  to reach the range where her offense actually lands, then punish.
- a punish when he attacks - hit him out of or right after his attack,
- an anti-air when he jumps that she can do while GROUNDED: an uppercut / Shoryuken (e.g. shoryuken_hp for Ryu/Ken), a
  standing or crouching heavy, or Chun-Li's spinning_bird_kick or lightning_legs.
Do NOT propose a jump attack (a "j." or "jf." move) as an answer: those need her to ALREADY be airborne, but the "when"
is about HIS state, so she can almost never be in the air on his cue - she would fall back to block. For an anti-air,
name a GROUNDED move, not a jump attack.

She has ALREADY TRIED these moves - pick one she is NOT leaning on, or a common new situation for one of them: {tried}.
The STOP side should drop a defensive habit she overuses."""


def coach_mode(digest: Dict) -> str:
    """escalate when she is losing (find offense), else consolidate (promote what works)."""
    return "escalate" if digest.get("verdict") == "losing" else "consolidate"


def strategize_messages(me: str, opp: str, reg: L.Registry, digest: Dict, moves: Sequence[str] = (),
                        refused: Sequence[Dict] = (), mode: Optional[str] = None) -> List[Dict]:
    """Stage 2 COACH prompt (higher temperature): the Scout digest + the rules in play + the trend, in one of two
    modes. ``mode`` defaults to ``coach_mode(digest)``."""
    mode = mode or coach_mode(digest)
    moves = lesson_moves(list(moves) or digest.get("already_tried", []))
    block = ESCALATE.format(defensive=", ".join(DEFENSIVE), tried=", ".join(digest.get("already_tried", [])) or "(none)") \
        if mode == "escalate" else CONSOLIDATE
    system = "\n\n".join([STRATEGIZE_HEAD.format(me=me, opp=opp), block,
                          _GRAMMAR.format(moves=", ".join(moves)), _ANSWER_JSON])
    parts = ["The scout's summary of the last game:\n%s" % (digest.get("notes") or "(no scout notes)"),
             "The measured facts behind it:\n%s" % _render_digest(digest),
             "Rules in play now:\n%s" % _registry(reg)]
    if refused:
        parts.append("Refused last time (do not propose again):\n%s" % "\n".join(
            "- %s: %s" % (L.render(o["claim"]) if _renderable(o["claim"]) else o["claim"], o["why"]) for o in refused))
    return [{"role": "system", "content": system}, {"role": "user", "content": "\n\n".join(parts)}]


# Jump-attack prefixes (air stance, sf2.system1.advice.STANCE_PREFIXES["air"]): she must already be airborne to use one.
JUMP_PREFIXES = ("j.", "jf.")


def _is_jump_attack(move) -> bool:
    """A jump attack (an air normal: "j.*" / "jf.*"). The lesson grammar's "when" names what HE is doing (jumping,
    attacking, ...), never that SHE is airborne, so a jump attack keyed on his state/range can almost never be done on
    cue (she is grounded when she decides) - it is stance-invalid and she falls back to block."""
    return isinstance(move, str) and move.startswith(JUMP_PREFIXES)


def coach_filter(claims: Sequence[Dict], mode: str) -> Tuple[List[Dict], List[str]]:
    """Mechanically enforce stance validity and the escalate rule. Returns (kept claims, dropped-reason strings); the
    reasons are logged in the trace (scripts/play_loop_screen.py folds them into ``problems``).
    [SCRIPT] STANCE (both modes): a move whose stance she can't reliably be in ON HIS CUE is DROPPED - a JUMP attack
    ("j."/"jf.*", she can't be airborne) AND a CROUCH normal ("c.*", she is not reliably crouching when a lesson keyed
    on his state fires; "c.mk when he stands" voided to block in play). Prefer a grounded STANDING move (s.*).
    [SCRIPT] ESCALATE: when she is losing by turtling, a new DEFENSIVE "use more"/"always" answer is dropped (she is
    already losing by blocking); an "avoid" of a defensive move is kept (it removes defense). Consolidate keeps blocks.
    (A move not in char_menu_moves(me) is refused downstream in lessons.propose - left there, not duplicated here.)"""
    kept, dropped = [], []
    for c in claims:
        move = c.get("move")
        if _is_jump_attack(move):
            dropped.append("stance: %s is a jump attack - she must be airborne, but the lesson is keyed on his "
                           "state/range, so she cannot do it on cue (use a grounded move instead)" % move)
            continue
        if isinstance(move, str) and move.startswith("c."):            # crouch normal (c.lp/c.mk/c.hk...): stance-unreliable
            dropped.append("stance: %s is a crouch normal - she is not reliably crouching on his cue (the lesson is "
                           "keyed on his state/range, not her own crouch), so it voids to block like a jump attack; "
                           "use a standing s.* move instead" % move)
            continue
        if mode == "escalate" and c.get("kind") in ("use_more", "always") and c.get("move") in DEFENSIVE:
            dropped.append("escalate: she is losing by blocking, a new defensive answer (%s) is forbidden" % c.get("move"))
            continue
        kept.append(c)
    return kept, dropped


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
