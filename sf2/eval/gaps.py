"""Where the learning loop loses: every link measured from one session's game log, by opponent
(scripts/report.py gaps).

  1 SEE      laya-vision's rating vs what really happened (is "likely works" really likely?)
  2 CHOOSE   how System 1 decided: advice (soft/hard), vision's best, or walk in; text laya's faithfulness
  3 ADVICE   when she followed an advised move, did it land more than vision's own picks?
  4 WALK     what walking in costs: hit while walking, and how often a walk-in ends in a hit on her
  5 DEFEND   her own bar: hits taken per round, blocks, and what she was doing when hit
  6 LESSONS  Qwen's claims while a lesson was in play vs what the lesson's move did then
"""
import collections
import re
from typing import Dict, List

from ..system1.advice import FAILS, MAY, WORKS, parse


def pct(a: int, b: int) -> str:
    return "%3.0f%% (%d/%d)" % (100 * a / b, a, b) if b else "   - (0/0)"


def see(acts: List[Dict]) -> List[str]:
    lines: List[str] = []
    lines.append("1 SEE - laya-vision's rating of the move she did vs what happened")
    by = collections.defaultdict(list)
    for a in acts:
        if a["kind"] == "attack" and a.get("shortlist") and a["action"] in a["shortlist"]:
            by[a["shortlist"][a["action"]]].append(a["actual"] == "hit")
    for r in (WORKS, MAY, FAILS):
        lines.append("    rated %-13s hit %s" % (r, pct(sum(by[r]), len(by[r]))))
    return lines


def choose(acts: List[Dict]) -> List[str]:
    lines: List[str] = []
    lines.append("2 CHOOSE - what decided each move (text laya's rule label) and faithfulness")
    ad = [a for a in acts if "rule" in a]
    c = collections.Counter(a["rule"] for a in ad)
    lines.append("    " + "  ".join("%s %s" % (k, pct(v, len(ad))) for k, v in c.most_common()))
    bad = collections.Counter(a["rule"] for a in ad if not a["follows_rule"])
    lines.append("    text laya broke the rule %s; by rule: %s" % (pct(sum(bad.values()), len(ad)), dict(bad)))
    neg = [a for a in ad if any(parse(t, list(a["shortlist"])).polarity == "neg" and
                                parse(t, list(a["shortlist"])).move == a["action"] and
                                parse(t, list(a["shortlist"])).applies(a["range"], _doing(a))
                                for t in _lessons(a))]
    lines.append("    picked a move an applying 'avoid' lesson ruled out: %s" % pct(len(neg), len(ad)))
    return lines


def _lessons(a: Dict) -> List[str]:
    text = a.get("advice_text", "")
    body = text.split("Advice: ", 1)[1].rstrip(".") if "Advice: " in text else ""
    return [] if body in ("none", "") else [t.strip() for t in body.split(";")]


def _doing(a: Dict) -> str:
    m = re.search(r"and (\w+)\.", a.get("advice_text", ""))
    return m.group(1) if m else "standing"


def advice(acts: List[Dict]) -> List[str]:
    lines: List[str] = []
    lines.append("3 ADVICE - attacks chosen by advice vs by laya-vision's rating")
    for rule in ("soft", "hard", "vision"):
        xs = [a for a in acts if a.get("rule") == rule and a["kind"] == "attack" and a["action"] in a.get("rule_answers", [])]
        hit = sum(a["actual"] == "hit" for a in xs)
        pun = sum(a["i_was_hit"] for a in xs)
        lines.append("    %-7s attacks: hit %s  punished %s  dealt/try %.1f  taken/try %.1f" % (
            rule, pct(hit, len(xs)), pct(pun, len(xs)), sum(a["dealt"] for a in xs) / max(1, len(xs)),
            sum(a["taken"] for a in xs) / max(1, len(xs))))
    return lines


def walk(acts: List[Dict]) -> List[str]:
    lines: List[str] = []
    lines.append("4 WALK - walking in")
    w = [a for a in acts if a["action"] == "forward"]
    lines.append("    walks %s of decisions; hit while walking %s; damage taken on walks %d of %d total" % (
        pct(len(w), len(acts)), pct(sum(a["i_was_hit"] for a in w), len(w)), sum(a["taken"] for a in w),
        sum(a["taken"] for a in acts)))
    rng = collections.Counter(a["range"] for a in w)
    lines.append("    walks by range: %s" % dict(rng))
    return lines


def defend(acts: List[Dict], rounds: List[Dict]) -> List[str]:
    lines: List[str] = []
    lines.append("5 DEFEND - her own bar")
    hit = [a for a in acts if a["i_was_hit"]]
    lines.append("    hit %.1f times per round; blocks chosen %d; opponent attacked in %s of decisions" % (
        len(hit) / max(1, len(rounds)), sum(a["kind"] == "defense" for a in acts),
        pct(sum(a["opp_attacked"] for a in acts), len(acts))))
    lines.append("    when hit she was doing: %s" % dict(collections.Counter(a["kind"] for a in hit)))
    ratings = collections.Counter(a["shortlist"].get(b) for a in acts if a.get("shortlist") and a["opp_attacked"]
                                  for b in ("block_high", "block_low") if b in a["shortlist"])
    lines.append("    blocks on the shortlist while he attacked: %s" % dict(ratings))
    return lines


def lessons(acts: List[Dict]) -> List[str]:
    lines: List[str] = []
    lines.append("6 LESSONS - each advised move used where its lesson applied: what it did")
    reach = [a for a in acts if _lessons(a)]
    applies = [a for a in reach if any(parse(t, list(a.get("shortlist", {})) + [a["action"]]).move and
                                       parse(t, list(a.get("shortlist", {})) + [a["action"]]).applies(a["range"], _doing(a))
                                       for t in _lessons(a))]
    lines.append("    decisions with any lesson whose condition held: %s" % pct(len(applies), len(acts)))
    seen = collections.defaultdict(lambda: [0, 0, 0])      # (polarity, move, range) -> tries, hits, punished
    for a in acts:
        for t in _lessons(a):
            les = parse(t, list(a.get("shortlist", {})) + [a["action"]])
            if les.move and les.move == a["action"] and les.applies(a["range"], _doing(a)):
                k = (les.polarity, les.move, les.where or "any")
                s = seen[k]
                s[0] += 1
                s[1] += a["actual"] == "hit"
                s[2] += a["i_was_hit"]
    for (pol, move, where), (n, h, p) in sorted(seen.items(), key=lambda kv: -kv[1][0])[:10]:
        lines.append("    %-4s %-20s %-5s used %4d  hit %s  punished %s" % (pol, move, where, n, pct(h, n), pct(p, n)))
    return lines


def ab(rounds: List[Dict], games: List[Dict]) -> List[str]:
    """The A/B test (learn_loop --ab-advice): the same loop with and without the short memory's advice."""
    lines: List[str] = []
    if not any("advice" in r for r in rounds):
        return []
    lines.append("\n===== A/B: advice on vs off (same opponents, alternate games) =====")
    for opp in ["all"] + sorted({r["opp"] for r in rounds}):
        line = []
        for arm in ("on", "off"):
            rs = [r for r in rounds if r.get("advice") == arm and opp in ("all", r["opp"])]
            gs = [g for g in games if g.get("advice") == arm and opp in ("all", g["opp"])]
            line.append("%s: rounds won %s, games won %s, dealt %.0f taken %.0f per round" % (
                arm, pct(sum(r["result"] == "win" for r in rs), len(rs)), pct(sum(g["result"] == "win" for g in gs),
                                                                               len(gs)),
                sum(r["dealt"] for r in rs) / max(1, len(rs)), sum(r["taken"] for r in rs) / max(1, len(rs))))
        lines.append("  %-8s %s\n           %s" % (opp, line[0], line[1]))
    return lines


def report(session: str, acts: List[Dict], rounds: List[Dict], games: List[Dict]) -> List[str]:
    lines = ["session %s: %d rounds, %d decisions" % (session, len(rounds), len(acts))]
    for opp in ["all"] + sorted({a["opp"] for a in acts}):
        a = acts if opp == "all" else [x for x in acts if x["opp"] == opp]
        r = rounds if opp == "all" else [x for x in rounds if x["opp"] == opp]
        lines.append("\n===== %s: %d rounds, won %d =====" % (opp, len(r), sum(x["result"] == "win" for x in r)))
        for f in (see, choose, advice, walk):
            lines += f(a)
        lines += defend(a, r)
        if opp == "all":
            lines += lessons(a)
    return lines + ab(rounds, games)
