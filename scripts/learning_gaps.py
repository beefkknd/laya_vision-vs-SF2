"""Where the learning loop loses: every link measured from one session's game log, by opponent.

    python scripts/learning_gaps.py                                  # the newest chunli session
    python scripts/learning_gaps.py --session rollouts/learn/chunli/20260928-062237

  1 SEE      laya-vision's rating vs what really happened (is "likely works" really likely?)
  2 CHOOSE   how System 1 decided: advice (soft/hard), vision's best, or walk in; text laya's faithfulness
  3 ADVICE   when she followed an advised move, did it land more than vision's own picks?
  4 WALK     what walking in costs: hit while walking, and how often a walk-in ends in a hit on her
  5 DEFEND   her own bar: hits taken per round, blocks, and what she was doing when hit
  6 LESSONS  Qwen's claims while a lesson was in play vs what the lesson's move did then
Log: logs/gaps_<me>.log
"""
import argparse
import collections
import glob
import os
import re
from typing import Dict, List

import _path  # noqa: F401
from sf2.advice import FAILS, MAY, WORKS, parse
from sf2.dataset import read

LOG_LINES: List[str] = []


def out(line: str = "") -> None:
    print(line)
    LOG_LINES.append(line)


def pct(a: int, b: int) -> str:
    return "%3.0f%% (%d/%d)" % (100 * a / b, a, b) if b else "   - (0/0)"


def see(acts: List[Dict]) -> None:
    out("1 SEE - laya-vision's rating of the move she did vs what happened")
    by = collections.defaultdict(list)
    for a in acts:
        if a["kind"] == "attack" and a.get("shortlist") and a["action"] in a["shortlist"]:
            by[a["shortlist"][a["action"]]].append(a["actual"] == "hit")
    for r in (WORKS, MAY, FAILS):
        out("    rated %-13s hit %s" % (r, pct(sum(by[r]), len(by[r]))))


def choose(acts: List[Dict]) -> None:
    out("2 CHOOSE - what decided each move (text laya's rule label) and faithfulness")
    ad = [a for a in acts if "rule" in a]
    c = collections.Counter(a["rule"] for a in ad)
    out("    " + "  ".join("%s %s" % (k, pct(v, len(ad))) for k, v in c.most_common()))
    bad = collections.Counter(a["rule"] for a in ad if not a["follows_rule"])
    out("    text laya broke the rule %s; by rule: %s" % (pct(sum(bad.values()), len(ad)), dict(bad)))
    neg = [a for a in ad if any(parse(t, list(a["shortlist"])).polarity == "neg" and
                                parse(t, list(a["shortlist"])).move == a["action"] and
                                parse(t, list(a["shortlist"])).applies(a["range"], _doing(a))
                                for t in _lessons(a))]
    out("    picked a move an applying 'avoid' lesson ruled out: %s" % pct(len(neg), len(ad)))


def _lessons(a: Dict) -> List[str]:
    text = a.get("advice_text", "")
    body = text.split("Advice: ", 1)[1].rstrip(".") if "Advice: " in text else ""
    return [] if body in ("none", "") else [t.strip() for t in body.split(";")]


def _doing(a: Dict) -> str:
    m = re.search(r"and (\w+)\.", a.get("advice_text", ""))
    return m.group(1) if m else "standing"


def advice(acts: List[Dict]) -> None:
    out("3 ADVICE - attacks chosen by advice vs by laya-vision's rating")
    for rule in ("soft", "hard", "vision"):
        xs = [a for a in acts if a.get("rule") == rule and a["kind"] == "attack" and a["action"] in a.get("rule_answers", [])]
        hit = sum(a["actual"] == "hit" for a in xs)
        pun = sum(a["i_was_hit"] for a in xs)
        out("    %-7s attacks: hit %s  punished %s  dealt/try %.1f  taken/try %.1f" % (
            rule, pct(hit, len(xs)), pct(pun, len(xs)), sum(a["dealt"] for a in xs) / max(1, len(xs)),
            sum(a["taken"] for a in xs) / max(1, len(xs))))


def walk(acts: List[Dict]) -> None:
    out("4 WALK - walking in")
    w = [a for a in acts if a["action"] == "forward"]
    out("    walks %s of decisions; hit while walking %s; damage taken on walks %d of %d total" % (
        pct(len(w), len(acts)), pct(sum(a["i_was_hit"] for a in w), len(w)), sum(a["taken"] for a in w),
        sum(a["taken"] for a in acts)))
    rng = collections.Counter(a["range"] for a in w)
    out("    walks by range: %s" % dict(rng))


def defend(acts: List[Dict], rounds: List[Dict]) -> None:
    out("5 DEFEND - her own bar")
    hit = [a for a in acts if a["i_was_hit"]]
    out("    hit %.1f times per round; blocks chosen %d; opponent attacked in %s of decisions" % (
        len(hit) / max(1, len(rounds)), sum(a["kind"] == "defense" for a in acts),
        pct(sum(a["opp_attacked"] for a in acts), len(acts))))
    out("    when hit she was doing: %s" % dict(collections.Counter(a["kind"] for a in hit)))
    ratings = collections.Counter(a["shortlist"].get(b) for a in acts if a.get("shortlist") and a["opp_attacked"]
                                  for b in ("block_high", "block_low") if b in a["shortlist"])
    out("    blocks on the shortlist while he attacked: %s" % dict(ratings))


def lessons(acts: List[Dict]) -> None:
    out("6 LESSONS - each advised move used where its lesson applied: what it did")
    reach = [a for a in acts if _lessons(a)]
    applies = [a for a in reach if any(parse(t, list(a.get("shortlist", {})) + [a["action"]]).move and
                                       parse(t, list(a.get("shortlist", {})) + [a["action"]]).applies(a["range"], _doing(a))
                                       for t in _lessons(a))]
    out("    decisions with any lesson whose condition held: %s" % pct(len(applies), len(acts)))
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
        out("    %-4s %-20s %-5s used %4d  hit %s  punished %s" % (pol, move, where, n, pct(h, n), pct(p, n)))


def ab(rounds: List[Dict], games: List[Dict]) -> None:
    """The A/B test (learn_loop --ab-advice): the same loop with and without the short memory's advice."""
    if not any("advice" in r for r in rounds):
        return
    out("\n===== A/B: advice on vs off (same opponents, alternate games) =====")
    for opp in ["all"] + sorted({r["opp"] for r in rounds}):
        line = []
        for arm in ("on", "off"):
            rs = [r for r in rounds if r.get("advice") == arm and opp in ("all", r["opp"])]
            gs = [g for g in games if g.get("advice") == arm and opp in ("all", g["opp"])]
            line.append("%s: rounds won %s, games won %s, dealt %.0f taken %.0f per round" % (
                arm, pct(sum(r["result"] == "win" for r in rs), len(rs)), pct(sum(g["result"] == "win" for g in gs),
                                                                               len(gs)),
                sum(r["dealt"] for r in rs) / max(1, len(rs)), sum(r["taken"] for r in rs) / max(1, len(rs))))
        out("  %-8s %s\n           %s" % (opp, line[0], line[1]))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--char", default="chunli")
    ap.add_argument("--session", default=None)
    args = ap.parse_args()
    session = args.session or sorted(glob.glob("rollouts/learn/%s/*" % args.char))[-1]
    acts, rounds, games = (read(os.path.join(session, k + ".jsonl")) for k in ("actions", "rounds", "games"))
    out("session %s: %d rounds, %d decisions" % (session, len(rounds), len(acts)))
    for opp in ["all"] + sorted({a["opp"] for a in acts}):
        a = acts if opp == "all" else [x for x in acts if x["opp"] == opp]
        r = rounds if opp == "all" else [x for x in rounds if x["opp"] == opp]
        out("\n===== %s: %d rounds, won %d =====" % (opp, len(r), sum(x["result"] == "win" for x in r)))
        for f in (see, choose, advice, walk):
            f(a)
        defend(a, r)
        if opp == "all":
            lessons(a)
    ab(rounds, games)
    os.makedirs("logs", exist_ok=True)
    with open("logs/gaps_%s.log" % args.char, "w") as f:
        f.write("\n".join(LOG_LINES) + "\n")


if __name__ == "__main__":
    main()
