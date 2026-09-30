"""Where does an advice line's chain break? One line L (her move M, its polarity, its range / what he is doing) traced
through every component, each judged on its own data (scripts/boundary.py loads it):

    V  laya-vision   in the no-advice arm, where L applies: is M in laya-vision's top 3, rated how (``advice.rating``)
                     - would vision offer M on its own
    R  label rule    in the arm with L in play, where L applies: do the rule's answers (recomputed from the logged
                     shortlist and the arm's lines, ``advice.answers``) contain M (for an avoid line: rule M out where it
                     is offered); if not, why (not offered, ruled out, a hard lesson overrides, soft vs "likely fails").
                     The logged ``rule_answers`` are rechecked against the recomputation (a harness fault if they differ)
    T  text laya     same rows: follows_rule, and does the pick == M when the rule says M
    G  the game      M's net per try vs her other decisions there (``lessons.condition_evidence``), in the arm and in
                     no advice; how often she picks M; the round-level A/B (seed as the unit) of the arm whose lines are
                     exactly [L]
    Q  Qwen          over the opponent's loop ledgers: proposals of exactly L (and the verifier's state / why), claims
                     naming M by kind; what her history showed for L (``condition_evidence``: tries, cls)
    H  verifier      does the verifier's judgement of Qwen's L agree with the A/B (helps -> registered, hurts ->
                     rejected)

``diagnose`` names the breaks in chain order; the first is the diagnosis. Pure functions: no files, no games.
"""
import re
from collections import Counter
from typing import Dict, List, Optional, Sequence

from ..system1.advice import FAILS, FORWARD, Lesson, answers, opp_doing, parse, rating, read
from ..system2.lessons import condition_evidence
from .stats import paired, run_level

__all__ = ["rating", "lesson_of", "claim_of", "stage_vision", "stage_rule", "stage_text", "stage_game", "round_ab",
           "stage_qwen", "stage_verifier", "diagnose", "trace"]

KIND = {"soft": "use_more", "hard": "always", "neg": "avoid"}
POLARITY = {k: p for p, k in KIND.items()}
EXPECT = {"HELPS": "registered", "HURTS": "rejected"}      # what the verifier should have said, given the A/B
DECISIVE = ("registered", "rejected", "retired")
VISION_MIN = 0.10      # M in laya-vision's top 3 less often than this where L applies: vision rarely offers it
RULE_MIN = 0.50        # the rule says M (or rules it out, for avoid) less often than this: the rule blocks the line
TEXT_MIN = 0.80        # text laya follows the rule / picks M when told less often than this


def _share(n: int, d: int) -> Optional[float]:
    return n / d if d else None


def _pct(x: Optional[float]) -> str:
    return "n/a" if x is None else "%.0f%%" % (100 * x)


def lesson_of(line: str, moves: Sequence[str]) -> Lesson:
    """The line as text laya's rule reads it (``advice.parse``: a line it cannot read is an error here)."""
    return parse(line, list(moves) + ([FORWARD] if FORWARD not in moves else []))


def claim_of(les: Lesson) -> Dict:
    return {"kind": KIND.get(les.polarity), "move": les.move, "range": les.where, "when": les.when}


def _applies(rows: Sequence[Dict], les: Lesson) -> List[Dict]:
    return [r for r in rows if les.applies(r["range"], opp_doing(r))]


def _offered(r: Dict, move: str) -> bool:
    return move == FORWARD or move in r["shortlist"]


def stage_vision(none_rows: Sequence[Dict], les: Lesson) -> Dict:
    """V: laya-vision with no advice, where L applies."""
    here = _applies(none_rows, les)
    m = les.move
    scores = [dict((mv, s) for mv, s in r["top3"]).get(m) for r in here]
    ratings = Counter(rating(s, m) for s in scores if s is not None)
    top3 = sum(s is not None for s in scores)
    offered = sum(_offered(r, m) for r in here)
    best = sum(m in answers(r["range"], opp_doing(r), r["shortlist"], [])[0] for r in here)
    picked = sum(r["action"] == m for r in here)
    n = len(here)
    return {"decisions": n, "top3": top3, "top3_share": _share(top3, n), "offered": offered,
            "offered_share": _share(offered, n), "ratings": dict(ratings), "vision_best": best,
            "vision_best_share": _share(best, n), "picked": picked, "picked_share": _share(picked, n)}


def _why_not(r: Dict, les: Lesson, lessons: Sequence[Lesson], rule: str) -> str:
    m, rng, doing = les.move, r["range"], opp_doing(r)
    options = r["shortlist"]
    if m not in options:
        return "not offered"
    if any(x.polarity == "neg" and x.move == m and x.applies(rng, doing) for x in lessons):
        return "ruled out"
    if rule == "hard":
        return "hard lesson overrides"
    if les.polarity == "soft" and options[m] == FAILS:
        return "likely fails"
    return "other (%s)" % rule


def _rule_rows(rows: Sequence[Dict], les: Lesson, lines: Sequence[str], moves: Sequence[str]) -> List[Dict]:
    """Each applying row with the rule's answers recomputed."""
    lessons = [read(t, list(moves) + [FORWARD]) for t in lines]
    out = []
    for r in _applies(rows, les):
        ans, rule = answers(r["range"], opp_doing(r), r["shortlist"], lessons)
        out.append({"row": r, "answers": ans, "rule": rule, "lessons": lessons})
    return out


def _says(x: Dict, les: Lesson) -> bool:
    """The rule does what L asks: M among the answers (use more / always), M not an answer (avoid: the shortlist
    already drops a move an applying avoid rules out, ``advisor.shortlist``)."""
    if les.polarity == "neg":
        return les.move not in x["answers"]
    return les.move in x["answers"]


def stage_rule(rows: Sequence[Dict], les: Lesson, lines: Sequence[str], moves: Sequence[str]) -> Dict:
    """R: the label rule in the arm with L in play, where L applies."""
    rr = _rule_rows(rows, les, lines, moves)
    m = les.move
    says = sum(_says(x, les) for x in rr)
    offered = sum(_offered(x["row"], m) for x in rr)
    if les.polarity == "neg":
        why = Counter("still an answer (%s)" % x["rule"] for x in rr if not _says(x, les))
    else:
        why = Counter(_why_not(x["row"], les, x["lessons"], x["rule"]) for x in rr if not _says(x, les))
    ratings = Counter(x["row"]["shortlist"][m] for x in rr if m != FORWARD and m in x["row"]["shortlist"])
    logged = [x for x in rr if "rule_answers" in x["row"]]
    mismatch = sum(sorted(x["row"]["rule_answers"]) != sorted(x["answers"]) for x in logged)
    return {"decisions": len(rr), "offered": offered, "says": says, "says_share": _share(says, len(rr)),
            "why_not": dict(why), "rules": dict(Counter(x["rule"] for x in rr)), "ratings": dict(ratings), "logged": len(logged), "logged_mismatch": mismatch}


def stage_text(rows: Sequence[Dict], les: Lesson, lines: Sequence[str], moves: Sequence[str]) -> Dict:
    """T: text laya in the same rows - does it follow the rule, and pick M (not M, for avoid) when the rule says so."""
    rr = _rule_rows(rows, les, lines, moves)
    follows = sum(bool(x["row"].get("follows_rule")) for x in rr)
    told = [x for x in rr if _says(x, les)]
    if les.polarity == "neg":
        picked = sum(x["row"]["action"] != les.move for x in told)
    else:
        picked = sum(x["row"]["action"] == les.move for x in told)
    return {"decisions": len(rr), "follows": follows, "follows_share": _share(follows, len(rr)),
            "rule_says": len(told), "picked": picked, "pick_share": _share(picked, len(told))}


def stage_game(arm_rows: Sequence[Dict], none_rows: Sequence[Dict], les: Lesson) -> Dict:
    """G: M's net per try vs her other decisions where L applies, with and without the advice; how often she picks M."""
    c = claim_of(les)
    a, n = _applies(arm_rows, les), _applies(none_rows, les)
    return {"arm": condition_evidence(arm_rows, c), "none": condition_evidence(none_rows, c),
            "use_arm": _share(sum(r["action"] == les.move for r in a), len(a)),
            "use_none": _share(sum(r["action"] == les.move for r in n), len(n)), "ab": None}


def round_ab(arm_rounds: Sequence[Sequence[Dict]], none_rounds: Sequence[Sequence[Dict]]) -> Dict:
    """The arm vs no advice, hp per round, paired round by round within a seed; the seed as the unit."""
    if len(arm_rounds) != len(none_rounds):
        raise ValueError("%d arm runs vs %d no-advice runs" % (len(arm_rounds), len(none_rounds)))
    return run_level([paired(a, n) for a, n in zip(arm_rounds, none_rounds)])


def _same(c: Dict, les: Lesson) -> bool:
    return (POLARITY.get(c.get("kind")) == les.polarity and c.get("move") == les.move
            and c.get("range") == les.where and c.get("when") == les.when)


def _why(text: str) -> str:
    return re.sub(r"\s*\(.*?\)", "", str(text).split(":", 1)[0]).strip()


def stage_qwen(ledgers: Sequence[Sequence[Dict]], history: Sequence[Dict], les: Lesson,
               moves: Sequence[str]) -> Dict:
    """Q: over the loop ledgers (one list of rows per run), what Qwen proposed and what the verifier said; what her
    history showed for L."""
    exact, runs = 0, 0
    states, whys, naming, final = Counter(), Counter(), Counter(), Counter()
    for ledger in ledgers:
        hit = False
        for row in ledger:
            for o in row.get("outcome") or []:
                c = o.get("claim")
                if not isinstance(c, dict) or c.get("move") != les.move:
                    continue
                naming[c.get("kind")] += 1
                if _same(c, les):
                    exact, hit = exact + 1, True
                    states[o.get("state")] += 1
                    whys[_why(o.get("why", ""))] += 1
        last = [row["registry"] for row in ledger if row.get("registry") is not None]
        for r in (last[-1] if last else []):
            if _same(r.get("claim", {}), les):
                final[r.get("state")] += 1
        runs += hit
    hist = condition_evidence(history, claim_of(les)) if les.move else None
    return {"ledgers": len(ledgers), "exact": exact, "exact_runs": runs, "exact_states": dict(states),
            "exact_whys": dict(whys), "final_states": dict(final), "naming": dict(naming), "history": hist}


def stage_verifier(q: Dict, ab: Optional[Dict]) -> Dict:
    """H: the verifier's decisive judgements of Qwen's L (registered / rejected / retired, when proposed and at the
    end of each run) against what the A/B says it should be. None: nothing to compare."""
    verdict = (ab or {}).get("verdict")
    expected = EXPECT.get(verdict)
    said = Counter()
    for st, n in list(q["exact_states"].items()) + list(q["final_states"].items()):
        if st in DECISIVE:
            said["rejected" if st == "retired" else st] += n
    agree = None if expected is None or not said else all(s == expected for s in said)
    return {"ab": verdict, "expected": expected, "states": dict(said), "agree": agree}


def _top(why: Dict) -> str:
    if not why:
        return ""
    k, n = max(why.items(), key=lambda kv: kv[1])
    return "; mostly %s (%d of %d)" % (k, n, sum(why.values()))


def _breaks(les: Lesson, s: Dict) -> List[str]:
    out = []
    m, v, r, t, g, q, h = les.move, s["V"], s["R"], s["T"], s["G"], s["Q"], s["H"]
    share = v.get("offered_share") if m == FORWARD else v.get("top3_share")
    if v.get("decisions") and share is not None and share < VISION_MIN:
        out.append("vision: %s in top 3 only %s of %d decisions where the line applies -> %s" % (
            m, _pct(share), v["decisions"],
            "avoiding it changes little" if les.polarity == "neg" else "rarely offered without advice"))
    if r.get("decisions") and r.get("says_share") is not None and r["says_share"] < RULE_MIN:
        out.append("rule: %s %s only %s%s" % ("rules out" if les.polarity == "neg" else "says", m,
                                              _pct(r["says_share"]), _top(r.get("why_not", {}))))
    fol = t.get("follows_share")
    if t.get("decisions") and ((fol is not None and fol < TEXT_MIN) or
                               (t.get("rule_says") and (t.get("pick_share") or 0) < TEXT_MIN)):
        out.append("text laya: follows the rule %s, %s %s %s when the rule says so" % (
            _pct(t.get("follows_share")), "avoids" if les.polarity == "neg" else "picks", m, _pct(t.get("pick_share"))))
    ab = g.get("ab") or {}
    if ab.get("verdict") == "HURTS":
        out.append("game: the A/B says it hurts (%+.1f hp per round)" % ab["mean"])
    elif les.polarity != "neg" and (g.get("arm") or {}).get("cls") == "worse":
        out.append("game: %s nets worse per try than her other decisions there (%d tries)" % (m, g["arm"]["tries"]))
    if q.get("exact", 0) == 0 and ab.get("verdict") in (None, "HELPS"):
        hist = q.get("history") or {}
        naming = q.get("naming") or {}
        out.append("Qwen: never proposed it; her history had %d tries (%s)%s%s" % (
            hist.get("tries", 0), hist.get("cls"),
            "; with no advice she picks %s %s there" % (m, _pct(v["picked_share"]))
            if v.get("picked_share") is not None else "",
            "; claims naming %s: %s" % (m, ", ".join("%s x%d" % kv for kv in sorted(naming.items())))
            if naming else ""))
    if h.get("agree") is False:
        out.append("verifier: judged it %s but the A/B says %s (expected %s)" % (
            ", ".join("%s x%d" % kv for kv in sorted(h["states"].items())), h.get("ab"), h.get("expected")))
    return out


def diagnose(les: Lesson, stages: Dict) -> Dict:
    """The breaks in chain order (V, R, T, G, Q, H); the first is the diagnosis."""
    br = _breaks(les, stages)
    ab = (stages["G"].get("ab") or {}).get("verdict")
    return {"breaks": br, "diagnosis": br[0] if br else "no break: the chain holds (A/B %s)" % (ab or "n/a")}


def trace(line: str, moves: Sequence[str], arms: Dict[str, Dict], ledgers: Sequence[Sequence[Dict]],
          history: Sequence[Dict]) -> Dict:
    """One line through every stage. ``arms``: name -> {"lines", "rows" (actions pooled over seeds), "rounds" (one list
    per seed, in seed order)}, with "none" the no-advice arm. The arm tracing R/T/G is the one whose lines are exactly
    [line], else the first (by name) that has it."""
    les = lesson_of(line, moves)
    with_line = sorted(a for a, d in arms.items() if line in d["lines"])
    arm = next((a for a in with_line if arms[a]["lines"] == [line]), with_line[0] if with_line else None)
    none = arms["none"]
    d = arms[arm] if arm else {"lines": [], "rows": [], "rounds": []}
    out = {"line": line, "move": les.move, "polarity": les.polarity, "where": les.where, "when": les.when,
           "arm": arm, "arms_with_line": with_line,
           "V": stage_vision(none["rows"], les),
           "R": stage_rule(d["rows"], les, d["lines"], moves),
           "T": stage_text(d["rows"], les, d["lines"], moves),
           "G": stage_game(d["rows"], none["rows"], les),
           "Q": stage_qwen(ledgers, history, les, moves)}
    if arm and d["lines"] == [line]:
        out["G"]["ab"] = round_ab(d["rounds"], none["rounds"])
    out["H"] = stage_verifier(out["Q"], out["G"]["ab"])
    out.update(diagnose(les, out))
    return out
