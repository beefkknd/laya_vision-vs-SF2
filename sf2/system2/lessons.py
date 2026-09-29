"""The lesson registry: Qwen proposes what to learn, code verifies it and keeps the books (scripts/qwen_lessons.py).

A claim is one text-laya line: {"kind": "use_more" | "always" | "avoid", "move", "range" (or None: anywhere), "when"
(what he is doing, or None)}; ``render`` writes it in text laya's grammar ("always" is the one kind that overrides a
"likely fails" rating, e.g. for blocks). Each claim is judged by code RELATIVE to what she does now in the same
situation: in the claim's condition (range, what he is doing), the claim move's net hit points per decision (damage
dealt - taken, to her next decision) minus her average over every other decision there (attacks, walks, blocks), with
the 95% interval of that difference: better, worse, unclear, or few. (A block nets about -4 per try, "bad" on its own,
while her average when he attacks is about -10: it is better than what she does.)

    avoid          judged at once: clearly worse -> registered, else rejected (she should not keep doing a worse move
                   to test it; with too few tries: "too few", and it may be proposed again)
    use more /     clearly better -> registered, clearly worse -> rejected, else testing: in play for up to TEST_GAMES
    always         games, then registered if it became clearly better, rejected if not ("too few" if thin)
    registered -> retired when its evidence stops holding

``in_play``: the lines text laya reads, at most MAX_LINES: the tests first (at most MAX_TESTS), then the registered
lessons worth the most (``strength``: the difference per decision x decisions). ``cause`` names where damage came
from, for the defense view. Nothing here calls Qwen.
"""
from typing import Dict, List, Optional, Sequence, Tuple

from ..system1.advice import opp_doing
from ..vocab import RANGE_WORDS, RANGES
from .move_coach import MIN_TRIES

KINDS = ("use_more", "always", "avoid")
RIGHT = {"use_more": "better", "always": "better", "avoid": "worse"}
LEAD = {"use_more": "use more", "always": "always", "avoid": "avoid"}
WHEN_WORDS = {"jumping": "when he jumps", "crouching": "when he crouches", "attacking": "when he attacks",
              "standing": "when he stands", "stunned": "when he is stunned"}
MAX_LINES = 5          # text laya reads at most 5 advice lines
MAX_TESTS = 2          # claims being tried at once
TEST_GAMES = 3         # games a "use more" claim is tried before it is judged

Claim = Dict
Registry = List[Dict]


def render(c: Claim) -> str:
    words = [LEAD[c["kind"]], c["move"]]
    if c.get("range"):
        words.append(RANGE_WORDS[c["range"]])
    if c.get("when"):
        words.append(WHEN_WORDS[c["when"]])
    return " ".join(words)


def key(c: Claim) -> Tuple:
    return (c["kind"], c["move"], c.get("range"), c.get("when"))


def _overlap(a: Claim, b: Claim) -> bool:
    """Same move, and conditions that can hold at the same moment."""
    return a["move"] == b["move"] and all(a.get(k) is None or b.get(k) is None or a.get(k) == b.get(k)
                                          for k in ("range", "when"))


def _covers(broad: Claim, narrow: Claim) -> bool:
    """``broad`` applies wherever ``narrow`` does (same move; each condition of broad is absent or the same)."""
    return broad["move"] == narrow["move"] and all(broad.get(k) is None or broad.get(k) == narrow.get(k)
                                                   for k in ("range", "when"))


def _mean_var(xs: Sequence[float]) -> Tuple[float, float]:
    m = sum(xs) / len(xs)
    return m, sum((x - m) ** 2 for x in xs) / (len(xs) - 1) if len(xs) > 1 else float("inf")


def condition_evidence(rows: Sequence[Dict], c: Claim) -> Dict:
    """c's move against her average over every other decision in c's condition (range, what he was doing)."""
    here = [a for a in rows if (c.get("range") is None or a["range"] == c["range"])
            and (c.get("when") is None or opp_doing(a) == c["when"])]
    mine = [a["dealt"] - a["taken"] for a in here if a["action"] == c["move"]]
    rest = [a["dealt"] - a["taken"] for a in here if a["action"] != c["move"]]
    out = {"tries": len(mine), "others": len(rest), "net": sum(mine) / len(mine) if mine else 0.0,
           "base": sum(rest) / len(rest) if rest else 0.0, "diff": 0.0, "lo": 0.0, "hi": 0.0, "cls": "few"}
    if len(mine) < MIN_TRIES or len(rest) < MIN_TRIES:
        return out
    (m1, v1), (m2, v2) = _mean_var(mine), _mean_var(rest)
    half = 1.96 * (v1 / len(mine) + v2 / len(rest)) ** 0.5
    d = m1 - m2
    return dict(out, diff=d, lo=d - half, hi=d + half, cls="better" if d - half > 0 else "worse" if d + half < 0
                else "unclear")


def strength(r: Dict) -> float:
    """What a lesson is worth: its difference per decision x its decisions."""
    return abs(r["evidence"]["diff"]) * r["evidence"]["tries"]


def cause(a: Dict) -> Optional[str]:
    """Where a decision's damage taken came from: traded (her attack hit, and he hit her), punished (she whiffed or
    was blocked, then got hit), stuffed (her attack did not land and she got hit), caught (hit while walking or
    blocking); None if she took nothing."""
    if not a["taken"]:
        return None
    if a.get("kind", "attack") != "attack":
        return "caught"
    return {"hit": "traded", "whiff": "punished", "blocked": "punished"}.get(a.get("actual"), "stuffed")


def _refusal(reg: Registry, c, tried: set, moves: set) -> Optional[str]:
    if not isinstance(c, dict) or c.get("kind") not in KINDS or not isinstance(c.get("move"), str):
        return "not a claim: %.80r" % (c,)
    if c.get("range") not in (None,) + RANGES or c.get("when") not in (None,) + tuple(WHEN_WORDS):
        return "range must be close / mid / far or none, when one of %s or none" % ", ".join(WHEN_WORDS)
    if c["move"] not in moves:
        return "%s is not one of her moves" % c["move"]
    if c["move"] not in tried and c["kind"] == "avoid":
        return "she never used %s: nothing to avoid" % c["move"]
    for r in reg:
        if key(r["claim"]) == key(c) and r["state"] in ("testing", "registered"):
            return "already %s" % r["state"]
        if key(r["claim"]) == key(c) and r["state"] == "rejected" and not r["why"].startswith("too few"):
            return "already rejected: %s" % r["why"]
        same = RIGHT[r["claim"]["kind"]] == RIGHT[c["kind"]]
        if r["state"] in ("testing", "registered") and same and _covers(r["claim"], c):
            return "covered by the %s lesson %r" % (r["state"], r["line"])
        exception = _covers(r["claim"], c) and key(r["claim"])[2:] != key(c)[2:]     # strictly narrower: allowed
        if r["state"] in ("testing", "registered") and not same and _overlap(r["claim"], c) and not exception:
            return "contradicts the %s lesson %r" % (r["state"], r["line"])
    return None


def _entry(c: Claim, state: str, why: str, game: int, ev: Dict, qwen_why: str = "") -> Dict:
    return {"claim": {k: c.get(k) for k in ("kind", "move", "range", "when", "view")}, "line": render(c), "state": state,
            "why": why, "since": game, "evidence": ev, "qwen_why": qwen_why}


def _vs(ev: Dict) -> str:
    return "%+.1f per decision vs her %+.1f there (difference %+.1f, 95%% %+.1f to %+.1f, %d tries)" % (
        ev["net"], ev["base"], ev["diff"], ev["lo"], ev["hi"], ev["tries"])


def _judge(c: Claim, ev: Dict) -> Tuple[str, str]:
    if ev["cls"] == "few":
        return ("rejected", "too few tries to judge (%d)" % ev["tries"]) if c["kind"] == "avoid" else \
            ("testing", "to be tried in play")
    if c["kind"] == "avoid":
        return ("registered", "clearly worse: " + _vs(ev)) if ev["cls"] == "worse" else \
            ("rejected", "not clearly worse: " + _vs(ev))
    if ev["cls"] == "better":
        return "registered", "clearly better: " + _vs(ev)
    if ev["cls"] == "worse":
        return "rejected", "clearly worse: " + _vs(ev)
    return "testing", "to be tried in play"


def propose(reg: Registry, claims: Sequence, rows: Sequence[Dict], game: int,
            moves: Optional[Sequence[str]] = None) -> Tuple[Registry, List[Dict]]:
    """Take Qwen's claims: each is refused (invalid, known, contradicting) or judged. ``moves``: her whole move set (a
    "use more" / "always" may name a move she never used: it is tried in play); default: the moves in ``rows``.
    A claim opposite to a lesson but for a strictly narrower situation is an exception, judged on its own data (text
    laya's rule: an applying avoid rules the move out even where a use more applies). Returns a new registry."""
    reg = [dict(r) for r in reg]
    tried = {a["action"] for a in rows}
    moves = set(moves) if moves is not None else tried
    out = []
    for c in claims:
        why = _refusal(reg, c, tried, moves)
        if why:
            out.append({"claim": c, "state": "refused", "why": why})
            continue
        ev = condition_evidence(rows, c)
        state, why = _judge(c, ev)
        if state == "testing" and sum(r["state"] == "testing" for r in reg) >= MAX_TESTS:
            out.append({"claim": c, "state": "refused", "why": "already %d claims in test" % MAX_TESTS})
            continue
        entry = _entry(c, state, why, game, ev, c.get("why", ""))
        reg.append(entry)
        out.append(entry)
    return reg, out


def review(reg: Registry, rows: Sequence[Dict], game: int) -> Registry:
    """After a game: judge the claims in test, retire registered lessons whose evidence stopped holding."""
    out = []
    for r in reg:
        r = dict(r)
        ev = condition_evidence(rows, r["claim"])
        right = RIGHT[r["claim"]["kind"]]
        if r["state"] == "testing":
            state, why = _judge(r["claim"], ev)
            if state == "testing" and game - r["since"] >= TEST_GAMES:
                state, why = "rejected", (
                    "too few tries after %d games (%d): may be proposed again" % (TEST_GAMES, ev["tries"])
                    if ev["cls"] == "few" else "not shown after %d games: %s" % (TEST_GAMES, _vs(ev)))
            r.update(state=state, why=why, evidence=ev)
        elif r["state"] == "registered" and ev["cls"] != right:
            r.update(state="retired", why="no longer clearly %s: %s" % (right, _vs(ev)), evidence=ev)
        elif r["state"] == "registered":
            r["evidence"] = ev
        out.append(r)
    return out


def in_play(reg: Registry) -> List[str]:
    tests = [r["line"] for r in reg if r["state"] == "testing"][:MAX_TESTS]
    lessons = sorted((r for r in reg if r["state"] == "registered"), key=lambda r: -strength(r))
    return tests + [r["line"] for r in lessons][:MAX_LINES - len(tests)]


def violations(reg: Registry, rows: Sequence[Dict]) -> List[str]:
    """Registered lessons their own evidence does not support (must always be empty after ``review``)."""
    return [r["line"] for r in reg if r["state"] == "registered"
            and condition_evidence(rows, r["claim"])["cls"] != RIGHT[r["claim"]["kind"]]]

