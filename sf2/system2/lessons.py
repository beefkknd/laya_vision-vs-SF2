"""The lesson registry: Qwen proposes what to learn, code verifies it and keeps the books (scripts/qwen_lessons.py).

A claim is one text-laya line: {"kind": "use_more" | "avoid", "move", "range" (or None: anywhere), "when" (what he is
doing, or None)}; ``render`` writes it in text laya's grammar. Each claim is judged by code on the rounds inside its
own condition (net hit points per try: damage dealt - taken, to her next decision; its 95% interval):

    avoid      judged at once: clearly bad -> registered, else rejected (she should not play a bad move to test it)
    use more   clearly good -> registered, clearly bad -> rejected, else testing: in play for up to TEST_GAMES games,
               then registered if it became clearly good, rejected if not (with too few tries in its condition:
               "too few", and it may be proposed again, like an avoid with too few tries)
    registered -> retired when its evidence stops holding (a "use more" no longer clearly good, an "avoid" no longer
               clearly bad)

``in_play``: the lines text laya reads, at most MAX_LINES: the tests first (at most MAX_TESTS), then the registered
lessons that cost or gain her the most in total. Nothing here calls Qwen.
"""
from typing import Dict, List, Optional, Sequence, Tuple

from ..system1.advice import opp_doing
from ..vocab import RANGE_WORDS, RANGES
from .move_coach import classify

KINDS = ("use_more", "avoid")
RIGHT = {"use_more": "good", "avoid": "bad"}
WHEN_WORDS = {"jumping": "when he jumps", "crouching": "when he crouches", "attacking": "when he attacks",
              "standing": "when he stands", "stunned": "when he is stunned"}
MAX_LINES = 5          # text laya reads at most 5 advice lines
MAX_TESTS = 2          # claims being tried at once
TEST_GAMES = 3         # games a "use more" claim is tried before it is judged

Claim = Dict
Registry = List[Dict]


def render(c: Claim) -> str:
    words = ["use more" if c["kind"] == "use_more" else "avoid", c["move"]]
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


def stat(xs: Sequence[float]) -> Dict:
    n = len(xs)
    if not n:
        return {"tries": 0, "net": 0.0, "lo": 0.0, "hi": 0.0, "total": 0, "cls": "few"}
    m = sum(xs) / n
    half = 1.96 * (sum((x - m) ** 2 for x in xs) / (n - 1)) ** 0.5 / n ** 0.5 if n > 1 else float("inf")
    return {"tries": n, "net": m, "lo": m - half, "hi": m + half, "total": sum(xs), "cls": classify(n, m - half, m + half)}


def condition_evidence(rows: Sequence[Dict], c: Claim) -> Dict:
    """What c's move did in c's condition: her attacks with that move, at that range (if any), while he did that
    (if any)."""
    return stat([a["dealt"] - a["taken"] for a in rows
                  if a.get("kind", "attack") == "attack" and a["action"] == c["move"]
                  and (c.get("range") is None or a["range"] == c["range"])
                  and (c.get("when") is None or opp_doing(a) == c["when"])])


def _refusal(reg: Registry, c, tried: set) -> Optional[str]:
    if not isinstance(c, dict) or c.get("kind") not in KINDS or not isinstance(c.get("move"), str):
        return "not a claim: %.80r" % (c,)
    if c.get("range") not in (None,) + RANGES or c.get("when") not in (None,) + tuple(WHEN_WORDS):
        return "range must be close / mid / far or none, when one of %s or none" % ", ".join(WHEN_WORDS)
    if c["move"] not in tried:
        return "she never tried %s" % c["move"]
    for r in reg:
        if key(r["claim"]) == key(c) and r["state"] in ("testing", "registered"):
            return "already %s" % r["state"]
        if key(r["claim"]) == key(c) and r["state"] == "rejected" and not r["why"].startswith("too few"):
            return "already rejected: %s" % r["why"]
        if r["state"] in ("testing", "registered") and r["claim"]["kind"] != c["kind"] and _overlap(r["claim"], c):
            return "contradicts the %s lesson %r" % (r["state"], r["line"])
    return None


def _entry(c: Claim, state: str, why: str, game: int, ev: Dict, qwen_why: str = "") -> Dict:
    return {"claim": {k: c.get(k) for k in ("kind", "move", "range", "when")}, "line": render(c), "state": state,
            "why": why, "since": game, "evidence": ev, "qwen_why": qwen_why}


def _judge(c: Claim, ev: Dict) -> Tuple[str, str]:
    if c["kind"] == "avoid":
        if ev["cls"] == "bad":
            return "registered", "clearly bad: %+.1f per try over %d tries" % (ev["net"], ev["tries"])
        if ev["cls"] == "few":
            return "rejected", "too few tries to judge (%d)" % ev["tries"]
        return "rejected", "not clearly bad: %+.1f per try, 95%% %+.1f to %+.1f" % (ev["net"], ev["lo"], ev["hi"])
    if ev["cls"] == "good":
        return "registered", "clearly good: %+.1f per try over %d tries" % (ev["net"], ev["tries"])
    if ev["cls"] == "bad":
        return "rejected", "clearly bad: %+.1f per try over %d tries" % (ev["net"], ev["tries"])
    return "testing", "to be tried in play"


def propose(reg: Registry, claims: Sequence, rows: Sequence[Dict], game: int) -> Tuple[Registry, List[Dict]]:
    """Take Qwen's claims: each is refused (invalid, known, contradicting) or judged. Returns a new registry."""
    reg = [dict(r) for r in reg]
    tried = {a["action"] for a in rows}
    out = []
    for c in claims:
        why = _refusal(reg, c, tried)
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
                    if ev["cls"] == "few" else "not shown after %d games: %+.1f per try over %d tries, 95%% %+.1f "
                    "to %+.1f" % (TEST_GAMES, ev["net"], ev["tries"], ev["lo"], ev["hi"]))
            r.update(state=state, why=why, evidence=ev)
        elif r["state"] == "registered" and ev["cls"] != right:
            r.update(state="retired", why="no longer clearly %s: %+.1f per try, 95%% %+.1f to %+.1f" % (
                right, ev["net"], ev["lo"], ev["hi"]), evidence=ev)
        elif r["state"] == "registered":
            r["evidence"] = ev
        out.append(r)
    return out


def in_play(reg: Registry) -> List[str]:
    tests = [r["line"] for r in reg if r["state"] == "testing"][:MAX_TESTS]
    lessons = sorted((r for r in reg if r["state"] == "registered"), key=lambda r: -abs(r["evidence"]["total"]))
    return tests + [r["line"] for r in lessons][:MAX_LINES - len(tests)]


def violations(reg: Registry, rows: Sequence[Dict]) -> List[str]:
    """Registered lessons their own evidence does not support (must always be empty after ``review``)."""
    return [r["line"] for r in reg if r["state"] == "registered"
            and condition_evidence(rows, r["claim"])["cls"] != RIGHT[r["claim"]["kind"]]]

