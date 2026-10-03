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
    stopped        a claim in test is rejected at once when her rounds tank after it began (``stop``: the mean hp per
                   round of the games since it began below that of the games before it by more than STOP_DROP, with at
                   least STOP_BEFORE games before it) - the per-decision yardstick cannot see harm to the round (0f)

    verified       a players' tip whose single-line fixed-advice arm helped vs no advice (lessons/book.json,
                   scripts/book.py; ``from_book``): in the registry from the start, never retired by ``review`` (her
                   history has too few tries to judge it), and a claim repeating, covered by or contradicting it is
                   refused like any other

A claim naming a move in UNFOLLOWABLE (forward) is refused: text laya was never trained on lessons naming it
(docs/component_boundaries.md), so System 1 cannot follow them.

``in_play``: the lines text laya reads, at most MAX_LINES: the verified first, then the tests (at most MAX_TESTS, and
only as many as the verified leave room for), then the registered lessons worth the most (``strength``: the
difference per decision x decisions). ``cause`` names where damage came
from, for the defense view. Nothing here calls Qwen.
"""
from typing import Dict, List, Optional, Sequence, Tuple

from ..system1.advice import FORWARD, opp_doing
from ..vocab import RANGE_WORDS, RANGES
from .move_coach import MIN_TRIES

KINDS = ("use_more", "always", "avoid")
RIGHT = {"use_more": "better", "always": "better", "avoid": "worse"}
LEAD = {"use_more": "use more", "always": "always", "avoid": "avoid"}
WHEN_WORDS = {"jumping": "when he jumps", "crouching": "when he crouches", "attacking": "when he attacks",
              "standing": "when he stands", "stunned": "when he is stunned"}
MAX_LINES = 5          # text laya reads at most 5 advice lines
MAX_TESTS = 2          # claims being tried at once (+1 for a "what if" when she is stuck in a streak)
TEST_GAMES = 3         # games a "use more" claim is tried before it is judged
# Churn fix (docs/plan_two_stage_qwen.md): a registered rule that has survived PROMOTE_GAMES games, still tracks good
# per-decision outcomes AND is correlated with winning becomes "sticky" (verified by play) so a single per-decision CI
# flip no longer drops it; a registered/sticky rule correlated with LOSING games is retired on the round outcomes, not
# the per-decision yardstick. Both fire ONLY when the loop passes ``games_wl`` to ``review``; without it (the unit
# tests that pin the per-decision behaviour), review is byte-identical to before.
PROMOTE_GAMES = 3      # games a registered rule must survive before it can stick
STICK_WINDOW = 2       # games of win/loss outcome needed before a trend can promote or retire a rule
# Early stop. Calibrated on the 89 finished no-advice arms (lock lesson_loop_v1 + rollouts/qwen_lessons, 2026-09-29):
# a game's mean hp per round varies by SD ~36 between games of one arm, so a 40 drop is crossed by chance in 10% of
# the checks a test would get; 60 in 4.4% (per 3-game test 8.3%). tests/test_early_stop.py holds it under 5%.
STOP_DROP = 60.0       # hp per round
STOP_BEFORE = 2        # games before the claim began, at least

UNFOLLOWABLE = (FORWARD,)
UNFOLLOWABLE_WHY = "System 1 cannot follow lessons naming %s (text laya untrained, docs/component_boundaries.md)"
LIVE = ("verified", "testing", "registered", "sticky")   # the states a new claim must not repeat or contradict
VERIFIED_NOTE = ("(A verified lesson is a players' tip proven in play against him, vs no advice: it stays in play. "
                 "Do not repeat or contradict it.)")

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


def _refusal(reg: Registry, c, tried: set, moves: set, unfollowable: Sequence[str] = UNFOLLOWABLE) -> Optional[str]:
    if not isinstance(c, dict) or c.get("kind") not in KINDS or not isinstance(c.get("move"), str):
        return "not a claim: %.80r" % (c,)
    if c.get("range") not in (None,) + RANGES or c.get("when") not in (None,) + tuple(WHEN_WORDS):
        return "range must be close / mid / far or none, when one of %s or none" % ", ".join(WHEN_WORDS)
    if c["move"] not in moves:
        return "%s is not one of her moves" % c["move"]
    if c["move"] in unfollowable:
        return UNFOLLOWABLE_WHY % c["move"]
    if c["move"] not in tried and c["kind"] == "avoid":
        return "she never used %s: nothing to avoid" % c["move"]
    for r in reg:
        if key(r["claim"]) == key(c) and r["state"] in LIVE:
            return "already %s" % r["state"]
        if key(r["claim"]) == key(c) and r["state"] == "rejected" and not r["why"].startswith("too few"):
            return "already rejected: %s" % r["why"]
        same = RIGHT[r["claim"]["kind"]] == RIGHT[c["kind"]]
        if r["state"] in LIVE and same and _covers(r["claim"], c):
            return "covered by the %s lesson %r" % (r["state"], r["line"])
        exception = _covers(r["claim"], c) and key(r["claim"])[2:] != key(c)[2:]     # strictly narrower: allowed
        if r["state"] in LIVE and not same and _overlap(r["claim"], c) and not exception:
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
            moves: Optional[Sequence[str]] = None,
            unfollowable: Sequence[str] = UNFOLLOWABLE) -> Tuple[Registry, List[Dict]]:
    """Take Qwen's claims: each is refused (invalid, known, contradicting) or judged. ``moves``: her whole move set (a
    "use more" / "always" may name a move she never used: it is tried in play); default: the moves in ``rows``.
    A claim opposite to a lesson but for a strictly narrower situation is an exception, judged on its own data (text
    laya's rule: an applying avoid rules the move out even where a use more applies). The track record
    (sf2.system2.track_record) is only shown to Qwen, never a reason to refuse. A claim naming a move in
    ``unfollowable`` is refused (() = the loop before 2026-09-30). A test is taken only while it can be in play
    (MAX_LINES minus the verified lines). Returns a new registry."""
    reg = [dict(r) for r in reg]
    tried = {a["action"] for a in rows}
    moves = set(moves) if moves is not None else tried
    out = []
    for c in claims:
        why = _refusal(reg, c, tried, moves, unfollowable)
        if why:
            out.append({"claim": c, "state": "refused", "why": why})
            continue
        ev = condition_evidence(rows, c)
        state, why = _judge(c, ev)
        limit = MAX_TESTS + (c.get("view") == "what_if")       # a what-if gets its own slot
        testing = sum(r["state"] == "testing" for r in reg)
        if state == "testing" and testing >= limit:
            out.append({"claim": c, "state": "refused", "why": "already %d claims in test" % limit})
            continue
        room = MAX_LINES - sum(r["state"] == "verified" for r in reg)
        if state == "testing" and testing >= room:
            out.append({"claim": c, "state": "refused", "why": "already %d claims in test: the other %d lines in play "
                        "are verified" % (testing, MAX_LINES - room)})
            continue
        entry = _entry(c, state, why, game, ev, c.get("why", ""))
        reg.append(entry)
        out.append(entry)
    return reg, out


def stop(game_hp: Sequence[float], since: int, game: int) -> Optional[str]:
    """Why a claim proposed after game ``since`` is stopped after game ``game``, or None. ``game_hp``: her mean hp
    (dealt - taken) per round in each game of this run, by game number; only games up to ``game`` count."""
    before, after = list(game_hp[:since + 1]), list(game_hp[since + 1:game + 1])
    if len(before) < STOP_BEFORE or not after:
        return None
    b, a = sum(before) / len(before), sum(after) / len(after)
    if b - a <= STOP_DROP:
        return None
    return "stopped: rounds went badly since it began: %+.1f hp per round in %d games vs %+.1f in the %d before" % (
        a, len(after), b, len(before))


def _trend(games_wl: Sequence[Dict], since: int, game: int) -> Tuple[int, int, int]:
    """(won, lost, games) over the games played AFTER a rule began (games_wl[since + 1 : game + 1]). ``games_wl`` is
    the loop's per-game {"won", "lost"} list, by game number."""
    window = list(games_wl[since + 1: game + 1])
    won = sum(int(g.get("won", 0)) for g in window)
    lost = sum(int(g.get("lost", 0)) for g in window)
    return won, lost, len(window)


def losing_since(games_wl: Optional[Sequence[Dict]], since: int, game: int) -> Optional[str]:
    """Why a registered/sticky rule is retired because the games since it began went to losses (round outcomes, not the
    per-decision yardstick), or None. Needs at least STICK_WINDOW games of outcome."""
    if games_wl is None:
        return None
    won, lost, n = _trend(games_wl, since, game)
    if n < STICK_WINDOW or lost <= won:
        return None
    return ("retired: correlated with losing - %d rounds lost vs %d won in the %d games since it began "
            "(round outcomes, not the per-decision yardstick)" % (lost, won, n))


def _promotable(r: Dict, ev: Dict, right: str, games_wl: Optional[Sequence[Dict]], game: int) -> bool:
    """A registered rule sticks when it has survived PROMOTE_GAMES games, still tracks clearly good per-decision
    outcomes, and the games since it began are won at least as often as lost."""
    if games_wl is None or game - r["since"] < PROMOTE_GAMES or ev["cls"] != right:
        return False
    won, lost, n = _trend(games_wl, r["since"], game)
    return n >= STICK_WINDOW and won >= lost


def review(reg: Registry, rows: Sequence[Dict], game: int, game_hp: Optional[Sequence[float]] = None,
           games_wl: Optional[Sequence[Dict]] = None) -> Registry:
    """After a game: judge the claims in test, retire registered lessons whose evidence stopped holding. ``game_hp``
    (her mean hp per round per game of this run, see ``stop``): a claim in test is stopped first when its rounds
    tanked; without it, no claim is stopped. ``games_wl`` (the per-game {"won", "lost"} list): with it, a surviving
    registered lesson that keeps tracking good outcomes and is winning STICKS (verified by play), and a
    registered/sticky lesson correlated with losing games is retired - the churn fix. Without it, registered lessons
    follow only the per-decision CI flip (byte-identical to before)."""
    out = []
    for r in reg:
        r = dict(r)
        ev = condition_evidence(rows, r["claim"])
        right = RIGHT[r["claim"]["kind"]]
        halt = stop(game_hp, r["since"], game) if game_hp and r["state"] == "testing" else None
        lose = losing_since(games_wl, r["since"], game) if r["state"] in ("registered", "sticky") else None
        if halt:
            r.update(state="rejected", why=halt, evidence=ev)
        elif r["state"] == "testing":
            state, why = _judge(r["claim"], ev)
            if state == "testing" and game - r["since"] >= TEST_GAMES:
                state, why = "rejected", (
                    "too few tries after %d games (%d): may be proposed again" % (TEST_GAMES, ev["tries"])
                    if ev["cls"] == "few" else "not shown after %d games: %s" % (TEST_GAMES, _vs(ev)))
            r.update(state=state, why=why, evidence=ev)
        elif lose:
            r.update(state="retired", why=lose, evidence=ev)                       # losing: drop it on the round outcomes
        elif r["state"] == "registered" and ev["cls"] != right:
            r.update(state="retired", why="no longer clearly %s: %s" % (right, _vs(ev)), evidence=ev)
        elif r["state"] == "registered" and _promotable(r, ev, right, games_wl, game):
            r.update(state="sticky", why="promoted (verified by play): still clearly %s since game %d and winning: %s"
                     % (right, r["since"], _vs(ev)), evidence=ev)
        elif r["state"] in ("registered", "sticky"):
            r["evidence"] = ev                       # sticky survives a per-decision CI flip; only losing retires it
        # verified: kept as it is - its evidence is the A/B in play, which her per-situation history cannot judge
        out.append(r)
    return out


def in_play(reg: Registry) -> List[str]:
    verified = [r["line"] for r in reg if r["state"] == "verified"]
    sticky = [r["line"] for r in reg if r["state"] == "sticky"]          # promoted: stays in play like a verified line
    tests = [r["line"] for r in reg if r["state"] == "testing"][:MAX_TESTS + 1]
    lessons = sorted((r for r in reg if r["state"] == "registered"), key=lambda r: -strength(r))
    return (verified + sticky + tests + [r["line"] for r in lessons])[:MAX_LINES]


def violations(reg: Registry, rows: Sequence[Dict]) -> List[str]:
    """Registered lessons their own evidence does not support (must always be empty after ``review``). Verified
    lessons are not judged here: their evidence is the A/B in play (``from_book``)."""
    return [r["line"] for r in reg if r["state"] == "registered"
            and condition_evidence(rows, r["claim"])["cls"] != RIGHT[r["claim"]["kind"]]]



ORDER = {"use_more": 0, "always": 1, "avoid": 2}     # on a tie in the book, the softer line (the owner-named one)


def _tip_why(t: Dict) -> str:
    return "players' tip, verified in play: %+.0f hp/round vs no advice (95%% %+.0f to %+.0f, %d seeds, %s)" % (
        t["mean"], t["ci95"][0], t["ci95"][1], t["runs"], t["batch"])


def from_book(tips: Sequence[Dict], book: str, game: int = -1) -> Registry:
    """One opponent's verified lines (lessons/book.json, scripts/book.py) as registry entries, the best A/B mean
    first. A line covered by one already taken in the same direction (the same move and conditions, "use more" or
    "always"; or a narrower one) is left out: it would only take a slot. A line whose claim does not render to it
    stops the loop (ValueError)."""
    reg: Registry = []
    for t in sorted(tips, key=lambda t: (-t["mean"], ORDER[t["claim"]["kind"]], t["line"])):
        c = dict(t["claim"], view="book")
        if render(c) != t["line"]:
            raise ValueError("book line %r does not render from its claim (%r)" % (t["line"], render(c)))
        if any(RIGHT[r["claim"]["kind"]] == RIGHT[c["kind"]] and _covers(r["claim"], c) for r in reg):
            continue
        ev = {"source": "players' tip", "book": book, "arm": t["arm"], "mean": t["mean"], "ci95": list(t["ci95"]),
              "runs": t["runs"], "seeds": list(t["seeds"]), "batch": t["batch"]}
        reg.append(_entry(c, "verified", _tip_why(t), game, ev))
    return reg
