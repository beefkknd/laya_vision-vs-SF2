"""The SIMPLE live early-game policy: while losing, change one line; while winning, freeze.

This is the LIVE short-memory controller (what text-laya reads each round). It replaces the old timer
tangle (testing/registered/sticky states + TEST_GAMES/PROMOTE_GAMES/STICK_WINDOW/STOP_DROP/MIN_TRIES +
MAX_TESTS slot cap) that deadlocked -- a carried rule's stale `since` locked the test slots and refused
every Coach claim for 12 losing rounds (playbooks/chun/round_03_ryu). See docs/plan_simple_learning.md.

Two states, one counter, one threshold:
  - states: `trying` (on probation) and `kept` (earned its place, immune to swaps). A book `verified`
    entry (lessons.from_book) reads as kept-equivalent: always in play, never swapped.
  - counter: `rounds` per entry = rounds it has been in the short memory. Carries across blocks/restarts
    trivially (it lives in the entry) -- there is NO since/idx coordinate, so the deadlock cannot recur.
  - threshold: SWAP_AFTER straight lost rounds forces exactly one line change.

The owner's rule, encoded: not losing -> freeze (stop churning once winning); losing -> drop the weakest
`trying` line and admit one fresh claim (the Coach's, else the explore pool), giving the newcomer one
SWAP_AFTER window of chance before it too can be swapped. The measurement (lessons.condition_evidence) is
kept only as the ADVISORY graduation scorer (trying -> kept); it never blocks an admission or a change.

Pure: no I/O, no Qwen, returns NEW registries (owner immutability). The loop (play_loop_screen.update)
and the career driver call ``step`` once per round.
"""
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from . import lessons as L
from . import explore_pool

Claim = Dict
Registry = List[Dict]
Scorer = Callable[[Sequence[Dict], Claim], Dict]

SWAP_AFTER = 2                      # straight lost rounds that force one line change (owner: a single number)
TRYING, KEPT = "trying", "kept"
IN_PLAY_STATES = (KEPT, "verified", TRYING)      # what text-laya actually reads; kept/verified are immune to swaps
IMMUNE = (KEPT, "verified")


# --------------------------------------------------------------------------- the one signal
def loss_streak(round_wl: Sequence[Dict]) -> int:
    """Trailing lost rounds (a round where lost > won), newest last. A win breaks the streak."""
    n = 0
    for g in reversed(list(round_wl)):
        if int(g.get("lost", 0)) > int(g.get("won", 0)):
            n += 1
        else:
            break
    return n


# --------------------------------------------------------------------------- entries
def _trying_entry(c: Claim, why: str = "trying: admitted while losing") -> Dict:
    """A claim -> a fresh `trying` registry entry (same shape the carry file / TUI / trace already read)."""
    return {"claim": {k: c.get(k) for k in ("kind", "move", "range", "when", "view")},
            "line": L.render(c), "state": TRYING, "why": why, "rounds": 0, "evidence": {},
            "qwen_why": c.get("why", "")}


def _in_play_entries(reg: Registry) -> List[Dict]:
    return [r for r in reg if r["state"] in IN_PLAY_STATES]


def in_play(reg: Registry) -> List[str]:
    """The lines text-laya reads: the immune (kept/verified) ones first, then the trying ones, capped at
    MAX_LINES. Stable set first so a swap only ever churns the trying tail."""
    immune = [r["line"] for r in reg if r["state"] in IMMUNE]
    trying = [r["line"] for r in reg if r["state"] == TRYING]
    return (immune + trying)[:L.MAX_LINES]


# --------------------------------------------------------------------------- validity / novelty
def _valid(c: Claim, moves: set) -> bool:
    if not isinstance(c, dict) or c.get("kind") not in L.KINDS or not isinstance(c.get("move"), str):
        return False
    if c.get("range") not in (None,) + L.RANGES or c.get("when") not in (None,) + tuple(L.WHEN_WORDS):
        return False
    return c["move"] in moves and c["move"] not in L.UNFOLLOWABLE


def _novel(reg: Registry, c: Claim) -> bool:
    """Not already in play (same key), and not covered by an in-play line in the same direction."""
    for r in _in_play_entries(reg):
        rc = r["claim"]
        if L.key(rc) == L.key(c):
            return False
        if L.RIGHT[rc["kind"]] == L.RIGHT[c["kind"]] and L._covers(rc, c):
            return False
    return True


def _pick_admit(reg: Registry, claims: Sequence[Claim], moves: set, rotate: int) -> Optional[Claim]:
    """The fresh line to bring in: the Coach's first valid & novel claim, else the next explore-pool rule."""
    for c in claims or []:
        if _valid(c, moves) and _novel(reg, c):
            return c
    pool = explore_pool.pick(in_play(reg), moves, rotate)
    return pool if (pool and _novel(reg, pool)) else None


# --------------------------------------------------------------------------- swap / graduate
def _score(ev: Dict, kind: str) -> float:
    """Signed goodness: higher is better in the claim's intended direction (so min() is the weakest)."""
    sign = 1.0 if L.RIGHT[kind] == "better" else -1.0
    return sign * float(ev.get("diff", 0.0))


def _weakest_trying(reg: Registry, rows: Sequence[Dict], scorer: Scorer) -> Optional[int]:
    """Index of the trying line to drop: the lowest-scoring one that has actually fired; if none has fired,
    the oldest (most rounds in play -- it had the most chance and showed nothing). None if no trying line."""
    fired, idle = [], []
    for i, r in enumerate(reg):
        if r["state"] != TRYING:
            continue
        ev = scorer(rows, r["claim"])
        (fired if ev.get("tries", 0) > 0 else idle).append((i, ev, r))
    if fired:
        return min(fired, key=lambda t: _score(t[1], t[2]["claim"]["kind"]))[0]
    if idle:
        return max(idle, key=lambda t: t[2].get("rounds", 0))[0]
    return None


def _graduate(reg: Registry, rows: Sequence[Dict], scorer: Scorer) -> Registry:
    """Advisory: a trying line the scorer judges clearly good (in its direction) AND that has fired becomes
    kept (immune). Never blocks anything; a rule it cannot yet judge just stays trying."""
    out = []
    for r in reg:
        r = dict(r)
        if r["state"] == TRYING:
            ev = scorer(rows, r["claim"])
            if ev.get("tries", 0) > 0 and ev.get("cls") == L.RIGHT[r["claim"]["kind"]]:
                r.update(state=KEPT, evidence=ev,
                         why="kept: measured clearly %s (%d tries)" % (ev["cls"], ev["tries"]))
        out.append(r)
    return out


def _age(reg: Registry) -> Registry:
    """+1 round-in-play to every trying line (the fair-chance clock). Immune lines are untouched."""
    return [dict(r, rounds=r.get("rounds", 0) + 1) if r["state"] == TRYING else dict(r) for r in reg]


def _should_swap(reg: Registry, round_wl: Sequence[Dict], swap_after: int) -> bool:
    """Swap when she is on a loss streak AND the current experiment has had its SWAP_AFTER window of chance
    (the youngest trying line is at least SWAP_AFTER rounds old). With no trying line, bring one in at once."""
    if loss_streak(round_wl) < swap_after:
        return False
    tries = [r for r in reg if r["state"] == TRYING]
    if not tries:
        return True
    return min(r.get("rounds", 0) for r in tries) >= swap_after


def _swap(reg: Registry, claims: Sequence[Claim], rows: Sequence[Dict], moves: set, rotate: int,
          scorer: Scorer) -> Tuple[Registry, Dict]:
    reg = [dict(r) for r in reg]
    admit = _pick_admit(reg, claims, moves, rotate)
    if admit is None:
        return reg, {"added": [], "removed": []}
    removed: List[str] = []
    if len(_in_play_entries(reg)) >= L.MAX_LINES:
        victim = _weakest_trying(reg, rows, scorer)
        if victim is None:
            return reg, {"added": [], "removed": []}        # every slot is immune: cannot swap (she is winning on kept)
        removed = [reg.pop(victim)["line"]]
    entry = _trying_entry(admit)
    reg.append(entry)
    return reg, {"added": [entry["line"]], "removed": removed}


def _default_scorer(rows: Sequence[Dict], claim: Claim) -> Dict:
    return L.condition_evidence(rows, claim)


def step(reg: Registry, round_wl: Sequence[Dict], claims: Sequence[Claim], rows: Sequence[Dict],
         moves: set, rotate: int = 0, scorer: Optional[Scorer] = None,
         swap_after: int = SWAP_AFTER) -> Tuple[Registry, Dict]:
    """One round of the live policy: age the trying lines, graduate the proven ones (advisory), and -- only
    while losing past the threshold -- swap exactly one line (drop the weakest trying, admit one fresh claim).
    Returns (new registry, event) where event = {added, removed, streak}. Pure; never raises on empty inputs."""
    scorer = scorer or _default_scorer
    moves = set(moves) if moves is not None else {a.get("action") for a in rows}
    reg = _age(reg)
    reg = _graduate(reg, rows, scorer)
    event = {"added": [], "removed": [], "streak": loss_streak(round_wl)}
    if _should_swap(reg, round_wl, swap_after):
        reg, ch = _swap(reg, claims, rows, moves, rotate, scorer)
        event.update(ch)
    return reg, event
