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
from . import rule_stats

Claim = Dict
Registry = List[Dict]
Scorer = Callable[[Sequence[Dict], Claim], Dict]

SWAP_AFTER = 2                      # straight lost rounds that force one line change (owner: a single number)
MAX_LINES = 10                      # the LIVE short memory holds up to 10 lines (the offline lessons cap stays 5);
#                                     only the situation-applicable subset is ever shown to text-laya per decision
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
    return (immune + trying)[:MAX_LINES]


# --------------------------------------------------------------------------- validity / novelty
Followable = Callable[[str, Optional[str]], bool]
_ALLOW: Followable = lambda move, rng: True        # default: no enforceability check (pure tests / callers w/o a char)


def _valid(c: Claim, moves: set, followable: Followable = _ALLOW) -> bool:
    if not isinstance(c, dict) or c.get("kind") not in L.KINDS or not isinstance(c.get("move"), str):
        return False
    if c.get("range") not in (None,) + L.RANGES or c.get("when") not in (None,) + tuple(L.WHEN_WORDS):
        return False
    if c["move"] not in moves or c["move"] in L.UNFOLLOWABLE:
        return False
    return followable(c["move"], c.get("range"))   # UNFOLLOWABLE at its range (e.g. 's.mk up close') -> refused


def drop_unfollowable(reg: Registry, followable: Followable) -> Tuple[Registry, List[str]]:
    """Prune in-play lines text-laya can never play at their range (e.g. a carried 's.* up close'). Immutable:
    returns (new reg, dropped lines). Non-in-play entries (rejected history) are left untouched."""
    kept, dropped = [], []
    for r in reg:
        if r["state"] in IN_PLAY_STATES and not followable(r["claim"]["move"], r["claim"].get("range")):
            dropped.append(r["line"])
        else:
            kept.append(dict(r))
    return kept, dropped


def _novel(reg: Registry, c: Claim) -> bool:
    """Not already in play (same key), and not covered by an in-play line in the same direction."""
    for r in _in_play_entries(reg):
        rc = r["claim"]
        if L.key(rc) == L.key(c):
            return False
        if L.RIGHT[rc["kind"]] == L.RIGHT[c["kind"]] and L._covers(rc, c):
            return False
    return True


def _pick_admit(reg: Registry, claims: Sequence[Claim], moves: set, rotate: int,
                allow_pool: bool, followable: Followable = _ALLOW) -> Optional[Claim]:
    """The fresh line to bring in: the Coach's first valid & novel & FOLLOWABLE claim, else (only while losing,
    via ``allow_pool``) the next followable explore-pool rule. None if nothing valid/novel/followable is available."""
    for c in claims or []:
        if _valid(c, moves, followable) and _novel(reg, c):
            return c
    if allow_pool:
        pool = explore_pool.pick(in_play(reg), moves, rotate, followable)
        if pool and _novel(reg, pool):
            return pool
    return None


# --------------------------------------------------------------------------- swap / graduate
def _ev(entry: Dict, rows: Sequence[Dict], scorer: Scorer) -> Dict:
    """A rule's evidence: its POOLED cross-block stats when present (Step 1B), else the block-local scorer. Pooling
    is what lets MIN_TRIES be reached so a rule can graduate at all; the scorer fallback keeps pure tests working."""
    st = entry.get("stats")
    return rule_stats.evidence_from_stats(st) if st else scorer(rows, entry["claim"])


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
        ev = _ev(r, rows, scorer)
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
            ev = _ev(r, rows, scorer)
            if ev.get("tries", 0) > 0 and ev.get("cls") == L.RIGHT[r["claim"]["kind"]]:
                r.update(state=KEPT, evidence=ev,
                         why="kept: measured clearly %s (%d tries)" % (ev["cls"], ev["tries"]))
        out.append(r)
    return out


def _age(reg: Registry) -> Registry:
    """+1 round-in-play to every trying line (the fair-chance clock). Immune lines are untouched."""
    return [dict(r, rounds=r.get("rounds", 0) + 1) if r["state"] == TRYING else dict(r) for r in reg]


def _retone(reg: Registry, claims: Sequence[Claim], moves: set,
            followable: Followable = _ALLOW) -> Tuple[Registry, Dict]:
    """TONE ADJUSTMENT between rounds: a Coach claim naming the SAME move+range+when as a line in play but a
    DIFFERENT kind (use_more <-> always <-> avoid) dials that line's tone in place -- keep its state and its
    rounds-in-play, just change the tone. At most one retone per round (one change at a time). Returns
    (new registry, event) where a retone fills event added=[new line] / removed=[old line]; else ({}, {})."""
    for c in claims or []:
        if not _valid(c, moves, followable):
            continue
        for i, r in enumerate(reg):
            if r["state"] not in IN_PLAY_STATES:
                continue
            rc = r["claim"]
            if (rc["move"] == c["move"] and rc.get("range") == c.get("range")
                    and rc.get("when") == c.get("when") and rc["kind"] != c["kind"]):
                new_reg = [dict(x) for x in reg]
                old = r["line"]
                new_reg[i] = dict(r, claim={k: c.get(k) for k in ("kind", "move", "range", "when", "view")},
                                  line=L.render(c), why="retoned: %s -> %s" % (rc["kind"], c["kind"]))
                return new_reg, {"added": [L.render(c)], "removed": [old], "retoned": True}
    return reg, {}


def _fair_chance(reg: Registry, swap_after: int) -> bool:
    """A just-admitted trying line gets one SWAP_AFTER window before it can be evicted (no thrashing). True
    when the youngest trying line is at least SWAP_AFTER rounds old (or there is none)."""
    tries = [r for r in reg if r["state"] == TRYING]
    return (not tries) or min(r.get("rounds", 0) for r in tries) >= swap_after


def _swap(reg: Registry, claims: Sequence[Claim], rows: Sequence[Dict], moves: set, rotate: int,
          scorer: Scorer, followable: Followable = _ALLOW) -> Tuple[Registry, Dict]:
    """FULL memory + losing: drop the weakest trying line and admit one fresh claim (Coach's, else pool)."""
    reg = [dict(r) for r in reg]
    admit = _pick_admit(reg, claims, moves, rotate, allow_pool=True, followable=followable)
    if admit is None:
        return reg, {"added": [], "removed": []}
    victim = _weakest_trying(reg, rows, scorer)
    if victim is None:
        return reg, {"added": [], "removed": []}            # every slot is immune: cannot swap (winning on kept lines)
    removed = [reg.pop(victim)["line"]]
    entry = _trying_entry(admit)
    reg.append(entry)
    return reg, {"added": [entry["line"]], "removed": removed}


def _default_scorer(rows: Sequence[Dict], claim: Claim) -> Dict:
    return L.condition_evidence(rows, claim)


def step(reg: Registry, round_wl: Sequence[Dict], claims: Sequence[Claim], rows: Sequence[Dict],
         moves: set, rotate: int = 0, scorer: Optional[Scorer] = None,
         swap_after: int = SWAP_AFTER, followable: Optional[Followable] = None) -> Tuple[Registry, Dict]:
    """One round of the live policy. Age the trying lines, drop any now-unfollowable ones, graduate the proven
    ones (advisory), then change the short memory by the owner's rule:
      - ROOM in the memory (< MAX_LINES): admit one fresh line -- the Coach's valid&novel&FOLLOWABLE claim always
        (early-game growth), or, only while losing, an explore-pool rule when the Coach is silent.
      - FULL + losing past SWAP_AFTER (and the newest trying line has had its fair-chance window): SWAP --
        drop the weakest trying line and admit one fresh claim.
      - FULL + not losing: freeze (stop churning once the established set is winning).
    ``followable(move, range)`` refuses/drops rules text-laya can never play at their range (default: allow all --
    pure tests). Returns (new registry, event={added, removed, streak}). Pure; never raises on empty inputs."""
    scorer = scorer or _default_scorer
    fol = followable or _ALLOW
    moves = set(moves) if moves is not None else {a.get("action") for a in rows}
    reg = _age(reg)
    reg, dead = drop_unfollowable(reg, fol)                         # prune carried rules she can't play at their range
    reg = _graduate(reg, rows, scorer)
    streak = loss_streak(round_wl)
    losing = streak >= swap_after
    event = {"added": [], "removed": list(dead), "streak": streak}
    reg, retone = _retone(reg, claims, moves, fol)                  # tone change counts as this round's one change
    if retone.get("retoned"):
        event["added"] = retone["added"]
        event["removed"] = list(dead) + retone["removed"]
        event["retoned"] = True
        return reg, event
    if MAX_LINES - len(_in_play_entries(reg)) > 0:                  # room: grow the memory
        admit = _pick_admit(reg, claims, moves, rotate, allow_pool=losing, followable=fol)
        if admit is not None:
            reg = [dict(r) for r in reg] + [_trying_entry(admit)]
            event["added"] = [L.render(admit)]
    elif losing and _fair_chance(reg, swap_after):                  # full + losing: swap one
        reg, ch = _swap(reg, claims, rows, moves, rotate, scorer, fol)
        event["added"] = ch["added"]
        event["removed"] = list(dead) + ch["removed"]
    return reg, event
