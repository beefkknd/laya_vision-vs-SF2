"""The self-learning (when -> action) VALUE TABLE: System-1 policy core (docs/plan_table_system1.md, Stage 2).

A contextual bandit. `credit` folds each round's decision outcomes (per-decision net hp, already delayed-hit-fixed
by screen_evidence) into running Welford stats per (when, action). `choose` picks the move for a 'when' with
ε-greedy exploration that decays as the cell fills -- so it explores alternatives (its own counterfactual) and
pushes the better action up. Winning by a dominant move is a legitimate win (owner): the policy may converge on one
move in a cell, and `choose` will still adapt if that move's measured value falls.

Gate-clean: lives in sf2/system1, imports only stdlib + sf2.system1.advice (the play-path-safe helper). NO sf2.data,
no RAM, no table-module import. JSON-serialisable (string keys, list values) so it persists like the registry via
--save-table / --carry-table.

2a (this file): the base-keyed core. 2b adds the his_label shadow tally + cell split (the `shadow`/`depth` fields are
present and empty here so 2b slots in without a format change).
"""
import random
from typing import Dict, List, Optional, Sequence, Tuple

from .advice import opp_doing

MIN_TRIES = 20          # samples before a (when, action) mean is trusted / a cell may split (same threshold family
#                         as move_coach.MIN_TRIES; kept local so sf2/system1 does not import sf2/system2)
EPS0 = 0.3              # base exploration rate; per-cell eps = EPS0 * MIN_TRIES / (MIN_TRIES + n_cell)

Table = Dict           # {"cells": {wkey: {action: [n, sum, sumsq]}}, "shadow": {...}, "depth": {base: "his_label"}}


def blank() -> Table:
    return {"cells": {}, "shadow": {}, "depth": {}}


# --------------------------------------------------------------------------- keys
def base_key(rng: Optional[str], doing: str, fireball) -> str:
    return "%s|%s|%d" % (rng, doing, 1 if fireball else 0)


def when_key(rng: Optional[str], doing: str, fireball, his_label, depth: Dict[str, str]) -> str:
    """The 'when' string. Base = (range, doing, fireball); a base cell flagged in ``depth`` is split by his_label."""
    b = base_key(rng, doing, fireball)
    return (b + "|" + str(his_label)) if depth.get(b) == "his_label" else b


def row_when(row: Dict, depth: Dict[str, str]) -> str:
    """The 'when' for a decision row (screen_evidence drow): range + opp_doing + fireball (+ his_label if split)."""
    return when_key(row.get("range"), opp_doing(row), row.get("opp_shot"), row.get("his_label"), depth)


# --------------------------------------------------------------------------- stat helpers (Welford via n/sum/sumsq)
def _acc(cell: Dict[str, List], action: str, net: float) -> None:
    s = cell.get(action)
    if s is None:
        s = cell[action] = [0, 0.0, 0.0]
    s[0] += 1
    s[1] += net
    s[2] += net * net


def mean(s: Optional[List]) -> float:
    return s[1] / s[0] if s and s[0] else 0.0


def count(cell: Dict[str, List], action: str) -> int:
    s = cell.get(action)
    return s[0] if s else 0


def net_of(row: Dict) -> float:
    return (row.get("dealt", 0) or 0) - (row.get("taken", 0) or 0)


def _clone(t: Table) -> Table:
    return {"cells": {k: {a: list(v) for a, v in c.items()} for k, c in t["cells"].items()},
            "shadow": {b: {l: {a: list(v) for a, v in acts.items()} for l, acts in labs.items()}
                       for b, labs in t.get("shadow", {}).items()},
            "depth": dict(t.get("depth", {}))}


# --------------------------------------------------------------------------- credit (learn) and choose (play)
def _var(s: List) -> float:
    n, su, sq = s
    return (sq - su * su / n) / (n - 1) if n > 1 else float("inf")      # inf for n<=1 -> never 'separated'


def _separated(s1: List, s2: List) -> bool:
    """s1's mean is confidently ABOVE s2's (Welch 95% lower bound of the difference > 0)."""
    half = 1.96 * (_var(s1) / s1[0] + _var(s2) / s2[0]) ** 0.5
    return (mean(s1) - mean(s2)) > half


def _top_covered(acts: Dict[str, List], min_tries: int) -> Optional[Tuple[str, List]]:
    cov = [(a, s) for a, s in acts.items() if s[0] >= min_tries]
    return max(cov, key=lambda kv: mean(kv[1])) if cov else None


def _maybe_split(t: Table, min_tries: int) -> None:
    """Split a base cell by his_label when two labels confidently prefer DIFFERENT actions -- i.e. the label the
    coarse cell ignores actually flips the best move (close/jumping: empty-jump vs jump-attack). Evidence bar: in
    some label l1 the preferred action a1 is confidently better (Welch) than a2 (a different label's preferred
    action) MEASURED IN l1, both covered (n>=min_tries). On split, seed the per-label cells from the shadow tally.
    One split per credit call; the parent cell stays as the fallback for sparse labels."""
    for b, labs in list(t["shadow"].items()):
        if t["depth"].get(b):
            continue
        tops = {l: tc for l, tc in ((l, _top_covered(acts, min_tries)) for l, acts in labs.items()) if tc}
        if len(tops) < 2:
            continue
        for l1, (a1, s1) in tops.items():
            for l2, (a2, _s2) in tops.items():
                if l2 == l1 or a1 == a2:
                    continue
                other = labs[l1].get(a2)                     # a2 (another label's pick) as measured IN l1
                if other and other[0] >= min_tries and _separated(s1, other):
                    t["depth"][b] = "his_label"
                    for lab, acts in labs.items():           # seed the split cells from what we already saw per label
                        t["cells"][b + "|" + lab] = {a: list(s) for a, s in acts.items()}
                    return


def credit(table: Table, drows: Sequence[Dict], min_tries: int = MIN_TRIES) -> Table:
    """Fold one round's decision rows into a NEW table (immutable). Each row's net hp (dealt-taken, already
    delayed-hit-corrected upstream) updates the (when, action) Welford stats for the move she played, plus a
    his_label SHADOW tally per base cell (while unsplit) used to detect a needed split (``_maybe_split``)."""
    t = _clone(table)
    for r in drows:
        act = r.get("action")
        if not act:
            continue
        rng, doing, fb, lab = r.get("range"), opp_doing(r), r.get("opp_shot"), str(r.get("his_label"))
        b = base_key(rng, doing, fb)
        _acc(t["cells"].setdefault(when_key(rng, doing, fb, lab, t["depth"]), {}), act, net_of(r))
        if not t["depth"].get(b):                            # track shadow only while the base cell is unsplit
            _acc(t["shadow"].setdefault(b, {}).setdefault(lab, {}), act, net_of(r))
    _maybe_split(t, min_tries)
    return t


def choose(table: Table, when: str, actions: Sequence[str], rng: random.Random,
           min_tries: int = MIN_TRIES, eps0: float = EPS0) -> Tuple[str, bool]:
    """Pick an action for ``when`` over the FOLLOWABLE ``actions``. ε-greedy: with probability eps (decaying as the
    cell fills) explore -- preferring an under-sampled action (forced coverage) -- else exploit the highest mean
    (an unsampled action is treated as 0, i.e. optimistic vs a known-bad one). Returns (action, explored)."""
    actions = list(actions)
    if not actions:
        return "", False
    cell = table["cells"].get(when, {})
    n_cell = sum(count(cell, a) for a in actions)
    eps = eps0 * min_tries / (min_tries + n_cell)
    if rng.random() < eps:
        under = [a for a in actions if count(cell, a) < min_tries]
        return rng.choice(under or actions), True
    best = max(actions, key=lambda a: mean(cell.get(a)))      # unsampled -> 0; ties -> first (exploration covers it)
    return best, False
