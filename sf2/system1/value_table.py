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
def credit(table: Table, drows: Sequence[Dict]) -> Table:
    """Fold one round's decision rows into a NEW table (immutable). Each row's net hp (dealt-taken, already
    delayed-hit-corrected upstream) updates the (when, action) Welford stats for the move she played."""
    t = _clone(table)
    for r in drows:
        act = r.get("action")
        if not act:
            continue
        _acc(t["cells"].setdefault(row_when(r, t["depth"]), {}), act, net_of(r))
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
