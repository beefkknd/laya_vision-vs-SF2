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
OVERRIDE_TRIES = 8      # HYBRID override threshold (owner 2026-10-04: laya is limited, give the table MORE weight) --
#                         the table may override text-laya on a clearly-good cell with only this many samples (< the
#                         MIN_TRIES used to TRUST a mean / split), so the table speaks sooner. The mean>0 floor stays,
#                         so a cell where everything is net-negative still defers to laya (no defensive drift).
# Density by distance (owner 2026-10-04): the closer the fighters, the more DETAIL worth carrying -- up close there
# are many interacting options (throws, close normals, mix-ups), so allow a cell to split by his_label there; far
# away she has few options and the situation is coarse, so far cells stay coarse (no split). (Action COUNT is
# already distance-scaled by `advice.available_moves`: far offers fewer moves, no throws.)
SPLIT_RANGES = ("close", "mid")

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
    return max(0.0, (sq - su * su / n) / (n - 1)) if n > 1 else float("inf")   # clamp roundoff >=0; inf for n<=1


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
        if t["depth"].get(b) or b.split("|")[0] not in SPLIT_RANGES:    # far cells stay coarse (density by distance)
            continue
        tops = {l: tc for l, tc in ((l, _top_covered(acts, min_tries)) for l, acts in labs.items()) if tc}
        if len(tops) < 2:
            continue
        for l1, (a1, s1) in tops.items():
            for l2, (a2, s2) in tops.items():
                if l2 == l1 or a1 == a2:
                    continue
                # a TWO-DIRECTIONAL reversal: a1 confidently beats a2 in l1 AND a2 beats a1 in l2, all four covered.
                # (one-sided was a false-positive source: an under-covered arm in l2 faked a disagreement -- review.)
                a2_l1, a1_l2 = labs[l1].get(a2), labs[l2].get(a1)
                if (a2_l1 and a2_l1[0] >= min_tries and _separated(s1, a2_l1)
                        and a1_l2 and a1_l2[0] >= min_tries and _separated(s2, a1_l2)):
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
        if not act or r.get("range") is None:        # a malformed row with no range can't be keyed reliably -> skip
            continue
        rng, doing, fb, lab = r.get("range"), opp_doing(r), r.get("opp_shot"), str(r.get("his_label"))
        b = base_key(rng, doing, fb)
        _acc(t["cells"].setdefault(when_key(rng, doing, fb, lab, t["depth"]), {}), act, net_of(r))
        if not t["depth"].get(b):                            # track shadow only while the base cell is unsplit
            _acc(t["shadow"].setdefault(b, {}).setdefault(lab, {}), act, net_of(r))
    _maybe_split(t, min_tries)
    return t


def decider(table: Table, me: str, rng: random.Random, min_tries: int = MIN_TRIES, eps0: float = EPS0):
    """A `decide(moment) -> {action, when, explored, category}` closure for loop_runner.play_round. Keys the moment
    ((range, doing, fireball)+his_label split), restricts to the FOLLOWABLE action set at that range, and lets
    `choose` pick. Reads ``table`` as-is (rebuild the decider each round against the freshly-credited table)."""
    from .advice import available_moves, stance_of, char_categories
    from .action_menu import category_of
    from ..vocab import range_of
    cats = char_categories(me)

    def decide(m) -> Dict:
        rng_ = range_of(abs(m.dx))
        actions = sorted(available_moves(stance_of("stand", rng_), cats))
        when = when_key(rng_, m.doing, m.fireball, m.his_label, table.get("depth", {}))
        action, explored = choose(table, when, actions, rng, min_tries, eps0)
        return {"action": action, "when": when, "explored": explored, "category": category_of(action)}
    return decide


def _cell_view(table: Table, when: str) -> Dict[str, List]:
    """The stats to read for ``when``, with PARENT FALLBACK: for a split-child key (base|doing|fb|label), an action
    the child has not seen yet falls back to the base cell's accumulated stats (so an unseen/sparse label keeps the
    parent's knowledge instead of starting blank). The base cell holds the pre-split data (frozen at split time)."""
    cell = dict(table["cells"].get(when, {}))
    if when.count("|") >= 3:                                  # a split child: fall back to the parent (base) cell
        base = "|".join(when.split("|")[:3])
        for a, s in table["cells"].get(base, {}).items():
            cell.setdefault(a, s)
    return cell


def hybrid_decider(table: Table, me: str, rng: random.Random, base_decide,
                   min_tries: int = MIN_TRIES, explore: float = 0.1, override_tries: int = OVERRIDE_TRIES):
    """HYBRID policy (owner 2026-10-04, after the pure table lost to ryu by tanking the neutral game): text-laya is
    the PLAYER via ``base_decide(moment) -> {action, ...}`` (its trained neutral play); the value table OVERRIDES in a
    cell where it is confident a move is clearly good (n >= ``override_tries`` AND mean net hp > 0) -- so it adds the
    wins it actually learned (punish-stunned, lightning_legs) without forcing the from-scratch defensive-losing policy.
    ``override_tries`` (default OVERRIDE_TRIES, well below ``min_tries``) is the owner's "give the table more weight"
    knob: the table speaks with less data, since laya is limited. The mean>0 floor stays, so a cell where every
    sampled move is net-negative defers to laya's neutral aggression (never the least-bad defensive move -- that is
    what tanked the pure table). A small fixed ``explore`` rate still tries an under-sampled followable move
    (``min_tries`` is the coverage target) so the table keeps learning even from losing games, but rarely enough not
    to wreck text-laya's neutral. The table is credited every round regardless of who chose. Returns the decision dict
    tagged with ``source`` (laya / table / table-explore)."""
    from .advice import available_moves, stance_of, char_categories
    from .action_menu import category_of
    from ..vocab import range_of
    cats = char_categories(me)

    def decide(m) -> Dict:
        rng_ = range_of(abs(m.dx))
        actions = sorted(available_moves(stance_of("stand", rng_), cats))
        when = when_key(rng_, m.doing, m.fireball, m.his_label, table.get("depth", {}))
        cell = _cell_view(table, when)
        under = [a for a in actions if count(cell, a) < min_tries]
        if under and rng.random() < explore:                 # a little discovery so the table keeps learning
            a = rng.choice(under)
            return {"action": a, "when": when, "explored": True, "source": "table-explore", "category": category_of(a)}
        good = [(mean(cell.get(a)), a) for a in actions if count(cell, a) >= override_tries and mean(cell.get(a)) > 0]
        if good:                                             # override text-laya where the table is confident-good
            a = max(good)[1]
            return {"action": a, "when": when, "explored": False, "source": "table", "category": category_of(a)}
        d = dict(base_decide(m))                             # otherwise defer to text-laya's neutral play
        d.setdefault("when", when)
        d["source"] = "laya"
        return d
    return decide


def choose(table: Table, when: str, actions: Sequence[str], rng: random.Random,
           min_tries: int = MIN_TRIES, eps0: float = EPS0) -> Tuple[str, bool]:
    """Pick an action for ``when`` over the FOLLOWABLE ``actions``. ε-greedy: with probability eps (decaying as the
    cell fills) explore -- preferring an under-sampled action (forced coverage) -- else exploit the highest mean
    (unsampled -> 0, optimistic vs a known-bad one), ties broken RANDOMLY (so a blank cell has no accidental
    alphabetical/defensive prior). Reads with parent fallback for split children. Returns (action, explored)."""
    actions = list(actions)
    if not actions:
        return "", False
    cell = _cell_view(table, when)
    n_cell = sum(count(cell, a) for a in actions)
    eps = eps0 * min_tries / (min_tries + n_cell)
    if rng.random() < eps:
        under = [a for a in actions if count(cell, a) < min_tries]
        return rng.choice(under or actions), True
    best = max(mean(cell.get(a)) for a in actions)
    return rng.choice([a for a in actions if mean(cell.get(a)) == best]), False   # random tie-break
