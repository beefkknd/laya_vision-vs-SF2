"""Offline RULE SCORER (README G5): score how good a Qwen/seed rule is against the value lookup table
(``lessons/value_oracle_v1.json`` via ``sf2.data.value_oracle``). This is ANALYSIS ONLY, OUTSIDE play: it
uses the table freely and is never imported by the play path (``scripts/hard_gate.py`` keeps the loop runner
clean). Nothing here calls Qwen or the emulator.

A rule (a text-laya line / ``sf2.system1.advice`` Lesson, or a ``sf2.system2.lessons`` claim) names a MOVE, a
RANGE and a CONDITION. We map its situation to a table cell and compare the move's expected net (hp dealt -
taken to the next decision) to WALKING IN (``forward`` in the same cell) and to the cell's best move:

    range   -> the table range (close / mid / far); no range -> every range, aggregated by the mean
    "when he attacks"  -> opp_attacking = 1      "when he jumps" -> opp_airborne = 1
    everything else (crouches / stands / is stunned / nothing) -> the neutral cell (0, 0)

Verdict, on the table's net scale relative to forward (the same margin text laya rates "net" moves on,
``sf2.system1.advice.NET_WORKS_MARGIN``):

    good  value >= forward + NET_WORKS_MARGIN   (clearly beats walking in)
    ok    forward <  value <  forward + margin  (beats walking in, modestly)
    bad   value <= forward                      (no better than walking in)

NOT_SCORABLE (said explicitly, never guessed): the table has no axis for the rule's situation --
  * a fireball condition (no fireball axis),
  * a per-opponent rule (no opponent axis -- the table is opponent-agnostic by design),
  * a move the table never samples (an aerial / combo / movement not in its vocabulary, or a move with no
    data in the required cell).
"""
from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Sequence, Tuple

from ..system1.action_menu import CATEGORIES
from ..system1.advice import FORWARD, NET_WORKS_MARGIN, read as read_lesson
from ..vocab import RANGES

MENU_MOVES = tuple(m for moves in CATEGORIES.values() for m in moves)   # Chun-Li's two-stage menu names

CHAR = "chunli"                         # laya plays Chun-Li; the table is keyed by the character acting ("me")

GOOD, OK, BAD, NOT_SCORABLE = "good", "ok", "bad", "not_scorable"
VERDICTS = (GOOD, OK, BAD, NOT_SCORABLE)

Cell = Tuple[str, str, int, int]
Table = Dict[Cell, Dict[str, float]]


@dataclass(frozen=True)
class Rule:
    """A scorable rule, independent of where it was read from."""
    move: Optional[str]                 # the move's name (oracle-native, e.g. "throw"/"c.mk", or a menu name)
    range: Optional[str] = None         # close / mid / far, or None: anywhere
    when: Optional[str] = None          # jumping / crouching / attacking / standing / stunned, or None
    fireball: bool = False              # the rule holds only while a fireball is coming
    opp: Optional[str] = None           # a specific opponent the rule targets, or None: opponent-agnostic
    line: str = ""                      # the rendered line, for reporting
    source: str = ""                    # where it was read from (book / registry / in_play), for reporting


@dataclass(frozen=True)
class CellScore:
    cell: Cell
    move: str                           # the oracle move the rule's move mapped to
    value: float                        # that move's value in the cell
    forward: float                      # walking in, in the same cell
    best_move: str
    best_value: float
    rank: int                           # 1 = best among the cell's moves
    n_moves: int
    verdict: str                        # good / ok / bad for this cell alone


@dataclass(frozen=True)
class RuleScore:
    rule: Rule
    scorable: bool
    reason: str                         # why not_scorable (empty when scorable)
    cells: Tuple[CellScore, ...]        # 1 cell for a ranged rule, 3 (close/mid/far) for range=None
    value: float                        # the move's value (mean over the scored cells)
    forward: float                      # walking in (mean over the scored cells)
    verdict: str                        # good / ok / bad / not_scorable


# ---- move mapping: menu / Lesson names (throw_F+hp, s.mk, c.hk...) -> the table's action vocabulary ----
# The value collection recorded a simplified action set: standing normals lose their prefix (s.mk -> mk), the
# crouching fierce is "sweep" (c.hk), throws collapse to "throw", walking in is "forward". Aerials, combos and
# the other movement moves were never a value action, so they have no table vocabulary.

def _menu_to_oracle(move: str) -> Optional[str]:
    if move == "walk_forward":
        return FORWARD
    if move.startswith("throw_"):
        return "throw"
    if move == "c.hk":
        return "sweep"
    for p in ("s.", "cl."):
        if move.startswith(p):
            return move[len(p):]
    if move.startswith("c."):
        return move
    return None


def char_vocab(table: Table, char: str) -> frozenset:
    """Every move name the table carries for ``char`` (the union over its cells)."""
    return frozenset(m for cell, values in table.items() if cell[0] == char for m in values)


def map_move(move: Optional[str], vocab: frozenset) -> Optional[str]:
    """The table-vocabulary move a rule's move names, or None when the table has no such move. An oracle-native
    name (as book / registry claims carry) passes straight through; a menu / Lesson name is translated."""
    if move is None:
        return None
    if move in vocab:
        return move
    mapped = _menu_to_oracle(move)
    return mapped if mapped in vocab else None


def flags_of(when: Optional[str]) -> Tuple[int, int]:
    """(opp_attacking, opp_airborne) for a rule's condition; everything but attacks / jumps is the neutral cell."""
    return int(when == "attacking"), int(when == "jumping")


def verdict_of(value: float, forward: float) -> str:
    if value >= forward + NET_WORKS_MARGIN:
        return GOOD
    if value > forward:
        return OK
    return BAD


def _mean(xs: Sequence[float]) -> float:
    return sum(xs) / len(xs)


def _score_cell(table: Table, char: str, rng: str, att: int, air: int, omove: str) -> Optional[CellScore]:
    """The rule's move in one cell, or None when the table never sampled the move there."""
    cell: Cell = (char, rng, att, air)
    values = table.get(cell, {})
    if omove not in values:
        return None
    ranked = sorted(values.items(), key=lambda kv: (-kv[1], kv[0]))
    best_move, best_value = ranked[0]
    rank = [m for m, _ in ranked].index(omove) + 1
    value = values[omove]
    forward = values.get(FORWARD, 0.0)
    return CellScore(cell, omove, round(value, 3), round(forward, 3), best_move, round(best_value, 3),
                     rank, len(values), verdict_of(value, forward))


def _not_scorable(rule: Rule, reason: str) -> RuleScore:
    return RuleScore(rule, False, reason, (), 0.0, 0.0, NOT_SCORABLE)


def score_rule(rule: Rule, table: Table, char: str = CHAR) -> RuleScore:
    """Score one rule against the table. Immutable: returns a fresh ``RuleScore``, never mutates its inputs."""
    if rule.fireball:
        return _not_scorable(rule, "fireball condition: the table has no fireball axis")
    if rule.opp is not None:
        return _not_scorable(rule, "per-opponent rule (opp=%r): the table has no opponent axis" % rule.opp)
    if rule.move is None:
        return _not_scorable(rule, "no move named in the rule")
    omove = map_move(rule.move, char_vocab(table, char))
    if omove is None:
        return _not_scorable(rule, "move %r is not in the table's vocabulary (aerial / combo / movement)" % rule.move)
    att, air = flags_of(rule.when)
    ranges = [rule.range] if rule.range in RANGES else list(RANGES)
    cells = tuple(cs for cs in (_score_cell(table, char, r, att, air, omove) for r in ranges) if cs is not None)
    if not cells:
        where = rule.range if rule.range in RANGES else "any range"
        return _not_scorable(rule, "move %r (->%s) has no data in cell char=%s range=%s att=%d air=%d"
                             % (rule.move, omove, char, where, att, air))
    value, forward = _mean([cs.value for cs in cells]), _mean([cs.forward for cs in cells])
    return RuleScore(rule, True, "", cells, round(value, 3), round(forward, 3), verdict_of(value, forward))


# ---- reading rules from the two systems' artifacts ----

def rule_from_claim(claim: Dict, source: str = "", opp: Optional[str] = None) -> Rule:
    """A rule from a ``sf2.system2.lessons`` claim / a ``lessons/book.json`` line's claim. The claim's move is
    oracle-native (as the registry and book store it), so no menu translation is needed to score it."""
    return Rule(move=claim.get("move"), range=claim.get("range"), when=claim.get("when"),
                fireball=bool(claim.get("fireball", False)), opp=opp, line=claim.get("line", ""), source=source)


def rule_from_line(line: str, source: str = "", moves: Sequence[str] = MENU_MOVES) -> Rule:
    """A rule from a rendered text-laya line (``sf2.system2.lessons.in_play`` / a verdict's in_play_end), parsed
    with ``sf2.system1.advice.read`` against the menu. The parsed move is a menu name, translated by ``map_move``
    when scored; an unreadable line names no move and scores not_scorable."""
    les = read_lesson(line, moves)
    return Rule(move=les.move, range=les.where, when=les.when, fireball=les.fireball, line=line, source=source)


def summarize(scores: Iterable[RuleScore]) -> Dict[str, int]:
    """Counts by verdict; total under 'rules'."""
    out = {v: 0 for v in VERDICTS}
    n = 0
    for s in scores:
        out[s.verdict] += 1
        n += 1
    return dict(out, rules=n)
