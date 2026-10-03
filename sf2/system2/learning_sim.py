"""Deterministic SIMULATION of the self-learning loop's lifecycle - the catalog of Qwen-driven events
(SUGGEST -> measure -> PROMOTE / KEEP / DEMOTE / CEILING -> CONFIRM) with the LEARNING CONTEXT carried
through correctly. No games, no network: it threads the REAL context->mode rule (character_prompt.
coach_mode) and the REAL keep/drop gate (promotion.decide).

The honest information flow: each round the Coach sees a CONTEXT OBSERVED FROM THE LAST GAME'S PLAY (not
predicted) - the verdict (winning/losing -> consolidate/escalate), HIS COMMON SITUATIONS (the range/state
histogram it keys rules to), and how each in-play rule is tracking (fired / net / still-good). It proposes
a noise-controlled candidate playbook; text-laya plays it; the measurement decides promote/keep. A promoted
rule then shows up in the NEXT round's observed context, and a rule whose net has gone negative there is the
next DEMOTE - the same loop the live system runs. text-laya is the PLAYER; this only SELECTS the short
memory by outcome (feedback_textlaya_is_the_player).

Context and scenarios are seeded from REAL Coach proposals (out/.../probe_guile|honda), so the catalog is
evidence, not imagination.
"""
from dataclasses import dataclass
from typing import List, Optional, Tuple

from sf2.system2 import character_prompt as CP
from sf2.system2.outcome_loop import Playbook
from sf2.system2.promotion import BlockStat, Cfg, decide, is_gain, warrants_held


@dataclass(frozen=True)
class RuleTrack:
    """How one in-play rule is tracking in the observed context (mirrors digest rule_tracking)."""
    line: str
    fired: int
    net: float
    good: bool


@dataclass(frozen=True)
class Context:
    """The learning context the Coach conditions on, OBSERVED from the last game (subset of digest_facts)."""
    verdict: str                                   # winning | losing | stable
    his_cells: Tuple[Tuple[str, int, int], ...]    # (range/state, count, pct) histogram
    rules: Tuple[RuleTrack, ...] = ()              # per in-play rule tracking

    @property
    def mode(self) -> str:
        """escalate when losing, else consolidate - the REAL rule the live Coach uses."""
        return CP.coach_mode({"verdict": self.verdict})

    def negative_rules(self) -> Tuple[RuleTrack, ...]:
        """In-play rules that have stopped working - the DEMOTE candidates."""
        return tuple(r for r in self.rules if not r.good)


@dataclass(frozen=True)
class Suggestion:
    """One round: the context the Coach OBSERVED, what it proposed (the whole resulting in-play playbook,
    so adding a rule and dropping/avoiding one are both just a new candidate set), and the measurement
    text-laya's play produced. kind is a catalog label only."""
    context: Context
    kind: str                       # "suggest" | "demote" | "sharpen"
    candidate: Playbook
    dev: BlockStat
    held: Optional[BlockStat] = None
    why: str = ""


@dataclass(frozen=True)
class RoundLog:
    round: int
    mode: str
    kind: str
    candidate: str
    verdict: str          # decide(): promote | keep
    ceiling: bool
    reason: str
    incumbent_after: str


@dataclass(frozen=True)
class SimResult:
    final: Playbook
    rounds: Tuple[RoundLog, ...]
    confirmed: Optional[bool]       # terminal untouched test for the final incumbent (None if no promotion)


def simulate(incumbent0: Playbook, rounds: List[Suggestion],
             terminal: Optional[BlockStat] = None, cfg: Cfg = Cfg()) -> SimResult:
    """Walk the playbook through its lifecycle. Each round carries its own observed context; the held-out
    block is consulted only when the dev result warrants it. Returns the trajectory + terminal confirm."""
    incumbent, logs = incumbent0, []
    for k, sug in enumerate(rounds):
        held = sug.held if warrants_held(sug.dev, cfg) else None
        dec = decide(sug.dev, held, cfg)
        if dec.verdict == "promote":
            incumbent = sug.candidate
        logs.append(RoundLog(round=k, mode=sug.context.mode, kind=sug.kind, candidate=sug.candidate.id,
                             verdict=dec.verdict, ceiling=dec.ceiling, reason=dec.reason,
                             incumbent_after=incumbent.id))
    confirmed = None
    if terminal is not None and incumbent.id != incumbent0.id:
        confirmed = is_gain(terminal)
    return SimResult(final=incumbent, rounds=tuple(logs), confirmed=confirmed)
