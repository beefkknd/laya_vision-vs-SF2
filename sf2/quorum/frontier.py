"""Gap-filling voters: fill the table's thin/blind cells instead of forcing a category.

Owner 2026-10-05, after A (rollouts/loop_screen/A_20261005_084309) showed the force-combo bee hurt (quorum 65% <
71% baseline): it spammed a move the thick trunk already owned. The study of that table found the real gaps -- 7
blind contexts (no move at n>=20), half the cells thin (n<20), and the whole fireball-up slice nearly unexplored
(fb=1: 273 samples vs fb=0's 7845). These voters target exactly those gaps.

  frontier_proposal  votes the least-sampled followable action; confidence = k / (k + covered_n), where covered_n is
                     the sample count the table has on a CONFIDENT POSITIVE winner in the cell. So confidence is 1.0
                     in a blind cell (no winner yet) and fades as a winner accrues evidence -- the bee LEADS the
                     exploration where the table is uncertain and DEFERS to the thick trunk where it is sure.
  fireball_proposal  the same, but only when a fireball is out (the 'when' key's fireball field is '1'); at fb=1
                     both bees fire on the same gap, doubling the push into the slice A found most under-explored.

Pure: value_table stats only, no models, no I/O. ``cell`` is value_table's per-action stats for the 'when'
({action: [n, sum, sumsq]}); ``actions`` the followable set at that range (decider restricts it).
"""
from typing import Dict, List, Optional, Sequence

from ..system1 import value_table as VT
from .config import QuorumConfig
from .tally import Proposal


def _covered_n(cell: Dict[str, List], actions: Sequence[str], cfg: QuorumConfig) -> int:
    """The table's grip on this cell: the largest sample count among confident (n >= table_min_n) positive moves.
    0 when nothing good is known yet -- a blind or all-losing cell is still a gap worth exploring."""
    ns = [VT.count(cell, a) for a in actions
          if VT.count(cell, a) >= cfg.table_min_n and VT.mean(cell.get(a)) > 0]
    return max(ns, default=0)


def _least_sampled(cell: Dict[str, List], actions: Sequence[str]) -> Optional[str]:
    """The followable action the table knows least (ties -> name, so a replay is deterministic). None if no actions."""
    acts = sorted(actions)
    return min(acts, key=lambda a: (VT.count(cell, a), a)) if acts else None


def _confidence(cell: Dict[str, List], actions: Sequence[str], cfg: QuorumConfig) -> float:
    """1.0 in a blind cell, fading to k / (k + covered_n) as a confident positive winner accrues evidence."""
    return cfg.k / (cfg.k + _covered_n(cell, actions, cfg))


def frontier_proposal(cell: Dict[str, List], actions: Sequence[str], cfg: QuorumConfig) -> Optional[Proposal]:
    """Vote the least-sampled followable action, loud where the table is uncertain. None = abstain (off / no actions)."""
    if not cfg.frontier:
        return None
    target = _least_sampled(cell, actions)
    if target is None:
        return None
    return Proposal("frontier", target, _confidence(cell, actions, cfg))


def _slice_proposal(name: str, cell: Dict[str, List], actions: Sequence[str], when: str, cfg: QuorumConfig,
                    field: int, value: str) -> Optional[Proposal]:
    """The frontier push, gated to one slice of the 'when' key (field == value). None otherwise."""
    parts = when.split("|")
    if len(parts) <= field or parts[field] != value:
        return None
    target = _least_sampled(cell, actions)
    if target is None:
        return None
    return Proposal(name, target, _confidence(cell, actions, cfg))


def fireball_proposal(cell: Dict[str, List], actions: Sequence[str], when: str,
                      cfg: QuorumConfig) -> Optional[Proposal]:
    """The frontier push, gated to fireball-up (the 'when' key's fireball field == '1'). None otherwise."""
    if not cfg.fireball:
        return None
    return _slice_proposal("fireball", cell, actions, when, cfg, field=2, value="1")


def pressure_proposal(cell: Dict[str, List], actions: Sequence[str], when: str,
                      cfg: QuorumConfig) -> Optional[Proposal]:
    """The frontier push, gated to the opponent ATTACKING (the 'when' posture field == 'attacking') -- the
    'being-pressured' slice, where a zoner (e.g. ryu) is blind because it cannot zone. None otherwise."""
    if not cfg.pressure:
        return None
    return _slice_proposal("pressure", cell, actions, when, cfg, field=1, value="attacking")
