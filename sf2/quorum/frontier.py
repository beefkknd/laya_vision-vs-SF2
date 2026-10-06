"""Gap-filling (EXPLORE) voters: fill the table's thin/blind cells instead of forcing a category. TRAIN stage only.

Owner 2026-10-05, after A (rollouts/loop_screen/A_20261005_084309) showed the force-combo bee hurt (quorum 65% <
71% baseline): it spammed a move the thick trunk already owned. The study of that table found the real gaps -- 7
blind contexts (no move at n>=20), half the cells thin (n<20), and the whole fireball-up slice nearly unexplored
(fb=1: 273 samples vs fb=0's 7845). These voters target exactly those gaps.

  frontier_proposal  votes the least-sampled followable action; confidence = k / (k + covered_n), where covered_n is
                     the sample count the table has on a CONFIDENT POSITIVE winner in the cell. So confidence is 1.0
                     in a blind cell (no winner yet) and fades as a winner accrues evidence -- the bee LEADS the
                     exploration where the table is uncertain and DEFERS to the thick trunk where it is sure.
  the gated bees     the same push, each gated to one slice of the 'when' key (GATES): fireball (fb=1), pressure /
                     punish / vs_crouch / antiair (the opponent's posture). Open gates all fire on the same gap.
  explore_proposals  the decider's one call: every bee the config turns on, in VOTER order; [] at the EVAL stage.

Pure: value_table stats only, no models, no I/O. ``cell`` is value_table's per-action stats for the 'when'
({action: [n, sum, sumsq]}); ``actions`` the followable set at that range (decider restricts it).
"""
from typing import Dict, List, Optional, Sequence

from ..system1 import value_table as VT
from .config import EXPLORE_BEES, QuorumConfig
from .tally import Proposal

GATES = {"fireball": (2, "1"), "pressure": (1, "attacking"), "punish": (1, "stunned"),    # bee -> ('when' field, value
         "vs_crouch": (1, "crouching"), "antiair": (1, "jumping")}                        # that opens its gate)


def _covered_n(cell: Dict[str, List], actions: Sequence[str], cfg: QuorumConfig) -> int:
    """The table's grip on this cell: the largest sample count among confident (n >= table_min_n) positive moves.
    0 when nothing good is known yet -- a blind or all-losing cell is still a gap worth exploring."""
    ns = [VT.count(cell, a) for a in actions
          if VT.count(cell, a) >= cfg.table_min_n and VT.mean(cell.get(a)) > 0]
    return max(ns, default=0)


def _push(name: str, cell: Dict[str, List], actions: Sequence[str], cfg: QuorumConfig) -> Optional[Proposal]:
    """The frontier push under ``name``: the followable action the table knows least (ties -> name, so a replay is
    deterministic), with confidence 1.0 in a blind cell fading to k / (k + covered_n) as a confident positive winner
    accrues evidence. None = no actions."""
    acts = sorted(actions)
    if not acts:
        return None
    return Proposal(name, min(acts, key=lambda a: (VT.count(cell, a), a)), cfg.k / (cfg.k + _covered_n(cell, acts, cfg)))


def frontier_proposal(cell: Dict[str, List], actions: Sequence[str], cfg: QuorumConfig) -> Optional[Proposal]:
    """Vote the least-sampled followable action, loud where the table is uncertain. None = abstain (off / no actions)."""
    return _push("frontier", cell, actions, cfg) if cfg.frontier else None


def _gated(name: str):
    """One slice bee: the frontier push only where the 'when' key opens its gate (GATES) and the config turns it on."""
    field, value = GATES[name]

    def proposal(cell: Dict[str, List], actions: Sequence[str], when: str, cfg: QuorumConfig) -> Optional[Proposal]:
        parts = when.split("|")
        if not getattr(cfg, name) or len(parts) <= field or parts[field] != value:
            return None
        return _push(name, cell, actions, cfg)
    return proposal


_GATED = [_gated(b) for b in EXPLORE_BEES[1:]]                 # VOTER order: fireball, pressure, punish, vs_crouch, antiair
fireball_proposal, pressure_proposal, punish_proposal, vs_crouch_proposal, antiair_proposal = _GATED


def explore_proposals(cell: Dict[str, List], actions: Sequence[str], when: str, cfg: QuorumConfig) -> List[Proposal]:
    """Every explore bee's vote (VOTER order); [] at the EVAL stage, where only the exploit voters (laya, table) play."""
    bees = [] if cfg.stage == "eval" else [frontier_proposal(cell, actions, cfg)] + [g(cell, actions, when, cfg) for g in _GATED]
    return [p for p in bees if p is not None]
