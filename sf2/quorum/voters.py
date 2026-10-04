"""The laya voters: the existing two-stage pick, and the same two checkpoints asked once per flavour.

A flavour is a subset of the 7 round-1 categories (config.FLAVORS). It asks the CATEGORY model to choose only among
its categories that the stance offers (skipped when there is just one), then the MOVE model inside that category --
the same two-step shape both checkpoints were trained on, over a narrower menu. Same weights, different questions:
each flavour scouts its own region of the move set. Nothing here forces a move; the models' picks stand.
"""
from typing import Dict, List, Optional, Sequence

from ..system1.advice import CAT_INSTRUCTIONS, move_question, moves_in_stance
from .tally import Proposal


def base_proposal(decision: Dict) -> Optional[Proposal]:
    """The "laya" voter = today's two-stage decision (loop_runner.two_stage_decide's dict). Confidence = P(category)
    x P(move | category), the probability the two models gave the path they took."""
    action = decision.get("action")
    if not action:
        return None
    pc = (decision.get("cat_probs") or {}).get(decision.get("category"), 1.0)
    pm = (decision.get("move_probs") or {}).get(action, 1.0)
    return Proposal("laya", action, float(pc) * float(pm))


def flavor_proposal(name: str, cats: Sequence[str], cat_advisor, move_advisor, text: str, stance: str,
                    categories: Dict[str, Sequence[str]]) -> Optional[Proposal]:
    """One flavour's vote, or None when the stance offers none of its categories (e.g. 'move' in the air)."""
    live = [c for c in cats if c in categories and moves_in_stance(c, stance, categories)]
    if not live:
        return None
    if len(live) == 1:
        cat, pc = live[0], 1.0
    else:
        probs = cat_advisor.ask(text, {"type": "choice", "instructions": CAT_INSTRUCTIONS,
                                       "criteria": {c: c for c in live}})
        cat = max(live, key=lambda c: probs.get(c, 0.0))
        pc = float(probs.get(cat, 0.0))
    options = moves_in_stance(cat, stance, categories)
    probs = move_advisor.ask(text, move_question(options))
    move = max(options, key=lambda m: probs.get(m, 0.0))
    return Proposal(name, move, pc * float(probs.get(move, 0.0)))


def flavor_proposals(flavors: Dict[str, List[str]], cat_advisor, move_advisor, text: str, stance: str,
                     categories: Dict[str, Sequence[str]]) -> List[Proposal]:
    out = []
    for name, cats in flavors.items():
        p = flavor_proposal(name, cats, cat_advisor, move_advisor, text, stance, categories)
        if p is not None:
            out.append(p)
    return out
