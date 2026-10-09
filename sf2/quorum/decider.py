"""quorum_decider: the swarm as a drop-in ``decide(moment)`` for loop_runner.play_round(decide=...).

Per decision:
  1. key the moment like value_table (range, his doing, fireball [+ his_label split]) and list the followable moves;
  2. run today's two-stage text-laya pick (``base_decide``) -- it is the "laya" voter and the fallback;
  3. ask the flavours (if any), let the table vote from its stats, and add the explore bees (TRAIN stage only);
  4. score the candidates (tally.score) and compute the top move's share;
  5. act by ``cfg.mode`` (config.py), tag the decision ``source`` (laya / quorum / table / explore / qwen /
     fallback) and attach a ``quorum`` record (every proposal with its weight, the scores, the share, what a
     quorum would have done) that reliability.credit reads back after the round.
The table and reliability state are read as given; rebuild the decider each round against the credited state.
"""
import random
from typing import Callable, Dict, Optional

from ..system1 import value_table as VT
from ..system1.action_menu import DEFAULT_MOVE
from ..system1.advice import available_moves, char_categories, prompt, stance_of
from ..system1.screen_words import sentence
from ..vocab import range_of
from . import reliability as R
from .config import QuorumConfig
from .frontier import counter_proposals, explore_proposals
from .tally import ranking, score, table_proposal
from .voters import base_proposal, flavor_proposals

QwenPick = Callable[[str, list, list], Optional[str]]


def quorum_decider(table: VT.Table, rel: R.State, me: str, rng: random.Random, base_decide, cat_advisor,
                   move_advisor, cfg: QuorumConfig, qwen_pick: Optional[QwenPick] = None):
    cats = char_categories(me)

    def decide(m) -> Dict:
        rng_ = range_of(abs(m.dx))
        stance = stance_of("stand", rng_)                     # decisions are only taken when she can act (grounded)
        actions = set(available_moves(stance, cats)) | {DEFAULT_MOVE}
        when = VT.when_key(rng_, m.doing, m.fireball, VT.split_label_moment(m), table.get("depth", {}))
        cell = VT._cell_view(table, when)
        base = dict(base_decide(m))
        text = prompt(sentence(m), base.get("prompt_lines", []))
        props = [p for p in [base_proposal(base)] if p is not None]
        props += flavor_proposals(cfg.flavors, cat_advisor, move_advisor, text, stance, cats)
        acts = sorted(actions)
        tp = table_proposal(cell, acts, cfg, laya_move=base["action"])
        if tp is not None:
            props.append(tp)
        props += explore_proposals(cell, acts, when, cfg)       # the gap-filling bees: TRAIN only, [] at eval
        props += counter_proposals(cell, acts, when, cfg)       # purposed counter-bees: TRAIN only, off by default
        props = [p for p in props if p.action in actions]
        scores = score(props, cell, when, rel, cfg)
        order, share = ranking(scores) if scores else ([base["action"]], 0.0)
        quorum_move = order[0] if share >= cfg.theta else None

        action, source = base["action"], "laya"
        if cfg.mode == "candidates":
            action, source = _candidates(order, cell, base["action"], rng, cfg)
        elif cfg.mode == "vote":
            if len(order) > 1 and rng.random() < cfg.epsilon:
                action, source = order[1], "explore"
            elif quorum_move is not None:
                action, source = quorum_move, "quorum"
            else:
                picked = qwen_pick(text, order, props) if (cfg.qwen and qwen_pick) else None
                action, source = (picked, "qwen") if picked in order else (base["action"], "fallback")

        out = dict(base)
        out.update(action=action, when=when, source=source, explored=source == "explore",
                   category=_category(action, cats))
        out["quorum"] = {"mode": cfg.mode, "share": round(share, 4), "theta": cfg.theta, "top": order[0],
                         "quorum_move": quorum_move, "laya_move": base["action"],
                         "proposals": [[p.voter, p.action, round(p.confidence, 4),
                                        round(R.weight(rel, when, p.voter, cfg), 4)] for p in props],
                         "scores": {a: round(s["score"], 4) for a, s in scores.items()}}
        return out
    return decide


def _category(move: str, cats) -> str:
    """The round-1 category of ``move`` in this character's menu (action_menu.category_of knows only Chun-Li's)."""
    return next((c for c, moves in cats.items() if move in moves), "block")


def _candidates(order, cell, laya_move: str, rng: random.Random, cfg: QuorumConfig):
    """Phase 1: the move must be one of the swarm's candidates. Explore an under-sampled candidate now and then;
    else the best candidate the table is confident is good (n >= table_min_n, mean > 0); else text-laya's pick."""
    under = [a for a in order if VT.count(cell, a) < cfg.table_min_n]
    if under and rng.random() < cfg.epsilon:
        return rng.choice(under), "explore"
    good = [(VT.mean(cell.get(a)), a) for a in order if VT.count(cell, a) >= cfg.table_min_n and VT.mean(cell.get(a)) > 0]
    if good:
        return max(good)[1], "table"
    return laya_move, "laya"
