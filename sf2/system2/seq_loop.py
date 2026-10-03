"""Option-D session driver (pure orchestration; measurer + proposer injected).

Per opponent, the WIN objective comes first: if the incumbent already reliably beats the opponent it is
SOLVED and we FREEZE (no proposals). Otherwise we climb: each round the Coach proposes a candidate,
which must pass the cheap dev SCREEN and then a sequential, pooled, powered CONFIRM on FRESH blocks
(sf2.system2.sequential). One promotion per round (noise control); after each promotion we re-check
SOLVED. Every screen/confirm result is written to the evidence ledger.

Seeds are handed out as FRESH blocks from a pool (dev + up to max_looks confirm blocks per candidate),
never reused, so the confirm stage is honest. The solved block measures the incumbent's own win-rate.
"""
from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence, Tuple

from sf2.system2.outcome_loop import Playbook
from sf2.system2.sequential import Block, SeqCfg, confirm, screen, solved


@dataclass(frozen=True)
class SeqSeeds:
    solved: Tuple[int, ...]              # block to measure the incumbent's win-rate (solved gate)
    pool: Tuple[Tuple[int, ...], ...]    # fresh blocks consumed in order for dev + confirm looks
    terminal: Tuple[int, ...]            # reserved untouched block (rollback check), never in-loop


@dataclass(frozen=True)
class SeqResult:
    final: Playbook
    status: str                          # "solved" | "unsolved"
    rows: Tuple[dict, ...]
    looks_used: int


# measure_block(incumbent, candidate, seeds) -> Block ; measure_wins(playbook, seeds) -> (wins, n)
def run_opponent(opp: str, incumbent: Playbook, propose, measure_block, measure_wins,
                 seeds: SeqSeeds, n_rounds: int, cfg: SeqCfg = SeqCfg()) -> SeqResult:
    rows: List[dict] = []
    pool = list(seeds.pool)
    looks_used = 0

    def log(**kw):
        rows.append({"opp": opp, **kw})

    def is_solved(pb):
        w, n = measure_wins(pb, seeds.solved)
        s = solved(w, n, cfg)
        log(stage="solved_check", incumbent=pb.id, inc_wins=w, n=n, solved=s)
        return s

    if is_solved(incumbent):
        return SeqResult(incumbent, "solved", tuple(rows), looks_used)

    for k in range(n_rounds):
        for cand in propose(opp, incumbent, k):
            if not pool:
                log(stage="out_of_seeds", round=k, candidate=cand.id)
                return SeqResult(incumbent, "unsolved", tuple(rows), looks_used)
            dev = measure_block(incumbent, cand, pool.pop(0))
            ok, why = screen(dev, cfg)
            log(stage="screen", round=k, candidate=cand.id, rules=list(cand.rules),
                dev_fire=dev.fire_rate, dev_follows=dev.follows, screened=ok, reason=why)
            if not ok:
                continue
            blocks: List[Block] = []
            dec = None
            for _ in range(cfg.max_looks):
                if not pool:
                    log(stage="out_of_seeds", round=k, candidate=cand.id)
                    break
                blocks.append(measure_block(incumbent, cand, pool.pop(0)))
                looks_used += 1
                dec = confirm(blocks, cfg)
                log(stage="confirm", round=k, candidate=cand.id, look=len(blocks),
                    verdict=dec.verdict, z=round(dec.z, 2), delta=round(dec.delta, 1), reason=dec.reason)
                if dec.verdict in ("promote", "reject"):
                    break
            if dec is not None and dec.verdict == "promote":
                incumbent = cand
                log(stage="promote", round=k, candidate=cand.id, rules=list(cand.rules))
                if is_solved(incumbent):
                    return SeqResult(incumbent, "solved", tuple(rows), looks_used)
                break  # one promotion per round (noise control)

    return SeqResult(incumbent, "unsolved", tuple(rows), looks_used)
