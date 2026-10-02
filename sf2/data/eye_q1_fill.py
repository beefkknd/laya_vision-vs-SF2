"""docs/eye_questions_v1.md, "Next: more fireball data": how full q1 (fireball yes / no, v1.1 rules) would be after a
collection round - the one line per round (scripts/fill_eye_q1.py). Counts exactly what the q1 build selects
(sf2.data.eye_data.cands_q1 + select over the eye frame pool, same matching, no cap):

counts        yes / no rows per split (train / val / test, sf2.data.pairs_train.split3 by whole match)
short_strata  train strata (split, match pair, game range, poses) with more "yes" candidates than "no": the matching
              drops their extra "yes" rows (``yes_lost``)
shots         cells not full: (match pair, game >= from_game, thrower slot, flight stage) whose projectile samples are
              under the per-game cap (sf2.data.pairs_shots.ShotSampler) - all full means the cap, not play, limits
"""
import collections
import os
from typing import Dict, Iterable, Optional

from . import eye_data as E
from . import pairs_collect_io as IO
from . import pairs_shots as S
from . import pairs_train as T
from .movement_collect import STAGES

ANSWERS = E.Q["q1"]["answers"]
TARGET = 2000
FROM_GAME = 32


def fill(pool: Iterable[Dict], root: str, prefer: Optional[set] = None, seed: int = 0, from_game: int = FROM_GAME,
         cap: int = S.SHOT_PER_GAME) -> Dict:
    cands, dropped = E.cands_q1(pool, prefer or set())
    rows = E.select(cands, ANSWERS, E.Q["q1"]["cap"], seed)
    got = collections.Counter((r["answer"], r["split"]) for r in rows)
    per: Dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    for c in cands:
        if c["split"] == "train":
            per[c["stratum"]][c["answer"]] += 1
    short = {k: v for k, v in per.items() if v["yes"] > v["no"]}
    ycand = collections.Counter(c["split"] for c in cands if c["answer"] == "yes")
    return {"counts": {a: {f: got[(a, f)] for f in T.FILES} for a in ANSWERS},
            "yes_candidates": {f: ycand[f] for f in T.FILES}, "short_strata": len(short),
            "yes_lost": sum(v["yes"] - v["no"] for v in short.values()), "dropped": dropped,
            "matches": {f: len({(r["pair_name"], r["game"]) for r in rows if r["split"] == f}) for f in T.FILES},
            "shots": shot_cells(root, from_game, cap), "from_game": from_game, "shot_cap": cap}


def shot_cells(root: str, from_game: int = FROM_GAME, cap: int = S.SHOT_PER_GAME) -> Dict[str, int]:
    """Cells (match pair, committed game >= from_game, thrower slot, stage) and how many hold fewer than ``cap``
    projectile samples."""
    cells = not_full = 0
    for name in sorted(os.listdir(root)):
        base = os.path.join(root, name)
        if "_vs_" not in name or not os.path.isdir(base):
            continue
        chars = dict(zip((1, 2), name.split("_vs_")))
        games = {g["game"] for g in IO.committed(base) if g["game"] >= from_game}
        taken = collections.Counter((s["game"], s["slot"], s["flight_stage"]) for s in S.committed_shots(base)[0]
                                    if s["game"] in games)
        for g in games:
            for slot in (1, 2):
                if chars[slot] in S.THROWERS:
                    for stage in STAGES:
                        cells += 1
                        not_full += taken[(g, slot, stage)] < cap
    return {"cells": cells, "not_full": not_full}


def met(rep: Dict, target: int = TARGET) -> bool:
    return all(rep["counts"][a]["train"] >= target for a in ANSWERS)


def line(rep: Dict, target: int = TARGET) -> str:
    c = rep["counts"]
    return ("q1 train yes %d no %d (val %d, test %d) | yes candidates train %d, lost to matching %d in %d strata"
            " | shot cells not full %d/%d | %.0f s | target %d %s") % (
        c["yes"]["train"], c["no"]["train"], c["yes"]["val"], c["yes"]["test"], rep["yes_candidates"]["train"],
        rep["yes_lost"], rep["short_strata"], rep["shots"]["not_full"], rep["shots"]["cells"],
        rep.get("seconds", 0.0), target, "MET" if met(rep, target) else "not met")
