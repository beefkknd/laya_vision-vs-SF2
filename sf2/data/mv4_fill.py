"""Round 4 step 1 of docs/prereg_movement_finetunes.md: how full the round-4 datasets would be after a round, the
one line per round (scripts/fill_mv4.py). Counts exactly what the builds select:

act       sf2.data.pairs_data.eligible + select with the per-movement caps (attack / special 100 train / 30 test,
          the others unchanged) and the kept round-3 build; attack / special rows per split (train / val / test,
          sf2.data.pairs_train.split3). The act build later drops the few rows whose two x are equal at t.
fireball  sf2.data.mv3_fireball.checked + select with its caps (120 train / 30 test per thrower x side x flight
          stage); left / right rows per split.
Cells not full: act (split, char, attack|special, facing) and fireball (split, thrower, side, stage) under their cap.
"""
import collections
from typing import Dict, Optional

from . import mv3_fireball as F
from . import pairs_data as D
from . import pairs_train as T

ACT = ("attack", "special")
FIRE = ("left", "right")
CAPS = {"train": 40, "test": 20}
MOVEMENT_CAPS = {"attack": {"train": 100, "test": 30}, "special": {"train": 100, "test": 30}}
FIRE_CAPS = {"train": 120, "test": 30}
TARGET = 1000


def fill(root: str, caps: Dict[str, int] = CAPS, movement_caps: Optional[Dict] = None, keep_from: Optional[str] = None,
         fire_caps: Dict[str, int] = FIRE_CAPS, bands: Optional[Dict[str, int]] = None, seed: int = 0) -> Dict:
    mcaps = MOVEMENT_CAPS if movement_caps is None else movement_caps
    pool, _, names, _ = D.eligible(root, seed, D.CONTROLLERS, bands)
    keep = D.keep_keys(keep_from) if keep_from else set()
    chosen = D.select(pool, caps, seed, mcaps, keep)
    act = {a: dict.fromkeys(T.FILES, 0) for a in ACT}
    cells = collections.Counter()
    for p in chosen:
        mv = D.movement_answer(p["movement"], p["direction"])
        if mv in ACT:
            act[mv][T.split3(p["pair_name"], p["game"])] += 1
            cells[(D.split_of(p),) + D.cell(p)] += 1
    chars = {c for n in names for c in n.split("_vs_")}
    act_not_full = {"|".join(k): "%d/%d" % (cells[k], n) for k, n in D.cell_caps(chars, caps, mcaps).items()
                    if k[2] in ACT and cells[k] < n}
    kept, _ = F.checked(root, F.shot_samples(root))
    fire_rows = F.select(kept, fire_caps, seed)
    fire = {a: dict.fromkeys(T.FILES, 0) for a in FIRE}
    fcells = collections.Counter(F._cell(r) for r in fire_rows)
    for r in fire_rows:
        fire[r["side"]][T.split3(r["pair_name"], r["game"])] += 1
    fire_not_full = {"|".join(k): "%d/%d" % (fcells[k], fire_caps[k[0]]) for k in F.universe(fire_caps)
                     if fcells[k] < fire_caps[k[0]]}
    return {"act": act, "fireball": fire, "act_not_full": act_not_full, "fire_not_full": fire_not_full,
            "act_cells": len([k for k in D.cell_caps(chars, caps, mcaps) if k[2] in ACT]),
            "fire_cells": len(F.universe(fire_caps)), "kept": len(keep)}


def met(rep: Dict, target: int = TARGET) -> bool:
    return all(rep["act"][a]["train"] >= target for a in ACT) and \
        all(rep["fireball"][a]["train"] >= target for a in FIRE)


def line(rep: Dict, target: int = TARGET) -> str:
    return ("act train attack %d special %d | fireball train left %d right %d | not full: act %d/%d, fireball %d/%d"
            " | target %d %s") % (
        rep["act"]["attack"]["train"], rep["act"]["special"]["train"], rep["fireball"]["left"]["train"],
        rep["fireball"]["right"]["train"], len(rep["act_not_full"]), rep["act_cells"], len(rep["fire_not_full"]),
        rep["fire_cells"], target, "MET" if met(rep, target) else "not met")
