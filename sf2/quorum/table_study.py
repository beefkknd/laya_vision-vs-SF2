"""Study one value table, then suggest which bees fill its gaps -- the trunk-study -> bee-suggestion process, made
callable (owner 2026-10-05). ``study`` reads a table's cells and returns what it knows (the thick trunk), where it
is blind/thin, how well the fireball slice is covered, and which cells are promising but under-sampled. ``suggest_bees``
maps that analysis to a bee set (which gap-fillers to enable). Pure: table stats only, no models, no I/O.

The "when" key is value_table's range|posture|fireball (+ optional his_label). A "confident" cell has a move at
n >= confident_n; a "promising-thin" cell has a positive-mean move with min_n <= n < confident_n (worth more trials).
"""
from typing import Dict, List, Sequence

CONFIDENT_N = 20       # MIN_TRIES: samples before a cell is "known" (value_table.MIN_TRIES)
MIN_N = 4              # a move needs at least this many trials before its mean is worth reading


def _mean(stat: Sequence) -> float:
    n = stat[0]
    return stat[1] / n if n else 0.0


def _fireball_field(when: str) -> str:
    parts = when.split("|")
    return parts[2] if len(parts) >= 3 else "0"


def study(cells: Dict[str, Dict[str, List]], confident_n: int = CONFIDENT_N, min_n: int = MIN_N) -> Dict:
    """What the table knows and lacks. Returns trunk (best confident+ move per context), blind contexts (none
    confident), fb slice coverage, promising-thin cells, and the headline counts."""
    trunk, blind, promising = [], [], []
    cell_count = confident = thin = 0
    fb = {"fb0_cells": 0, "fb0_n": 0, "fb1_cells": 0, "fb1_n": 0}
    for when in sorted(cells):
        moves = cells[when]
        is_fb1 = _fireball_field(when) == "1"
        best = None
        for mv, stat in moves.items():
            n, m = stat[0], _mean(stat)
            cell_count += 1
            confident += n >= confident_n
            thin += n < confident_n
            fb["fb1_cells" if is_fb1 else "fb0_cells"] += 1
            fb["fb1_n" if is_fb1 else "fb0_n"] += n
            if n >= confident_n and m > 0 and (best is None or m > best[0]):
                best = (m, mv, n)
            if min_n <= n < confident_n and m > 0:
                promising.append((when, mv, m, n))
        if best is not None:
            trunk.append((when, best[1], best[0], best[2]))
        else:
            blind.append((when, sum(s[0] for s in moves.values())))
    total_n = fb["fb0_n"] + fb["fb1_n"]
    fb["fb1_share"] = fb["fb1_n"] / total_n if total_n else 0.0
    promising.sort(key=lambda t: -t[2])
    trunk.sort(key=lambda t: -t[2])
    return {"contexts": len(cells), "cells": cell_count, "confident": confident, "thin": thin,
            "blind": blind, "trunk": trunk, "fb": fb, "promising": promising}


def suggest_bees(analysis: Dict, fb_share_floor: float = 0.15) -> Dict:
    """Map the gaps to a bee set. frontier whenever anything is thin/blind; fireball when the fb=1 slice is
    under-represented; never re-force a category (that is what A showed hurts)."""
    notes: List[str] = []
    n_blind, n_thin = len(analysis["blind"]), analysis["thin"]
    frontier = n_thin > 0 or n_blind > 0
    if frontier:
        notes.append("frontier bee: %d blind context(s) + %d thin cell(s) -> explore the least-sampled followable move"
                     % (n_blind, n_thin))
    fb = analysis["fb"]
    fireball = fb["fb1_share"] < fb_share_floor
    if fireball:
        where = "no fb=1 contexts seen yet" if fb["fb1_cells"] == 0 else "fb=1 only %.1f%% of samples" % (
            100 * fb["fb1_share"])
        notes.append("fireball bee: %s -> fill the fireball slice" % where)
    for when, mv, m, n in analysis["promising"][:5]:
        notes.append("priority gap: %-22s %-20s mean %+.2f but only n=%d" % (when, mv, m, n))
    return {"frontier": frontier, "fireball": fireball, "flavors": {}, "notes": notes}
