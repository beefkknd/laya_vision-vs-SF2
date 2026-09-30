"""The lookup-table value ranking (Dr Fable's review 2026-09-30; docs/prereg_value_oracle.md): the mean net (dealt -
taken until the next decision) per (character, range, opp_attacking, opp_airborne, move), from the EXPLORED decisions
of the value collection's training games (a random move: no bias from the policy's timing), shrunk toward 0 by
n / (n + SHRINK). General: the note's fields only, never the opponent's name. Chun-Li vs Guile stays held out."""
import glob
import json
import os
import re
from typing import Dict, Iterable, List, Tuple

from .vs_sweep import TEST_INDEX

SHRINK = 10               # a cell with n tries keeps n / (n + SHRINK) of its mean
HELDOUT = ("chunli", "guile")
Cell = Tuple[str, str, int, int]

_NOTE = re.compile(r"^me=(\w+) dist=(\w+) .* opp_airborne=([01]) opp_crouch=[01] opp_attacking=([01])$")


def cell(note: str) -> Cell:
    """(me, range, opp_attacking, opp_airborne) from a v2 note."""
    m = _NOTE.match(note)
    if not m:
        raise ValueError("not a v2 note: %r" % note)
    return m.group(1), m.group(2), int(m.group(4)), int(m.group(3))


def _cell_of(e: Dict) -> Cell:
    return e["me"], e["range"], int(e["opp_state"] in ("attack", "special")), int(bool(e["opp_air"]))


def build(entries: Iterable[Dict]) -> Dict[Cell, Dict[str, float]]:
    sums: Dict[Cell, Dict[str, List[float]]] = {}
    for e in entries:
        if not e.get("explored") or e["game"] % 10 in TEST_INDEX or (e["me"], e["opp"]) == HELDOUT:
            continue
        acc = sums.setdefault(_cell_of(e), {}).setdefault(e["action"], [0.0, 0])
        acc[0] += e["dealt"] - e["taken"]
        acc[1] += 1
    return {c: {m: s / n * n / (n + SHRINK) for m, (s, n) in moves.items()} for c, moves in sums.items()}


def rank(table: Dict[Cell, Dict[str, float]], note: str, moves: Iterable[str]) -> Dict[str, float]:
    row = table.get(cell(note), {})
    return {m: round(float(row.get(m, 0.0)), 3) for m in moves}


def from_logs(root: str) -> Dict[Cell, Dict[str, float]]:
    paths = sorted(glob.glob(os.path.join(root, "*", "*", "actions.jsonl")))
    if not paths:
        raise SystemExit("no value-collection logs under %s" % root)
    return build(json.loads(line) for p in paths for line in open(p))


def save(table: Dict[Cell, Dict[str, float]], path: str) -> None:
    with open(path, "w") as f:
        json.dump([{"cell": list(c), "values": v} for c, v in sorted(table.items())], f, indent=1, sort_keys=True)


def load(path: str) -> Dict[Cell, Dict[str, float]]:
    return {tuple(r["cell"]): r["values"] for r in json.load(open(path))}
