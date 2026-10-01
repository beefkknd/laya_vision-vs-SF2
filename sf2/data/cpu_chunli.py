"""CPU Chun-Li games (docs/prereg_movement_data.md, "Addition (2026-10-01): CPU Chun-Li games"): each of the 7 other
characters is player 1 (runs/all8, explore 0.5), the arcade CPU's Chun-Li is player 2. Her action rows only, in their
own dataset (test_data_act_cpu), never merged silently with test_data_act: every pair and every row carries its
provenance (PROVENANCE_KEYS), equal to the collection's run settings (<root>/run.json).
"""
import json
import os
from typing import Dict, List, Optional, Sequence

from ..vocab import FIGHTERS

COLLECTION = "mv4"
CPU = "chunli"                    # the CPU's character, player 2
SLOT = "p2"
CONTROLLER = "cpu"
P1_CHARS = ("blanka", "dhalsim", "guile", "honda", "ken", "ryu", "zangief")
PROVENANCE_KEYS = ("collection", "chunli_slot", "controller", "p1_char", "note")
WATCHED = (23, 24, 25, 26, 27, 32)   # her jump attacks the policy never made (absent from mv3)


def note(p1_char: str) -> str:
    return "me=%s" % p1_char


def provenance(p1_char: str) -> Dict[str, str]:
    """The provenance of one player-1 character's games."""
    if p1_char not in FIGHTERS or p1_char == CPU:
        raise ValueError("player 1 must be one of the other characters, got %r" % p1_char)
    return {"collection": COLLECTION, "chunli_slot": SLOT, "controller": CONTROLLER, "p1_char": p1_char,
            "note": note(p1_char)}


def provenance_problems(rec: Dict, want: Dict[str, str]) -> List[str]:
    """What is missing or different in ``rec``'s provenance fields; empty = OK."""
    out = []
    for k in PROVENANCE_KEYS:
        if k not in rec:
            out.append("provenance %s missing" % k)
        elif rec[k] != want[k]:
            out.append("provenance %s = %r, run says %r" % (k, rec[k], want[k]))
    return out


def run_settings(p1_chars: Sequence[str], **settings) -> Dict:
    """The collection's run.json: the provenance constants, the player-1 characters and the given settings."""
    for c in p1_chars:
        provenance(c)
    return dict(settings, collection=COLLECTION, chunli_slot=SLOT, controller=CONTROLLER, cpu=CPU,
                p1_chars=list(p1_chars))


def write_run(root: str, settings: Dict) -> None:
    os.makedirs(root, exist_ok=True)
    with open(os.path.join(root, "run.json"), "w") as f:
        json.dump(settings, f, indent=1)


def read_run(root: str) -> Optional[Dict]:
    path = os.path.join(root, "run.json")
    return json.load(open(path)) if os.path.exists(path) else None


def expected_provenance(run: Dict, p1_char: str) -> Dict[str, str]:
    """A row's provenance as the run's settings give it (not as the builder wrote it)."""
    if p1_char not in run.get("p1_chars", []):
        raise ValueError("%r is not a player-1 character of this run (%s)" % (p1_char, run.get("p1_chars")))
    return {"collection": run["collection"], "chunli_slot": run["chunli_slot"], "controller": run["controller"],
            "p1_char": p1_char, "note": note(p1_char)}


def compare_codes(table: Dict[str, Dict], mv3_codes: Sequence[int]) -> Dict:
    """Her codes in this dataset (the gate's coverage table, keys "chunli actNN") against mv3's Chun-Li codes: the
    codes new here, and the watched jump-attack IDs (None = absent)."""
    mine = {k.split()[1]: v for k, v in table.items() if k.startswith(CPU + " ")}
    with_rows = sorted(c for c, v in mine.items() if v["train"] + v["val"] + v["test"])
    old = {"act%02d" % c for c in mv3_codes}
    return {"codes": with_rows, "new_vs_mv3": [c for c in with_rows if c not in old],
            "in_mv3_not_here": sorted(old - set(with_rows)),
            "watched": {"act%02d" % i: mine.get("act%02d" % i) for i in WATCHED}}
