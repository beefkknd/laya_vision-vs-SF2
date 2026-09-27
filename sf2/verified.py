"""The ALL-8 gate (docs/TWO_SYSTEM_PLAN.md, "HARD CHECKPOINT ALL-8"): no training until every move of every character
is verified on the ROM from both facings.

The ROM move tests (tests/test_rom_moves.py) write one record file per (character, move, facing) they verified, so
any number of test processes can write at once:

    out/verified_moves/ryu.hadoken.right.json = {"stamp": "<sha256 of the harness stamp>", "test": "<pytest node id>"}

The stamp is sf2.cli's harness stamp (ROM sha1 + the harness code + the RAM map), so a record made before the harness
changed does not count. ``check_moves`` refuses unless every character has its specials in the code and every move on
its list (``sf2.actions.moves``) has a record under the current stamp, for both facings. There is no override.
"""
import hashlib
import json
import os
import re
from typing import List

from . import actions as A
from .cli import ROOT, _harness

FACINGS = ("right", "left")        # fighter's x-facing when the move was verified (ram.Fighters.facing_right)
RECORDS = os.path.join(ROOT, "out", "verified_moves")


def stamp_key(rom_sha1: str, ram_map: str) -> str:
    return hashlib.sha256(json.dumps(_harness(rom_sha1, ram_map), sort_keys=True).encode()).hexdigest()


def map_rom_sha1(ram_map: str) -> str:
    """The ROM the RAM map was made for (its header); the ROM harness test asserts the running ROM is this one."""
    with open(ram_map) as f:
        m = re.search(r"SHA1 ([0-9A-Fa-f]{40})", f.read())
    if not m:
        raise RuntimeError("%s names no ROM SHA1" % ram_map)
    return m.group(1).upper()


def _load(path: str) -> dict:
    """{"ryu/hadoken/right": record, ...} from a records directory (none if it does not exist)."""
    out = {}
    if os.path.isdir(path):
        for name in os.listdir(path):
            if name.endswith(".json") and name.count(".") == 3:
                with open(os.path.join(path, name)) as f:
                    out["/".join(name.split(".")[:3])] = json.load(f)
    return out


def record(character: str, move: str, facing: str, key: str, test: str, path: str = None) -> None:
    """Called by a ROM test after it verified ``move`` for ``character`` facing ``facing``."""
    path = path or RECORDS
    if facing not in FACINGS or move not in A.moves(character):
        raise ValueError("not on the move list: %s/%s/%s" % (character, move, facing))
    os.makedirs(path, exist_ok=True)
    out = os.path.join(path, "%s.%s.%s.json" % (character, move, facing))
    tmp = "%s.%d.tmp" % (out, os.getpid())
    with open(tmp, "w") as f:
        json.dump({"stamp": key, "test": test}, f, sort_keys=True)
    os.replace(tmp, out)


def missing(key: str, path: str = None) -> List[str]:
    """Everything that keeps the gate shut: characters without specials in the code, then each
    character/move/facing without a record under ``key``."""
    recs = _load(path or RECORDS)
    out = ["%s: specials not in the code" % c for c in A.CHARACTERS if not A.SPECIALS.get(c)]
    for c in A.CHARACTERS:
        for m in A.moves(c):
            for f in FACINGS:
                name = "%s/%s/%s" % (c, m, f)
                if recs.get(name, {}).get("stamp") != key:
                    out.append(name)
    return out


def check_moves(ram_map: str, path: str = None) -> None:
    """Refuse (RuntimeError) unless every (character, move, facing) of all 8 characters is verified on the ROM
    under the current harness stamp."""
    miss = missing(stamp_key(map_rom_sha1(ram_map), ram_map), path)
    if miss:
        raise RuntimeError("moves not verified on the ROM (%d missing, e.g. %s): code the specials and run "
                           "SF2_ROM=... pytest -q tests/test_rom_moves.py" % (len(miss), ", ".join(miss[:12])))
