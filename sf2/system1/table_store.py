"""Per-character value-table store (owner 2026-10-05: "support multiple characters and a storage design").

One JSON per character under a root dir -- each file is the UNCHANGED value_table structure ({cells, shadow, depth}),
so value_table's math is untouched and a character's table carries/merges/studies exactly as before. A manifest
alongside them tracks each character's accumulated games, last-updated time, source run, and cell count, so the
condensation of each character's knowledge is visible at a glance. This module is only path resolution + JSON I/O +
pooling a run's worker tables into the stored one (seed-dedup); the table math stays in value_table.

    runs/tables/
      manifest.json      {"chunli": {"games": 32, "updated": ..., "run": "A2", "cells": 25}, "ryu": {...}}
      chunli.json        {"cells": {...}, "shadow": {...}, "depth": {...}}
      ryu.json           {...}
"""
import json
import os
import time
from typing import Dict, Optional, Sequence

from . import value_table as VT

ROOT = "runs/tables"
MANIFEST = "manifest.json"


def path(char: str, root: str = ROOT) -> str:
    return os.path.join(root, char + ".json")


def manifest_path(root: str = ROOT) -> str:
    return os.path.join(root, MANIFEST)


def _read_json(p: str, default):
    if os.path.exists(p):
        with open(p) as f:
            return json.load(f)
    return default


def _write_json(p: str, obj) -> None:
    os.makedirs(os.path.dirname(p) or ".", exist_ok=True)
    tmp = p + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f)
    os.replace(tmp, p)


def load(char: str, root: str = ROOT) -> VT.Table:
    """The character's stored table, or a blank one if it has none yet."""
    return _read_json(path(char, root), VT.blank())


def read_manifest(root: str = ROOT) -> Dict:
    return _read_json(manifest_path(root), {})


def save(char: str, table: VT.Table, root: str = ROOT, games: int = 0, run: str = "") -> None:
    """Write the character's table and fold a new entry into the manifest (games ACCUMULATE across runs). The old
    manifest is never mutated: a fresh dict is written."""
    _write_json(path(char, root), table)
    man = read_manifest(root)
    prior_games = man.get(char, {}).get("games", 0)
    entry = {"games": prior_games + games, "updated": time.strftime("%Y-%m-%dT%H:%M:%S"),
             "run": run, "cells": len(table.get("cells", {}))}
    _write_json(manifest_path(root), {**man, char: entry})


def pool_into(char: str, worker_tables: Sequence[VT.Table], seed: Optional[VT.Table] = None,
              root: str = ROOT, games: int = 0, run: str = "") -> VT.Table:
    """Merge a run's per-worker tables and store under ``char``. ``seed`` = the table the workers all carried (so the
    common seed is subtracted K-1 times, not counted K times -- the seed-double-count bug); None if they started blank."""
    merged = VT.merge(worker_tables, shared=seed)
    save(char, merged, root=root, games=games, run=run)
    return merged
