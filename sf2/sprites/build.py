"""Join the per-pair collections (sf2.sprites.collect) into the catalog, labels from RAM.

Labels of a sprite seen at stream row k are read at the displayed row t = k - lag (default LAG = 1, the repo's
convention sf2.data.perception.LAG; ``lag`` = 0 is computed alongside as a check). Per fighter (p1 / p2):
  act2       the 7-answer action (sf2.data.eye_v2.act2, with the pressed-word rule), "unknown" when RAM's is unknown
  state      the raw state byte 0x?C03, hex
  air        ground / air (sf2.data.pairs_labels.air)
  facing     RAM's facing byte vs the drawn sprite's (oam_facing): a mismatch is counted
  aid        the move id 0x?C3E
Projectile groups (pal5 / pal7): the shot slots' bytes (shot1 0x1000, shot2 0x1050) and the thrower candidates.

out/sprite_catalog/<char>/<key>.png, _projectiles/<key>.png, catalog.json:
  {"<char>/<key>": {char, key, index_key, size, frames, labels: {act2, state, air, aid}, lag0: {act2},
                    facing_mismatch, examples: [[pair, game, k], ...], snap: pair}}
"""
import glob
import gzip
import json
import os
import random
import shutil
from collections import Counter, defaultdict
from typing import Dict, Iterator, List, Tuple

import numpy as np

from ..data import pairs_labels as L
from ..data.eye_v2 import act2
from ..data.perception import LAG
from .catalog import FACE_LEFT, FACE_RIGHT
from .collect import FIGHTER_SLOT

N_EXAMPLES = 3
PROJ = "_projectiles"
LABEL_NAMES = ("act2", "state", "air", "aid")


def load_game(path: str) -> Tuple[List[Dict[str, int]], Dict[int, List[str]]]:
    """(rows as dicts of the named fields - not the raw bytes, pressed class per slot) of one g<game>.npz."""
    z = np.load(path)
    names = [str(n) for n in z["names"]]
    keep = [i for i, n in enumerate(names) if "_raw" not in n]
    rows = [dict(zip((names[i] for i in keep), map(int, r))) for r in z["rows"][:, keep]]
    pressed = {p: [c or None for c in z["pressed"][p - 1].tolist()] for p in (1, 2)}
    return rows, pressed


def fighter_labels(rows, pressed, t: int, p: int) -> Dict[str, str]:
    if not 0 <= t < len(rows):
        return dict(act2="unknown", state="none", air="unknown", aid="none", facing=None)
    r = rows[t]
    return dict(act2=act2(rows, t, p, pressed[p][t]) or "unknown", state="%02x" % r["p%d_state" % p],
                air=L.air(rows, t, p), aid="%02x" % r["p%d_aid" % p], facing=r["p%d_facing" % p])


def frames_of(pair_dir: str) -> Iterator[Tuple[int, List[Dict], List[Dict], Dict]]:
    """(game, frame records, rows, pressed) per collected game of one pair."""
    for npz in sorted(glob.glob(os.path.join(pair_dir, "g*.npz"))):
        game = int(os.path.basename(npz)[1:-4])
        rows, pressed = load_game(npz)
        with gzip.open(os.path.join(pair_dir, "g%d_frames.jsonl.gz" % game), "rt") as f:
            frames = [json.loads(line) for line in f]
        yield game, frames, rows, pressed


def _entry(char: str, fr: Dict, pair: str) -> Dict:
    return dict(char=char, key=fr["key"], index_key=fr["index_key"], size=fr["size"], frames=0,
                labels={n: Counter() for n in LABEL_NAMES}, lag0={"act2": Counter()}, facing_mismatch=0,
                examples=[], snap=pair, pairs=set(), owners=Counter(), shots=Counter())


def build(out: str, lag: int = LAG) -> Dict:
    """Merge every out/_pairs/* into the catalog dict (not written)."""
    cat: Dict[str, Dict] = {}
    stats = dict(pal_frames=Counter(), proj=Counter(), frames=Counter())
    for pair_dir in sorted(glob.glob(os.path.join(out, "_pairs", "*_vs_*"))):
        pair = os.path.basename(pair_dir)
        a, b = pair.split("_vs_")
        chars = {1: a, 2: b}
        for game, frames, rows, pressed in frames_of(pair_dir):
            seen_k = set()
            for fr in frames:
                k = fr["k"]
                if k not in seen_k:
                    seen_k.add(k)
                    for pal, n in fr["pals"].items():
                        stats["pal_frames"][int(pal)] += 1
                    _proj_stats(stats["proj"], fr["pals"], rows[k - lag] if k - lag >= 0 else None, chars)
                slot = FIGHTER_SLOT.get(fr["group"])
                char = chars[slot] if slot else PROJ
                ck = "%s/%s" % (char, fr["key"])
                e = cat.get(ck) or cat.setdefault(ck, _entry(char, fr, pair))
                e["frames"] += 1
                e["pairs"].add(pair)
                if len(e["examples"]) < N_EXAMPLES:
                    e["examples"].append([pair, game, k])
                stats["frames"][char] += 1
                if slot:
                    lab = fighter_labels(rows, pressed, k - lag, slot)
                    for n in LABEL_NAMES:
                        e["labels"][n][lab[n]] += 1
                    e["lag0"]["act2"][fighter_labels(rows, pressed, k, slot)["act2"]] += 1
                    if fr["oam_facing"] in (FACE_LEFT, FACE_RIGHT) and lab["facing"] != fr["oam_facing"]:
                        e["facing_mismatch"] += 1
                else:
                    r = rows[k - lag]
                    e["shots"]["shot1=%d shot2=%d" % (r["shot1"] != 0, r["shot2"] != 0)] += 1
                    e["owners"][chars[1] if fr["group"] == "pal5" else chars[2]] += 1
    for e in cat.values():
        e["pairs"] = len(e["pairs"])
    return dict(catalog=cat, stats=stats)


def _proj_stats(c: Counter, pals: Dict, r, chars: Dict[int, str]) -> None:
    """Which palettes are up while each shot slot is active, by which side can throw."""
    if r is None:
        return
    from ..data.pairs_shots import THROWERS
    who = "+".join(s for s, p in (("p1", 1), ("p2", 2)) if chars[p] in THROWERS) or "none"
    c["throwers=%s shot1=%d shot2=%d pal5=%d pal7=%d" % (who, r["shot1"] != 0, r["shot2"] != 0, "5" in pals,
                                                         "7" in pals)] += 1


def to_json(e: Dict) -> Dict:
    return {k: ({n: dict(v.most_common()) for n, v in e[k].items()} if k in ("labels", "lag0")
                else dict(v.most_common()) if isinstance(v, Counter) else v) for k, v in e.items()}


def write(out: str, result: Dict) -> str:
    """Copy each sprite's PNG to out/<char>/<key>.png and write out/catalog.json; the json path."""
    cat = result["catalog"]
    for ck, e in cat.items():
        dst = os.path.join(out, ck + ".png")
        if not os.path.exists(dst):
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copyfile(os.path.join(out, "_pairs", e["snap"], "sprites", e["key"] + ".png"), dst)
    path = os.path.join(out, "catalog.json")
    with open(path, "w") as f:
        json.dump({"lag": LAG, "sprites": {ck: to_json(e) for ck, e in sorted(cat.items())},
                   "stats": {k: dict(v) for k, v in result["stats"].items()}}, f, indent=1)
    return path
