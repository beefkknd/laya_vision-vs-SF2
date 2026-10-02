"""Probe: the fighter's pose pointer and the per-character pose tables (sf2/sprites/rom_table.py). Collection time
only; nothing here runs in play.

    .venv/bin/python scripts/probe_pose_table.py              # purity over out/sprite_catalog + tables per character

1. Purity: over every fighter frame of out/sprite_catalog/_pairs (the raw struct bytes 0x?C00-0x?C7F per row),
   (character, pose pointer 0x?C1E - buffer base 0x?C20) -> sprite key at lag 0 / 1 / 2.
2. Tables: one headless Mesen per character (2P versus state, the character as player 1), WRAM dumped after 3
   idle frames; the image / anim tables parsed from the player-1 buffer and the buffer mapped onto the ROM.
Writes out/sprite_rom/wram/<char>.bin (the WRAM dumps), out/sprite_rom/pose_tables.json, out/sprite_rom/probe.json.
"""
import argparse
import glob
import gzip
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from typing import Dict

import numpy as np

import _path  # noqa: F401
from sf2.config import DEFAULT_ROM, REPO
from sf2.sprites import rom_table as T
from collect_pairs import CHARS
from make_vs_pair_states import vs_state

CATALOG = os.path.join(REPO, "out", "sprite_catalog")
OUT = os.path.join(REPO, "out", "sprite_rom")
BASE_PORT = 52201            # 52201 .. 52208: outside sf2.config.PORTS and collect_sprites' 52101..52156


def load_pair(pair_dir: str):
    """(raw bytes per player per row, fighter frame records, chars) of one collected pair (game 0)."""
    z = np.load(os.path.join(pair_dir, "g0.npz"))
    names = [str(n) for n in z["names"]]
    raw = {p: z["rows"][:, [names.index("p%d_raw%02x" % (p, o)) for o in range(0x80)]].astype(np.int64).tolist()
           for p in (1, 2)}
    with gzip.open(os.path.join(pair_dir, "g0_frames.jsonl.gz"), "rt") as f:
        frames = [fr for fr in map(json.loads, f) if fr["group"] in ("p1", "p2")]
    a, b = os.path.basename(pair_dir).split("_vs_")
    return raw, frames, {1: a, 2: b}


def _samples(args):
    pair_dir, lag = args
    raw, frames, chars = load_pair(pair_dir)
    return T.pose_samples(raw, frames, chars, lag=lag)


def purity_all(catalog: str) -> Dict:
    pairs = sorted(glob.glob(os.path.join(catalog, "_pairs", "*_vs_*")))
    out = {}
    with ProcessPoolExecutor(min(32, os.cpu_count() or 4)) as ex:
        for lag in (0, 1, 2):
            vals, keys = [], []
            for v, k in ex.map(_samples, [(p, lag) for p in pairs]):
                vals += v
                keys += k
            r = T.purity(vals, keys)
            per_char = {}
            for c in CHARS:
                idx = [i for i, v in enumerate(vals) if v[0] == c]
                rc = T.purity([vals[i] for i in idx], [keys[i] for i in idx])
                per_char[c] = {k: rc[k] for k in ("n", "values", "keys", "purity", "reverse")}
            out[lag] = dict({k: r[k] for k in ("n", "values", "keys", "purity", "reverse")}, per_char=per_char,
                            majority={"%s|%d" % v: k for v, k in r["majority"].items()} if lag == 1 else None)
    return out


def dump_char(char: str, port: int, rom: str) -> bytes:
    from sf2.emu.vs import VARS
    from sf2.sprites.emu import open_mesen
    other = next(c for c in CHARS if c != char)
    with open(os.path.join(REPO, vs_state(char, other)), "rb") as f:
        state = f.read()
    with open_mesen(port, rom) as b:
        b.set_vars(VARS)
        b.load_state(state)
        b.run([["-"]] * 3)
        return b.dump_wram()


def tables(wram: bytes, rom: bytes, seen=()) -> Dict:
    """``seen``: pose pointers (relative to the buffer) this character showed in the catalog - each must be a
    record of the parsed image table."""
    p = 1
    images = T.image_table(wram, p)
    anims = T.anim_table(wram, p)
    used = T.poses_used(wram, p)
    base = T.BUFFER[p]
    segs = T.rom_segments(wram, rom, base, base + T.BUFFER_LEN)
    return {"n_poses": len(images), "n_anims": len(anims),
            "pose_wram": [a for a in images], "poses_used": {str(k): v for k, v in sorted(used.items())},
            "n_poses_used": sum(1 for i in range(len(images)) if used.get(i)),
            "used_out_of_range": sorted(k for k in used if k >= len(images)),
            "empty_slots": sum(1 for a in images if a is None),
            "seen_in_play": len(set(seen)),
            "seen_not_in_table": sorted("%04x" % s for s in set(seen) if base + s not in set(images)),
            "rom_segments": segs}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--catalog", default=CATALOG)
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--rom", default=os.environ.get("SF2_ROM") or os.path.join(REPO, DEFAULT_ROM))
    ap.add_argument("--skip-purity", action="store_true")
    args = ap.parse_args()
    os.makedirs(os.path.join(args.out, "wram"), exist_ok=True)
    with open(args.rom, "rb") as f:
        rom = f.read()
    probe = {}
    if not args.skip_purity:
        probe["purity"] = purity_all(args.catalog)
        for lag, r in probe["purity"].items():
            print("lag %d: n %d values %d keys %d purity %.4f reverse %.4f" % (lag, r["n"], r["values"], r["keys"],
                                                                             r["purity"], r["reverse"]))
    with ThreadPoolExecutor(len(CHARS)) as ex:
        dumps = dict(zip(CHARS, ex.map(lambda ci: dump_char(ci[1], BASE_PORT + ci[0], args.rom), enumerate(CHARS))))
    tabs = {}
    for c, w in dumps.items():
        with open(os.path.join(args.out, "wram", c + ".bin"), "wb") as f:
            f.write(w)
        maj = (probe.get("purity") or {}).get(1, {}).get("majority") or {}
        seen = [int(k.split("|")[1]) for k in maj if k.split("|")[0] == c]
        tabs[c] = tables(w, rom, seen)
        t = tabs[c]
        segs = ", ".join("%04x-%04x<-%s" % (s["wram"][0], s["wram"][1],
                                            "%02x:%04x" % tuple(s["rom"]) if s["rom"] else "?")
                         for s in t["rom_segments"])
        print("%-8s poses %3d (used by anims %3d, empty %d) anims %3d; seen in play %3d, not in table %d | %s" % (
            c, t["n_poses"], t["n_poses_used"], t["empty_slots"], t["n_anims"], t["seen_in_play"],
            len(t["seen_not_in_table"]), segs))
    with open(os.path.join(args.out, "pose_tables.json"), "w") as f:
        json.dump(tabs, f, indent=1)
    with open(os.path.join(args.out, "probe.json"), "w") as f:
        json.dump(probe, f, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
