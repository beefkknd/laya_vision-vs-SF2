"""The full stock art: every pose of each character's ROM pose table, forced and cut (sf2/sprites/rom_force.py).
Collection time only; nothing here runs in play. Needs scripts/probe_pose_table.py first (out/sprite_rom/wram/,
pose_tables.json).

    .venv/bin/python scripts/collect_full_sprites.py                  # all 8 characters, one headless Mesen each
    .venv/bin/python scripts/collect_full_sprites.py --chars ryu      # one

Writes out/sprite_rom/<char>/p<NNN>.png (the canonical RGBA, facing right, palette 4 = player 1's colours),
out/sprite_rom/<char>/index.json (per pose: WRAM record, ROM bank:addr, key, catalog match, label) and
out/sprite_rom/index.json (the summary). out/sprite_catalog is only read.

Catalog comparison: the catalog's frames give, per (character, pose pointer), the sprite keys drawn (lag 1). A forced
pose seen in play MATCHES when its key is the majority key of its pointer (``match``) / any key of it (``match_any``);
the PNGs are compared pixel for pixel as well (key = sha1 of the pixels, so the two agree by construction).
Label: the catalog's most common act2 over the pointer's sprites, else "unseen".
"""
import argparse
import glob
import json
import os
import sys
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from typing import Dict

import numpy as np
from PIL import Image

import _path  # noqa: F401
from sf2.config import DEFAULT_ROM, REPO
from sf2.sprites import rom_force as F
from sf2.sprites import rom_table as T
from sf2.sprites.emu import open_mesen
from collect_pairs import CHARS
from make_vs_pair_states import vs_state
from probe_pose_table import load_pair

CATALOG = os.path.join(REPO, "out", "sprite_catalog")
OUT = os.path.join(REPO, "out", "sprite_rom")
BASE_PORT = 52211            # 52211 .. 52218
TRIES = 3                    # scripts tried per pose (shortest prefix first) before the stance method


def catalog_pointers(catalog: str, char: str) -> Dict[int, Counter]:
    """pose pointer (relative to the buffer) -> Counter of sprite keys, over every catalog frame of ``char``."""
    out: Dict[int, Counter] = defaultdict(Counter)
    for pd in sorted(glob.glob(os.path.join(catalog, "_pairs", "*_vs_*"))):
        if char not in os.path.basename(pd).split("_vs_"):
            continue
        raw, frames, chars = load_pair(pd)
        vals, keys = T.pose_samples(raw, frames, chars, lag=1)
        for (c, ptr), k in zip(vals, keys):
            if c == char:
                out[ptr][k] += 1
    return out


def rom_address(segs, wram: int):
    for s in segs:
        if s["wram"][0] <= wram < s["wram"][1] and s["rom_offset"] is not None:
            return "%02x:%04x" % T.lorom_address(s["rom_offset"] + wram - s["wram"][0])
    return None


def same_pixels(a: np.ndarray, path: str) -> bool:
    if not os.path.exists(path):
        return False
    b = np.asarray(Image.open(path).convert("RGBA"))
    return a.shape == b.shape and bool((a == b).all())


def collect_char(char: str, port: int, out: str, catalog: str, rom: str) -> Dict:
    with open(os.path.join(out, "pose_tables.json")) as f:
        tab = json.load(f)[char]
    with open(os.path.join(catalog, "catalog.json")) as f:
        cat = json.load(f)["sprites"]
    ptrs = catalog_pointers(catalog, char)
    other = next(c for c in CHARS if c != char)
    with open(os.path.join(REPO, vs_state(char, other)), "rb") as f:
        state = f.read()
    d = os.path.join(out, char)
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(out, "wram", char + ".bin"), "rb") as f:
        wram = f.read()
    scripts = T.pose_scripts(wram)
    rows = []
    with open_mesen(port, rom) as b:
        stance = F.stance_records(b, state)
        for i, rec in enumerate(tab["pose_wram"]):
            if rec is None:
                rows.append(dict(pose=i, ok=False, why="empty slot"))
                continue
            tries = []
            for script, target, _ in scripts.get(i, [])[:TRIES]:
                r = F.force_chain(b, state, wram, stance, i, rec, script, target)
                tries.append("chain %04x>%04x: %s" % (script, target, r.why))
                if r.ok:
                    break
            if not tries or not r.ok:
                r = F.force(b, state, stance, i, rec)
                tries.append("stance: %s" % r.why)
            rel = rec - T.BUFFER[1]
            seen = ptrs.get(rel, Counter())
            row = dict(pose=i, wram="%04x" % rec, rom=rom_address(tab["rom_segments"], rec), ok=r.ok, why=r.why, method=r.method if r.ok else None, tries=tries,
                       frames=r.frames, used_by_anims=tab["poses_used"].get(str(i), 0),
                       catalog_frames=sum(seen.values()), catalog_keys=dict(seen.most_common()))
            if r.ok:
                name = "p%03d" % i
                Image.fromarray(r.rgba, "RGBA").save(os.path.join(d, name + ".png"))
                row.update(png=name + ".png", key=r.key, index_key=r.index_key, pal=r.pal,
                           size=[int(r.rgba.shape[1]), int(r.rgba.shape[0])])
                if seen:
                    maj = seen.most_common(1)[0][0]
                    ikeys = {cat.get("%s/%s" % (char, k), {}).get("index_key") for k in seen}
                    row.update(match=r.key == maj, match_any=r.key in seen, match_index=r.index_key in ikeys,
                               pixels_equal=same_pixels(r.rgba, os.path.join(catalog, char, maj + ".png")))
                    act = Counter()
                    for k in seen:
                        act.update(cat.get("%s/%s" % (char, k), {}).get("labels", {}).get("act2", {}))
                    row["label"] = act.most_common(1)[0][0] if act else "unknown"
                    row["labels"] = dict(act.most_common())
                else:
                    row["label"] = "unseen"
            rows.append(row)
    seen_rows = [r for r in rows if r.get("catalog_frames")]
    summ = dict(char=char, poses=len(rows), forced_ok=sum(r["ok"] for r in rows), stance_records=stance,
                seen_in_play=len(seen_rows), match=sum(bool(r.get("match")) for r in seen_rows),
                match_any=sum(bool(r.get("match_any")) for r in seen_rows),
                pixels_equal=sum(bool(r.get("pixels_equal")) for r in seen_rows),
                match_index=sum(bool(r.get("match_index")) for r in seen_rows),
                new=sum(1 for r in rows if r["ok"] and not r.get("catalog_frames")),
                new_distinct_keys=len({r["key"] for r in rows if r["ok"] and not r.get("catalog_frames")}
                                      - {k for r in seen_rows for k in r["catalog_keys"]}),
                failed=[[r["pose"], r["why"]] for r in rows if not r["ok"]],
                methods=dict(Counter(r.get("method") for r in rows if r["ok"])),
                mismatched=[r["pose"] for r in seen_rows if r["ok"] and not r.get("match_any")],
                mismatched_index=[r["pose"] for r in seen_rows if r["ok"] and not r.get("match_index")])
    with open(os.path.join(d, "index.json"), "w") as f:
        json.dump(dict(summary=summ, poses=rows), f, indent=1)
    return summ


def _one(args):
    return collect_char(*args)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--chars", default=",".join(CHARS))
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--catalog", default=CATALOG)
    ap.add_argument("--rom", default=os.environ.get("SF2_ROM") or os.path.join(REPO, DEFAULT_ROM))
    args = ap.parse_args()
    chars = [c for c in args.chars.split(",") if c]
    bad = [c for c in chars if c not in CHARS]
    if bad:
        raise SystemExit("unknown characters: %s" % bad)
    if os.path.abspath(args.out) == os.path.abspath(args.catalog):
        raise SystemExit("--out must not be the catalog")
    jobs = [(c, BASE_PORT + CHARS.index(c), args.out, args.catalog, args.rom) for c in chars]
    with ProcessPoolExecutor(len(jobs)) as ex:
        res = list(ex.map(_one, jobs))
    print("%-8s %5s %5s %5s %6s %6s %6s %6s %4s %s" % ("char", "poses", "ok", "seen", "match", "any", "pixEq", "index",
                                                        "new", "failed / mismatched (pixels) / mismatched (index)"))
    for s in res:
        print("%-8s %5d %5d %5d %6d %6d %6d %6d %4d %s / %s / %s" % (
            s["char"], s["poses"], s["forced_ok"], s["seen_in_play"], s["match"], s["match_any"], s["pixels_equal"],
            s["match_index"], s["new"], s["failed"], s["mismatched"], s["mismatched_index"]))
    with open(os.path.join(args.out, "index.json"), "w") as f:
        json.dump({s["char"]: s for s in res}, f, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
