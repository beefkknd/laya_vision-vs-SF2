"""VS BATTLE move test: Python plays both controllers, runs every move of both fighters (movement, normals, throws,
blocks, specials, combos), checks each against RAM, measures it, and sweeps the gap to find each attack's reach.

    python scripts/vs_moves.py --p1 ryu --p2 chunli --tag game1              # Ryu left, Chun-Li right
    python scripts/vs_moves.py --p1 chunli --p2 ryu --tag game2 --port 47992  # same pair, sides swapped

Needs $SF2_ROM (or --rom). Headless Mesen, one process per run (runs on different ports can go in parallel).
Writes out/vs_moves/<tag>/: start.state/.png, summary.csv + summary.json (one row per move: check, measurements,
reach), reach.csv (every sweep try), and per move <char>_p<n>/<move>/{ram.csv, sheet.png, frames/*.png}.
Exit code 1 if any move's check failed.
"""
import argparse
import csv
import json
import os
import sys
import time
from typing import Dict, List

import _path  # noqa: F401
from sf2.config import PAD
from sf2.dataset import save_png
from sf2.emu.headless import launch_argv
from sf2.emu.mesen import MesenBridge
from sf2.emu.vs import NAMES, boot_vs, gap_state, record, view
from sf2.vs_metrics import measure, reach, walk_speed
from sf2.vs_moves import CONDS, GAPS, MOVESETS, REACH_GAPS, Move, connected

TAIL = 240


def contact_sheet(images: Dict[int, object], path: str, cols: int = 6, rows: int = 4) -> None:
    from PIL import Image, ImageDraw

    keys = sorted(k for k, v in images.items() if v is not None)
    if not keys:
        return
    pick = [keys[round(i * (len(keys) - 1) / max(cols * rows - 1, 1))] for i in range(min(cols * rows, len(keys)))]
    pick = sorted(set(pick))
    h, w = images[pick[0]].shape[:2]
    tw, th = w // 2, h // 2
    sheet = Image.new("RGB", (tw * cols, th * ((len(pick) + cols - 1) // cols)))
    for i, k in enumerate(pick):
        tile = Image.fromarray(images[k]).resize((tw, th))
        ImageDraw.Draw(tile).text((3, 2), "f%d" % k, fill=(255, 255, 0))
        sheet.paste(tile, ((i % cols) * tw, (i // cols) * th))
    sheet.save(path)


def save_take(take, folder: str) -> None:
    os.makedirs(os.path.join(folder, "frames"), exist_ok=True)
    with open(os.path.join(folder, "ram.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame", "p1_in", "p2_in"] + NAMES)
        for i, r in enumerate(take.rows):
            w.writerow([i, "+".join(take.p1[i]) or "-", "+".join(take.p2[i]) or "-"] + [r[n] for n in NAMES])
    for k, img in take.images.items():
        if img is not None:
            save_png(img, os.path.join(folder, "frames", "%04d.png" % k))
    contact_sheet(take.images, os.path.join(folder, "sheet.png"))


class GapCache:
    def __init__(self, bridge, start: bytes):
        self.bridge, self.start, self.states = bridge, start, {}

    def get(self, g: int):
        if g not in self.states:
            self.states[g] = gap_state(self.bridge, self.start, g)
        return self.states[g]


def roles(move: Move, me: int):
    """(attacker player, attacker steps, defender steps) for the measured player ``me``."""
    if move.defends:
        return 3 - me, move.other, move.steps
    return me, move.steps, move.other


def run_move(bridge, gaps: GapCache, me: int, move: Move, every: int, sweep: bool) -> Dict[str, object]:
    state, got = gaps.get(GAPS[move.gap])
    bridge.load_state(state)
    attacker, a_steps, d_steps = roles(move, me)
    take = record(bridge, attacker, a_steps, d_steps, TAIL, CONDS, PAD, every)
    rows = [view(r, attacker) for r in take.rows]
    row: Dict[str, object] = {"move": move.name, "kind": move.kind, "setup_gap": got, "expect": move.expect,
                              "pass": bool(move.check(rows)),
                              # blocks: the measured fighter defends, so the timing columns are the blocked attack's
                              "stats_of": "the blocked attack" if move.defends else "this move"}
    row.update(measure(rows))
    if move.name.startswith("walk"):
        row["walk_speed"] = walk_speed(rows, 40)
    tries = []
    if sweep and move.sweep:
        for g in REACH_GAPS:
            s, real = gaps.get(g)
            bridge.load_state(s)
            t = record(bridge, attacker, a_steps, d_steps, 120, CONDS, PAD, every=0)
            rv = [view(r, attacker) for r in t.rows]
            m = measure(rv)
            tries.append({"move": move.name, "gap": real, "connected": connected(rv), "hits": m["hits"],
                          "damage": m["damage"], "startup": m["startup"]})
        row.update(reach(tries))
    return {"row": row, "take": take, "tries": tries}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--p1", default="ryu")
    ap.add_argument("--p2", default="chunli")
    ap.add_argument("--tag", default="game1")
    ap.add_argument("--out", default="out/vs_moves")
    ap.add_argument("--port", type=int, default=47991)
    ap.add_argument("--rom", default=os.environ.get("SF2_ROM"))
    ap.add_argument("--mesen", default=os.environ.get("SF2_MESEN"))
    ap.add_argument("--every", type=int, default=2, help="screenshot every N frames of each take")
    ap.add_argument("--only", help="comma-separated move names")
    ap.add_argument("--no-sweep", action="store_true", help="skip the reach sweeps")
    args = ap.parse_args()
    for c in (args.p1, args.p2):
        if c not in MOVESETS:
            ap.error("no move set for %r (have: %s)" % (c, ", ".join(MOVESETS)))
    only = set(args.only.split(",")) if args.only else None
    root = os.path.join(args.out, args.tag)
    os.makedirs(root, exist_ok=True)

    b = MesenBridge(args.port, launch=launch_argv(args.port, args.rom, args.mesen))
    t0 = time.time()
    try:
        b.set_capture("raw")
        start = boot_vs(b, args.p1, args.p2)
        with open(os.path.join(root, "start.state"), "wb") as f:
            f.write(start)
        save_png(b.load_state(start).images[0], os.path.join(root, "start.png"))
        gaps = GapCache(b, start)
        rows: List[Dict[str, object]] = []
        tries: List[Dict[str, object]] = []
        for me, char in ((1, args.p1), (2, args.p2)):
            side = "left" if me == 1 else "right"
            for move in MOVESETS[char]():
                if only and move.name not in only:
                    continue
                res = run_move(b, gaps, me, move, args.every, not args.no_sweep)
                head = {"game": args.tag, "char": char, "player": me, "side": side,
                        "facing": "right" if side == "left" else "left", "opponent": args.p2 if me == 1 else args.p1}
                row = dict(head, **res["row"])
                rows.append(row)
                tries += [dict(head, **t) for t in res["tries"]]
                save_take(res["take"], os.path.join(root, "%s_p%d" % (char, me), move.name))
                print("%-6s p%d %-24s %s  startup=%s dur=%s dmg=%s hits=%s reach=%s" % (
                    char, me, move.name, "PASS" if row["pass"] else "FAIL", row["startup"], row["duration"],
                    row["damage"], row["hits"], row.get("reach_max")), flush=True)
    finally:
        b.close()
    keys = sorted({k for r in rows for k in r}, key=lambda k: (list(rows[0]).index(k) if k in rows[0] else 99, k))
    with open(os.path.join(root, "summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, keys)
        w.writeheader()
        w.writerows(rows)
    with open(os.path.join(root, "summary.json"), "w") as f:
        json.dump(rows, f, indent=1)
    if tries:
        with open(os.path.join(root, "reach.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, list(tries[0]))
            w.writeheader()
            w.writerows(tries)
    failed = [r for r in rows if not r["pass"]]
    print("%d moves, %d passed, %d failed (%s) in %.0f s -> %s" % (
        len(rows), len(rows) - len(failed), len(failed), ", ".join("%s:%s" % (r["char"], r["move"]) for r in failed),
        time.time() - t0, root))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
