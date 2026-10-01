"""How well player 1 did the directed moves in a movement-pairs collection (sf2.data.pairs_moves.executed, logged per
move in each games.jsonl), per character and move word, plus seconds and pairs per game and MB per pair.

    python scripts/pairs_moves_report.py --root rollouts/pairs [--json out.json]
"""
import argparse
import collections
import json
import os
import sys

import _path  # noqa: F401
from sf2.data import pairs_collect_io as IO
from sf2.data import pairs_moves as PM


def report(root: str) -> dict:
    names = sorted(n for n in os.listdir(root) if "_vs_" in n)
    per = collections.defaultdict(collections.Counter)
    games, img_bytes = [], 0
    for n in names:
        a = n.split("_vs_")[0]
        for g in IO.committed(os.path.join(root, n)):
            games.append({"pair": n, "seconds": g["seconds"], "pairs": g["pairs"], "rows": g["rows"],
                          "result": g.get("result")})
            for w, _, _, st in g["moves"]:
                per[(a, w)][st] += 1
        d = os.path.join(root, n, "images")
        if os.path.isdir(d):
            img_bytes += sum(e.stat().st_size for e in os.scandir(d) if e.is_file())
    pairs = sum(g["pairs"] for g in games)
    by_char = {}
    for (a, w), c in sorted(per.items()):
        k = PM.kind(a, w)
        by_char.setdefault(a, {})[w] = dict(c, kind=k)
    return {"games": len(games), "seconds_per_game": round(sum(g["seconds"] for g in games) / max(1, len(games)), 1),
            "pairs_per_game": round(pairs / max(1, len(games)), 1), "image_mb": round(img_bytes / 1e6, 1),
            "mb_per_pair": round(img_bytes / 1e6 / max(1, pairs), 3), "per_game": games, "moves": by_char}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=os.path.join("rollouts", "pairs"))
    ap.add_argument("--json")
    args = ap.parse_args(argv)
    rep = report(args.root)
    print("%d games, %.1f s/game, %.1f pairs/game, %.1f MB images, %.3f MB/pair" % (
        rep["games"], rep["seconds_per_game"], rep["pairs_per_game"], rep["image_mb"], rep["mb_per_pair"]))
    for a, ws in rep["moves"].items():
        tot = collections.Counter()
        for c in ws.values():
            tot.update({k: v for k, v in c.items() if k != "kind"})
        print("%-8s %d words tried, %s" % (a, len(ws), dict(tot)))
        bad = {w: {k: v for k, v in c.items() if k != "kind"} for w, c in ws.items()
               if c.get("missed", 0) > c.get("done", 0) + c.get("held", 0)}
        if bad:
            print("   mostly missed:", bad)
    if args.json:
        with open(args.json, "w") as f:
            json.dump(rep, f, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
