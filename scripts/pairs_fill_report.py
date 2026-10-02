"""How full the movement-pairs grid is after a collection round (sf2.data.pairs_data.fill_report): per cell
(character, movement10, facing) min(collected, cap) / cap over both splits and both players, as a matrix (rows =
characters, columns = the 10 movements; labels re-derived from RAM + the move log as the builder does, each cell "R%/L%" = facing right / left), per character, overall, the empty
cells by name, and how often each fighter actually blocks (RAM rows whose movement label is block, and block episodes).

    python scripts/pairs_fill_report.py --root rollouts/pairs2p [--cap 60] [--json out.json]
"""
import argparse
import json
import os
import sys

import _path  # noqa: F401
from sf2.data import movement_collect_io as MIO
from sf2.data import pairs_collect_io as IO
from sf2.data import pairs_data as D
from sf2.data import pairs_labels as L
from sf2.data import pairs_moves as PM
from sf2.data.movement_collect import FIRST_T


def block_stats(root: str, names) -> dict:
    """Per character: RAM rows labelled block / all rows, and block episodes (runs of block rows), both slots. And
    the attack-vs-special source (owner after round 1) over the RAM rows in an attack state: "special pressed" /
    "attack pressed" (our word decided, RAM confirmed), "fallback <label>" (no attack word pressed: the RAM rule);
    and per special WORD we pressed: "special words confirmed" (RAM in an attack state in its rows) / "not"."""
    out = {}
    for n in names:
        chars = dict(enumerate(n.split("_vs_"), 1))
        for g in IO.committed(os.path.join(root, n)):
            rows = MIO.read_ram(os.path.join(root, n, "ram", "g%04d.json.gz" % g["game"]))
            words = PM.pressed_words(g["moves"], len(rows))
            for p, c in chars.items():
                src = out.setdefault(c, {}).setdefault("sources", {})
                for t in range(FIRST_T, len(rows)):
                    cls = PM.pressed_class(c, words[p][t])
                    mv, how = L.movement_pressed(rows, t, p, cls)
                    if how != "ram":
                        key = "%s %s" % (mv, how) if how == "pressed" else "fallback %s" % mv
                        src[key] = src.get(key, 0) + 1
                for m in g["moves"]:
                    if PM.slot_of(m) == p and PM.pressed_class(c, m[0]) == "special":
                        ok = any(L.movement(rows, t, p) in L.ATTACK_MOVES for t in range(m[1] + 1, m[2] + 1))
                        key = "special words confirmed" if ok else "special words not confirmed"
                        src[key] = src.get(key, 0) + 1
                s = out[c]
                for k in ("rows", "block_rows", "episodes", "games_with_block"):
                    s.setdefault(k, 0)
                prev, mine = False, 0
                for t in range(FIRST_T, len(rows)):
                    b = L.movement(rows, t, p) == "block"
                    mine += b
                    s["episodes"] += b and not prev
                    prev = b
                s["rows"] += len(rows) - FIRST_T
                s["block_rows"] += mine
                s["games_with_block"] += mine > 0
    for s in out.values():
        s["block_pct"] = round(100.0 * s["block_rows"] / max(1, s["rows"]), 2)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True)
    ap.add_argument("--cap", type=int, default=D.CAP)
    ap.add_argument("--json")
    ap.add_argument("--quick", action="store_true", help="skip the per-row block / source scan")
    args = ap.parse_args(argv)
    pairs, dropped, names = D.collection_pairs(args.root)
    # the labels and the episode filter as the builder makes them
    pairs, dropped_ep = D.in_episode_only(D.relabel(args.root, pairs, L.poke_bands()))
    pairs = D.cap_per_game(pairs)
    chars = sorted({c for n in names for c in n.split("_vs_")})
    rep = dict(D.fill_report(pairs, chars, args.cap), pair_dirs=len(names), pairs=len(pairs), dropped=dropped)
    rep["block"] = block_stats(args.root, names) if not args.quick else {}
    rep["dropped_episode"] = dropped_ep
    rep["not_full"] = sorted(k for k, n in rep["counts"].items() if n < args.cap)
    print("%d pair dirs, %d pairs; %d cells x cap %d; overall %.1f%%; %d cells not full; dropped outside an "
          "episode: %s" % (len(names), len(pairs), rep["cells"], args.cap, rep["overall_pct"], len(rep["not_full"]),
                           json.dumps(dropped_ep)))
    short = {"walk toward": "walk tw", "walk away": "walk aw"}
    print("%-8s %s  %6s" % ("char", " ".join("%-9s" % short.get(m, m) for m in L.MOVEMENTS10), "total"))
    for c in chars:
        cells = " ".join("%-9s" % ("%d/%d" % (rep["pct"][c]["%s|right" % m], rep["pct"][c]["%s|left" % m]))
                         for m in L.MOVEMENTS10)
        print("%-8s %s  %5.1f%%" % (c, cells, rep["per_char"][c]))
    for c in chars:
        if rep["zero"][c]:
            print("zero %-8s %s" % (c, "; ".join(rep["zero"][c])))
    for c, s in sorted(rep["block"].items()):
        print("block %-8s %.2f%% of rows, %d episodes, in %d games" % (c, s["block_pct"], s["episodes"],
                                                                      s["games_with_block"]))
    for c, s in sorted(rep["block"].items()):
        print("source %-8s %s" % (c, json.dumps(dict(sorted(s["sources"].items())))))
    if args.json:
        with open(args.json, "w") as f:
            json.dump(rep, f, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
