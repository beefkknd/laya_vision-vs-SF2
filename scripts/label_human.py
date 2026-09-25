"""Turn a record_human.py session log into dataset rows (gold = your inputs, mapped to the 12 options).

    python scripts/label_human.py --session human/s1 --name human_s1 [--me ryu --opp ken]

Frames where either life bar reads KO are dropped. A new "episode" starts whenever both bars refill, and every
10th one goes to val.
"""
import argparse
import os
from collections import Counter

import _path  # noqa: F401
from sf2 import dataset as D
from sf2 import labeler
from sf2.actions import question
from sf2.config import FULL_HP, HOLD, PREV_GAP
from sf2.ram import Fighters, text_state


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--session", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--out", default="data")
    ap.add_argument("--me", default="ryu")
    ap.add_argument("--opp", default="guile")
    ap.add_argument("--val-every", type=int, default=10)
    args = ap.parse_args()

    log = D.read(os.path.join(args.session, "log.jsonl"))
    by_frame = {r["frame"]: r for r in log}
    labels = labeler.label_frames([(r["names"], r["facing_right"]) for r in log], hold=HOLD)
    dst = os.path.join(args.out, args.name)
    os.makedirs(dst, exist_ok=True)
    splits = {"train": [], "val": []}
    episode, last, prev_full = 0, "idle", True
    for g, action in labels:
        r = log[g]
        full = r["my_hp"] == FULL_HP and r["opp_hp"] == FULL_HP
        if full and not prev_full:
            episode, last = episode + 1, "idle"
        prev_full = full
        if "image" not in r or r["my_hp"] < 0 or r["opp_hp"] < 0:
            continue
        prev = by_frame.get(r["frame"] - PREV_GAP, r)
        f = Fighters(r["my_hp"], r["opp_hp"], r["my_x"], r["opp_x"], r["my_y"], r["opp_y"])
        imgs = [os.path.relpath(os.path.join(args.session, x.get("image", r["image"])), dst) for x in (prev, r)]
        target = D.one_hot(action)
        rec = {"id": "%s-e%05d-s%07d" % (args.name, episode, r["frame"]), "images": imgs,
               "state_text": text_state(f, args.me, args.opp, last, r["my_air"], r["opp_air"]),
               "question": question(),
               "label": target.index(1.0), "target": target, "episode": episode, "step": r["frame"],
               "source": args.name, "meta": {"action": action, "actor": "human", "frame": r["frame"]}}
        sp = "val" if args.val_every and episode % args.val_every == args.val_every - 1 else "train"
        splits[sp].append(rec)
        last = action
    for sp, recs in splits.items():
        print("%s/%s.jsonl: %d rows" % (dst, sp, D.write_jsonl(os.path.join(dst, sp + ".jsonl"), recs)))
    print("label mix:", Counter(r["meta"]["action"] for s in splits.values() for r in s).most_common())


if __name__ == "__main__":
    main()
