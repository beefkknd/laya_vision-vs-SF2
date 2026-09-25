"""Turn a record_human.py session log into dataset rows (gold = your inputs, mapped to the 12 options).

    python scripts/label_human.py --session human/s1 --name human_s1 [--me ryu --opp ken]

Frames where either life reads KO are dropped. A new "episode" starts whenever both bars refill, and every
10th one goes to val.
"""
import argparse
import os
from collections import Counter

import _path  # noqa: F401
from sf2 import dataset as D
from sf2 import labeler
from sf2.actions import question
from sf2.config import HOLD, PREV_GAP
from sf2.env import AIR_DY
from sf2.ram import REQUIRED, Fighters, text_state


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--session", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--out", default="data")
    ap.add_argument("--me", default="ryu")
    ap.add_argument("--opp", default="ken")
    ap.add_argument("--val-every", type=int, default=10)
    args = ap.parse_args()

    log = D.read(os.path.join(args.session, "log.jsonl"))
    fighters = [Fighters(*(r["ram"][n] for n in REQUIRED)) for r in log]
    labels = labeler.label_frames([(r["names"], f.facing_right) for r, f in zip(log, fighters)], hold=HOLD)
    by_frame = {r["frame"]: r for r in log}
    full = max(max(f.my_hp, f.opp_hp) for f in fighters)
    dst = os.path.join(args.out, args.name)
    os.makedirs(dst, exist_ok=True)
    splits = {"train": [], "val": []}
    episode, last, prev_full, ground = 0, "idle", True, (fighters[0].my_y, fighters[0].opp_y)
    for g, action in labels:
        r, f = log[g], fighters[g]
        is_full = f.my_hp == full and f.opp_hp == full
        if is_full and not prev_full:
            episode, last, ground = episode + 1, "idle", (f.my_y, f.opp_y)
        prev_full = is_full
        if "image" not in r or f.my_hp < 0 or f.opp_hp < 0:
            continue
        prev = by_frame.get(r["frame"] - PREV_GAP, r)
        imgs = [os.path.relpath(os.path.join(args.session, x.get("image", r["image"])), dst) for x in (prev, r)]
        my_air, opp_air = abs(f.my_y - ground[0]) > AIR_DY, abs(f.opp_y - ground[1]) > AIR_DY
        target = D.one_hot(action)
        rec = {"id": "%s-e%05d-s%07d" % (args.name, episode, r["frame"]), "images": imgs,
               "state_text": text_state(f, args.me, args.opp, last, my_air, opp_air, full),
               "question": question(), "label": target.index(1.0), "target": target, "episode": episode,
               "step": r["frame"], "source": args.name,
               "meta": {"action": action, "actor": "human", "frame": r["frame"]}}
        sp = "val" if args.val_every and episode % args.val_every == args.val_every - 1 else "train"
        splits[sp].append(rec)
        last = action
    for sp, recs in splits.items():
        print("%s/%s.jsonl: %d rows" % (dst, sp, D.write_jsonl(os.path.join(dst, sp + ".jsonl"), recs)))
    print("label mix:", Counter(r["meta"]["action"] for s in splits.values() for r in s).most_common())


if __name__ == "__main__":
    main()
