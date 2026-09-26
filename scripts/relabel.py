"""Days 4-5: turn student rollouts into new training rows. This is the learning step.

    python scripts/relabel.py --rollout rollouts/r0 --name dagger_r1                  # (a) DAgger
    python scripts/relabel.py --rollout rollouts/r0 --name filter_r1 --mode filter    # (b) cheap filter

(a) dagger: the teacher's distribution is the gold on the *student's* frames. Frames around trouble (hits taken
    in the next 0.5 s, knockdowns, whiffed specials) are also written to a separate "<name>_hot" set, so
    train.py samples them as their own group instead of drowning them in neutral walking frames.
(b) filter: keep only student actions followed by damage_for > damage_against in the next 0.5 s, with the
    student's own action as a one-hot gold. No teacher needed; weaker signal.
Decisions where the stick did nothing (meta controllable False: hit or knocked down) are left out in both modes.
(c) never: mark every frame of a lost round as wrong. round_result is kept in meta for analysis only.

Val = every 10th match. Images are referenced from the rollout dir (relative paths), not copied.
"""
import argparse
import os

import _path  # noqa: F401
from sf2 import dataset as D


def rebase(r, rollout_dir, dst):
    """Image paths relative to the new dataset dir ``dst`` (laya joins them onto it)."""
    return dict(r, images=[os.path.relpath(os.path.join(rollout_dir, im), dst) for im in r["images"]])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rollout", action="append", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--out", default="data")
    ap.add_argument("--mode", choices=["dagger", "filter"], default="dagger")
    ap.add_argument("--val-every", type=int, default=10)
    ap.add_argument("--only-disagree", action="store_true",
                    help="dagger: keep only frames where student != teacher (smaller, sharper set)")
    args = ap.parse_args()

    out = {k: {"train": [], "val": []} for k in (args.name, args.name + "_hot")}
    for ro in args.rollout:
        for split in ("train", "val"):
            p = os.path.join(ro, split + ".jsonl")
            if not os.path.exists(p):
                continue
            for r in D.read(p):
                m = r["meta"]
                if m.get("controllable") is False:  # hit or knocked down: the stick did nothing, nothing to learn
                    continue
                sp = "val" if args.val_every and r["episode"] % args.val_every == args.val_every - 1 else "train"
                if args.mode == "filter":
                    if m["dmg_for_next"] <= m["dmg_against_next"]:
                        continue
                    r["target"] = D.one_hot(m["action"])
                    r["label"] = r["target"].index(1.0)
                    out[args.name][sp].append(rebase(r, ro, os.path.join(args.out, args.name)))
                    continue
                if args.only_disagree and m["action"] == m["teacher_action"]:
                    continue
                out[args.name][sp].append(rebase(r, ro, os.path.join(args.out, args.name)))  # label/target already = teacher's
                if m.get("hot"):
                    out[args.name + "_hot"][sp].append(rebase(r, ro, os.path.join(args.out, args.name + "_hot")))
    for name, splits in out.items():
        if not splits["train"]:
            continue
        d = os.path.join(args.out, name)
        os.makedirs(d, exist_ok=True)
        for sp, recs in splits.items():
            n = D.write_jsonl(os.path.join(d, sp + ".jsonl"), recs)
            print("%s/%s.jsonl: %d rows" % (d, sp, n))


if __name__ == "__main__":
    main()
