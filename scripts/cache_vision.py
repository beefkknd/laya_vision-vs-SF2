"""Build the frozen vision-feature cache (sf2.vision_cache) for datasets ahead of training.

    python scripts/cache_vision.py --init runs/chunli_r3/best --data data/seed_chunli_r5 --data data/dagger_chunli_r2

train.py builds whatever is missing on its own; this just lets the GPU do it while nothing else needs it. The cache
is keyed by the vision weights, which LoRA rounds never change, so one build serves every later round.
"""
import argparse
import os
import time

import _path  # noqa: F401
from sf2 import vision_cache


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--init", required=True, help="checkpoint whose vision tower to use")
    ap.add_argument("--data", action="append", required=True)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--device", default=None)
    ap.add_argument("--image-size", type=int, default=None, help="as train.py --image-size; default: the checkpoint's")
    args = ap.parse_args()

    import laya
    import laya.vlm_train as vt

    agent = laya.load_vlm(args.init, device=args.device, **({"image_size": args.image_size} if args.image_size else {}))
    fp = vision_cache.fingerprint(agent.model, agent.model.prep)
    todo = vision_cache.missing(args.data, fp)
    print("vision cache %s: %d to build, %d present" % (fp, len(todo), len(args.data) - len(todo)), flush=True)
    enc = vision_cache.encoder(agent.model, agent.processor, num_workers=args.workers)
    for d in todo:
        t = time.time()
        root, name = os.path.split(os.path.normpath(d))
        rows = vt.load_jsonl_examples(root, name, "train")
        if os.path.exists(os.path.join(d, "val.jsonl")):
            rows += vt.load_jsonl_examples(root, name, "val")
        vision_cache.build(d, rows, enc, fp, log=lambda m: print(m, flush=True))
        print("  %s done in %.0fs" % (d, time.time() - t), flush=True)


if __name__ == "__main__":
    main()
