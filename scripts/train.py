"""Days 3 and 6: LoRA laya-vision on the teacher's frames; early-stop on held-out frames from the same teacher.

    python scripts/train.py --data data/seed_teacher --out runs/r0
    python scripts/train.py --data data/seed_teacher --data data/dagger_r1 --data data/dagger_r1_hot \\
        --init runs/r0/best --out runs/r1

Each --data dir is sampled as its own group, in equal shares by default (``--mix name=weight`` to change it),
which is how seed and DAgger rows are merged 50/50 without one swamping the other.
Early stopping watches val *frame accuracy against the teacher*; the real verdict is scripts/gate.py.
``--max-minutes`` is wall time for the whole run: model load and every eval, the final one included, come out of it.
"""
import argparse
import json
import os
import random
import time

import _path  # noqa: F401
from sf2 import lora
from sf2.config import BASE_MODEL


class EarlyStop(Exception):
    pass


def split_val(train, val, limit, seed):
    """Cap val at ``limit`` frames; with no val.jsonl anywhere, hold out a slice of train instead."""
    rng = random.Random(seed)
    if len(val) > limit:
        val = rng.sample(val, limit)
    if not val:
        rng.shuffle(train)
        n = min(len(train) // 20, limit)
        val, train = train[:n], train[n:]
    return train, val


def training_minutes(max_minutes, spent_s, eval_s):
    """Budget handed to the training loop, given the seconds already spent and one eval's duration.

    The loop's clock covers its own evals; what is left over pays for the final eval and the save."""
    return max(1.0, max_minutes - (spent_s + eval_s) / 60 - 0.5)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", action="append", required=True, help="dataset dir (train.jsonl / val.jsonl)")
    ap.add_argument("--init", default=BASE_MODEL, help="checkpoint to start from (Hub id or runs/<x>/best)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--mode", choices=["lora", "head", "last_n", "full"], default="lora")
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--alpha", type=float, default=32.0)
    ap.add_argument("--n-last", type=int, default=4, help="layers for --mode last_n")
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr-head", type=float, default=1e-4)
    ap.add_argument("--lr-backbone", type=float, default=None, help="default 2e-4 for lora, 2e-5 otherwise")
    ap.add_argument("--eval-every", type=int, default=250)
    ap.add_argument("--patience", type=int, default=3, help="evals without val-accuracy gain before stopping")
    ap.add_argument("--val-limit", type=int, default=600, help="val frames per eval (~0.15 s each on MPS)")
    ap.add_argument("--mix", action="append", default=[], help="group=weight, e.g. dagger_r1_hot=0.5")
    ap.add_argument("--temperature", choices=["one", "keep"], default="one",
                    help="'one' resets the choice temperature to 1.0 (the base checkpoint's photo-fitted value "
                         "flattens game policies); 'keep' leaves it")
    ap.add_argument("--max-minutes", type=float, default=None, help="wall-clock budget for the whole run")
    ap.add_argument("--workers", type=int, default=4, help="data-loader processes (0 = load on the GPU thread)")
    ap.add_argument("--device", default=None, help="default: mps on Apple silicon")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    t_start = time.time()

    import laya
    import laya.vlm_train as vt

    agent = laya.load_vlm(args.init, device=args.device)
    train, val = [], []
    for d in args.data:
        root, name = os.path.split(os.path.normpath(d))
        train += vt.load_jsonl_examples(root, name, "train")
        if os.path.exists(os.path.join(d, "val.jsonl")):
            val += vt.load_jsonl_examples(root, name, "val")
    train, val = split_val(train, val, args.val_limit, args.seed)
    print("train %d frames, val %d frames, init %s, mode %s" % (len(train), len(val), args.init, args.mode))

    if args.mode == "lora":
        n = lora.inject(agent.model.encoder, rank=args.rank, alpha=args.alpha)
        print("LoRA r=%d on %d projections" % (args.rank, n))
        orig_set_trainable = vt.set_trainable

        def set_trainable(model, mode="head", n_last=4, train_vision=False):
            orig_set_trainable(model, "head", n_last, train_vision)  # backbone frozen, decision head trains
            for p in lora.lora_params(model):
                p.requires_grad = True
            return sum(p.numel() for p in model.parameters() if p.requires_grad)

        vt.set_trainable = set_trainable  # vlm_train.train looks it up by name
    lr_backbone = args.lr_backbone or (2e-4 if args.mode == "lora" else 2e-5)

    def save(path):
        model = lora.merge(agent.model) if args.mode == "lora" else agent.model
        keep_model, keep_t, keep_tb = agent.model, agent.temperature, agent.temperature_by_options
        agent.model = model
        if args.temperature == "one":
            agent.temperature = [1.0] + list(keep_t)[1:]
            agent.temperature_by_options = {k: v for k, v in keep_tb.items() if not k.startswith("choice")}
        agent.save(path, include_backbone=args.mode != "head")
        agent.model, agent.temperature, agent.temperature_by_options = keep_model, keep_t, keep_tb

    os.makedirs(args.out, exist_ok=True)
    hist, best = [], {"acc": -1.0, "step": None, "bad": 0}

    def eval_fn(step):
        t = time.time()
        m = vt.metrics_from(vt.collect_logits(agent.model, agent.processor, val, batch_size=16))
        agent.model.train()
        hist.append({"step": step, "seconds": time.time() - t, **{k: v for k, v in m.items()}})
        print("eval step %d (%.0fs): %s" % (step, hist[-1]["seconds"], vt.format_metrics(m)), flush=True)
        if m["all"]["acc"] > best["acc"]:
            best.update(acc=m["all"]["acc"], step=step, bad=0)
            save(os.path.join(args.out, "best"))
            print("  saved best -> %s/best" % args.out, flush=True)
        else:
            best["bad"] += 1
            if best["bad"] >= args.patience:
                raise EarlyStop()
        return True

    eval_fn(0)
    train_minutes = None
    if args.max_minutes is not None:
        train_minutes = training_minutes(args.max_minutes, time.time() - t_start, hist[0]["seconds"])
        print("budget %.1f min, %.1f for the training loop" % (args.max_minutes, train_minutes))
    steps = max(1, int(args.epochs * len(train) / args.batch_size))
    mix = {k: float(v) for k, v in (s.split("=") for s in args.mix)} or None
    try:
        vt.train(agent.model, agent.processor, train, steps=steps, batch_size=args.batch_size, freeze=args.mode,
                 n_last=args.n_last, lr_head=args.lr_head, lr_backbone=lr_backbone, device=str(agent.device),
                 seed=args.seed, log_every=25, max_minutes=train_minutes, num_workers=args.workers,
                 warmup=min(100, steps // 10), eval_fn=eval_fn, eval_every=args.eval_every, mix_weights=mix)
        eval_fn(steps)
    except EarlyStop:
        print("early stop: no val gain in %d evals" % args.patience)
    with open(os.path.join(args.out, "train_log.json"), "w") as f:
        json.dump({"args": vars(args), "best": best, "evals": hist}, f, indent=2)
    print("best val frame accuracy %.3f at step %s -> %s/best" % (best["acc"], best["step"], args.out))


if __name__ == "__main__":
    main()
