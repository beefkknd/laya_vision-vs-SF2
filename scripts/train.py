"""LoRA-train base laya-vision (sf2.config.BASE_MODEL, SmolVLM-256M) on one or more dataset dirs. Always starts from
the base checkpoint; no other init is supported. Images go in at 256x256 native pixels (sf2.config.IMAGE_CFG), no
upscale; the saved checkpoint keeps those settings, so play loads them too.

    python scripts/train.py --out runs/all8 \\
        --data test_data/ryu --data test_data/ken --data test_data/chunli --data test_data/guile \\
        --data test_data/honda --data test_data/blanka --data test_data/zangief --data test_data/dhalsim

Each --data dir supplies <dir>/train.jsonl (and <dir>/val.jsonl if it has one) in laya-vision's record layout
(laya.vlm_train.jsonl_example). Each dir is sampled as its own group, in equal shares, so every character gets the
same weight. Without any val.jsonl, 5% of the screen positions (every action asked there, and the mirrored twin)
are held out for early stopping, so validation never shows a frame the model trained on. The adapters are merged
before saving, so <out>/best is an ordinary laya-vision checkpoint.
"""
import argparse
import json
import os
import time

import _path  # noqa: F401
from sf2.data import lora
from sf2.config import BASE_MODEL, IMAGE_CFG
from sf2.data.train_data import checkpoint_tags, coverage_problems, coverage_table, load_data


class EarlyStop(Exception):
    pass


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", action="append", required=True, help="dataset dir (train.jsonl, optional val.jsonl)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--alpha", type=float, default=32.0)
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr-head", type=float, default=1e-4)
    ap.add_argument("--lr-backbone", type=float, default=2e-4, help="learning rate of the LoRA adapters")
    ap.add_argument("--eval-every", type=int, default=250)
    ap.add_argument("--patience", type=int, default=3, help="evals without val-accuracy gain before stopping")
    ap.add_argument("--val-limit", type=int, default=4000, help="stop if validation is larger (it is not cut at random)")
    ap.add_argument("--max-minutes", type=float, default=None)
    ap.add_argument("--device", default=None, help="default: mps on Apple silicon")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    # data first: a coverage gap stops the run before a model is loaded
    train, val = load_data(args.data, args.seed)
    if len(val) > args.val_limit:
        raise SystemExit("val has %d rows > --val-limit %d; a random cut would unbalance the characters"
                         % (len(val), args.val_limit))
    print("coverage:\n" + coverage_table(train, val, args.data), flush=True)
    problems = coverage_problems(train, val, args.data)
    if problems:
        raise SystemExit("coverage check failed (%d):\n  %s" % (len(problems), "\n  ".join(problems[:40])))

    tags = checkpoint_tags(train + val)      # the note and the questions play must use (sf2.system1.system1)
    print("checkpoint tags: %s" % tags, flush=True)

    import laya
    import laya.vlm_train as vt

    agent = laya.load_vlm(BASE_MODEL, device=args.device, **IMAGE_CFG)
    agent.cfg = dict(agent.cfg, **tags)      # saved in <out>/best/vlm_agent_config.json
    prep = agent.model.prep
    got = (prep.image_size, prep.backend, prep.interpolation,
           agent.processor.image_processor.max_image_size.get("longest_edge"))
    want = (IMAGE_CFG["image_size"], IMAGE_CFG["preprocess"], IMAGE_CFG["image_interpolation"], IMAGE_CFG["image_size"])
    if got != want:
        raise SystemExit("image prep is %s, expected %s (sf2.config.IMAGE_CFG)" % (got, want))
    print("base %s | images %s | train %d rows from %d dirs, val %d rows" % (
        BASE_MODEL, IMAGE_CFG, len(train), len(args.data), len(val)))

    n = lora.inject(agent.model.encoder, rank=args.rank, alpha=args.alpha)
    print("LoRA r=%d on %d projections" % (args.rank, n))
    orig_set_trainable = vt.set_trainable

    def set_trainable(model, mode="head", n_last=4, train_vision=False):
        orig_set_trainable(model, "head", n_last, train_vision)  # backbone frozen, decision head trains
        for p in lora.lora_params(model):
            p.requires_grad = True
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    vt.set_trainable = set_trainable  # vlm_train.train looks it up by name

    def save(path):
        # merged copy (training continues on the adapters); the choice temperature goes back to 1.0, because the
        # base checkpoint's photo-fitted value flattens game answers
        keep_model, keep_t, keep_tb = agent.model, agent.temperature, agent.temperature_by_options
        agent.model = lora.merge(keep_model)
        agent.temperature = [1.0] + list(keep_t)[1:]
        agent.temperature_by_options = {k: v for k, v in keep_tb.items() if not k.startswith("choice")}
        agent.save(path, include_backbone=True)
        agent.model, agent.temperature, agent.temperature_by_options = keep_model, keep_t, keep_tb

    os.makedirs(args.out, exist_ok=True)
    hist, best = [], {"acc": -1.0, "step": None, "bad": 0}

    def eval_fn(step):
        t = time.time()
        m = vt.metrics_from(vt.collect_logits(agent.model, agent.processor, val, batch_size=16))
        agent.model.train()
        hist.append({"step": step, **{k: v for k, v in m.items()}})
        print("eval step %d (%.0fs): %s" % (step, time.time() - t, vt.format_metrics(m)), flush=True)
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
    steps = max(1, int(args.epochs * len(train) / args.batch_size))
    try:
        vt.train(agent.model, agent.processor, train, steps=steps, batch_size=args.batch_size, freeze="lora",
                 n_last=4, lr_head=args.lr_head, lr_backbone=args.lr_backbone, device=str(agent.device),
                 seed=args.seed, log_every=25, max_minutes=args.max_minutes, num_workers=0,
                 warmup=min(100, steps // 10), eval_fn=eval_fn, eval_every=args.eval_every)
        eval_fn(steps)
    except EarlyStop:
        print("early stop: no val gain in %d evals" % args.patience)
    with open(os.path.join(args.out, "train_log.json"), "w") as f:
        json.dump({"args": vars(args), "base": BASE_MODEL, "tags": tags, "best": best, "evals": hist}, f, indent=2)
    print("best val accuracy %.3f at step %s -> %s/best" % (best["acc"], best["step"], args.out))


if __name__ == "__main__":
    main()
