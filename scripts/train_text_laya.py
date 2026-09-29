"""LoRA fine-tune of text laya (ModernBERT-large, MLX) to follow advice: test_data/advice (scripts/build_advice_data.py)
-> runs/text_laya/<name>/. One run from the pretrained base (aac6fef/laya-mlx), nothing else.

    ~/work/laya_mlx/.venv/bin/python scripts/train_text_laya.py --name advice_v1

Trains a LoRA on the encoder's linear layers plus the decision head (head, scorer, type_emb), soft-target cross-entropy
over the options (a row with tied answers spreads its target over them). Validation accuracy every --eval-every steps;
the best adapter is kept; stops after --patience evaluations without a gain. At the end: test accuracy of the base and
of the best adapter, by case / held-out wording / character.

Output: runs/text_laya/<name>/{adapter.safetensors, adapter.json, test.json}; log: logs/text_laya/<name>.log.
"""
import argparse
import json
import os
import random
import time

import numpy as np

import _path  # noqa: E402,F401
from sf2.config import TEST_DATA  # noqa: E402
from sf2.system1 import mlx_lora, text_laya  # noqa: E402

DATA = os.path.join(TEST_DATA, "advice")


class Say:
    """One line to the console and the run's log, with the time."""

    def __init__(self, path: str):
        self.log = open(path, "a")

    def __call__(self, msg: str) -> None:
        line = "%s  %s" % (time.strftime("%H:%M:%S"), msg)
        print(line, flush=True)
        self.log.write(line + "\n")
        self.log.flush()


def batches(encoded, size, rng):
    order = list(range(len(encoded)))
    rng.shuffle(order)
    for i in range(0, len(order), size):
        yield [encoded[j] for j in order[i:i + size]]


def collate(agent, chunk):
    import mlx.core as mx
    from laya_mlx.agent import collate_items

    b = collate_items([item for item, _, _ in chunk], agent.tok.pad_token_id, max_length=agent.cfg.get("max_len", 512))
    k = b["marker_pos"].shape[1]
    target = np.zeros((len(chunk), k), dtype=np.float32)
    for i, (_, _, t) in enumerate(chunk):
        target[i, :len(t)] = t
    return {name: mx.array(v) for name, v in b.items()}, mx.array(target)


def accuracy(agent, rows) -> float:
    probs = text_laya.predict_rows(agent, rows)
    return text_laya.score(rows, probs)["accuracy"]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--eval-every", type=int, default=100)
    ap.add_argument("--patience", type=int, default=4)
    ap.add_argument("--val-limit", type=int, default=400, help="validation rows per evaluation (speed)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0, help="smoke test: this many rows of each split")
    return ap.parse_args()


def load_splits(args, rng):
    """(train, val rows evaluated during training, test); ``--limit`` cuts each split for a smoke test."""
    train = text_laya.read(os.path.join(DATA, "train.jsonl"))
    val = text_laya.read(os.path.join(DATA, "val.jsonl"))
    test = text_laya.read(os.path.join(DATA, "test.jsonl"))
    if args.limit:
        train, val, test = train[:args.limit], val[:args.limit], test[:args.limit]
    return train, rng.sample(val, min(args.val_limit, len(val))), test


def train_lora(args, agent, train, val_eval, rng, say):
    """LoRA on the encoder plus the trainable heads, early stopping on validation accuracy: (best val, the best
    adapter's weights, steps taken)."""
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim
    from mlx.utils import tree_map

    model = agent.model
    n_wrapped = mlx_lora.apply_lora(model.encoder, r=args.rank, alpha=2.0 * args.rank)
    for name in text_laya.TRAINABLE:
        getattr(model, name).unfreeze()
    model.train()
    encoded = [text_laya.encode(agent, r) for r in train]
    steps = args.epochs * ((len(encoded) + args.batch_size - 1) // args.batch_size)
    say("LoRA r=%d on %d linears, %d trainable params, %d train rows, %d steps" % (
        args.rank, n_wrapped, mlx_lora.trainable_count(model), len(encoded), steps))

    def loss_fn(m, batch, target):
        logits, _ = m(**batch)
        return -(target * nn.log_softmax(logits, axis=-1)).sum(axis=-1).mean()

    loss_and_grad = nn.value_and_grad(model, loss_fn)
    opt = optim.AdamW(learning_rate=optim.cosine_decay(args.lr, steps), weight_decay=0.0)
    best, best_state, stale, step, t0 = -1.0, None, 0, 0, time.time()
    losses = []
    for epoch in range(args.epochs):
        for chunk in batches(encoded, args.batch_size, rng):
            batch, target = collate(agent, chunk)
            loss, grads = loss_and_grad(model, batch, target)
            opt.update(model, grads)
            mx.eval(model.parameters(), opt.state)
            losses.append(float(loss))
            step += 1
            if step % args.eval_every == 0 or step == steps:
                model.eval()
                acc = accuracy(agent, val_eval)
                model.train()
                gain = acc > best
                if gain:
                    best, stale = acc, 0
                    best_state = tree_map(lambda p: mx.array(p), mlx_lora.adapter_state(model))
                    mx.eval(best_state)
                else:
                    stale += 1
                say("epoch %d step %d/%d  loss %.4f  val %.3f%s  (%.0f s)" % (
                    epoch, step, steps, float(np.mean(losses[-args.eval_every:])), acc, "  best" if gain else "",
                    time.time() - t0))
                if stale >= args.patience:
                    break
        if stale >= args.patience:
            say("no gain in %d evaluations: stop" % args.patience)
            break
    return best, best_state, step


def save_and_test(args, out, best, best_state, step, base_test, test, say) -> None:
    """Save the best adapter, reload it, and score it on the test split next to the base model."""
    import mlx.core as mx
    from mlx.utils import tree_flatten

    mx.save_safetensors(os.path.join(out, "adapter.safetensors"), dict(tree_flatten(best_state)))
    with open(os.path.join(out, "adapter.json"), "w") as f:
        json.dump({"r": args.rank, "alpha": 2.0 * args.rank, "targets": list(mlx_lora.TARGETS), "top_k": None,
                   "base": text_laya.BASE, "data": DATA, "best_val": best, "steps": step, "args": vars(args)}, f,
                  indent=1)
    tuned = text_laya.load(out)
    result = {"base": base_test, "tuned": text_laya.score(test, text_laya.predict_rows(tuned, test))}
    with open(os.path.join(out, "test.json"), "w") as f:
        json.dump(result, f, indent=1)
    say("best val %.3f; test base %.3f -> tuned %.3f" % (best, base_test["accuracy"], result["tuned"]["accuracy"]))
    say("tuned test %s" % json.dumps(result["tuned"]))
    say("saved %s" % out)


def main() -> None:
    args = parse_args()
    import mlx.core as mx

    out = os.path.join("runs", "text_laya", args.name)
    if os.path.exists(os.path.join(out, "adapter.safetensors")):
        raise SystemExit("%s already has a checkpoint; pick a new --name" % out)
    os.makedirs(out, exist_ok=True)
    os.makedirs("logs/text_laya", exist_ok=True)
    say = Say(os.path.join("logs", "text_laya", args.name + ".log"))
    rng = random.Random(args.seed)
    mx.random.seed(args.seed)
    train, val_eval, test = load_splits(args, rng)
    agent = text_laya.load()
    base_test = text_laya.score(test, text_laya.predict_rows(agent, test))
    say("base %s: val %.3f  test %s" % (text_laya.BASE, accuracy(agent, val_eval), json.dumps(base_test)))
    best, best_state, step = train_lora(args, agent, train, val_eval, rng, say)
    save_and_test(args, out, best, best_state, step, base_test, test, say)
    say.log.close()


if __name__ == "__main__":
    main()
