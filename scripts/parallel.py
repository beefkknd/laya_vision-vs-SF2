"""Run a collection / play script on N headless Mesen workers at once, then merge their output.

    python scripts/parallel.py --workers 4 collect_teacher --name seed_teacher --decisions 40000 --eps 0.25
    python scripts/parallel.py --workers 4 play_student --model runs/r0/best --name r0 --matches 12
    python scripts/parallel.py --workers 4 play_teacher --name teacher --matches 12

Each worker is `scripts/<script>.py ... --headless --port <base+i> --seed <i> --name <name>_w<i>`, with
--decisions / --matches split between workers. Every worker starts its own windowless Mesen (`--testrunner`),
so set SF2_ROM (and SF2_MESEN if Mesen is not in /Applications) or pass --rom / --mesen after the script name.
Logs: out/parallel/<name>_w<i>.log. The merged result lands where a single run would have put it
(data/<name> or rollouts/<name>), so train.py / relabel.py / gate.py use it unchanged.

This speeds up *collection and evaluation* (the emulator, PNG writing and the rollouts). Training itself is one
process on the GPU; run the next round's collection while it trains if you want both busy.
"""
import argparse
import json
import os
import subprocess
import sys
import time

import _path  # noqa: F401
from sf2 import dataset as D
from sf2.rollout import gate

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_OUT = {"collect_teacher": "data", "play_student": "rollouts", "play_teacher": "rollouts"}


def split(total: int, n: int):
    return [total // n + (1 if i < total % n else 0) for i in range(n)]


def worker_seed_args(seed, i):
    """What makes worker i's matches differ from every other worker's."""
    return ["--seed", str(seed * 1000 + i), "--jitter-base", str(i * 30)]


def worker_dirs(out: str, name: str, n: int):
    """Workers write inside the batch dir, so ``<out>/<name>`` alone is the whole batch (copyable between machines)."""
    return [os.path.join(out, name, "%s_w%d" % (name, i)) for i in range(n)]  # batch name keeps row ids unique


def merge(src_dirs, dst, script, model=None):
    os.makedirs(dst, exist_ok=True)
    for sp in ("train", "val"):
        open(os.path.join(dst, sp + ".jsonl"), "w").close()
    counts, rows, rounds = {}, [], []
    for i, src in enumerate(src_dirs):
        off = i * 100_000  # keep episode ids unique across workers (val split, damage windows)
        for sp in ("train", "val"):
            p = os.path.join(src, sp + ".jsonl")
            if not os.path.exists(p):
                continue
            recs = []
            for r in D.read(p):
                r["images"] = [os.path.relpath(os.path.join(src, im), dst) for im in r["images"]]
                r["episode"] += off
                r["meta"]["episode"] = r["meta"].get("episode", 0) + off
                recs.append(r)
            with open(os.path.join(dst, sp + ".jsonl"), "a") as f:
                for r in recs:
                    f.write(json.dumps(r) + "\n")
            counts[sp] = counts.get(sp, 0) + len(recs)
            rows += recs
        p = os.path.join(src, "rows.jsonl")
        if os.path.exists(p):
            rows += D.read(p)
        p = os.path.join(src, "rounds.jsonl")
        if os.path.exists(p):
            for r in D.read(p):
                r["episode"] += off
                rounds.append(r)
    D.write_jsonl(os.path.join(dst, "rounds.jsonl"), rounds)
    with open(os.path.join(dst, "_READY"), "w") as f:
        json.dump(counts, f)
    if script != "collect_teacher" or rounds:
        g = gate(rows, rounds)
        g.update(model=model or script, workers=len(src_dirs))
        with open(os.path.join(dst, "gate.json"), "w") as f:
            json.dump(g, f, indent=2)
        return counts, g
    return counts, None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--base-port", type=int, default=47810)
    ap.add_argument("script", choices=sorted(DEFAULT_OUT))
    args, rest = ap.parse_known_args()

    sub = argparse.ArgumentParser(add_help=False)
    sub.add_argument("--name", required=True)
    sub.add_argument("--out", default=DEFAULT_OUT[args.script])
    sub.add_argument("--decisions", type=int)
    sub.add_argument("--matches", type=int)
    sub.add_argument("--seed", type=int, default=0)
    sub.add_argument("--port", type=int)
    sub.add_argument("--model")
    known, passthrough = sub.parse_known_args(rest)
    if known.model:
        passthrough += ["--model", known.model]

    n = args.workers
    dec = split(known.decisions, n) if known.decisions else [None] * n
    mat = split(known.matches, n) if known.matches else [None] * n
    os.makedirs("out/parallel", exist_ok=True)
    procs, dirs = [], worker_dirs(known.out, known.name, n)
    for i in range(n):
        name = "%s_w%d" % (known.name, i)
        argv = [sys.executable, os.path.join(HERE, args.script + ".py"), *passthrough, "--headless",
                "--port", str(args.base_port + i), *worker_seed_args(known.seed, i), "--name", name,
                "--out", os.path.join(known.out, known.name)]
        if dec[i] is not None:
            argv += ["--decisions", str(dec[i])]
        if mat[i] is not None:
            argv += ["--matches", str(mat[i])]
        log = open("out/parallel/%s.log" % name, "w")
        procs.append((subprocess.Popen(argv, stdout=log, stderr=subprocess.STDOUT), log, name))
        print("worker %d: port %d -> %s (log out/parallel/%s.log)" % (i, args.base_port + i, dirs[i], name),
              flush=True)
    t0, failed = time.time(), []
    for p, log, name in procs:
        if p.wait():
            failed.append(name)
        log.close()
    print("workers done in %.0fs" % (time.time() - t0))
    if failed:
        print("FAILED: %s (see out/parallel/<name>.log); merging the rest" % ", ".join(failed))
    ok = [d for d, (_, _, name) in zip(dirs, procs) if name not in failed]
    if not ok:  # an empty dataset marked _READY would look trainable and block re-using the name
        sys.exit("every worker failed; nothing merged")
    counts, g = merge(ok, os.path.join(known.out, known.name), args.script, known.model)
    print("merged %s -> %s" % (counts, os.path.join(known.out, known.name)))
    if g:
        print(json.dumps(g, indent=2))
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
