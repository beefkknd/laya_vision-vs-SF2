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
import threading
import time

import _path  # noqa: F401
from sf2 import dataset as D
from sf2.rollout import gate

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_OUT = {"collect_teacher": "data", "play_student": "rollouts", "play_teacher": "rollouts"}
LIVE_LOG = "out/live.log"  # every run's workers, interleaved and timestamped: tail -F out/live.log
RESULTS = "out/results.jsonl"  # one row per finished run, appended forever: the data for learning curves
LEDGER_KEYS = ["matches", "distinct_matches", "rounds", "round_win_rate", "net_damage_per_round", "net_damage_se",
               "dmg_dealt_per_round", "dmg_taken_per_round", "decisions", "teacher_agreement"]


def record_result(path, name, script, model, openings, workers, argv, gate):
    """Append one run's summary to the results ledger (never rewritten)."""
    opt = lambda flag: argv[argv.index(flag) + 1] if flag in argv else None  # noqa: E731
    row = {"time": time.strftime("%Y-%m-%dT%H:%M:%S"), "name": name, "script": script, "model": model,
           "savestate": opt("--savestate"), "me": opt("--me"), "opp": opt("--opp"), "openings": openings,
           "memory": opt("--memory"), "tau": opt("--tau"),
           "workers": workers, **{k: gate.get(k) for k in LEDGER_KEYS if k in gate}}
    with open(path, "a") as f:
        f.write(json.dumps(row) + "\n")


def pump(proc, own_log, name, live, lock):
    """Copy a worker's output to its own log and, prefixed with the time and its name, to the shared live log."""
    for line in proc.stdout:
        own_log.write(line)
        own_log.flush()
        with lock:
            live.write("%s [%s] %s" % (time.strftime("%H:%M:%S"), name, line))
            live.flush()


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
        if g["distinct_matches"] < g["matches"]:
            print("WARNING: only %d of %d matches differ; the others replayed an identical fight"
                  % (g["distinct_matches"], g["matches"]), flush=True)
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
    sub.add_argument("--openings", help="a saved opening schedule; split across workers, one match per opening")
    known, passthrough = sub.parse_known_args(rest)
    if known.model:
        passthrough += ["--model", known.model]

    n = args.workers
    chunks = None
    if known.openings:  # pairing comes from the schedule, not from the worker layout
        from sf2 import openings as O

        chunks = O.split(O.parse(known.openings), n)
        n = len(chunks)
    dec = split(known.decisions, n) if known.decisions else [None] * n
    mat = [len(c) for c in chunks] if chunks else split(known.matches, n) if known.matches else [None] * n
    os.makedirs("out/parallel", exist_ok=True)
    procs, dirs = [], worker_dirs(known.out, known.name, n)
    live, lock, pumps = open(LIVE_LOG, "a"), threading.Lock(), []

    def say(msg):
        print(msg, flush=True)
        with lock:
            live.write("%s [%s] %s\n" % (time.strftime("%H:%M:%S"), known.name, msg))
            live.flush()

    say("start: %d workers, %s %s" % (n, args.script, " ".join(rest)))
    for i in range(n):
        name = "%s_w%d" % (known.name, i)
        argv = [sys.executable, "-u", os.path.join(HERE, args.script + ".py"), *passthrough, "--headless",
                "--port", str(args.base_port + i), *worker_seed_args(known.seed, i), "--name", name,
                "--out", os.path.join(known.out, known.name)]
        if dec[i] is not None:
            argv += ["--decisions", str(dec[i])]
        if mat[i] is not None:
            argv += ["--matches", str(mat[i])]
        if chunks:
            argv += ["--openings", ",".join(map(str, chunks[i]))]
        log = open("out/parallel/%s.log" % name, "w")
        p = subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        pumps.append(threading.Thread(target=pump, args=(p, log, name, live, lock), daemon=True))
        pumps[-1].start()
        procs.append((p, log, name))
        print("worker %d: port %d -> %s (log out/parallel/%s.log)" % (i, args.base_port + i, dirs[i], name),
              flush=True)
    t0, failed = time.time(), []
    for (p, log, name), t in zip(procs, pumps):
        if p.wait():
            failed.append(name)
        t.join()
        log.close()
    say("workers done in %.0fs" % (time.time() - t0))
    if failed:
        say("FAILED: %s (see out/parallel/<name>.log); merging the rest" % ", ".join(failed))
    ok = [d for d, (_, _, name) in zip(dirs, procs) if name not in failed]
    if not ok:  # an empty dataset marked _READY would look trainable and block re-using the name
        sys.exit("every worker failed; nothing merged")
    counts, g = merge(ok, os.path.join(known.out, known.name), args.script, known.model)
    say("merged %s -> %s" % (counts, os.path.join(known.out, known.name)))
    if g:
        say("gate: rounds %d, round win rate %.3f, net damage per round %.1f +- %.1f"
            % (g["rounds"], g["round_win_rate"], g["net_damage_per_round"], g["net_damage_se"]))
        record_result(RESULTS, known.name, args.script, known.model, known.openings, n, rest, g)
    if g:
        print(json.dumps(g, indent=2))
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
