"""Collect the movement dataset (docs/prereg_movement_data.md): Chun-Li (runs/all8, explore 0.5, no memory, no Qwen,
no text laya) vs each of the 7 opponents, headless, one worker (Mesen + her model on MPS) per opponent in parallel.
Every frame is captured; when one of his actions ends, frame pairs are sampled from it at start / middle / end
(sf2.data.movement_collect) until every (answer, split) quota is full or --cap games are played.

    python scripts/collect_movement.py --out rollouts/mv2 --cap 1000
    python scripts/collect_movement.py --out /tmp/x --cap 1 --opps ryu,ken          # a smoke

Writes <out>/<opp>/{images/, ram/, pairs.jsonl, games.jsonl, stop.json} (sf2.data.movement_collect_io) and
<out>/run.json; process logs <log-dir>/<opp>.log. Re-running the same command resumes from the committed games.
Memory: each worker reserves --job-gb in the machine-wide ledger (sf2.eval.budget) with a --budget-gb total, and
stops itself (exit 3) when its own peak RSS plus Mesen's exceeds --mem-cap-gb.
"""
import argparse
import json
import os
import random
import subprocess
import sys
import time

import _path  # noqa: F401
from sf2.config import PORTS
from sf2.data import movement_collect as C
from sf2.data import movement_collect_io as IO
from sf2.emu.vs import NAMES
from sf2.eval.budget import Budget
from sf2.eval.runner import exit_on_sigterm, fan_out, open_fight

ME = "chunli"
OPPS = ("blanka", "dhalsim", "guile", "honda", "ken", "ryu", "zangief")
MODEL = os.path.join("runs", "all8", "best")
EXPLORE = 0.5
CAP = 1000                   # games per opponent: Ken's rarest answer (walking away, ~2 pairs/game) needs ~770
JOB_GB = 4.0                 # one worker: laya-vision (~3 GB, config.VISION_JOB_GB) + the ring + frames in flight
BUDGET_GB = 60.0             # all workers together (the owner's ~60 GB)


def rss_of(pid: int) -> float:
    out = subprocess.run(["ps", "-o", "rss=", "-p", str(pid)], capture_output=True, text=True).stdout.strip()
    return int(out) * 1024 / 1e9 if out.isdigit() else 0.0


def play_one(args, opp: str, port: int) -> int:
    from sf2.system1.system1 import System1, play_round

    base = os.path.join(args.out, opp)
    s1 = System1(None if args.model in ("random", "none") else args.model, ME, args.threshold, args.device, args.seed,
                 explore=args.explore)
    t0 = time.time()
    with open_fight(ME, opp, port, args.rom) as (b, state):
        proxy = C.CapturingBridge(b, NAMES)

        def play(game: int, sampler) -> dict:
            proxy.sink = sampler.feed
            s1.rng = random.Random("s1:%d:%s:%d" % (args.seed, opp, game))
            rnd = play_round(proxy, s1, opp, state, random.Random("start:%d:%s:%d" % (args.seed, opp, game)), None,
                             game)
            proxy.sink = None
            return {k: rnd.summary.get(k) for k in ("result", "frames", "dealt", "taken", "actions")}

        def rss() -> float:
            return IO.peak_rss_gb() + (rss_of(b.proc.pid) if getattr(b, "proc", None) else 0.0)

        try:
            stop = IO.collect_opponent(base, play, opp=opp, cap=args.cap, seed=args.seed, ring=args.ring,
                                       mem_cap_gb=args.mem_cap_gb, rss_gb=rss,
                                       log=lambda m: print(m, flush=True))
        except IO.MemoryCapExceeded as e:
            print("STOP:", e, flush=True)
            return 3
    print("%s: %s after %d games, %.0f s; short: %s" % (opp, stop["reason"], stop["games"], time.time() - t0,
                                                       stop["short"]), flush=True)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=os.path.join("rollouts", "mv2"))
    ap.add_argument("--opps", default=",".join(OPPS))
    ap.add_argument("--cap", type=int, default=CAP, help="games per opponent at most")
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--explore", type=float, default=EXPLORE)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--ring", type=int, default=C.RING)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=None)
    ap.add_argument("--rom", default=os.environ.get("SF2_ROM"))
    ap.add_argument("--base-port", type=int, default=PORTS["movement"][0])
    ap.add_argument("--job-gb", type=float, default=JOB_GB)
    ap.add_argument("--budget-gb", type=float, default=BUDGET_GB)
    ap.add_argument("--mem-cap-gb", type=float, default=IO.MEM_CAP_GB)
    ap.add_argument("--log-dir", default=os.path.join("logs", "movement"))
    ap.add_argument("--one", nargs=2, metavar=("OPP", "PORT"), help=argparse.SUPPRESS)
    args = ap.parse_args()
    exit_on_sigterm()
    if args.one:
        return play_one(args, args.one[0], int(args.one[1]))
    opps = args.opps.split(",")
    bad = sorted(set(opps) - set(OPPS))
    if bad:
        raise SystemExit("unknown opponents %s (Chun-Li meets %s)" % (bad, ",".join(OPPS)))
    if len(opps) > PORTS["movement"][1]:
        raise SystemExit("more workers than ports")
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "run.json"), "w") as f:
        json.dump({"me": ME, "opps": opps, "cap": args.cap, "model": args.model, "explore": args.explore,
                   "threshold": args.threshold, "ring": args.ring, "seed": args.seed, "quota": C.SPLIT_QUOTA,
                   "short": C.SHORT, "lag": C.LAG, "started": time.strftime("%Y-%m-%d %H:%M:%S")}, f, indent=1)
    common = ["--out", args.out, "--cap", str(args.cap), "--model", args.model, "--explore", str(args.explore),
              "--threshold", str(args.threshold), "--ring", str(args.ring), "--seed", str(args.seed),
              "--mem-cap-gb", str(args.mem_cap_gb)] + (["--rom", args.rom] if args.rom else []) + (
              ["--device", args.device] if args.device else [])
    cmds = [((o,), [sys.executable, os.path.abspath(__file__), "--one", o, str(args.base_port + i)] + common)
            for i, o in enumerate(opps)]
    print("%d workers (%s/<opp>.log), %.1f GB each, %.0f GB budget" % (len(cmds), args.log_dir, args.job_gb,
                                                                     args.budget_gb), flush=True)
    failed = [o for (o,) in fan_out(cmds, args.log_dir, job_gb=args.job_gb, budget=Budget(total_gb=args.budget_gb))]
    for o in opps:
        p = os.path.join(args.out, o, "stop.json")
        s = json.load(open(p)) if os.path.exists(p) else {"reason": "no stop.json"}
        print("%-8s %-12s games %s short %s" % (o, s["reason"], s.get("games"), s.get("short")))
    for o in failed:
        print("FAILED:", o, "see %s/%s.log" % (args.log_dir, o))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
