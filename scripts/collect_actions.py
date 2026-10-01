"""Collect the action dataset (docs/prereg_movement_data.md, "Owner decisions on the label"): Chun-Li (runs/all8,
explore 0.5, no memory, no Qwen, no text laya) vs each of the 7 opponents, headless, one worker per opponent in
parallel. Every frame is captured with both fighters' move bytes (sf2.data.action_codes.EXTRA_VARS); when either
fighter's action episode ends, pairs are sampled from it at start / middle / end under small per-(actor, code, stage)
caps (sf2.data.action_collect), until no new (actor, code) has come for --patience games after --min-games, or --cap.

    python scripts/collect_actions.py --out rollouts/mv3
    python scripts/collect_actions.py --out /tmp/x --cap 1 --min-games 1 --opps ryu,ken          # a smoke

Writes <out>/<opp>/{images/, ram/, pairs.jsonl, games.jsonl, stop.json} and <out>/run.json; process logs
<log-dir>/<opp>.log. Re-running the same command resumes from the committed games. Memory as collect_movement.py.
"""
import argparse
import json
import os
import random
import sys
import time

import _path  # noqa: F401
from collect_movement import BUDGET_GB, EXPLORE, JOB_GB, ME, MODEL, OPPS, rss_of
from sf2.config import PORTS
from sf2.data import action_codes as A
from sf2.data import action_collect as C
from sf2.data import action_collect_io as IO
from sf2.emu.vs import NAMES, VARS
from sf2.eval.budget import Budget
from sf2.eval.runner import exit_on_sigterm, fan_out, open_fight
from sf2.data.movement_collect import CapturingBridge


def play_one(args, opp: str, port: int) -> int:
    from sf2.system1.system1 import System1, play_round

    base = os.path.join(args.out, opp)
    s1 = System1(None if args.model in ("random", "none") else args.model, ME, args.threshold, args.device, args.seed,
                 explore=args.explore)
    t0 = time.time()
    with open_fight(ME, opp, port, args.rom) as (b, state):
        b.set_vars(VARS + A.EXTRA_VARS)      # play_round zips NAMES with each row: the extra bytes ride at the end
        proxy = CapturingBridge(b, NAMES + A.EXTRA_NAMES)

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
            stop = IO.collect_opponent(base, play, opp=opp, actors={1: ME, 2: opp}, seed=args.seed, cap=args.cap,
                                       min_games=args.min_games, patience=args.patience, ring=args.ring,
                                       mem_cap_gb=args.mem_cap_gb, rss_gb=rss, log=lambda m: print(m, flush=True))
        except IO.MemoryCapExceeded as e:
            print("STOP:", e, flush=True)
            return 3
    print("%s: %s after %d games (last new at game %s), %.0f s" % (opp, stop["reason"], stop["games"],
                                                                   stop["last_new_game"], time.time() - t0), flush=True)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=os.path.join("rollouts", "mv3"))
    ap.add_argument("--opps", default=",".join(OPPS))
    ap.add_argument("--cap", type=int, default=IO.GAME_CAP, help="games per opponent at most")
    ap.add_argument("--min-games", type=int, default=IO.MIN_GAMES)
    ap.add_argument("--patience", type=int, default=IO.PATIENCE, help="games without a new (actor, code) to stop")
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
    ap.add_argument("--log-dir", default=os.path.join("logs", "actions"))
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
        json.dump({"me": ME, "opps": opps, "cap": args.cap, "min_games": args.min_games, "patience": args.patience,
                   "caps": C.CAPS, "model": args.model, "explore": args.explore, "threshold": args.threshold,
                   "ring": args.ring, "seed": args.seed, "short": C.SHORT, "lag": C.LAG, "codes": A.table(),
                   "started": time.strftime("%Y-%m-%d %H:%M:%S")}, f, indent=1)
    common = ["--out", args.out, "--cap", str(args.cap), "--min-games", str(args.min_games), "--patience",
              str(args.patience), "--model", args.model, "--explore", str(args.explore), "--threshold",
              str(args.threshold), "--ring", str(args.ring), "--seed", str(args.seed), "--mem-cap-gb",
              str(args.mem_cap_gb)] + (["--rom", args.rom] if args.rom else []) + (
              ["--device", args.device] if args.device else [])
    cmds = [((o,), [sys.executable, os.path.abspath(__file__), "--one", o, str(args.base_port + i)] + common)
            for i, o in enumerate(opps)]
    print("%d workers (%s/<opp>.log), %.1f GB each, %.0f GB budget" % (len(cmds), args.log_dir, args.job_gb,
                                                                     args.budget_gb), flush=True)
    failed = [o for (o,) in fan_out(cmds, args.log_dir, job_gb=args.job_gb, budget=Budget(total_gb=args.budget_gb))]
    for o in opps:
        p = os.path.join(args.out, o, "stop.json")
        s = json.load(open(p)) if os.path.exists(p) else {"reason": "no stop.json"}
        print("%-8s %-14s games %s, last new at game %s, missing in train/test: %d codes" % (
            o, s["reason"], s.get("games"), s.get("last_new_game"), len(s.get("missing_in", {}))))
    for o in failed:
        print("FAILED:", o, "see %s/%s.log" % (args.log_dir, o))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
