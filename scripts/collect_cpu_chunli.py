"""Collect CPU Chun-Li's actions (docs/prereg_movement_data.md, "Addition (2026-10-01): CPU Chun-Li games"): each of
the 7 other characters as player 1 (runs/all8, explore 0.5, no memory, no Qwen, no text laya) vs the arcade CPU's
Chun-Li as player 2 (states/p1_<char>_vs_chunli.state, made by scripts/make_cpu_chunli_states.py), headless, one
worker per player-1 character in parallel. Same frames, sampler, caps and stop rule as scripts/collect_actions.py
(mv3), except that only Chun-Li's episodes are sampled and only her new (code, split) reset the patience. Every game's
RAM must hold (<char>, Chun-Li) on every row, or it is not committed; every pair and games.jsonl line carries the
provenance (sf2.data.cpu_chunli).

    python scripts/collect_cpu_chunli.py --out rollouts/mv4
    python scripts/collect_cpu_chunli.py --out /tmp/x --cap 1 --min-games 1 --p1 ken          # a smoke

Writes <out>/<p1 char>/{images/, ram/, pairs.jsonl, games.jsonl, stop.json} and <out>/run.json (the run's settings,
which the gate checks every row's provenance against); process logs <log-dir>/<p1 char>.log. Re-running the same
command resumes; a run.json with other settings is refused.
"""
import argparse
import json
import os
import random
import sys
import time

import _path  # noqa: F401
from collect_movement import BUDGET_GB, EXPLORE, JOB_GB, MODEL, rss_of
from sf2.config import PORTS
from sf2.data import action_codes as A
from sf2.data import action_collect as C
from sf2.data import action_collect_io as IO
from sf2.data import cpu_chunli as K
from sf2.data.movement_collect import CapturingBridge
from sf2.emu.vs import NAMES, VARS
from sf2.eval.budget import Budget
from sf2.eval.runner import exit_on_sigterm, fan_out, open_fight
from sf2.vocab import IDS

SETTINGS = ("model", "explore", "threshold", "ring", "seed", "cap", "min_games", "patience")


def play_one(args, p1: str, port: int) -> int:
    from sf2.system1.system1 import System1, play_round

    base = os.path.join(args.out, p1)
    s1 = System1(None if args.model in ("random", "none") else args.model, p1, args.threshold, args.device, args.seed,
                 explore=args.explore)
    t0 = time.time()
    with open_fight(p1, K.CPU, port, args.rom) as (b, state):      # checks (p1, Chun-Li) in the savestate's RAM
        b.set_vars(VARS + A.EXTRA_VARS)
        proxy = CapturingBridge(b, NAMES + A.EXTRA_NAMES)

        def play(game: int, sampler) -> dict:
            proxy.sink = sampler.feed
            s1.rng = random.Random("s1:%d:%s:%d" % (args.seed, p1, game))
            rnd = play_round(proxy, s1, K.CPU, state, random.Random("start:%d:%s:%d" % (args.seed, p1, game)),
                             None, game)
            proxy.sink = None
            return {k: rnd.summary.get(k) for k in ("result", "frames", "dealt", "taken", "actions")}

        def rss() -> float:
            return IO.peak_rss_gb() + (rss_of(b.proc.pid) if getattr(b, "proc", None) else 0.0)

        try:
            stop = IO.collect_opponent(base, play, opp=p1, actors={1: p1, 2: K.CPU}, seed=args.seed, cap=args.cap,
                                       min_games=args.min_games, patience=args.patience, ring=args.ring,
                                       mem_cap_gb=args.mem_cap_gb, rss_gb=rss, log=lambda m: print(m, flush=True),
                                       sample_actors={K.CPU}, provenance=K.provenance(p1),
                                       expect_ids={1: IDS[p1], 2: IDS[K.CPU]})
        except (IO.MemoryCapExceeded, IO.WrongCharacters) as e:
            print("STOP:", e, flush=True)
            return 3
    print("%s: %s after %d games (last new at game %s), %.0f s" % (p1, stop["reason"], stop["games"],
                                                                   stop["last_new_game"], time.time() - t0), flush=True)
    return 0


def check_run(out: str, settings: dict) -> None:
    """Write <out>/run.json, or refuse a resume whose settings differ from the existing one's."""
    old = K.read_run(out)
    if old is not None:
        keys = SETTINGS + ("collection", "chunli_slot", "controller", "cpu", "p1_chars")
        diff = {k: (old.get(k), settings.get(k)) for k in keys if old.get(k) != settings.get(k)}
        if diff:
            raise SystemExit("%s/run.json has other settings %s: use another --out" % (out, diff))
        return
    K.write_run(out, settings)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=os.path.join("rollouts", K.COLLECTION))
    ap.add_argument("--p1", default=",".join(K.P1_CHARS), help="player-1 characters (comma-separated)")
    ap.add_argument("--cap", type=int, default=IO.GAME_CAP, help="games per player-1 character at most")
    ap.add_argument("--min-games", type=int, default=IO.MIN_GAMES)
    ap.add_argument("--patience", type=int, default=IO.PATIENCE, help="games without a new Chun-Li (code, split)")
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
    ap.add_argument("--log-dir", default=os.path.join("logs", "cpu_chunli"))
    ap.add_argument("--one", nargs=2, metavar=("P1", "PORT"), help=argparse.SUPPRESS)
    args = ap.parse_args()
    exit_on_sigterm()
    if args.one:
        return play_one(args, args.one[0], int(args.one[1]))
    p1s = args.p1.split(",")
    bad = sorted(set(p1s) - set(K.P1_CHARS))
    if bad:
        raise SystemExit("unknown player-1 characters %s (CPU Chun-Li meets %s)" % (bad, ",".join(K.P1_CHARS)))
    if len(p1s) > PORTS["movement"][1]:
        raise SystemExit("more workers than ports")
    missing = [c for c in p1s if not os.path.exists(os.path.join("states", "p1_%s_vs_%s.state" % (c, K.CPU)))]
    if missing:
        raise SystemExit("no savestates for %s: run scripts/make_cpu_chunli_states.py first" % missing)
    check_run(args.out, K.run_settings(
        p1s, **{k: getattr(args, k) for k in SETTINGS}, caps=C.CAPS, short=C.SHORT, lag=C.LAG,
        sample_actors=[K.CPU], codes=A.table(), started=time.strftime("%Y-%m-%d %H:%M:%S")))
    common = ["--out", args.out, "--cap", str(args.cap), "--min-games", str(args.min_games), "--patience",
              str(args.patience), "--model", args.model, "--explore", str(args.explore), "--threshold",
              str(args.threshold), "--ring", str(args.ring), "--seed", str(args.seed), "--mem-cap-gb",
              str(args.mem_cap_gb)] + (["--rom", args.rom] if args.rom else []) + (
              ["--device", args.device] if args.device else [])
    cmds = [((c,), [sys.executable, os.path.abspath(__file__), "--one", c, str(args.base_port + i)] + common)
            for i, c in enumerate(p1s)]
    print("%d workers (%s/<p1>.log), %.1f GB each, %.0f GB budget" % (len(cmds), args.log_dir, args.job_gb,
                                                                    args.budget_gb), flush=True)
    failed = [c for (c,) in fan_out(cmds, args.log_dir, job_gb=args.job_gb, budget=Budget(total_gb=args.budget_gb))]
    for c in p1s:
        p = os.path.join(args.out, c, "stop.json")
        s = json.load(open(p)) if os.path.exists(p) else {"reason": "no stop.json"}
        print("%-8s %-17s games %s, last new at game %s, Chun-Li codes missing in train/test: %d" % (
            c, s["reason"], s.get("games"), s.get("last_new_game"), len(s.get("missing_in", {}))))
    for c in failed:
        print("FAILED:", c, "see %s/%s.log" % (args.log_dir, c))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
