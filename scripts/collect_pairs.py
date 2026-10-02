"""Collect the movement pairs (docs/prereg_movement_pairs.md, "Fallback"): for each ordered pair (A, B) of the 8
characters, player 1 = A directed through its full move list (sf2.data.pairs_moves; no model, no Qwen), player 2 =
B, the arcade CPU; ``--games`` games (rounds) per pair, then it stops. Every frame is captured; both fighters' episodes
are sampled at start / middle / end (sf2.data.pairs_collect). One worker (one headless Mesen) per pair, as many at
once as the memory budget allows.

    python scripts/collect_pairs.py --out rollouts/pairs --games 3                       # all 56 pairs
    python scripts/collect_pairs.py --out /tmp/x --games 1 --pairs ryu:ken,ken:ryu        # a smoke
    python scripts/collect_pairs.py --mode vs --out rollouts/pairs2p --games 1           # Plan B round 1

--mode vs (Plan B): 2P versus, BOTH controllers ours, each driven by its own character's list and seeded cycle
(sf2.data.pairs_collect.play_both); rows record controller p1 / p2. Needs states/vs_<A>_vs_<B>.state
(scripts/make_vs_pair_states.py).

Writes <out>/<A>_vs_<B>/{images/, ram/, pairs.jsonl, games.jsonl, stop.json} (sf2.data.pairs_collect_io) and
<out>/run.json; process logs <log-dir>/<A>_vs_<B>.log. Re-running the same command resumes from the committed games.
Needs states/p1_<A>_vs_<B>.state (scripts/make_pair_states.py).
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
from sf2.data import pairs_collect as PC
from sf2.data import pairs_collect_io as IO
from sf2.data import pairs_labels as L
from sf2.data import pairs_moves as PM
from sf2.data.action_codes import EXTRA_NAMES, EXTRA_VARS
from sf2.emu.vs import NAMES, VARS
from sf2.eval.budget import Budget
from sf2.eval.runner import exit_on_sigterm, fan_out, open_fight, savestate
from make_vs_pair_states import vs_state

CHARS = ("blanka", "chunli", "dhalsim", "guile", "honda", "ken", "ryu", "zangief")
GAMES = 3
JOB_GB = 1.5                 # one worker: Python + the ring + Mesen
BUDGET_GB = 60.0
WORKERS = 24                 # of the 32 cores: Mesen runs one core flat out per worker
ALL_NAMES = NAMES + EXTRA_NAMES


def all_pairs():
    return [(a, b) for a in CHARS for b in CHARS if a != b]


def parse_pairs(text: str):
    if text == "all":
        return all_pairs()
    out = []
    for item in text.split(","):
        a, _, b = item.partition(":")
        if a not in CHARS or b not in CHARS or a == b:
            raise SystemExit("bad pair %r (A:B, two different of %s)" % (item, ",".join(CHARS)))
        out.append((a, b))
    if len(set(out)) != len(out):
        raise SystemExit("a pair is listed twice")
    return out


def rss_of(pid: int) -> float:
    out = subprocess.run(["ps", "-o", "rss=", "-p", str(pid)], capture_output=True, text=True).stdout.strip()
    return int(out) * 1024 / 1e9 if out.isdigit() else 0.0


def state_path(mode: str, a: str, b: str) -> str:
    return vs_state(a, b) if mode == "vs" else savestate(a, b)


def play_one(args, a: str, b: str, port: int) -> int:
    base = os.path.join(args.out, IO.pair_name(a, b))
    bands = L.poke_bands()
    chars = {1: a, 2: b}
    words = {p: list(PM.moves(c)) for p, c in chars.items()}
    t0 = time.time()
    with open_fight(a, b, port, args.rom, state=state_path(args.mode, a, b)) as (br, state):
        br.set_vars(VARS + EXTRA_VARS)
        proxy = C.CapturingBridge(br, ALL_NAMES)

        def play(game: int, sampler) -> dict:
            proxy.sink = sampler.feed
            start = random.Random("start:%d:%s:%s:%d" % (args.seed, a, b, game))
            if args.mode == "vs":
                cycles = {p: PM.Cycle(words[p], random.Random("cycle%d:%d:%s:%s:%d" % (p, args.seed, a, b, game)))
                          for p in chars}
                out = PC.play_both(proxy, ALL_NAMES, chars, cycles, state, start, lambda: len(sampler.rows),
                                   on_word=sampler.press)
            else:
                cycle = PM.Cycle(words[1], random.Random("cycle:%d:%s:%s:%d" % (args.seed, a, b, game)))
                out = PC.play_directed(proxy, ALL_NAMES, a, cycle, state, start, lambda: len(sampler.rows))
            proxy.sink = None
            return out

        def rss() -> float:
            return IO.peak_rss_gb() + (rss_of(br.proc.pid) if getattr(br, "proc", None) else 0.0)

        try:
            stop = IO.collect_pair(base, play, a, b, args.games, args.seed, bands, ring=args.ring,
                                   per_game=args.per_game, mem_cap_gb=args.mem_cap_gb, rss_gb=rss,
                                   log=lambda m: print(m, flush=True),
                                   controllers=PC.VS_SLOTS if args.mode == "vs" else PC.SLOTS)
        except IO.MemoryCapExceeded as e:
            print("STOP:", e, flush=True)
            return 3
    print("%s vs %s: %s, %d games, %.0f s" % (a, b, stop["reason"], stop["games"], time.time() - t0), flush=True)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=os.path.join("rollouts", "pairs"))
    ap.add_argument("--mode", choices=("directed", "vs"), default="directed",
                    help="directed: player 1 ours vs the CPU; vs: Plan B, 2P versus, both ours")
    ap.add_argument("--pairs", default="all", help="all, or A:B,A:B,... (A = player 1, directed)")
    ap.add_argument("--games", type=int, default=GAMES, help="games per ordered pair (the fixed budget)")
    ap.add_argument("--per-game", type=int, default=PC.PER_GAME)
    ap.add_argument("--ring", type=int, default=C.RING)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--rom", default=os.environ.get("SF2_ROM"))
    ap.add_argument("--base-port", type=int, default=PORTS["pairs"][0])
    ap.add_argument("--workers", type=int, default=WORKERS, help="at most this many at once")
    ap.add_argument("--job-gb", type=float, default=JOB_GB)
    ap.add_argument("--budget-gb", type=float, default=BUDGET_GB)
    ap.add_argument("--mem-cap-gb", type=float, default=IO.MEM_CAP_GB)
    ap.add_argument("--log-dir", default=os.path.join("logs", "pairs"))
    ap.add_argument("--one", nargs=3, metavar=("A", "B", "PORT"), help=argparse.SUPPRESS)
    args = ap.parse_args()
    exit_on_sigterm()
    if args.one:
        return play_one(args, args.one[0], args.one[1], int(args.one[2]))
    pairs = parse_pairs(args.pairs)
    missing = [p for p in pairs if not os.path.exists(state_path(args.mode, *p))]
    if missing:
        raise SystemExit("no savestate for %s: run scripts/make_%spair_states.py" % (
            missing, "vs_" if args.mode == "vs" else ""))
    if args.workers < 1 or len(pairs) > PORTS["pairs"][1]:
        raise SystemExit("--workers must be >= 1 and at most %d pairs (one port each)" % PORTS["pairs"][1])
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "run.json"), "w") as f:
        json.dump({"mode": args.mode, "pairs": ["%s:%s" % p for p in pairs], "games": args.games, "per_game": args.per_game,
                   "ring": args.ring, "seed": args.seed, "short": C.SHORT, "lag": C.LAG,
                   "moves": {c: list(PM.moves(c)) for c in sorted({c for p in pairs for c in p})},
                   "started": time.strftime("%Y-%m-%d %H:%M:%S")}, f, indent=1)
    common = ["--mode", args.mode, "--out", args.out, "--games", str(args.games), "--per-game", str(args.per_game), "--ring",
              str(args.ring), "--seed", str(args.seed), "--mem-cap-gb", str(args.mem_cap_gb)] + (
              ["--rom", args.rom] if args.rom else [])
    # one port per pair (never shared); at most ``workers`` at once (the budget's job_gb * workers)
    budget = Budget(total_gb=min(args.budget_gb, args.job_gb * args.workers))
    cmds = [((IO.pair_name(a, b),), [sys.executable, os.path.abspath(__file__), "--one", a, b,
                                     str(args.base_port + i)] + common)
            for i, (a, b) in enumerate(pairs)]
    print("%d pairs x %d games, up to %d workers (%s/<pair>.log)" % (len(pairs), args.games, args.workers,
                                                                   args.log_dir), flush=True)
    failed = [k for (k,) in fan_out(cmds, args.log_dir, job_gb=args.job_gb, budget=budget)]
    for a, b in pairs:
        p = os.path.join(args.out, IO.pair_name(a, b), "stop.json")
        s = json.load(open(p)) if os.path.exists(p) else {"reason": "no stop.json"}
        print("%-18s %-12s games %s pairs %s" % (IO.pair_name(a, b), s["reason"], s.get("games"), s.get("pairs")))
    for k in failed:
        print("FAILED:", k, "see %s/%s.log" % (args.log_dir, k))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
