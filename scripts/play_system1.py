"""System 1 (sf2/system1.py) plays every character against the arcade CPU, in parallel (one headless Mesen and one
model per character), and records every decision: frames, note, the model's P(hit) for each attack, the choice, and
what really happened.

    python scripts/play_system1.py --model runs/all8/best                  # all 8 characters, 6 rounds each
    python scripts/play_system1.py --model runs/all8/best --chars ryu --rounds 2

Opponent: states/p1_<char>_vs_ryu.state (Ryu: p1_ryu_vs_ken.state), round 1 against the CPU.
Writes rollouts/system1/<char>/{decisions.jsonl, rounds.jsonl, images/} and rollouts/system1/summary.json; logs in
logs/system1/<char>.log.
"""
import argparse
import collections
import json
import os
import random
import subprocess
import sys
import time

import _path  # noqa: F401
from sf2.headless import launch_argv
from sf2.mesen import MesenBridge
from sf2.short_memory import ShortMemory
from sf2.system1 import System1, play_round
from sf2.vs import IDS, NAMES, VARS

CHARS = ["ryu", "ken", "chunli", "guile", "honda", "blanka", "zangief", "dhalsim"]


def opponent(me: str) -> str:
    return "ken" if me == "ryu" else "ryu"


def play_one(args, me: str, port: int) -> int:
    opp = opponent(me)
    out = os.path.join(args.out, me)
    img_dir = os.path.join(out, "images")
    os.makedirs(img_dir, exist_ok=True)
    state = open("states/p1_%s_vs_%s.state" % (me, opp), "rb").read()
    s1 = System1(None if args.model == "random" else args.model, me, args.threshold, args.device, args.seed)
    if args.memory:
        s1.memory = ShortMemory()            # blank at the start; kept across this character's rounds
    b = MesenBridge(port, launch=launch_argv(port, args.rom))
    rounds, t0 = [], time.time()
    try:
        b.set_capture("raw")
        b.set_vars(VARS)
        r = dict(zip(NAMES, b.load_state(state).rams[-1]))
        if (r["p1_char"], r["p2_char"]) != (IDS[me], IDS[opp]):
            raise SystemExit("%s: savestate holds characters %s, expected %s" % (
                me, (r["p1_char"], r["p2_char"]), (IDS[me], IDS[opp])))
        rng = random.Random(args.seed)
        with open(os.path.join(out, "decisions.jsonl"), "w") as dec:
            for i in range(args.rounds):
                rnd = play_round(b, s1, opp, state, rng, img_dir, "r%02d" % i)
                for e in rnd.log:
                    dec.write(json.dumps(dict(e, round=i)) + "\n")
                rounds.append({"round": i, "result": rnd.result, "frames": rnd.frames, "decisions": rnd.decisions,
                               "dealt": rnd.dealt, "taken": rnd.taken})
                print("%-8s round %d: %-10s dealt %3d taken %3d  %d decisions" % (
                    me, i, rnd.result, rnd.dealt, rnd.taken, rnd.decisions), flush=True)
    finally:
        b.close()
    if s1.memory:
        s1.memory.save(os.path.join(out, "memory.json"))
    with open(os.path.join(out, "rounds.jsonl"), "w") as f:
        f.write("".join(json.dumps(x) + "\n" for x in rounds))
    print("%s done in %.0f s" % (me, time.time() - t0))
    return 0


def summarize(out_dir: str, chars) -> dict:
    summary = {}
    for c in chars:
        rpath, dpath = os.path.join(out_dir, c, "rounds.jsonl"), os.path.join(out_dir, c, "decisions.jsonl")
        if not os.path.exists(rpath):
            continue
        rounds = [json.loads(x) for x in open(rpath)]
        dec = [json.loads(x) for x in open(dpath)]
        att = [d for d in dec if d["action"] != "forward"]
        pred_hit = [d for d in att if d["predicted"] == "hit"]
        res = collections.Counter(r["result"] for r in rounds)
        summary[c] = {
            "rounds": len(rounds), "win": res["win"], "loss": res["loss"], "draw": res["draw"],
            "dealt_per_round": sum(r["dealt"] for r in rounds) / max(len(rounds), 1),
            "taken_per_round": sum(r["taken"] for r in rounds) / max(len(rounds), 1),
            "decisions": len(dec), "attack_share": len(att) / max(len(dec), 1),
            "predicted_hit_really_hit": sum(d["actual"] == "hit" for d in pred_hit) / max(len(pred_hit), 1),
            "attack_outcomes": dict(collections.Counter(d["actual"] for d in att)),
            "top_actions": collections.Counter(d["action"] for d in dec).most_common(5),
        }
    return summary


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="checkpoint dir, or 'random' for the random-attack baseline")
    ap.add_argument("--chars", default=",".join(CHARS))
    ap.add_argument("--rounds", type=int, default=6)
    ap.add_argument("--threshold", type=float, default=0.5, help="attack only if P(hit) is at least this")
    ap.add_argument("--out", default="rollouts/system1")
    ap.add_argument("--base-port", type=int, default=48901)
    ap.add_argument("--rom", default=os.environ.get("SF2_ROM"))
    ap.add_argument("--device", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--memory", action="store_true", help="play with a short memory of real outcomes (starts blank)")
    ap.add_argument("--one", nargs=2, metavar=("CHAR", "PORT"), help=argparse.SUPPRESS)
    args = ap.parse_args()
    if args.one:
        return play_one(args, args.one[0], int(args.one[1]))
    chars = args.chars.split(",")
    os.makedirs("logs/system1", exist_ok=True)
    jobs = []
    for i, c in enumerate(chars):
        cmd = [sys.executable, os.path.abspath(__file__), "--one", c, str(args.base_port + i), "--model", args.model,
               "--rounds", str(args.rounds), "--threshold", str(args.threshold), "--out", args.out,
               "--seed", str(args.seed)] + (["--rom", args.rom] if args.rom else []) + (
                   ["--memory"] if args.memory else [])
        log = open(os.path.join("logs", "system1", c + ".log"), "w")
        jobs.append((c, subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)))
    print("%d characters playing (logs/system1/<char>.log)" % len(jobs), flush=True)
    failed = [c for c, p in jobs if p.wait() != 0]
    summary = summarize(args.out, chars)
    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1)
    print("%-8s %6s %6s %6s %6s %7s %8s  %s" % ("char", "W-L-D", "dealt", "taken", "attack", "P(hit)ok", "decis.",
                                                "real outcome of attacks"))
    for c, s in summary.items():
        print("%-8s %6s %6.0f %6.0f %6.0f%% %7.0f%% %8d  %s" % (
            c, "%d-%d-%d" % (s["win"], s["loss"], s["draw"]), s["dealt_per_round"], s["taken_per_round"],
            100 * s["attack_share"], 100 * s["predicted_hit_really_hit"], s["decisions"], s["attack_outcomes"]))
    for c in failed:
        print("FAILED:", c, "see logs/system1/%s.log" % c)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
