"""System 1 (sf2/system1.py) plays every character against the arcade CPU, in parallel (one headless Mesen and one
model per character), and records every decision: frames, note, the model's P(hit) for each attack, the choice, and
what really happened.

    python scripts/play_system1.py --model runs/all8/best                  # all 8 characters, 10 games each
    python scripts/play_system1.py --model runs/all8/best --chars ryu --games 2

Opponent: states/p1_<char>_vs_ryu.state (Ryu: p1_ryu_vs_ken.state), round 1 against the CPU.
Writes the game log for System 2 (sf2/game_log.py): rollouts/<run>/<char>/{actions.jsonl, games.jsonl, images/}
and rollouts/<run>/summary.json; process logs in logs/system1/<char>.log.
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
    b = MesenBridge(port, launch=launch_argv(port, args.rom))
    games, t0 = [], time.time()
    try:
        b.set_capture("raw")
        b.set_vars(VARS)
        r = dict(zip(NAMES, b.load_state(state).rams[-1]))
        if (r["p1_char"], r["p2_char"]) != (IDS[me], IDS[opp]):
            raise SystemExit("%s: savestate holds characters %s, expected %s" % (
                me, (r["p1_char"], r["p2_char"]), (IDS[me], IDS[opp])))
        rng = random.Random(args.seed)
        with open(os.path.join(out, "actions.jsonl"), "w") as act, open(os.path.join(out, "games.jsonl"), "w") as gm:
            for i in range(args.games):
                rnd = play_round(b, s1, opp, state, rng, img_dir, i)
                act.write("".join(json.dumps(e) + "\n" for e in rnd.log))
                gm.write(json.dumps(rnd.summary) + "\n")
                act.flush()
                gm.flush()
                games.append(rnd.summary)
                print("%-8s game %d: %-10s dealt %3d taken %3d  %d actions" % (
                    me, i, rnd.result, rnd.summary["dealt"], rnd.summary["taken"], len(rnd.log)), flush=True)
    finally:
        b.close()
    print("%s done in %.0f s" % (me, time.time() - t0))
    return 0


def summarize(out_dir: str, chars) -> dict:
    summary = {}
    for c in chars:
        gpath, apath = os.path.join(out_dir, c, "games.jsonl"), os.path.join(out_dir, c, "actions.jsonl")
        if not os.path.exists(gpath):
            continue
        games = [json.loads(x) for x in open(gpath)]
        acts = [json.loads(x) for x in open(apath)]
        att = [a for a in acts if a["kind"] == "attack"]
        pred_hit = [a for a in att if a["predicted"] == "hit"]
        res = collections.Counter(g["result"] for g in games)
        summary[c] = {
            "games": len(games), "win": res["win"], "loss": res["loss"], "draw": res["draw"],
            "dealt_per_game": sum(g["dealt"] for g in games) / max(len(games), 1),
            "taken_per_game": sum(g["taken"] for g in games) / max(len(games), 1),
            "actions": len(acts), "attack_share": len(att) / max(len(acts), 1),
            "predicted_hit_really_hit": sum(a["actual"] == "hit" for a in pred_hit) / max(len(pred_hit), 1),
            "attack_outcomes": dict(collections.Counter(a["actual"] for a in att)),
            "times_hit": sum(a["i_was_hit"] for a in acts),
        }
    return summary


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="checkpoint dir, or 'random' for the random-attack baseline")
    ap.add_argument("--chars", default=",".join(CHARS))
    ap.add_argument("--games", type=int, default=10, help="games per character (a game = round 1 vs the CPU)")
    ap.add_argument("--threshold", type=float, default=0.5, help="attack only if P(hit) is at least this")
    ap.add_argument("--out", default="rollouts/system1")
    ap.add_argument("--base-port", type=int, default=48901)
    ap.add_argument("--rom", default=os.environ.get("SF2_ROM"))
    ap.add_argument("--device", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--one", nargs=2, metavar=("CHAR", "PORT"), help=argparse.SUPPRESS)
    args = ap.parse_args()
    if args.one:
        return play_one(args, args.one[0], int(args.one[1]))
    chars = args.chars.split(",")
    os.makedirs("logs/system1", exist_ok=True)
    jobs = []
    for i, c in enumerate(chars):
        cmd = [sys.executable, os.path.abspath(__file__), "--one", c, str(args.base_port + i), "--model", args.model,
               "--games", str(args.games), "--threshold", str(args.threshold), "--out", args.out,
               "--seed", str(args.seed)] + (["--rom", args.rom] if args.rom else [])
        log = open(os.path.join("logs", "system1", c + ".log"), "w")
        jobs.append((c, subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)))
    print("%d characters playing (logs/system1/<char>.log)" % len(jobs), flush=True)
    failed = [c for c, p in jobs if p.wait() != 0]
    summary = summarize(args.out, chars)
    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1)
    print("%-8s %6s %6s %6s %6s %7s %8s  %s" % ("char", "W-L-D", "dealt", "taken", "attack", "P(hit)ok", "actions",
                                                "real outcome of attacks"))
    for c, s in summary.items():
        print("%-8s %6s %6.0f %6.0f %6.0f%% %7.0f%% %8d  %s" % (
            c, "%d-%d-%d" % (s["win"], s["loss"], s["draw"]), s["dealt_per_game"], s["taken_per_game"],
            100 * s["attack_share"], 100 * s["predicted_hit_really_hit"], s["actions"], s["attack_outcomes"]))
    for c in failed:
        print("FAILED:", c, "see logs/system1/%s.log" % c)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
