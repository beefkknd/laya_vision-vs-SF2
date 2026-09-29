"""System 1 (sf2/system1/system1.py) plays every character against the arcade CPU, in parallel (one headless Mesen and one
model per character), and records every decision: frames, note, the model's P(hit) for each attack, the choice, and
what really happened.

    python scripts/play_system1.py --model runs/all8/best                  # all 8 characters, 10 games each
    python scripts/play_system1.py --model runs/all8/best --chars ryu --games 2

Opponent: states/p1_<char>_vs_ryu.state (Ryu: p1_ryu_vs_ken.state), round 1 against the CPU.
Writes the game log for System 2 (sf2/system1/game_log.py): rollouts/<run>/<char>/{actions.jsonl, games.jsonl, images/}
and rollouts/<run>/summary.json; process logs in logs/system1/<char>.log.
"""
import argparse
import collections
import json
import os
import random
import sys
import time

import _path  # noqa: F401
from sf2.config import PORTS
from sf2.eval.runner import exit_on_sigterm, fan_out, open_fight, open_logs
from sf2.data.dataset import read
from sf2.system2.memory import load, short_path
from sf2.system1.system1 import System1, play_round
from sf2.vocab import FIGHTERS
from sf2.data.vs_sweep import actions



def opponent(me: str, choice: str = "ryu") -> str:
    """The CPU opponent: ``choice`` (a savestate states/p1_<me>_vs_<choice>.state); arcade mode has no mirror match,
    so a character never meets itself: Ryu meets Ken instead of Ryu, Dhalsim Ken instead of Dhalsim."""
    return choice if choice != me else "ken"


def play_one(args, me: str, port: int) -> int:
    opp = opponent(me, args.opp)
    out = os.path.join(args.out, me)
    img_dir = os.path.join(out, "images")
    os.makedirs(img_dir, exist_ok=True)
    s1 = System1(None if args.model == "random" else args.model, me, args.threshold, args.device, args.seed)
    if args.memory != "none":     # the short memory vs THIS opponent: a new opponent has its own (or no) file
        s1.short = load(short_path(me, opp, args.memory), list(actions(me)))
    print("%s vs %s: short memory %s" % (me, opp, "%d lessons" % len(s1.short["lessons"]) if s1.short else "empty"),
          flush=True)
    t0 = time.time()
    with open_fight(me, opp, port, args.rom) as (b, state), open_logs(out, ("actions", "games")) as logs:
        rng = random.Random(args.seed)
        for i in range(args.games):
            rnd = play_round(b, s1, opp, state, rng, img_dir, i)
            logs["actions"].write("".join(json.dumps(e) + "\n" for e in rnd.log))
            logs["games"].write(json.dumps(rnd.summary) + "\n")
            for f in logs.values():
                f.flush()
            print("%-8s game %d: %-10s dealt %3d taken %3d  %d actions" % (
                me, i, rnd.result, rnd.summary["dealt"], rnd.summary["taken"], len(rnd.log)), flush=True)
    print("%s done in %.0f s" % (me, time.time() - t0))
    return 0


def summarize(out_dir: str, chars) -> dict:
    summary = {}
    for c in chars:
        gpath, apath = os.path.join(out_dir, c, "games.jsonl"), os.path.join(out_dir, c, "actions.jsonl")
        if not os.path.exists(gpath):
            continue
        games, acts = read(gpath), read(apath)
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
    ap.add_argument("--chars", default=",".join(FIGHTERS))
    ap.add_argument("--games", type=int, default=10, help="games per character (a game = round 1 vs the CPU)")
    ap.add_argument("--threshold", type=float, default=0.5, help="attack only if P(hit) is at least this")
    ap.add_argument("--out", default="rollouts/system1")
    ap.add_argument("--base-port", type=int, default=PORTS["system1"][0])
    ap.add_argument("--rom", default=os.environ.get("SF2_ROM"))
    ap.add_argument("--device", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--opp", default="ryu", help="the CPU opponent (dhalsim: the novice training opponent)")
    ap.add_argument("--memory", default="memory",
                    help="memory dir (sf2/system2/memory.py); 'none' plays with an empty short memory")
    ap.add_argument("--one", nargs=2, metavar=("CHAR", "PORT"), help=argparse.SUPPRESS)
    args = ap.parse_args()
    exit_on_sigterm()
    if args.one:
        return play_one(args, args.one[0], int(args.one[1]))
    chars = args.chars.split(",")
    cmds = [((c,), [sys.executable, os.path.abspath(__file__), "--one", c, str(args.base_port + i), "--model",
                    args.model, "--games", str(args.games), "--threshold", str(args.threshold), "--out", args.out,
                    "--seed", str(args.seed), "--memory", args.memory, "--opp", args.opp]
             + (["--rom", args.rom] if args.rom else [])) for i, c in enumerate(chars)]
    print("%d characters playing (logs/system1/<char>.log)" % len(cmds), flush=True)
    failed = [c for (c,) in fan_out(cmds, os.path.join("logs", "system1"))]
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
