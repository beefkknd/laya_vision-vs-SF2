"""Does the short memory help? A headless A/B test with the opponent locked: for each opponent, two headless Mesen runs
play the same rounds from the same savestate (states/p1_<me>_vs_<opp>.state) with the same random start delays,
one with text laya reading System 2's short memory (frozen: copied at the start, nothing revises it), one with
"Advice: none". Everything else is identical, so a paired difference is the memory's effect.

    python scripts/ab_memory.py                               # chunli, every opponent that has a short memory
    python scripts/ab_memory.py --opps ryu,ken --rounds 40

Output: rollouts/ab/<stamp>/<opp>_<arm>/{actions,rounds}.jsonl, memory.json (what was tested), summary.json;
process logs in logs/ab/<opp>_<arm>.log.
"""
import argparse
import json
import os
import random
import shutil
import subprocess
import sys
import time
from typing import Dict, List

import _path  # noqa: F401
from sf2.advisor import Advisor
from sf2.headless import launch_argv
from sf2.memory import load, short_path
from sf2.mesen import MesenBridge
from sf2.system1 import System1, play_round
from sf2.vs import IDS, NAMES, VARS
from sf2.vs_sweep import actions

ARMS = ("on", "off")


def play_arm(args, opp: str, arm: str, port: int, out: str) -> int:
    me = args.char
    os.makedirs(out, exist_ok=True)
    state = open("states/p1_%s_vs_%s.state" % (me, opp), "rb").read()
    advisor, b = None, None
    try:                                  # everything started here is closed in `finally`
        advisor = Advisor(args.advisor)
        s1 = System1(args.model, me, advisor=advisor)
        s1.short = json.load(open(os.path.join(os.path.dirname(out), "memory_%s.json" % opp)))
        s1.advice_on = arm == "on"
        b = MesenBridge(port, launch=launch_argv(port, None))
        b.set_capture("raw")
        b.set_vars(VARS)
        r = dict(zip(NAMES, b.load_state(state).rams[-1]))
        if (r["p1_char"], r["p2_char"]) != (IDS[me], IDS[opp]):
            raise SystemExit("savestate holds %s, expected %s" % ((r["p1_char"], r["p2_char"]), (IDS[me], IDS[opp])))
        rng = random.Random(args.seed)            # the same seed in both arms: the same start delays, round by round
        with open(os.path.join(out, "actions.jsonl"), "w") as act, open(os.path.join(out, "rounds.jsonl"), "w") as rf:
            for i in range(args.rounds):
                rnd = play_round(b, s1, opp, state, rng, None, i)
                where = {"round": i, "opp": opp, "advice": arm}
                act.write("".join(json.dumps(dict(e, **where)) + "\n" for e in rnd.log))
                rf.write(json.dumps(dict(rnd.summary, **where)) + "\n")
                act.flush()
                rf.flush()
                print("%s %s round %d: %s dealt %d taken %d" % (opp, arm, i, rnd.result, rnd.summary["dealt"],
                                                                 rnd.summary["taken"]), flush=True)
    finally:
        if b:
            b.close()
        if advisor:
            advisor.close()
    return 0


def summarize(root: str, opps: List[str]) -> Dict:
    out = {}
    for opp in opps:
        arms = {}
        for arm in ARMS:
            path = os.path.join(root, "%s_%s" % (opp, arm), "rounds.jsonl")
            arms[arm] = [json.loads(x) for x in open(path)] if os.path.exists(path) else []
        n = min(len(arms["on"]), len(arms["off"]))
        margin = [(a["dealt"] - a["taken"]) - (b["dealt"] - b["taken"]) for a, b in zip(arms["on"][:n], arms["off"][:n])]
        out[opp] = {"rounds": n, "paired_margin_on_minus_off": sum(margin) / max(1, n),
                    "on_better": sum(m > 0 for m in margin), "off_better": sum(m < 0 for m in margin),
                    **{arm: {"won": sum(r["result"] == "win" for r in rs[:n]),
                             "dealt": sum(r["dealt"] for r in rs[:n]) / max(1, n),
                             "taken": sum(r["taken"] for r in rs[:n]) / max(1, n)} for arm, rs in arms.items()}}
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--char", default="chunli")
    ap.add_argument("--opps", default=None, help="comma list; default: every opponent with a short memory")
    ap.add_argument("--rounds", type=int, default=30, help="rounds per arm per opponent")
    ap.add_argument("--model", default="runs/all8/best")
    ap.add_argument("--advisor", default="runs/text_laya/advice_v1")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--base-port", type=int, default=49101)
    ap.add_argument("--one", nargs=4, metavar=("OPP", "ARM", "PORT", "OUT"), help=argparse.SUPPRESS)
    args = ap.parse_args()
    if args.one:
        opp, arm, port, out = args.one
        return play_arm(args, opp, arm, int(port), out)
    me = args.char
    opps = args.opps.split(",") if args.opps else sorted(
        o for o in ("blanka", "chunli", "dhalsim", "guile", "honda", "ken", "ryu", "zangief")
        if o != me and os.path.exists(short_path(me, o)) and os.path.exists("states/p1_%s_vs_%s.state" % (me, o)))
    root = os.path.join("rollouts", "ab", time.strftime("%Y%m%d-%H%M%S"))
    os.makedirs(root, exist_ok=True)
    os.makedirs("logs/ab", exist_ok=True)
    for opp in opps:                    # freeze what is tested: the memory as it is now
        mem = load(short_path(me, opp), list(actions(me)))
        if not mem or not mem.get("lessons"):
            raise SystemExit("no short memory for %s vs %s" % (me, opp))
        shutil.copy(short_path(me, opp), os.path.join(root, "memory_%s.json" % opp))
    jobs = []
    for i, (opp, arm) in enumerate((o, a) for o in opps for a in ARMS):
        cmd = [sys.executable, os.path.abspath(__file__), "--one", opp, arm, str(args.base_port + i),
               os.path.join(root, "%s_%s" % (opp, arm)), "--char", me, "--rounds", str(args.rounds),
               "--model", args.model, "--advisor", args.advisor, "--seed", str(args.seed)]
        log = open(os.path.join("logs", "ab", "%s_%s.log" % (opp, arm)), "w")
        jobs.append(((opp, arm), subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)))
    print("%d headless runs (%s x on/off), %d rounds each; logs/ab/" % (len(jobs), ",".join(opps), args.rounds),
          flush=True)
    failed = [k for k, p in jobs if p.wait() != 0]
    summary = summarize(root, opps)
    with open(os.path.join(root, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1)
    print("%-8s %6s  %-24s %-24s %s" % ("opp", "rounds", "advice ON won/dealt/taken", "advice OFF won/dealt/taken",
                                         "paired margin on-off (on better / off better)"))
    for opp, s in summary.items():
        print("%-8s %6d  %-24s %-24s %+6.1f  (%d / %d)" % (
            opp, s["rounds"], "%d  %.0f  %.0f" % (s["on"]["won"], s["on"]["dealt"], s["on"]["taken"]),
            "%d  %.0f  %.0f" % (s["off"]["won"], s["off"]["dealt"], s["off"]["taken"]),
            s["paired_margin_on_minus_off"], s["on_better"], s["off_better"]))
    for k in failed:
        print("FAILED:", k, "see logs/ab/%s_%s.log" % k)
    print("saved", root)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
