"""Does advice help? A headless paired test with the opponent locked: every arm plays the same rounds from the same
savestate (states/p1_<me>_vs_<opp>.state) with the same random start delays; only the advice text laya reads differs.
Each arm's memory is frozen (copied into the run folder at the start; nothing revises it).

Arms:
    none           "Advice: none" (the control)
    qwen           System 2's current short memory (memory/short/<me>_vs_<opp>.json)
    code_short     written by code from my rounds against THIS opponent (sf2.code_coach)
    code_playbook  written by code from my rounds against every OTHER opponent (general knowledge of myself)

    python scripts/ab_memory.py --arms none,code_short,code_playbook --opps ryu,honda,ken,zangief,dhalsim,guile,blanka

PRE-REGISTERED TEST (decided before running; the script prints the verdict):
    score per round = hit points (damage dealt - taken); per arm, the paired difference to "none" round by round,
    pooled over every opponent the arm played. An arm HELPS when the 95% confidence interval of that mean difference
    is entirely above 0, HURTS when entirely below, else NOT SHOWN.
        proof 1 (a short memory works):   code_short helps
        proof 2 (the playbook helps):     code_playbook helps (incl. opponents never met: guile, blanka)
        proof 3 (Qwen, static part):      qwen helps, and qwen - code_short is not below 0

Output: rollouts/ab/<stamp>/<opp>_<arm>/{actions,rounds}.jsonl, memory_<opp>_<arm>.json, summary.json;
process logs in logs/ab/<opp>_<arm>.log.
"""
import argparse
import json
import os
import random
import sys
import time
from typing import Dict, List

import _path  # noqa: F401
from sf2 import code_coach
from sf2.advisor import Advisor
from sf2.eval import stats
from sf2.eval.logs import sources
from sf2.eval.runner import exit_on_sigterm, fan_out
from sf2.headless import launch_argv
from sf2.memory import short_path
from sf2.mesen import MesenBridge
from sf2.system1 import System1, play_round
from sf2.vocab import IDS
from sf2.vs import NAMES, VARS

ARMS = ("none", "qwen", "code_short", "code_playbook")
OPPS = ("ryu", "honda", "ken", "zangief", "dhalsim", "guile", "blanka")


def arm_memory(me: str, opp: str, arm: str, rows: List[Dict]):
    """The frozen memory an arm plays with (None for "none"), or False when this arm has nothing for ``opp``."""
    if arm == "none":
        return None
    if arm == "qwen":
        path = short_path(me, opp)
        return json.load(open(path)) if os.path.exists(path) else False
    mem = code_coach.memory(me, opp, arm.split("_", 1)[1], rows)
    return mem if mem["lessons"] else False


def play_arm(args, opp: str, arm: str, port: int, out: str) -> int:
    me = args.char
    os.makedirs(out, exist_ok=True)
    state = open("states/p1_%s_vs_%s.state" % (me, opp), "rb").read()
    mem_path = os.path.join(os.path.dirname(out), "memory_%s_%s.json" % (opp, arm))
    advisor, b = None, None
    try:                                  # everything started here is closed in `finally`
        advisor = Advisor(args.advisor)
        s1 = System1(args.model, me, advisor=advisor)
        s1.short = json.load(open(mem_path)) if os.path.exists(mem_path) else None
        s1.advice_on = arm != "none"
        b = MesenBridge(port, launch=launch_argv(port, None))
        b.set_capture("raw")
        b.set_vars(VARS)
        r = dict(zip(NAMES, b.load_state(state).rams[-1]))
        if (r["p1_char"], r["p2_char"]) != (IDS[me], IDS[opp]):
            raise SystemExit("savestate holds %s, expected %s" % ((r["p1_char"], r["p2_char"]), (IDS[me], IDS[opp])))
        rng = random.Random(args.seed)            # the same seed in every arm: the same start delays, round by round
        with open(os.path.join(out, "actions.jsonl"), "w") as act, open(os.path.join(out, "rounds.jsonl"), "w") as rf:
            for i in range(args.rounds):
                rnd = play_round(b, s1, opp, state, rng, None, i)
                where = {"round": i, "opp": opp, "arm": arm}
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


def _rounds(root: str, opp: str, arm: str) -> List[Dict]:
    path = os.path.join(root, "%s_%s" % (opp, arm), "rounds.jsonl")
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return [json.loads(x) for x in f if x.strip()]


def summarize(root: str, opps: List[str], arms: List[str], failed=()) -> Dict:
    return stats.summarize({o: {a: _rounds(root, o, a) for a in arms} for o in opps}, arms, failed)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--char", default="chunli")
    ap.add_argument("--arms", default=",".join(ARMS))
    ap.add_argument("--opps", default=",".join(OPPS))
    ap.add_argument("--rounds", type=int, default=30, help="rounds per arm per opponent")
    ap.add_argument("--model", default="runs/all8/best")
    ap.add_argument("--advisor", default="runs/text_laya/advice_v1")
    ap.add_argument("--seed", type=int, default=None,
                    help="start delays for every arm (default: a new one per run, so no earlier run replays the test)")
    ap.add_argument("--base-port", type=int, default=49101)
    ap.add_argument("--one", nargs=4, metavar=("OPP", "ARM", "PORT", "OUT"), help=argparse.SUPPRESS)
    args = ap.parse_args()
    exit_on_sigterm()
    if args.one:
        opp, arm, port, out = args.one
        return play_arm(args, opp, arm, int(port), out)
    me, arms, opps = args.char, args.arms.split(","), args.opps.split(",")
    if "none" not in arms:
        raise SystemExit("the control arm 'none' is required")
    seed = int(time.time()) % 100000 if args.seed is None else args.seed
    root = os.path.join("rollouts", "ab", time.strftime("%Y%m%d-%H%M%S"))
    os.makedirs(root, exist_ok=True)
    rows = code_coach.attacks(me)                  # play data only: A/B, notebook and --fresh runs are excluded
    runs = []
    for opp in opps:                               # freeze what is tested, before anything plays
        for arm in arms:
            mem = arm_memory(me, opp, arm, rows)
            if mem is False:
                print("skip %s %s: no memory for this opponent" % (opp, arm))
                continue
            if mem is not None:
                with open(os.path.join(root, "memory_%s_%s.json" % (opp, arm)), "w") as f:
                    json.dump(mem, f, indent=1)
            runs.append((opp, arm))
    with open(os.path.join(root, "run.json"), "w") as f:
        json.dump({"seed": seed, "rounds": args.rounds, "arms": arms, "opps": opps, "model": args.model,
                   "advisor": args.advisor, "coach_logs": sources(rows)}, f, indent=1)
    cmds = [((opp, arm), [sys.executable, os.path.abspath(__file__), "--one", opp, arm, str(args.base_port + i),
                          os.path.join(root, "%s_%s" % (opp, arm)), "--char", me, "--rounds", str(args.rounds),
                          "--model", args.model, "--advisor", args.advisor, "--seed", str(seed)])
            for i, (opp, arm) in enumerate(runs)]
    print("%d headless runs, %d rounds each, seed %d; logs/ab/" % (len(cmds), args.rounds, seed), flush=True)
    failed = fan_out(cmds, os.path.join("logs", "ab"))
    s = summarize(root, opps, arms, failed)
    with open(os.path.join(root, "summary.json"), "w") as f:
        json.dump(s, f, indent=1)
    for opp, row in s["per_opp"].items():
        print("%-8s " % opp + "  ".join("%s: won %d hp %+.0f%s" % (
            arm, v["won"], v["hp"], " (%s)" % v["error"] if "error" in v else
            "" if "vs_none" not in v else " (vs none %+.0f)" % v["vs_none"][0]) for arm, v in row.items()))
    for arm, v in s["pooled"].items():
        if "ci95" in v:
            print("POOLED %-22s %d opponents%s, %4d paired rounds  mean %+6.1f  95%% CI [%+.1f, %+.1f]  %s%s" % (
                arm, v["opponents"], " (not played: %s)" % ",".join(v["missing"]) if v.get("missing") else "",
                v["paired_rounds"], v["mean"], v["ci95"][0], v["ci95"][1], v["verdict"],
                "  (%s)" % v["why"] if "why" in v else ""))
        else:
            print("POOLED %-22s %s  (%s)" % (arm, v["verdict"], v.get("why", "")))
    for k in failed:
        print("FAILED:", k, "see logs/ab/%s_%s.log" % k)
    print("saved", root)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
