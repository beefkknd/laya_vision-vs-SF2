"""Does the notebook make her learn? Headless, opponent locked (states/p1_<me>_vs_<opp>.state), two arms playing the
same rounds (same savestate, same start delays), side by side:

    learn   after every round, win or lose, Qwen reads the round's facts (sf2.system2.round_facts), updates the notebook a
            little (sf2.system2.notebook) and writes the next round's plan; text laya follows the plan
    none    text laya with "Advice: none" every round (the control)

    python scripts/notebook_run.py --opp ryu --rounds 40
Score: hit points per round (dealt - taken); the question is whether the learn arm's curve rises over the rounds.
Nothing here touches memory/ (the loop's own memory); the notebook starts empty and lives in the run folder.
Output: rollouts/notebook/<stamp>_<opp>/<arm>/{rounds,actions}.jsonl, notebook.jsonl (every version) + summary.json;
process logs in logs/notebook/<opp>_<arm>.log.
"""
import argparse
import json
import os
import random
import sys
import time
from typing import Dict, List

import _path  # noqa: F401
from sf2.config import LAYA_VISION, MODEL_JOB_GB, PORTS, TEXT_LAYA
from sf2.system2 import notebook as nbk
from sf2.system1.advisor import Advisor
from sf2.data.dataset import read
from sf2.eval.runner import exit_on_sigterm, fan_out, open_fight, open_logs
from sf2.eval.stats import slope
from sf2.system2.notebook_prompts import messages, reflect_prompt
from sf2.system2.qwen import chat, json_reply
from sf2.system2.round_facts import facts, plan_check, text, unexpected
from sf2.system1.system1 import System1, play_round
from sf2.data.vs_sweep import actions

ARMS = ("learn", "none")


def reflect(me: str, opp: str, nb: Dict, plan: List[str], rnd, results: List[str], moves: List[str]):
    """One Qwen call after a round: (notebook, plan, log entry). A part that fails its checks is not taken."""
    f = facts(rnd.log, rnd.summary)
    allowed = nbk.tries_allowed(results)
    prompt = reflect_prompt(me, opp, nb, text(f, unexpected(f, nbk.lines_of(nb, opp), moves),
                                               plan_check(f, plan, moves), opp), allowed)
    entry = {"facts": f, "allowed_tries": allowed}
    try:
        reply = json_reply(chat(messages(me, prompt), "notebook_%s" % opp))
    except Exception as e:                               # Qwen down or unreadable: keep what we have
        return nb, plan, dict(entry, error="%s: %s" % (type(e).__name__, e))
    new_nb, new_plan, what = nbk.apply_reply(nb, plan, reply, opp, moves, allowed)
    return new_nb, new_plan, dict(entry, **what)


def play_arm(args, arm: str, port: int, out: str) -> int:
    me, opp = args.char, args.opp
    moves = list(actions(me))
    nb, plan, results = nbk.empty(me), [], []
    with Advisor(args.advisor) as advisor, open_fight(me, opp, port) as (b, state), \
            open_logs(out, ("rounds", "actions", "notebook")) as files:
        s1 = System1(args.model, me, advisor=advisor)
        s1.advice_on = arm == "learn"
        rng = random.Random(args.seed)                  # same seed in both arms: same start delays round by round
        for i in range(args.rounds):
            s1.short = nbk.shortlist_memory(plan, me, opp)
            rnd = play_round(b, s1, opp, state, rng, None, i)
            results.append(rnd.result)
            log = {"round": i, "arm": arm, "result": rnd.result, "dealt": rnd.summary["dealt"],
                   "taken": rnd.summary["taken"], "hp": rnd.summary["dealt"] - rnd.summary["taken"], "plan": plan}
            if arm == "learn":
                t = time.time()
                nb, plan, entry = reflect(me, opp, nb, plan, rnd, results, moves)
                log.update(qwen_s=round(time.time() - t, 1), **{k: v for k, v in entry.items() if k != "facts"})
                files["notebook"].write(json.dumps(dict(nb, round=i, next_plan=plan)) + "\n")
            files["rounds"].write(json.dumps(log) + "\n")
            files["actions"].write("".join(json.dumps(dict(e, round=i, arm=arm)) + "\n" for e in rnd.log))
            for f in files.values():
                f.flush()
            print("%s round %d: %s hp %+d  next plan %s" % (arm, i, rnd.result, log["hp"], plan), flush=True)
    return 0


def summarize(root: str) -> Dict:
    out = {}
    for arm in ARMS:
        rs = read(os.path.join(root, arm, "rounds.jsonl"), missing_ok=True)
        hp = [r["hp"] for r in rs]
        k = max(1, len(hp) // 3)
        out[arm] = {"rounds": len(rs), "won": sum(r["result"] == "win" for r in rs),
                    "hp_mean": sum(hp) / max(1, len(hp)), "hp_first_third": sum(hp[:k]) / k,
                    "hp_last_third": sum(hp[-k:]) / k, "hp_slope_per_round": slope(hp)}
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--char", default="chunli")
    ap.add_argument("--opp", default="ryu")
    ap.add_argument("--rounds", type=int, default=40)
    ap.add_argument("--model", default=LAYA_VISION)
    ap.add_argument("--advisor", default=TEXT_LAYA)
    ap.add_argument("--seed", type=int, default=None, help="start delays for both arms (default: a new one per run)")
    ap.add_argument("--base-port", type=int, default=PORTS["notebook"][0])
    ap.add_argument("--one", nargs=3, metavar=("ARM", "PORT", "OUT"), help=argparse.SUPPRESS)
    args = ap.parse_args()
    exit_on_sigterm()
    if args.one:
        return play_arm(args, args.one[0], int(args.one[1]), args.one[2])
    args.seed = int(time.time()) % 100000 if args.seed is None else args.seed
    root = os.path.join("rollouts", "notebook", "%s_%s" % (time.strftime("%Y%m%d-%H%M%S"), args.opp))
    cmds = [((args.opp, arm), [sys.executable, os.path.abspath(__file__), "--one", arm, str(args.base_port + i),
                               os.path.join(root, arm), "--char", args.char, "--opp", args.opp, "--rounds",
                               str(args.rounds), "--model", args.model, "--advisor", args.advisor, "--seed",
                               str(args.seed)])
            for i, arm in enumerate(ARMS)]
    print("%s vs %s: arms %s, %d rounds each, seed %d; logs/notebook/" % (args.char, args.opp, ",".join(ARMS),
                                                                         args.rounds, args.seed),
          flush=True)
    failed = [arm for _, arm in fan_out(cmds, os.path.join("logs", "notebook"), job_gb=MODEL_JOB_GB)]
    s = summarize(root)
    with open(os.path.join(root, "summary.json"), "w") as f:
        json.dump(s, f, indent=1)
    for arm, v in s.items():
        print("%-5s rounds %d won %d  hp mean %+.1f  first third %+.1f -> last third %+.1f  slope %+.2f/round" % (
            arm, v["rounds"], v["won"], v["hp_mean"], v["hp_first_third"], v["hp_last_third"], v["hp_slope_per_round"]))
    for a in failed:
        print("FAILED:", a, "see logs/notebook/%s_%s.log" % (args.opp, a))
    print("saved", root)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
