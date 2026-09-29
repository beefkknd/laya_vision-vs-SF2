"""Can Qwen tell Chun-Li's good moves from her bad ones, and keep doing it game after game? One opponent at a time.
Code computes the evidence and grades Qwen's picks (sf2.system2.move_coach); Qwen only chooses.

    python scripts/qwen_moves.py identify --opp ken --level table     # Q1: from her play data, 10 orders x 2 repeats
    python scripts/qwen_moves.py loop --opp ken --games 10            # Q2: headless, memory starts empty

Q1 passes when every call passes ``grade`` (a product-LLM step: no retry until green; one miss is a fail).
Q2 passes when every update passes ``grade_update`` and the last picks hold the biggest drain and a good move.
Ledgers: rollouts/qwen_moves/<stamp>_<opp>_<what>/ledger.jsonl (one row per Qwen call, with its grade).
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
from sf2.eval.logs import mark_run
from sf2.eval.runner import exit_on_sigterm, fan_out, open_fight, open_logs
from sf2.system1.advisor import Advisor
from sf2.system1.system1 import System1, play_round
from sf2.system2 import move_coach as mc
from sf2.system2.code_coach import attacks
from sf2.system2.qwen import chat, json_reply

ME = "chunli"
ROOT = os.path.join("rollouts", "qwen_moves")


def ask(opp: str, ev: List[Dict], level: str, order: int, current=None, since=None, task: str = "moves"):
    """One Qwen call: (picks, problems, raw reply)."""
    sig = mc.signal(ev, level, random.Random(order))
    try:
        raw = chat(mc.messages(ME, opp, sig, current, since, ev), "%s_%s" % (task, opp))
        reply = json_reply(raw)
    except Exception as e:                   # Qwen down, cut off or not JSON: an empty answer, graded as such
        return {k: [] for k in mc.KINDS}, ["%s: %s" % (type(e).__name__, e)], None
    picks, problems = mc.parse_reply(reply, ev)
    return picks, problems, raw


def identify(args) -> int:
    rows = [a for a in attacks(ME) if a["opp"] == args.opp]
    ev = mc.evidence(rows)
    out = os.path.join(ROOT, "%s_%s_identify_%s" % (time.strftime("%Y%m%d-%H%M%S"), args.opp, args.level))
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "evidence.json"), "w") as f:
        json.dump(ev, f, indent=1)
    print("%d attacks vs %s; classes: %s" % (len(rows), args.opp, {c: sum(e["cls"] == c for e in ev)
                                                                  for c in ("good", "bad", "unclear", "few")}))
    results = []
    with open(os.path.join(out, "ledger.jsonl"), "w") as led:
        for order in range(args.orders):
            for rep in range(args.repeats):
                picks, problems, raw = ask(args.opp, ev, args.level, order)
                g = mc.grade(picks, ev)
                row = {"order": order, "repeat": rep, "picks": mc.picks_json(picks), "problems": problems,
                       "grade": {k: (list(v) if isinstance(v, tuple) else v) for k, v in g.items()}, "reply": raw}
                led.write(json.dumps(row) + "\n")
                led.flush()
                results.append((order, g["ok"] and not problems))
                print("order %d repeat %d: %s  %s%s" % (order, rep, "PASS" if results[-1][1] else "FAIL",
                                                        mc.lines(picks), "  " + str(problems) if problems else ""),
                      flush=True)
    by_order = {}
    for order, ok in results:
        by_order.setdefault(order, []).append(ok)
    flips = [o for o, v in by_order.items() if len(set(v)) > 1]
    passed = sum(ok for _, ok in results)
    print("Q1 %s level=%s: %d/%d calls pass; orders that flip between repeats: %s -> %s" % (
        args.opp, args.level, passed, len(results), flips or "none",
        "PASS" if passed == len(results) else "FAIL"))
    print("saved", out)
    return 0 if passed == len(results) else 1


def since_added(acts: List[Dict], added: Dict[mc.Pick, int]) -> Dict[mc.Pick, Dict]:
    """For each pick in play: what that (move, range) did in the rounds since it was added."""
    out = {}
    for p, first_round in added.items():
        xs = [a["dealt"] - a["taken"] for a in acts if a["round"] >= first_round and (a["action"], a["range"]) == p]
        out[p] = {"tries": len(xs), "net": sum(xs) / len(xs) if xs else 0.0}
    return out


def play_arm(args, arm: str, port: int, out: str) -> int:
    """One arm: "loop" (Qwen updates the picks after every game) or "none" (no advice), same seed, same rounds."""
    picks: mc.Picks = {k: [] for k in mc.KINDS}
    added: Dict[mc.Pick, int] = {}
    acts: List[Dict] = []
    base = [a for a in attacks(ME) if a["opp"] == args.opp] if args.history else []   # her play data vs him
    os.makedirs(out, exist_ok=True)
    mark_run(out, test=True, arm=arm, opp=args.opp, seed=args.seed)       # never play data (sf2.eval.logs)
    with Advisor(args.advisor) as advisor, open_fight(ME, args.opp, port) as (b, state), \
            open_logs(out, ("actions", "rounds", "ledger")) as logs:
        s1 = System1(args.model, ME, advisor=advisor)
        s1.advice_on = arm == "loop"
        rng = random.Random(args.seed)

        def update(game: int) -> None:
            nonlocal picks, added
            ev = mc.evidence(base + acts)
            new, problems, raw = ask(args.opp, ev, args.level, game, current=picks, since=since_added(acts, added),
                                     task="moves_loop")
            g = mc.grade_update(picks, new, ev)
            nxt = (game + 1) * args.rounds
            added = {p: added.get(p, nxt) for k in mc.KINDS for p in new[k]}
            logs["ledger"].write(json.dumps({
                "game": game, "picks_before": mc.picks_json(picks), "picks": mc.picks_json(new), "problems": problems,
                "ok": g["ok"] and not problems, "grade": {k: (list(v) if isinstance(v, tuple) else v)
                                                          for k, v in g.items()},
                "evidence": [{k: e[k] for k in ("move", "range", "tries", "net", "total", "cls")} for e in ev],
                "reply": raw}) + "\n")
            logs["ledger"].flush()
            print("loop update after game %d: %s %s%s" % (game, "PASS" if g["ok"] and not problems else "FAIL",
                                                        mc.lines(new), "  " + str(problems) if problems else ""),
                  flush=True)
            picks = new

        if arm == "loop" and base:
            update(-1)                            # before the first game: from her play data alone
        for game in range(args.games):
            s1.short = {"me": ME, "opp": args.opp, "lessons": [{"text": t} for t in mc.lines(picks)]}
            for r in range(args.rounds):
                i = game * args.rounds + r
                rnd = play_round(b, s1, args.opp, state, rng, None, i)
                where = {"round": i, "game": game, "arm": arm}
                acts += [dict(e, **where) for e in rnd.log]
                logs["actions"].write("".join(json.dumps(dict(e, **where)) + "\n" for e in rnd.log))
                logs["rounds"].write(json.dumps(dict(rnd.summary, **where, lines=mc.lines(picks))) + "\n")
                for f in logs.values():
                    f.flush()
                print("%s game %d round %d: %s hp %+d" % (arm, game, r, rnd.result,
                                                          rnd.summary["dealt"] - rnd.summary["taken"]), flush=True)
            if arm == "loop":
                update(game)
    return 0


def loop_verdict(out: str) -> Dict:
    """Q2 from the ledger: every update passes; the last one holds the biggest drain and a good move (``grade``
    already requires both whenever the evidence has them); hit points per round vs the paired no-advice arm."""
    from sf2.data.dataset import read
    from sf2.eval.stats import ci, paired
    led = read(os.path.join(out, "loop", "ledger.jsonl"), missing_ok=True)
    loop, none = (read(os.path.join(out, a, "rounds.jsonl"), missing_ok=True) for a in ("loop", "none"))
    d = paired(loop, none) if loop and len(loop) == len(none) else []
    half = len(d) // 2
    return {"updates": len(led), "updates_ok": sum(r["ok"] for r in led),
            "failed_games": [r["game"] for r in led if not r["ok"]],
            "last_picks": led[-1]["picks"] if led else None,
            "last_evidence_bad": [e for e in (led[-1]["evidence"] if led else []) if e["cls"] in ("bad", "good")],
            "hp_vs_none_all": ci(d) if d else None, "hp_vs_none_second_half": ci(d[half:]) if d else None,
            "Q2": "FAIL" if not led or not all(r["ok"] for r in led) else
            "PASS" if any(e["cls"] in ("bad", "good") for e in led[-1]["evidence"]) else
            "NOT TESTED (nothing was clear yet: saying nothing is not learning)"}


def loop(args) -> int:
    if args.one:
        arm, port, out = args.one
        return play_arm(args, arm, int(port), out)
    args.seed = int(time.time()) % 100000 if args.seed is None else args.seed
    root = os.path.join(ROOT, "%s_%s_loop" % (time.strftime("%Y%m%d-%H%M%S"), args.opp))
    cmds = [((args.opp, arm), [sys.executable, os.path.abspath(__file__), "loop", "--opp", args.opp, "--games",
                               str(args.games), "--rounds", str(args.rounds), "--level", args.level, "--seed",
                               str(args.seed), "--model", args.model, "--advisor", args.advisor]
                              + ([] if args.history else ["--no-history"]) + [
                               "--one", arm, str(args.base_port + i), os.path.join(root, arm)])
            for i, arm in enumerate(("loop", "none"))]
    print("Chun-Li vs %s: %d games x %d rounds, seed %d, signal %s; logs/qwen_moves/" % (
        args.opp, args.games, args.rounds, args.seed, args.level), flush=True)
    failed = fan_out(cmds, os.path.join("logs", "qwen_moves"), job_gb=MODEL_JOB_GB)
    v = dict(loop_verdict(root), seed=args.seed, failed_jobs=[list(k) for k in failed])
    if failed:
        v["Q2"] = "NO VERDICT"
    with open(os.path.join(root, "verdict.json"), "w") as f:
        json.dump(v, f, indent=1)
    print(json.dumps(v, indent=1))
    print("saved", root)
    return 0 if v["Q2"] == "PASS" else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    i = sub.add_parser("identify", help="Q1: can Qwen pick good and bad moves from her play data")
    i.add_argument("--opp", required=True)
    i.add_argument("--level", choices=("table", "classes"), default="table")
    i.add_argument("--orders", type=int, default=10)
    i.add_argument("--repeats", type=int, default=2)
    lo = sub.add_parser("loop", help="Q2: game after game, headless, memory starts empty")
    lo.add_argument("--opp", required=True)
    lo.add_argument("--games", type=int, default=10)
    lo.add_argument("--rounds", type=int, default=3, help="rounds per game (Qwen updates after each game)")
    lo.add_argument("--level", choices=("table", "classes"), default="classes")
    lo.add_argument("--seed", type=int, default=None)
    lo.add_argument("--no-history", dest="history", action="store_false",
                    help="start from nothing: only the loop's own rounds (default: her play data vs him + new rounds)")
    lo.add_argument("--base-port", type=int, default=PORTS["qwen_moves"][0], help="two ports: loop and none")
    lo.add_argument("--model", default=LAYA_VISION)
    lo.add_argument("--advisor", default=TEXT_LAYA)
    lo.add_argument("--one", nargs=3, metavar=("ARM", "PORT", "OUT"), help=argparse.SUPPRESS)
    args = ap.parse_args()
    exit_on_sigterm()
    return identify(args) if args.cmd == "identify" else loop(args)


if __name__ == "__main__":
    sys.exit(main())
