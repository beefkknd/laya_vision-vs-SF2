"""The lesson loop, Chun-Li vs one opponent: Qwen proposes what to learn, code verifies and keeps the books
(sf2.system2.lessons), text laya plays with the lessons in play. Headless, paired with a no-advice arm (same savestate,
same seed, same start delays).

    python scripts/qwen_lessons.py --opp ken --games 10

After every game: code reviews the registry (claims in test judged, lessons whose evidence stopped holding retired),
then Qwen proposes at most 2 claims, code judges them. Starts from her play data against him (--no-history: nothing).
Verdict (rollouts/qwen_lessons/<stamp>_<opp>/verdict.json):
    invariant     no registered lesson ever contradicts its own evidence (code-enforced; a violation is a bug)
    hypotheses    Qwen's valid claims that hold on all the data at the end, vs 500 random valid claims (the same
                  moves, ranges and situations) judged the same way: does Qwen propose better than chance?
    outcome       hit points per round vs the no-advice arm, paired (reported, not a gate)
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
from sf2.data.dataset import read
from sf2.eval.logs import load_actions, mark_run, play_dirs
from sf2.eval.runner import exit_on_sigterm, fan_out, open_fight, open_logs
from sf2.eval.stats import ci, paired
from sf2.system1.advisor import Advisor
from sf2.system1.system1 import System1, play_round
from sf2.system2 import lessons as L
from sf2.system2.lesson_prompt import messages, parse_claims
from sf2.system2.qwen import chat, json_reply
from sf2.vocab import RANGES

ME = "chunli"
ROOT = os.path.join("rollouts", "qwen_lessons")
BASELINE = 500


def decisions(opp: str) -> List[Dict]:
    """Every decision of hers against ``opp`` in her play data (attacks, walks, blocks)."""
    return [a for d in play_dirs() for a in load_actions(d) if a.get("me") == ME and a.get("opp") == opp]


def ask(opp: str, reg: L.Registry, rows: List[Dict], last: List[Dict], all_rounds: List[Dict],
        last_rounds: List[Dict]):
    try:
        raw = chat(messages(ME, opp, reg, rows, last, all_rounds, last_rounds), "lessons_%s" % opp)
        claims, problems = parse_claims(json_reply(raw))
    except Exception as e:                   # Qwen down, cut off or not JSON: no claims this game
        return [], ["%s: %s" % (type(e).__name__, e)], None
    return claims, problems, raw


def play_arm(args, arm: str, port: int, out: str) -> int:
    base = decisions(args.opp) if args.history else []
    reg: L.Registry = []
    acts: List[Dict] = []
    played: List[Dict] = []
    os.makedirs(out, exist_ok=True)
    mark_run(out, test=True, arm=arm, opp=args.opp, seed=args.seed)       # never play data (sf2.eval.logs)
    with Advisor(args.advisor) as advisor, open_fight(ME, args.opp, port) as (b, state), \
            open_logs(out, ("actions", "rounds", "ledger")) as logs:
        s1 = System1(args.model, ME, advisor=advisor)
        s1.advice_on = arm == "loop"
        rng = random.Random(args.seed)

        def learn(game: int, last: List[Dict], last_rounds: List[Dict]) -> None:
            nonlocal reg
            rows = base + acts
            reg = L.review(reg, rows, game)
            claims, problems, raw = ask(args.opp, reg, rows, last, played, last_rounds)
            reg, outcome = L.propose(reg, claims, rows, game)
            logs["ledger"].write(json.dumps({"game": game, "claims": claims, "outcome": outcome, "problems": problems,
                                             "registry": reg, "in_play": L.in_play(reg),
                                             "violations": L.violations(reg, rows), "reply": raw}) + "\n")
            logs["ledger"].flush()
            print("after game %d: %s | in play %s" % (game, ["%s -> %s" % (o.get("line", o["claim"]), o["state"])
                                                            for o in outcome], L.in_play(reg)), flush=True)

        if arm == "loop" and base:
            learn(-1, base, [])                   # before the first game: from her play data
        for game in range(args.games):
            lines = L.in_play(reg)
            s1.short = {"me": ME, "opp": args.opp, "lessons": [{"text": t} for t in lines]}
            this, this_rounds = [], []
            for r in range(args.rounds):
                i = game * args.rounds + r
                rnd = play_round(b, s1, args.opp, state, rng, None, i)
                where = {"round": i, "game": game, "arm": arm}
                this += [dict(e, **where) for e in rnd.log]
                logs["actions"].write("".join(json.dumps(dict(e, **where)) + "\n" for e in rnd.log))
                logs["rounds"].write(json.dumps(dict(rnd.summary, **where, lines=lines)) + "\n")
                this_rounds.append(rnd.summary)
                for f in logs.values():
                    f.flush()
                print("%s game %d round %d: %s hp %+d" % (arm, game, r, rnd.result,
                                                          rnd.summary["dealt"] - rnd.summary["taken"]), flush=True)
            acts += this
            played += this_rounds
            if arm == "loop":
                learn(game, this, this_rounds)
    return 0


def random_claims(rows: List[Dict], n: int, seed: int = 0) -> List[Dict]:
    r = random.Random(seed)
    moves = sorted({a["action"] for a in rows})
    return [{"kind": r.choice(L.KINDS), "move": r.choice(moves), "range": r.choice((None,) + RANGES),
             "when": r.choice((None,) + tuple(L.WHEN_WORDS))} for _ in range(n)]


def holds(c: Dict, rows: List[Dict]) -> bool:
    return L.condition_evidence(rows, c)["cls"] == L.RIGHT[c["kind"]]


def verdict(root: str, opp: str, history: bool) -> Dict:
    led = read(os.path.join(root, "loop", "ledger.jsonl"), missing_ok=True)
    acts = read(os.path.join(root, "loop", "actions.jsonl"), missing_ok=True)
    rows = (decisions(opp) if history else []) + acts
    valid = [o["claim"] for r in led for o in r["outcome"] if o["state"] != "refused"]
    views = {v: [c for c in valid if c.get("view") == v] for v in ("attack", "defense")}
    rand = [c for c in random_claims(rows, BASELINE) if L.condition_evidence(rows, c)["tries"] > 0]
    loop, none = (read(os.path.join(root, a, "rounds.jsonl"), missing_ok=True) for a in ("loop", "none"))
    d = paired(loop, none) if loop and len(loop) == len(none) else []
    states = {}
    for r in led:
        for o in r["outcome"]:
            states[o["state"]] = states.get(o["state"], 0) + 1
    final = led[-1]["registry"] if led else []
    return {"updates": len(led), "violations": sum(len(r["violations"]) for r in led),
            "proposed": sum(len(r["outcome"]) for r in led), "outcomes": states,
            "qwen_hold_rate": sum(holds(c, rows) for c in valid) / len(valid) if valid else None,
            "qwen_hold_rate_by_view": {v: (sum(holds(c, rows) for c in cs) / len(cs) if cs else None, len(cs))
                                       for v, cs in views.items()},
            "kinds_registered": {k: sum(r["state"] == "registered" and r["claim"]["kind"] == k for r in final)
                                 for k in L.KINDS},
            "random_hold_rate": sum(holds(c, rows) for c in rand) / len(rand) if rand else None,
            "registered_at_end": [r["line"] for r in final if r["state"] == "registered"],
            "retired": [r["line"] for r in final if r["state"] == "retired"],
            "won": {"loop": sum(r["result"] == "win" for r in loop), "none": sum(r["result"] == "win" for r in none)},
            "hp_vs_none": ci(d) if d else None,
            "taken_vs_none": ci([b["taken"] - a["taken"] for a, b in zip(loop, none)]) if d else None}


def export(out: str = os.path.join("lessons", "chunli.json")) -> int:
    """Every lesson registered at the end of a run, per opponent, with the runs that registered it and its evidence
    there (a lesson registered in every run of that opponent is marked "all_runs")."""
    by_opp: Dict[str, Dict] = {}
    for root in sorted(d for d in os.listdir(ROOT) if os.path.isdir(os.path.join(ROOT, d))):
        led = read(os.path.join(ROOT, root, "loop", "ledger.jsonl"), missing_ok=True)
        v = os.path.join(ROOT, root, "verdict.json")
        if not led or not os.path.exists(v) or json.load(open(v)).get("updates", 0) < 10:
            continue                              # smoke runs and unfinished runs are left out
        opp = root.split("_", 1)[1]
        runs = by_opp.setdefault(opp, {"runs": [], "lessons": {}})
        runs["runs"].append(root)
        for r in led[-1]["registry"]:
            if r["state"] == "registered":
                e = runs["lessons"].setdefault(r["line"], {"claim": r["claim"], "runs": {}})
                e["runs"][root] = {k: r["evidence"][k] for k in ("tries", "net", "lo", "hi", "total")}
    doc = {"me": ME, "made_by": "scripts/qwen_lessons.py --export: Qwen proposed, code verified (sf2.system2.lessons)",
           "opponents": {o: {"runs": d["runs"], "lessons": [dict(line=line, all_runs=len(e["runs"]) == len(d["runs"]),
                                                                  **e) for line, e in sorted(d["lessons"].items())]}
                         for o, d in sorted(by_opp.items())}}
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(doc, f, indent=1)
    for o, d in doc["opponents"].items():
        print("%-8s %d runs, %d lessons (%d in every run)" % (o, len(d["runs"]), len(d["lessons"]),
                                                             sum(x["all_runs"] for x in d["lessons"])))
    print("saved", out)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--opp")
    ap.add_argument("--export", action="store_true", help="write lessons/chunli.json from every finished run")
    ap.add_argument("--games", type=int, default=10)
    ap.add_argument("--rounds", type=int, default=3, help="rounds per game (Qwen proposes after each game)")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--no-history", dest="history", action="store_false")
    ap.add_argument("--base-port", type=int, default=PORTS["qwen_moves"][0], help="two ports: loop and none")
    ap.add_argument("--model", default=LAYA_VISION)
    ap.add_argument("--advisor", default=TEXT_LAYA)
    ap.add_argument("--one", nargs=3, metavar=("ARM", "PORT", "OUT"), help=argparse.SUPPRESS)
    args = ap.parse_args()
    exit_on_sigterm()
    if args.export:
        return export()
    if not args.opp:
        ap.error("--opp is required")
    if args.one:
        return play_arm(args, args.one[0], int(args.one[1]), args.one[2])
    args.seed = int(time.time()) % 100000 if args.seed is None else args.seed
    root = os.path.join(ROOT, "%s_%s" % (time.strftime("%Y%m%d-%H%M%S"), args.opp))
    cmds = [((args.opp, arm), [sys.executable, os.path.abspath(__file__), "--opp", args.opp, "--games",
                               str(args.games), "--rounds", str(args.rounds), "--seed", str(args.seed), "--model",
                               args.model, "--advisor", args.advisor] + ([] if args.history else ["--no-history"])
                              + ["--one", arm, str(args.base_port + i), os.path.join(root, arm)])
            for i, arm in enumerate(("loop", "none"))]
    print("Chun-Li vs %s: %d games x %d rounds, seed %d; logs/qwen_lessons/" % (
        args.opp, args.games, args.rounds, args.seed), flush=True)
    failed = fan_out(cmds, os.path.join("logs", "qwen_lessons"), job_gb=MODEL_JOB_GB)
    v = dict(verdict(root, args.opp, args.history), seed=args.seed, failed_jobs=[list(k) for k in failed])
    with open(os.path.join(root, "verdict.json"), "w") as f:
        json.dump(v, f, indent=1)
    print(json.dumps(v, indent=1))
    print("saved", root)
    return 1 if failed or v["violations"] else 0


if __name__ == "__main__":
    sys.exit(main())
