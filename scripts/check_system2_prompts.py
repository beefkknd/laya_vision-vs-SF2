"""Live tests of System 2's prompts: send Qwen the exact chats the runtime sends (sf2/system2_prompts.py), built from
frozen fixtures (tests/fixtures/system2/, never memory/), and check its RAW replies mechanically
(sf2/system2_checks.py). Qwen runs inside the product, so every case runs --repeats times, some in several variations;
a case passes only if EVERY run passes. A mix of passes and fails is reported as FLIP: a sign the prompt is ambiguous,
not something to retry until green.

    python scripts/check_system2_prompts.py                 # needs omlx running (~/work/omlx/start); a few minutes
    python scripts/check_system2_prompts.py --repeats 3 --only revise_ignored

Cases (Chun-Li; her real logs vs Ryu and Dhalsim):
    new_opponent       short memory vs an opponent with no games: no habit or counter lessons (nothing backs them)
    known_opponent     short memory vs Ryu (36 rounds): at least one lesson about HIM (habit, counter, what he punishes)
    revise_ignored     the memory said "use more lp up close", the last round used lp up close 0 times: that lesson
                       must change (reworded or replaced), in 5 variations of wording and order
    revise_confirmed   the memory said "avoid spinning_bird_kick at mid", the last round used it there and was punished:
                       an avoid lesson on spinning_bird_kick must stay
    revise_failed      the memory recommended a move that just whiffed repeatedly: it must no longer be "use more",
                       in 5 variations
    playbook           the playbook rewrite (review) meets every reply rule
Every reply must also meet the reply rules (sf2.system2_checks.reply_problems).
Evidence: logs/system2/tests/<time>.json (every run: prompt task, reply, problems); every chat in logs/system2/.
Exit 1 unless every case is PASS.
"""
import argparse
import collections
import json
import os
import sys
import time
from typing import Callable, Dict, List

import _path  # noqa: F401
from sf2.qwen import chat, json_reply
from sf2.system2_checks import lessons_of, reply_problems
from sf2.system2_prompts import MAX_PLAYBOOK, MAX_PROMPT_LESSONS, messages, populate_prompt, review_prompt

FIX = os.path.join("tests", "fixtures", "system2")
ME = "chunli"
LOGS = json.load(open(os.path.join(FIX, "chunli_logs.json")))
MEM = json.load(open(os.path.join(FIX, "chunli_memory.json")))
RYU = LOGS["ryu"]["actions"]
EVERY = [a for v in LOGS.values() for a in v["actions"]]
ROUNDS: Dict[tuple, List[Dict]] = collections.defaultdict(list)
for _a in RYU:
    ROUNDS[(_a["log"], _a["game"])].append(_a)


def lesson(text, kind, action, rng, claim) -> Dict:
    return {"text": text, "kind": kind, "action": action, "range": rng, "claim": claim,
            "evidence": {"tries": 1, "count": 1, "rate": 1.0, "refs": ["fixture"]}}


def memory_with(first: Dict, rotate: int) -> Dict:
    """The fixture short memory vs Ryu with ``first`` put in (replacing its lesson on the same move), rotated."""
    others = [x for x in MEM["short_vs_ryu"]["lessons"] if x.get("action") != first["action"]]
    les = [first] + others
    k = rotate % len(les)
    return {"me": ME, "opp": "ryu", "lessons": les[k:] + les[:k]}


def pick_round(test: Callable[[List[Dict]], bool]) -> List[Dict]:
    for key in sorted(ROUNDS):
        if test(ROUNDS[key]):
            return ROUNDS[key]
    raise SystemExit("no round in the fixture fits this case")


def failed_move():
    """A move (not spinning_bird_kick) that whiffed at least 3 times vs Ryu and rarely landed."""
    by = collections.defaultdict(list)
    for a in RYU:
        if a["kind"] == "attack" and a["action"] != "spinning_bird_kick":
            by[(a["action"], a["range"])].append(a)
    cands = [(k, g) for k, g in by.items() if sum(x["actual"] == "whiff" for x in g) >= 3
             and sum(x["actual"] == "hit" for x in g) / len(g) <= 0.35]
    (act, rng), g = max(cands, key=lambda kg: sum(x["actual"] == "whiff" for x in kg[1]))
    return act, rng, [x for x in g if x["actual"] != "hit"][:6]


def case_list() -> List[Dict]:
    cases = []
    vs = {o: (v["actions"], v["rounds"]) for o, v in LOGS.items()}
    pb = MEM["playbook"]
    cases.append({"name": "new_opponent", "limit": MAX_PROMPT_LESSONS, "acts": [],
                  "prompt": populate_prompt(ME, "guile", pb, ([], []), EVERY),
                  "expect": lambda les: [] if not [x for x in les if x["kind"] in ("opponent_habit", "counter")]
                  else ["habit / counter lessons with no games against him"]})
    cases.append({"name": "known_opponent", "limit": MAX_PROMPT_LESSONS, "acts": RYU,
                  "prompt": populate_prompt(ME, "ryu", pb, vs["ryu"], EVERY),
                  "expect": lambda les: [] if [x for x in les if x["kind"] in ("opponent_habit", "counter")
                                               or x["claim"] == "punished"] else ["nothing about Ryu himself"]})
    no_lp = pick_round(lambda r: len(r) >= 5 and not [a for a in r if a["action"] == "lp" and a["range"] == "close"])
    for i, text in enumerate(["use more lp up close", "throw lp more when he is close", "lp up close: use it more",
                              "when close, use lp", "use more lp at close range"]):
        mem = memory_with(lesson(text, "use_more", "lp", "close", "lands"), i)
        cases.append({"name": "revise_ignored", "variant": i, "limit": MAX_PROMPT_LESSONS, "acts": RYU,
                      "prompt": populate_prompt(ME, "ryu", pb, vs["ryu"], EVERY, recent=no_lp, current=mem),
                      "expect": lambda les, t=text: ["the ignored lesson came back word for word"]
                      if [x for x in les if x.get("action") == "lp" and x.get("text") == t] else []})
    punished = pick_round(lambda r: len([a for a in r if a["action"] == "spinning_bird_kick" and a["range"] == "mid"
                                         and a["i_was_hit"]]) >= 2)
    mem = memory_with(lesson("avoid spinning_bird_kick at mid", "avoid", "spinning_bird_kick", "mid", "punished"), 0)
    cases.append({"name": "revise_confirmed", "limit": MAX_PROMPT_LESSONS, "acts": RYU,
                  "prompt": populate_prompt(ME, "ryu", pb, vs["ryu"], EVERY, recent=punished, current=mem),
                  "expect": lambda les: [] if [x for x in les if x["kind"] == "avoid"
                                               and x.get("action") == "spinning_bird_kick"]
                  else ["dropped the avoid lesson the last round confirmed"]})
    act, rng, whiffs = failed_move()
    for i, text in enumerate(["use more %s at %s range", "%s at %s range works: use it", "keep using %s at %s range",
                              "use %s more at %s range", "go for %s at %s range"]):
        mem = memory_with(lesson(text % (act, rng), "use_more", act, rng, "lands"), i)
        cases.append({"name": "revise_failed", "variant": i, "limit": MAX_PROMPT_LESSONS, "acts": RYU,
                      "prompt": populate_prompt(ME, "ryu", pb, vs["ryu"], EVERY, recent=whiffs, current=mem),
                      "expect": lambda les, a=act: ["still says use more %s after it whiffed" % a]
                      if [x for x in les if x["kind"] == "use_more" and x.get("action") == a] else []})
    cases.append({"name": "playbook", "limit": MAX_PLAYBOOK, "acts": EVERY, "fallback": None,
                  "prompt": review_prompt(ME, pb, vs), "expect": lambda les: []})
    return cases


def run_case(c: Dict, rep: int) -> Dict:
    task = "test_%s%s_r%d" % (c["name"], "_v%d" % c["variant"] if "variant" in c else "", rep)
    t = time.time()
    try:
        reply = json_reply(chat(messages(ME, c["prompt"]), task))
    except Exception as e:  # noqa: BLE001 - an unusable reply is a failed run, recorded as such
        return {"task": task, "reply": None, "problems": ["unusable reply: %s: %s" % (type(e).__name__, e)],
                "seconds": round(time.time() - t, 1)}
    problems = reply_problems(ME, reply, c["limit"], c["acts"], c.get("fallback", EVERY))
    les = lessons_of(reply) or []
    problems += c["expect"](les)
    return {"task": task, "reply": reply, "problems": problems, "seconds": round(time.time() - t, 1)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--only", help="run only this case name")
    args = ap.parse_args()
    cases = [c for c in case_list() if not args.only or c["name"] == args.only]
    runs: Dict[str, List[Dict]] = collections.defaultdict(list)
    t0 = time.time()
    for c in cases:
        for r in range(args.repeats):
            res = run_case(c, r)
            runs[c["name"]].append(dict(res, variant=c.get("variant")))
            print("%-17s %-4s run %d: %-4s %4.1fs %s" % (c["name"], "v%d" % c["variant"] if "variant" in c else "", r,
                                                        "ok" if not res["problems"] else "FAIL", res["seconds"],
                                                        "; ".join(res["problems"])[:160]), flush=True)
    print("\n%-17s %6s  %s" % ("case", "runs", "verdict"))
    verdicts = {}
    for name, rs in runs.items():
        ok = sum(not r["problems"] for r in rs)
        verdicts[name] = "PASS" if ok == len(rs) else "FAIL" if ok == 0 else "FLIP"
        print("%-17s %2d/%-3d  %s" % (name, ok, len(rs), verdicts[name]))
    os.makedirs(os.path.join("logs", "system2", "tests"), exist_ok=True)
    path = os.path.join("logs", "system2", "tests", time.strftime("%Y%m%d-%H%M%S") + ".json")
    with open(path, "w") as f:
        json.dump({"repeats": args.repeats, "seconds": round(time.time() - t0), "verdicts": verdicts, "runs": runs}, f,
                  indent=1)
    print("%s in %.0f s -> %s" % ("ALL PASS" if set(verdicts.values()) == {"PASS"} else "NOT ALL PASS",
                                   time.time() - t0, path))
    return 0 if set(verdicts.values()) == {"PASS"} else 1


if __name__ == "__main__":
    sys.exit(main())
