"""The batched two-system loop: System 1 plays a batch, System 2 (Qwen) reviews it and writes rules, and so on.

    QWEN_API_KEY=... python scripts/learn.py --session guile_ryu_s1 --model runs/base_all_v0/best \\
        --savestate states/p1_guile_vs_ryu.state --me guile --opp ryu --batches 5

Batch 0 plays with an empty memory: it is the control every later batch is compared to, opening by opening. After
each batch: moments (the hits, worst first, then audits) -> Qwen (the game's description, the memory, last batch's
feedback) -> validated rules merged into memories/sessions/<session>/qwen.txt. Everything is kept under
out/learn/<session>/ (a memory snapshot, the prompt and reply, rejected lines per batch) and one row per batch in
out/learn/<session>.jsonl, the learning curve. The playbook starts empty (owner's rule): Qwen gets no tactics.
"""
import argparse
import json
import os
import random
import shutil
import subprocess
import sys
import time
import urllib.request
from collections import Counter

import _path  # noqa: F401
from sf2 import dataset as D
from sf2 import memory as MEM
from sf2 import moments as MO
from sf2 import system2 as S2
from sf2.openings import paired

HERE = os.path.dirname(os.path.abspath(__file__))


def ask_qwen(url, model, messages, think, timeout=900):
    body = {"model": model, "messages": messages, "max_tokens": 8000 if think else 1500,
            "chat_template_kwargs": {"enable_thinking": think}}
    req = urllib.request.Request(url.rstrip("/") + "/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json",
                                          "Authorization": "Bearer " + os.environ["QWEN_API_KEY"]})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        d = json.load(r)
    return d["choices"][0]["message"]["content"], d.get("usage", {})


def play(args, name, memory_dir):
    argv = [sys.executable, os.path.join(HERE, "parallel.py"), "--workers", str(args.workers), "--base-port",
            str(args.base_port), "play_student", "--model", args.model, "--name", name, "--savestate", args.savestate,
            "--me", args.me, "--opp", args.opp, "--openings", args.openings, "--memory", memory_dir]
    with open(os.path.join("out", "learn", args.session, name + ".log"), "w") as log:
        if subprocess.call(argv, stdout=log, stderr=subprocess.STDOUT):
            sys.exit("batch run %s failed; see out/learn/%s/%s.log" % (name, args.session, name))
    return os.path.join("rollouts", name)


def usage(rows):
    ctl = [r["meta"] for r in rows if r["meta"].get("controllable")]
    fired = Counter(m["memory_rule"].split(" weight")[0] for m in ctl if m.get("memory_fired"))
    changed = Counter(m["memory_rule"].split(" weight")[0] for m in ctl if m.get("memory_changed"))
    n = max(1, len(ctl))
    return sum(fired.values()) / n, sum(changed.values()) / n, {k: (fired[k], changed[k]) for k in fired}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--session", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--savestate", required=True)
    ap.add_argument("--me", required=True)
    ap.add_argument("--opp", required=True)
    ap.add_argument("--openings", default="openings/dev.txt")
    ap.add_argument("--batches", type=int, default=5, help="batches after the control")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--base-port", type=int, default=47810)
    ap.add_argument("--qwen-url", default="http://192.168.1.199:8800/v1", help="claw; the Studio is :8000 on .216")
    ap.add_argument("--qwen-model", default="qwen38-27b-oq4e-mtp")
    ap.add_argument("--think", action="store_true", help="let Qwen reason before answering (slower)")
    ap.add_argument("--moments", type=int, default=30, help="moments shown to Qwen per batch")
    args = ap.parse_args()

    mem_dir = os.path.join("memories", "sessions", args.session)
    out = os.path.join("out", "learn", args.session)
    if os.path.exists(mem_dir) or os.path.exists(out):
        sys.exit("session %s exists; pick a new --session" % args.session)
    os.makedirs(mem_dir)
    os.makedirs(out)
    curve = os.path.join("out", "learn", args.session + ".jsonl")
    control_rounds, control_net, fb = None, None, None
    for k in range(args.batches + 1):
        snap = os.path.join(out, "b%d" % k)
        shutil.copytree(mem_dir, os.path.join(snap, "memory"))
        rollout = play(args, "learn_%s_b%d" % (args.session, k), mem_dir)
        rounds = D.read(os.path.join(rollout, "rounds.jsonl"))
        rows = D.read(os.path.join(rollout, "train.jsonl"))
        gate = json.load(open(os.path.join(rollout, "gate.json")))
        net = gate["net_damage_per_round"]
        if k == 0:
            control_rounds, control_net = rounds, net
            pr = {"mean_diff": 0.0, "ci95": (0.0, 0.0)}
        else:
            pr = paired(control_rounds, rounds)
        fired, changed, per_rule = usage(rows)
        row = {"batch": k, "rules": len(MEM.load(mem_dir).rules), "net_damage_per_round": net,
               "round_win_rate": gate["round_win_rate"], "paired_vs_control": pr["mean_diff"], "ci95": pr["ci95"],
               "fired": fired, "changed": changed}
        if k == args.batches:
            with open(curve, "a") as f:
                f.write(json.dumps(row) + "\n")
            print(json.dumps(row), flush=True)
            break
        for r in rows:
            r["images"] = [os.path.abspath(os.path.join(rollout, p)) for p in r["images"]]
        moments = MO.extract(rows, random.Random(k), unsure_margin=0.0)
        fb = S2.feedback(k, net, control_net, (pr["mean_diff"], pr["ci95"]), fired, changed, per_rule) if k else None
        msgs = S2.build_messages(S2.select_moments(moments, args.moments), MEM.load(mem_dir).rules, fb)
        t0 = time.time()
        reply, tokens = ask_qwen(args.qwen_url, args.qwen_model, msgs, args.think)
        seconds = time.time() - t0
        rules, rejected = S2.parse_reply(reply, thinking=args.think)
        added = S2.merge(mem_dir, rules, replace=True)
        json.dump({"messages": msgs, "reply": reply, "tokens": tokens, "seconds": seconds,
                   "rules": [str(r) for r in rules], "rejected": rejected}, open(os.path.join(snap, "qwen.json"), "w"),
                  indent=2)
        row.update(moments=len(moments), qwen_seconds=round(seconds, 1), rules_added=added, rejected=len(rejected))
        with open(curve, "a") as f:
            f.write(json.dumps(row) + "\n")
        print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
