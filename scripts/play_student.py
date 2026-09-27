"""The student plays; every decision is recorded.

    python scripts/play_student.py --model runs/r0/best --name r0 --matches 10

Writes rollouts/<name>/{train,val}.jsonl + images (same layout as a dataset: label/target = the move played),
rollouts/<name>/rounds.jsonl, and prints the gate. Each row's meta has: action (student), student_probs,
dmg_for / dmg_against (during the action), dmg_for_next / dmg_against_next (next 0.5 s),
whiff, hot, round_result, dx.
"""
import argparse
import json
from contextlib import nullcontext

import _path  # noqa: F401
from sf2 import memory
from sf2.dataset import Writer
from sf2.cli import add_env_args, make_env
from sf2.loop import play, save_rounds
from sf2.policy import LayaPolicy
from sf2.rollout import gate


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--out", default="rollouts")
    add_env_args(ap)
    ap.add_argument("--matches", type=int, default=10)
    ap.add_argument("--sample", action="store_true", help="sample from the probabilities instead of the top option")
    ap.add_argument("--device", default=None)
    ap.add_argument("--memory", default=None, help="a memory folder (owner.txt, claude.txt, qwen.txt): situation rules "
                                                  "that nudge the move (sf2/memory.py); unset = laya alone")
    ap.add_argument("--tau", type=float, default=None, help="one tau for every rule (default: each rule's weight): a "
                                                          "rule's move is played if laya's probability of it is within "
                                                          "tau of its top move")
    args = ap.parse_args()

    pol = LayaPolicy(args.model, device=args.device, sample=args.sample, seed=args.seed)
    mem = memory.load(args.memory) if args.memory else None
    if mem:
        print("memory %s: %d rules, tau %s" % (args.memory, len(mem.rules),
                                                "per rule" if args.tau is None else args.tau), flush=True)

    def choose(env, prev, cur, text):
        keep_alive = getattr(env.backend, "keep_alive", None)
        with keep_alive() if keep_alive else nullcontext():
            a, probs = pol.act(prev, cur, text)
        if mem is None:
            return a, {"actor": "student", "student_probs": probs}
        move, use = mem.apply(text, probs, a, args.tau)
        return move, {"actor": "student", "student_probs": probs, "laya_top": a,
                      **{"memory_" + k: v for k, v in use.items()}}

    env = make_env(args)
    w = Writer(args.out, args.name, source=args.name, val_every=0)  # all rows -> train.jsonl
    rows, rounds = play(env, choose, args.matches, writer=w)
    w.close()
    env.close()
    save_rounds("%s/%s/rounds.jsonl" % (args.out, args.name), rounds)
    g = gate(rows, rounds)
    g.update(model=args.model, savestate=args.savestate, memory=args.memory, tau=args.tau)
    with open("%s/%s/gate.json" % (args.out, args.name), "w") as f:
        json.dump(g, f, indent=2)
    print(json.dumps(g, indent=2))


if __name__ == "__main__":
    main()
