"""The gate's reference line: the scripted teacher (no randomness) against the same opponent and savestate.

    python scripts/play_teacher.py --name teacher --matches 10
"""
import argparse
import json
import os

import _path  # noqa: F401
from sf2.cli import add_env_args, make_env
from sf2.dataset import write_jsonl
from sf2.loop import play, save_rounds
from sf2.rollout import gate
from sf2.teacher import argmax


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="teacher")
    ap.add_argument("--out", default="rollouts")
    add_env_args(ap)
    ap.add_argument("--matches", type=int, default=10)
    ap.add_argument("--policy", choices=["teacher", "random", "idle"], default="teacher")
    args = ap.parse_args()
    import random

    rng = random.Random(args.seed)

    def choose(env, prev, cur, text, t_dist):
        if args.policy == "random":
            return rng.choice(list(t_dist)), {"actor": "random"}
        if args.policy == "idle":
            return "idle", {"actor": "idle"}
        return argmax(t_dist), {"actor": "teacher"}

    env = make_env(args)
    rows, rounds = play(env, choose, args.matches)
    env.close()
    os.makedirs(os.path.join(args.out, args.name), exist_ok=True)
    save_rounds("%s/%s/rounds.jsonl" % (args.out, args.name), rounds)
    write_jsonl("%s/%s/rows.jsonl" % (args.out, args.name), ({"episode": r["episode"], "meta": r["meta"]} for r in rows))
    g = gate(rows, rounds)
    g.update(model=args.policy, savestate=args.savestate)
    with open("%s/%s/gate.json" % (args.out, args.name), "w") as f:
        json.dump(g, f, indent=2)
    print(json.dumps(g, indent=2))


if __name__ == "__main__":
    main()
