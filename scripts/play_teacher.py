"""The gate's reference line: the scripted teacher (no randomness) against the same opponent and savestate.

    python scripts/play_teacher.py --name teacher --matches 10
"""
import argparse
import json
import os

import _path  # noqa: F401
from sf2.config import DEFAULT_STATE
from sf2.env import FightEnv
from sf2.loop import play, save_rounds
from sf2.rollout import gate
from sf2.teacher import argmax


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="teacher")
    ap.add_argument("--out", default="rollouts")
    ap.add_argument("--state", default=DEFAULT_STATE)
    ap.add_argument("--matches", type=int, default=10)
    ap.add_argument("--policy", choices=["teacher", "random", "idle"], default="teacher")
    args = ap.parse_args()
    import random

    rng = random.Random(0)

    def choose(env, prev, cur, text, t_dist):
        if args.policy == "random":
            return rng.choice(list(t_dist)), {"actor": "random"}
        if args.policy == "idle":
            return "idle", {"actor": "idle"}
        return argmax(t_dist), {"actor": "teacher"}

    env = FightEnv(args.state)
    rows, rounds = play(env, choose, args.matches)
    os.makedirs(os.path.join(args.out, args.name), exist_ok=True)
    save_rounds("%s/%s/rounds.jsonl" % (args.out, args.name), rounds)
    g = gate(rows, rounds)
    g.update(model=args.policy, state=args.state)
    with open("%s/%s/gate.json" % (args.out, args.name), "w") as f:
        json.dump(g, f, indent=2)
    print(json.dumps(g, indent=2))


if __name__ == "__main__":
    main()
