"""Day 2: dump the seed set from the scripted teacher (rung 2 of the teacher ladder).

    python scripts/collect_teacher.py --out data --name seed_teacher --decisions 30000 --eps 0.25

The teacher plays full matches from the savestate. With probability ``--eps`` it executes a random option
instead of its own (epsilon-expert), so the data covers states its own play would never reach; every frame is
still labelled with the teacher's distribution. Every 10th match goes to val. Decisions where the stick does
nothing (hit, knocked down) are played but not written.
"""
import argparse
import random

import _path  # noqa: F401
from sf2 import actions as A
from sf2.dataset import Writer
from sf2.cli import add_env_args, make_env
from sf2.loop import play, save_rounds
from sf2.rollout import gate


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_env_args(ap)
    ap.add_argument("--out", default="data")
    ap.add_argument("--name", default="seed_teacher")
    ap.add_argument("--decisions", type=int, default=30000)
    ap.add_argument("--max-matches", type=int, default=10000)
    ap.add_argument("--eps", type=float, default=0.25)
    args = ap.parse_args()
    rng = random.Random(args.seed)

    def choose(env, prev, cur, text, t_dist):
        if rng.random() < args.eps:
            return rng.choice(A.ACTIONS), {"actor": "random"}
        acts = list(t_dist)
        return rng.choices(acts, weights=[t_dist[a] for a in acts])[0], {"actor": "teacher"}

    env = make_env(args)
    w = Writer(args.out, args.name, source=args.name, skip_uncontrollable=True)
    rows, rounds = play(env, choose, args.max_matches, writer=w, max_decisions=args.decisions)
    off = sum(r["meta"]["controllable"] is False for r in rows)
    print("left out %d of %d decisions where the stick did nothing (hit / knocked down)" % (off, len(rows)))
    w.close()
    env.close()
    save_rounds("%s/%s/rounds.jsonl" % (args.out, args.name), rounds)
    print("wrote", w.n, "->", w.dir)
    print("teacher (eps=%.2f) gate:" % args.eps, gate(rows, rounds))


if __name__ == "__main__":
    main()
