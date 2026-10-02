"""docs/eye_questions_v1.md, "Next: more fireball data": the one q1 fill line after a collection round
(sf2.data.eye_q1_fill): yes / no train rows as the q1 build would select them, the yes rows lost to matching, the
shot-sample cells not full in the new games, the wall time; appends the full report to --log.

    python scripts/fill_eye_q1.py --root rollouts/pairs2p --label "games 32" [--log logs/eye_q1_fill.jsonl]

Exit 0 when both answers reach --target train rows, 2 when not yet.
"""
import argparse
import json
import os
import sys
import time

import _path  # noqa: F401
from sf2.data import eye_data as E
from sf2.data import eye_pool as P
from sf2.data import eye_q1_fill as F
from sf2.data import pairs_labels as L
from sf2.data import pairs_shots as S
from build_eye_data import PREFER


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="rollouts/pairs2p")
    ap.add_argument("--target", type=int, default=F.TARGET)
    ap.add_argument("--from-game", type=int, default=F.FROM_GAME, help="shot cells counted from this game on")
    ap.add_argument("--shot-per-game", type=int, default=S.SHOT_PER_GAME, help="the cap the new games ran with")
    ap.add_argument("--prefer", default=",".join(PREFER["q1"]), help="the build's prefer dirs ('none': no dirs)")
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--log", default=os.path.join("logs", "eye_q1_fill.jsonl"))
    ap.add_argument("--label", default="", help="e.g. the round: games 32")
    args = ap.parse_args(argv)
    t0 = time.time()
    pool = P.pool(args.root, L.poke_bands()["all"], args.workers)
    prefer = E.prefer_keys("q1", [] if args.prefer == "none" else args.prefer.split(","))
    rep = F.fill(pool, args.root, prefer, from_game=args.from_game, cap=args.shot_per_game)
    rep.update(label=args.label, pool=len(pool), at=time.strftime("%Y-%m-%d %H:%M:%S"),
               seconds=round(time.time() - t0, 1))
    os.makedirs(os.path.dirname(args.log) or ".", exist_ok=True)
    with open(args.log, "a") as f:
        f.write(json.dumps(rep) + "\n")
    print("%s %s" % (args.label, F.line(rep, args.target)))
    return 0 if F.met(rep, args.target) else 2


if __name__ == "__main__":
    sys.exit(main())
