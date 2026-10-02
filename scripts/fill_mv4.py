"""Round 4 step 1 (docs/prereg_movement_finetunes.md): the one fill line after a collection round
(sf2.data.mv4_fill): act attack / special train rows, fireball left / right train rows, cells not full; appends the
full report to --log.

    python scripts/fill_mv4.py --root rollouts/pairs2p --keep-from test_data_pairs2p_down [--log logs/mv4_fill.jsonl]

Exit 0 when the targets are met (each >= --target train rows), 2 when not yet.
"""
import argparse
import json
import os
import sys
import time

import _path  # noqa: F401
from sf2.data import mv4_fill as M


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="rollouts/pairs2p")
    ap.add_argument("--keep-from", default="test_data_pairs2p_down")
    ap.add_argument("--target", type=int, default=M.TARGET)
    ap.add_argument("--log", default=os.path.join("logs", "mv4_fill.jsonl"))
    ap.add_argument("--label", default="", help="e.g. the round: games 17")
    args = ap.parse_args(argv)
    t0 = time.time()
    rep = M.fill(args.root, keep_from=args.keep_from)
    rep.update(label=args.label, at=time.strftime("%Y-%m-%d %H:%M:%S"), seconds=round(time.time() - t0, 1))
    os.makedirs(os.path.dirname(args.log) or ".", exist_ok=True)
    with open(args.log, "a") as f:
        f.write(json.dumps(rep) + "\n")
    print("%s %s" % (args.label, M.line(rep, args.target)))
    return 0 if M.met(rep, args.target) else 2


if __name__ == "__main__":
    sys.exit(main())
