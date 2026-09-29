"""Reports on the learning loop's logs.

    python scripts/report.py gaps                   # where the loop loses: the newest chunli play session
    python scripts/report.py gaps --session rollouts/learn/chunli/20260928-062237
    python scripts/report.py churn                  # how much System 2 changed each memory, from today
    python scripts/report.py churn --since 20260928-062237 --show
Logs: logs/gaps_<me>.log, logs/system2/churn_<me>.log
"""
import argparse
import glob
import os
import sys
import time

import _path  # noqa: F401
from sf2.dataset import read
from sf2.eval import churn, gaps
from sf2.eval.logs import is_test


def newest_session(me: str) -> str:
    """The newest learning session that is play data (not a --fresh demo run)."""
    got = [d for d in sorted(glob.glob("rollouts/learn/%s/*" % me)) if not is_test(d)
           and os.path.exists(os.path.join(d, "actions.jsonl"))]
    if not got:
        raise SystemExit("no learning session of %s in rollouts/learn/" % me)
    return got[-1]


def save(path: str, lines) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("gaps", help="where the loop loses, from one session's game log")
    g.add_argument("--char", default="chunli")
    g.add_argument("--session", default=None)
    c = sub.add_parser("churn", help="how much System 2 changed each memory, version by version")
    c.add_argument("--char", default="chunli")
    c.add_argument("--since", default=time.strftime("%Y%m%d-000000"), help="version time stamp to start from")
    c.add_argument("--show", action="store_true", help="print every rewrite's changes")
    args = ap.parse_args()
    if args.cmd == "gaps":
        session = args.session or newest_session(args.char)
        logs = [read(os.path.join(session, k + ".jsonl")) for k in ("actions", "rounds", "games")]
        save(os.path.join("logs", "gaps_%s.log" % args.char), gaps.report(session, *logs))
    else:
        save(os.path.join("logs", "system2", "churn_%s.log" % args.char), churn.report(args.char, args.since, args.show))
    return 0


if __name__ == "__main__":
    sys.exit(main())
