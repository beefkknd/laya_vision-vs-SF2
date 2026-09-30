"""Where does each advice line's chain break? Traces lines through laya-vision, the label rule, text laya, the game,
Qwen's loop and the verifier (sf2/eval/boundary.py) over an ab_memory --fixed batch.

    python scripts/boundary.py logs/ab/ryu_expert --opp ryu [--arm expert | --lines "a;b"] [--base DIR] [--json out]

``--base``: the checkout the batch's relative paths (the logs' "saved rollouts/ab/..." lines, rollouts/, locks/)
resolve against; default the current directory. Lines: those of ``--arm``, or ``--lines``, else every arm's lines.
"""
import argparse
import glob
import json
import os
import sys
from contextlib import contextmanager
from typing import Dict, Iterator, List

import _path  # noqa: F401
from sf2.data.dataset import read
from sf2.eval.boundary import trace
from sf2.eval.logs import table_run


def roots(log_dir: str, base: str) -> List[str]:
    out = []
    for f in sorted(glob.glob(os.path.join(log_dir, "*.log"))):
        with open(f) as fh:
            saved = [x.split()[1] for x in fh if x.startswith("saved ")]
        if not saved:
            raise SystemExit("%s: no run saved" % f)
        root = os.path.join(base, saved[-1])
        why = table_run(root)
        if why:
            raise SystemExit("%s is a lookup-table run (%s): boundary traces laya-vision's P(hit) ranking only" % (
                root, why))
        out.append(root)
    if not out:
        raise SystemExit("%s: no ab_memory logs" % log_dir)
    return out


def arm_lines(root: str, opp: str, arm: str) -> List[str]:
    path = os.path.join(root, "memory_%s_%s.json" % (opp, arm))
    if arm == "none":
        return []
    with open(path) as f:
        return [x["text"] for x in json.load(f)["lessons"]]


def load_arms(rs: List[str], opp: str) -> Dict[str, Dict]:
    pre = opp + "_"
    names = sorted(n[len(pre):] for n in os.listdir(rs[0]) if n.startswith(pre)
                   and os.path.isdir(os.path.join(rs[0], n)))
    if "none" not in names:
        raise SystemExit("%s: no %snone arm" % (rs[0], pre))
    arms = {}
    for a in names:
        lines = arm_lines(rs[0], opp, a)
        for r in rs[1:]:
            if arm_lines(r, opp, a) != lines:
                raise SystemExit("%s: arm %s has other lines than in %s" % (r, a, rs[0]))
        rows = [x for r in rs for x in read(os.path.join(r, pre + a, "actions.jsonl"))
                if "shortlist" in x and "range" in x]
        rounds = [read(os.path.join(r, pre + a, "rounds.jsonl")) for r in rs]
        arms[a] = {"lines": lines, "rows": rows, "rounds": rounds}
    return arms


def ledgers(base: str, opp: str) -> List[List[Dict]]:
    pats = [os.path.join(base, "rollouts", "qwen_lessons", "*_%s*" % opp, "loop", "ledger.jsonl"),
            os.path.join(base, "rollouts", "locked", "*", "*_%s*" % opp, "loop", "ledger.jsonl")]
    found = sorted(set(p for pat in pats for p in glob.glob(pat)))
    return [read(p) for p in found if not table_run(os.path.dirname(os.path.dirname(p)))]    # runs/all8's only


@contextmanager
def cwd(path: str) -> Iterator[None]:
    old = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(old)


def history(base: str, lock: str, me: str, opp: str) -> List[Dict]:
    from sf2.eval.lock import play_rows
    with cwd(base):
        return play_rows(lock, me, opp)


def _p(x) -> str:
    return "  n/a" if x is None else "%4.0f%%" % (100 * x)


def show(t: Dict) -> str:
    v, r, tx, g, q, h = (t[k] for k in "VRTGQH")
    ab = g["ab"]
    hist = q["history"] or {}
    return "\n".join([
        "%s  [%s %s, arm %s]" % (t["line"], t["polarity"], t["move"], t["arm"]),
        "  V vision   %d decisions: top3 %s, ratings %s, vision-best %s, picked %s" % (
            v["decisions"], _p(v["top3_share"]), v["ratings"], _p(v["vision_best_share"]), _p(v["picked_share"])),
        "  R rule     %d decisions: says %s, why not %s, rule branch %s, ratings offered %s, logged mismatch %d/%d" % (
            r["decisions"], _p(r["says_share"]), r["why_not"], r["rules"], r["ratings"], r["logged_mismatch"],
            r["logged"]),
        "  T text     follows %s, picks when told %s (%d told)" % (
            _p(tx["follows_share"]), _p(tx["pick_share"]), tx["rule_says"]),
        "  G game     per try: arm %d tries diff %+.1f %s | none %d tries diff %+.1f %s | use none %s -> arm %s | "
        "A/B %s" % (g["arm"]["tries"], g["arm"]["diff"], g["arm"]["cls"], g["none"]["tries"], g["none"]["diff"],
                    g["none"]["cls"], _p(g["use_none"]), _p(g["use_arm"]),
                    "%+.1f [%+.1f, %+.1f] %s" % (ab["mean"], ab["ci95"][0], ab["ci95"][1], ab["verdict"])
                    if ab and "mean" in ab else "n/a"),
        "  Q Qwen     %d ledgers: exact %d (in %d runs) %s %s, final %s; naming %s by kind %s; history %d tries %s" % (
            q["ledgers"], q["exact"], q["exact_runs"], q["exact_states"], q["exact_whys"], q["final_states"],
            t["move"], q["naming"], hist.get("tries", 0), hist.get("cls")),
        "  H verifier A/B %s, expected %s, verifier %s -> agree %s" % (h["ab"], h["expected"], h["states"], h["agree"]),
        "  => %s" % t["diagnosis"]] + ["     also: %s" % x for x in t["breaks"][1:]])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("log_dir")
    ap.add_argument("--opp", required=True)
    ap.add_argument("--me", default="chunli")
    ap.add_argument("--arm", help="trace this arm's lines")
    ap.add_argument("--lines", help="lines to trace, separated by ';'")
    ap.add_argument("--base", default=".", help="checkout the relative paths resolve against")
    ap.add_argument("--lock", default="lesson_loop_v1", help="the lock whose play data is her history")
    ap.add_argument("--json", help="write the full traces here")
    args = ap.parse_args()
    from sf2.system1.system1 import choices
    base = os.path.abspath(args.base)
    arms = load_arms(roots(args.log_dir, base), args.opp)
    if args.lines:
        lines = [x.strip() for x in args.lines.split(";") if x.strip()]
    elif args.arm:
        if args.arm not in arms:
            raise SystemExit("no arm %r (have %s)" % (args.arm, ", ".join(sorted(arms))))
        lines = arms[args.arm]["lines"]
    else:
        lines = list(dict.fromkeys(x for a in sorted(arms) for x in arms[a]["lines"]))
    moves = choices(args.me)
    led, hist = ledgers(base, args.opp), history(base, args.lock, args.me, args.opp)
    traces = [trace(x, moves, arms, led, hist) for x in lines]
    print("%s: %d arms, %d loop ledgers, %d history decisions\n" % (args.opp, len(arms), len(led), len(hist)))
    print("\n\n".join(show(t) for t in traces))
    if args.json:
        with open(args.json, "w") as f:
            json.dump(traces, f, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
