"""The pre-registered in-play test of the laya-vision value fine-tune (docs/prereg_lv_value.md): System 1 alone, no
advice, Chun-Li vs 6 CPU opponents x 8 seeds x 15 rounds, the value checkpoint vs runs/all8, paired by seed.

    python scripts/value_inplay.py play --arm new --model runs/lv_value/best      # plays both arms' games (fan-out)
    python scripts/value_inplay.py play --arm old --model runs/all8/best
    python scripts/value_inplay.py report --new new --old old

Layout: rollouts/lv_inplay/<arm>/<opp>_s<seed>/chunli/{games,actions}.jsonl (scripts/play_system1.py, --memory none).
Unit: the seed (run) per opponent (sf2.eval.stats.run_level); pooled with the opponent as the unit. Success: pooled
CI above 0 and no opponent's CI entirely below 0; also reported: throws per close-range decision per arm.
"""
import argparse
import collections
import json
import os
import sys
from typing import Dict, List

import _path  # noqa: F401
from sf2.config import PORTS, VISION_JOB_GB
from sf2.data.dataset import read
from sf2.eval.runner import fan_out
from sf2.eval.stats import paired, pooled, run_level

ROOT = os.path.join("rollouts", "lv_inplay")
OPPS = ("ryu", "ken", "honda", "zangief", "guile", "dhalsim")
SEEDS = tuple(range(95001, 95009))
GAMES = 15


def _runs(root: str, arm: str) -> Dict[str, Dict[int, str]]:
    out = collections.defaultdict(dict)
    base = os.path.join(root, arm)
    for name in sorted(os.listdir(base)) if os.path.isdir(base) else []:
        opp, seed = name.rsplit("_s", 1)
        out[opp][int(seed)] = os.path.join(base, name, "chunli")
    return out


def _throw_close(dirs: List[str]) -> float:
    close = [a for d in dirs for a in read(os.path.join(d, "actions.jsonl")) if a["range"] == "close"]
    return sum(a["action"] == "throw" for a in close) / len(close) if close else None


def report(root: str, new: str, old: str) -> Dict:
    a, b = _runs(root, new), _runs(root, old)
    if set(a) != set(b) or any(set(a[o]) != set(b[o]) for o in a):
        raise ValueError("arms do not pair: %s vs %s" % ({o: sorted(s) for o, s in a.items()},
                                                      {o: sorted(s) for o, s in b.items()}))
    per_opp, by_opp = {}, {}
    for opp in sorted(a):
        runs = [paired(read(os.path.join(a[opp][s], "games.jsonl")), read(os.path.join(b[opp][s], "games.jsonl")))
                for s in sorted(a[opp])]
        per_opp[opp] = run_level(runs)
        by_opp[opp] = [x for r in runs for x in r]
    pool = pooled(by_opp)
    ok = pool.get("verdict") == "HELPS" and not any(r.get("verdict") == "HURTS" for r in per_opp.values())
    throws = {arm: {o: _throw_close(list(runs[o].values())) for o in sorted(runs)} for arm, runs in ((new, a), (old, b))}
    return {"per_opp": per_opp, "pooled": pool, "throw_close": throws, "success": ok}


def commands(arm: str, model: str, oracle=None) -> List:
    """One play_system1 job per (opponent, seed): (key of strings, argv)."""
    cmds = []
    for i, (opp, seed) in enumerate((o, s) for o in OPPS for s in SEEDS):
        out = os.path.join(ROOT, arm, "%s_s%d" % (opp, seed))
        if os.path.exists(out):
            raise SystemExit("%s exists: refusing to mix runs" % out)
        # play_system1's one-character mode: this fan-out reserves the job's memory once (its own fan-out would
        # reserve it a second time)
        cmds.append(((opp, str(seed)), [sys.executable, os.path.join("scripts", "play_system1.py"), "--one", "chunli",
                                        str(PORTS["ab"][0] + i), "--model", model, "--opp", opp, "--games",
                                        str(GAMES), "--memory", "none", "--seed", str(seed), "--out", out]
                    + (["--oracle", oracle] if oracle else [])))
    return cmds


def play(args) -> int:
    cmds = commands(args.arm, args.model, args.oracle)
    failed = fan_out(cmds, os.path.join("logs", "lv_inplay", args.arm), job_gb=VISION_JOB_GB)
    for f in failed:
        print("FAILED", f)
    return 1 if failed else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("play")
    p.add_argument("--arm", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--oracle", default=None, help="a lookup-table ranking (with --model none)")
    r = sub.add_parser("report")
    r.add_argument("--new", default="new")
    r.add_argument("--old", default="old")
    r.add_argument("--root", default=ROOT)
    args = ap.parse_args()
    if args.cmd == "play":
        return play(args)
    rep = report(args.root, args.new, args.old)
    print(json.dumps(rep, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
