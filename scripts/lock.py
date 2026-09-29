"""Lock the inputs of finished lesson-loop runs (sf2.eval.lock), or check a lock.

    python scripts/lock.py make lesson_loop_v1 rollouts/qwen_lessons/20260929-1554*_* rollouts/qwen_lessons/20260929-1555*_*
    python scripts/lock.py verify lesson_loop_v1
    python scripts/qwen_lessons.py --lock lesson_loop_v1 --run 0         # repeat run 0 from the lock

``make`` takes the runs' opponent, seed, games and rounds from their logs, and locks the checkpoints they played with
(--model / --advisor, default the config's), Chun-Li's savestates against those opponents and every play log with her
decisions against them (her history). Services outside the checkout (Qwen, the Hugging Face base weights, the ROM) are
recorded in the manifest's "extra", not copied.
"""
import argparse
import hashlib
import json
import os
import sys
from typing import Dict, List

import _path  # noqa: F401
from sf2.config import DEFAULT_ROM, HF_HOME, LAYA_VISION, QWEN_MODEL, QWEN_TEMPERATURE, TEXT_LAYA
from sf2.data.dataset import read
from sf2.eval import lock
from sf2.eval.logs import load_actions, play_dirs
from sf2.eval.runner import savestate

ME = "chunli"
HF_BASES = ("aac6fef/laya-mlx", "thaitea/laya-vision-smolvlm-256m")


def run_of(d: str) -> Dict:
    """What a finished qwen_lessons run was: opponent, seed, games, rounds (from its logs), and its verdict."""
    path = os.path.join(d, "verdict.json")
    if not os.path.isfile(path):
        raise SystemExit("%s: no verdict.json (unfinished run?)" % d)
    with open(path) as f:
        v = json.load(f)
    if v.get("failed_jobs"):
        raise SystemExit("%s: failed jobs %s" % (d, v["failed_jobs"]))
    with open(os.path.join(d, "loop", "run.json")) as f:
        meta = json.load(f)
    rounds = read(os.path.join(d, "loop", "rounds.jsonl"))
    games = max(r["game"] for r in rounds) + 1
    if len(rounds) % games:
        raise SystemExit("%s: %d rounds do not split into %d games" % (d, len(rounds), games))
    ledger = read(os.path.join(d, "loop", "ledger.jsonl"), missing_ok=True)
    return {"source": d, "opp": meta["opp"], "seed": v["seed"], "games": games, "rounds": len(rounds) // games,
            "history": any(r["game"] == -1 for r in ledger), "verdict": v}


def hf_revision(repo: str) -> str:
    path = os.path.join(HF_HOME, "hub", "models--" + repo.replace("/", "--"), "refs", "main")
    return open(path).read().strip() if os.path.isfile(path) else "?"


def sha256(path: str) -> str:
    if not os.path.isfile(path):
        return "?"
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()


def make(args) -> int:
    runs = [run_of(d.rstrip("/")) for d in args.runs]
    opps = sorted({r["opp"] for r in runs})
    play: List[str] = [d for d in play_dirs() if any(a.get("me") == ME and a.get("opp") in opps
                                                       for a in load_actions(d))]
    rom = os.environ.get("SF2_ROM", DEFAULT_ROM)
    extra = {"qwen": {"model": QWEN_MODEL, "temperature": QWEN_TEMPERATURE},
             "hf_bases": {r: hf_revision(r) for r in HF_BASES}, "rom": {"path": rom, "sha256": sha256(rom)},
             "note": args.note}
    m = lock.make(args.name, args.model, args.advisor, [savestate(ME, o) for o in opps], play, runs, extra)
    print("locked %s: %d files, %d runs (%s), %d play logs" % (args.name, len(m["files"]), len(runs),
                                                               ", ".join(opps), len(play)))
    return 0


def verify(args) -> int:
    bad = lock.verify(args.name)
    extra = lock.manifest(args.name)["extra"]
    rom = extra.get("rom", {})
    if rom and sha256(rom["path"]) != rom["sha256"]:
        bad.append("rom %s: changed or missing" % rom["path"])
    bad += ["hf base %s: now %s" % (r, hf_revision(r)) for r, rev in extra.get("hf_bases", {}).items()
            if hf_revision(r) != rev]
    print("\n".join(bad) if bad else "lock %s holds" % args.name)
    return 1 if bad else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("make")
    m.add_argument("name")
    m.add_argument("runs", nargs="+", help="finished rollouts/qwen_lessons/<stamp>_<opp> dirs")
    m.add_argument("--model", default=LAYA_VISION)
    m.add_argument("--advisor", default=TEXT_LAYA)
    m.add_argument("--note", default="")
    v = sub.add_parser("verify")
    v.add_argument("name")
    args = ap.parse_args()
    return make(args) if args.cmd == "make" else verify(args)


if __name__ == "__main__":
    sys.exit(main())
