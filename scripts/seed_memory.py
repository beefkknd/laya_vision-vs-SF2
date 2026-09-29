"""Build a seed memory for a blank-start run (learn_loop --fresh NAME --seed DIR): a playbook counted by code from
every logged round of the character (sf2.code_coach, the advice that tested +32 hit points per round), and a short
memory per opponent counted from the rounds against him. Qwen takes it from there. Nothing in memory/ is touched.

    python scripts/seed_memory.py --out memory_seeds/video
    python scripts/seed_memory.py --out memory_seeds/video --short-from dhalsim=rollouts/ab/20260928-091745/memory_dhalsim_qwen.json
"""
import argparse
import json
import os

import _path  # noqa: F401
from sf2 import code_coach
from sf2.memory import check, playbook_path, short_path
from sf2.system2 import fits_laya
from sf2.vs_sweep import actions


def write(path: str, mem: dict, me: str) -> None:
    problems = check(mem, list(actions(me)))
    if problems or not fits_laya(mem):
        raise SystemExit("%s would not load: %s" % (path, problems or "fails fits_laya"))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(mem, f, indent=1)
    print("%s\n  %s" % (path, "\n  ".join(x["text"] for x in mem["lessons"])))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--char", default="chunli")
    ap.add_argument("--out", required=True)
    ap.add_argument("--short-from", action="append", default=[], metavar="OPP=FILE",
                    help="use this memory file as the short memory vs OPP instead of the counted one")
    args = ap.parse_args()
    if os.path.exists(args.out):
        raise SystemExit("%s exists: pick a new --out" % args.out)
    me, rows = args.char, code_coach.attacks(args.char)
    write(playbook_path(me, args.out), code_coach.lesson_file(me, None, "all", rows), me)
    given = dict(x.split("=", 1) for x in args.short_from)
    for opp in sorted({a["opp"] for a in rows}):
        mem = json.load(open(given[opp])) if opp in given else code_coach.lesson_file(me, opp, "short", rows)
        if mem["lessons"]:
            write(short_path(me, opp, args.out), dict(mem, opp=opp), me)


if __name__ == "__main__":
    main()
