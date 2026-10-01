"""Write an act dataset without its rare actions (sf2.data.action_common; owner: "I don't want rare move").

    python scripts/filter_rare_actions.py --data test_data_act --out test_data_act_common [--min-episodes 10]

Rarity comes from <data>/gate.json (coverage table, episodes_seen). Each opponent dir is rewritten with the same
split files; frames is the same symlink target. Writes <out>/filter.json: the threshold, the dropped (actor, code)
with their episode counts, rows in/out per split. Refuses an existing --out.
"""
import argparse
import json
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sf2.data.action_common import MIN_EPISODES, filter_rows, rare_codes  # noqa: E402

SPLITS = ("train.jsonl", "val.jsonl", "test_real.jsonl")


def _rows(path):
    with open(path) as f:
        for line in f:
            yield json.loads(line)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--min-episodes", type=int, default=MIN_EPISODES)
    a = ap.parse_args(argv)
    if os.path.exists(a.out):
        sys.exit("refusing to overwrite %s" % a.out)
    table = json.load(open(os.path.join(a.data, "gate.json")))["coverage"]["table"]
    rare = rare_codes(table, a.min_episodes)
    report = {"source": a.data, "min_episodes": a.min_episodes,
              "dropped": {"%s act%02d" % k: table["%s act%02d" % k]["episodes_seen"] for k in sorted(rare)},
              "rows": {}}
    opps = sorted(d for d in os.listdir(a.data) if os.path.isdir(os.path.join(a.data, d, "frames")))
    for opp in opps:
        src, dst = os.path.join(a.data, opp), os.path.join(a.out, opp)
        os.makedirs(dst)
        os.symlink(os.path.realpath(os.path.join(src, "frames")), os.path.join(dst, "frames"))
        for split in SPLITS:
            path = os.path.join(src, split)
            if not os.path.exists(path):
                continue
            n_in = sum(1 for _ in _rows(path))
            n_out = 0
            with open(os.path.join(dst, split), "w") as f:
                for row in filter_rows(_rows(path), rare):
                    f.write(json.dumps(row) + "\n")
                    n_out += 1
            report["rows"]["%s/%s" % (opp, split)] = {"in": n_in, "out": n_out}
    shutil.copyfile(os.path.join(a.data, "build.json"), os.path.join(a.out, "build.json"))   # the gate reads it
    with open(os.path.join(a.out, "filter.json"), "w") as f:
        json.dump(report, f, indent=1)
    print("dropped %d (actor, code) seen in < %d episodes; rows %d -> %d" % (
        len(rare), a.min_episodes, sum(v["in"] for v in report["rows"].values()),
        sum(v["out"] for v in report["rows"].values())))
    return 0


if __name__ == "__main__":
    sys.exit(main())
