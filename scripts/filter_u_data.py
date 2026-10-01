"""Round 2's dataset (docs/prereg_u_round2.md): a U-data build (scripts/build_u_data.py) minus some questions. Every
jsonl is copied without those rows; frames/ is a symlink to the source's frames (no image copies).

    python scripts/filter_u_data.py --src test_data_u --out test_data_u2 \\
        --drop "How full is my health bar?" "How full is his health bar?" "Is the gap between us changing?"
"""
import argparse
import hashlib
import json
import os
import sys
from typing import Dict, List

import _path  # noqa: F401


def _sha(path: str) -> str:
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def filter_build(src: str, out: str, drop: List[str]) -> Dict:
    if os.path.exists(out):
        raise SystemExit("%s exists: refusing to overwrite" % out)
    chars = sorted(d for d in os.listdir(src) if os.path.isdir(os.path.join(src, d)))
    seen, counts = set(), {}
    plan = []
    for c in chars:
        for name in sorted(f for f in os.listdir(os.path.join(src, c)) if f.endswith(".jsonl")):
            rows = [l for l in open(os.path.join(src, c, name))]
            kept = []
            for l in rows:
                q = json.loads(l)["question"]["instructions"]
                if q in drop:
                    seen.add(q)
                else:
                    kept.append(l)
            plan.append((c, name, kept))
            counts["%s/%s" % (c, name)] = [len(rows), len(kept)]
    missing = [q for q in drop if q not in seen]
    if missing:
        raise SystemExit("no rows ask %s: refusing (check the exact wording)" % missing)
    for c in chars:
        os.makedirs(os.path.join(out, c))
        os.symlink(os.path.abspath(os.path.join(src, c, "frames")), os.path.join(out, c, "frames"))
    for c, name, kept in plan:
        with open(os.path.join(out, c, name), "w") as f:
            f.writelines(kept)
    build = os.path.join(src, "build.json")
    meta = {"src": src, "src_build_sha256": _sha(build) if os.path.exists(build) else None, "dropped": list(drop),
            "counts": counts}
    with open(os.path.join(out, "build.json"), "w") as f:
        json.dump(meta, f, indent=1, sort_keys=True)
    return meta


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--drop", nargs="+", required=True)
    args = ap.parse_args()
    meta = filter_build(args.src, args.out, args.drop)
    for k, (before, after) in sorted(meta["counts"].items()):
        print("%-40s %8d -> %8d" % (k, before, after))
    return 0


if __name__ == "__main__":
    sys.exit(main())
