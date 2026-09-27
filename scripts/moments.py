"""Pull System 2's moments out of a student rollout (sf2/moments.py), and check that "unsure" means something.

    python scripts/moments.py rollouts/<name> [--share 0.05] [--audit 0.01]

Writes out/moments/<name>.jsonl (one validated moment record per line; frame paths made absolute) and prints the
counts per reason, plus the rate of being hit in the next 0.5 s for unsure vs confident decisions. The plan's bar
(docs/TWO_SYSTEM_PLAN.md Phase 3) is at least 1.5x.
"""
import argparse
import json
import os
import random

import _path  # noqa: F401
from sf2 import contract as C
from sf2 import dataset as D
from sf2 import moments as MO


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("rollout")
    ap.add_argument("--share", type=float, default=0.05, help="share of decisions to flag as unsure")
    ap.add_argument("--audit", type=float, default=C.AUDIT_RATE)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="out/moments")
    args = ap.parse_args()
    rows = D.read(os.path.join(args.rollout, "train.jsonl"))
    for r in rows:
        r["images"] = [os.path.abspath(os.path.join(args.rollout, p)) for p in r["images"]]
    margin = MO.margin_for_share(rows, args.share)
    moments = MO.extract(rows, random.Random(args.seed), unsure_margin=margin, audit_rate=args.audit)
    name = os.path.basename(os.path.normpath(args.rollout))
    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, name + ".jsonl")
    D.write_jsonl(path, moments)

    ctl = [r for r in rows if r["meta"].get("controllable") and "student_probs" in r["meta"]]
    hit = lambda rs: sum(r["meta"]["dmg_against_next"] > 0 for r in rs) / max(1, len(rs))  # noqa: E731
    unsure = [r for r in ctl if C.confidence(r["meta"]["student_probs"])[0] < margin]
    sure = [r for r in ctl if C.confidence(r["meta"]["student_probs"])[0] >= margin]
    counts = {w: sum(m["why"] == w for m in moments) for w in C.WHY}
    ratio = hit(unsure) / hit(sure) if hit(sure) else float("inf")
    print(json.dumps({"moments": path, "decisions": len(rows), "controllable": len(ctl), "unsure_margin": margin,
                      "flagged": counts, "hit_rate_unsure": round(hit(unsure), 3),
                      "hit_rate_confident": round(hit(sure), 3), "unsure_over_confident": round(ratio, 2),
                      "unsure_predicts_damage": ratio >= 1.5}, indent=2))


if __name__ == "__main__":
    main()
