"""Build the CPU Chun-Li act set (sf2.data.action_data_cpu; docs/prereg_movement_data.md, "Addition: CPU Chun-Li
games") from the collection scripts/collect_cpu_chunli.py wrote:

    python scripts/build_action_data_cpu.py [--root rollouts/mv4] [--out test_data_act_cpu] [--workers 7]

Writes <out>/<p1 char>/{train,val,test_real}.jsonl, frames -> <root>/<p1 char>/images (symlink), <out>/build.json and
<out>/summary.json. Exit 1 on any problem (a pair that is not Chun-Li's, a provenance that is not its dir's, a
collector label the RAM does not give, a missing image, a duplicate row id, a question longer than laya's head).
Then: scripts/gate_action_data.py --data <out> --compare-act test_data_act.
"""
import argparse
import json
import os
import sys

import _path  # noqa: F401

from sf2.data import action_data_cpu as DC
from sf2.data.action_data import TOKENIZER


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=os.path.join("rollouts", "mv4"))
    ap.add_argument("--out", default=DC.OUT)
    ap.add_argument("--p1", help="only these player-1 characters (comma-separated)")
    ap.add_argument("--workers", type=int, default=7)
    ap.add_argument("--tokenizer", default=TOKENIZER, help="laya-vision processor dir ('none' skips the check)")
    args = ap.parse_args(argv)
    tok = None
    if args.tokenizer != "none":
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(args.tokenizer)
    res = DC.build(args.root, args.out, args.p1.split(",") if args.p1 else None, args.workers, tokenizer=tok)
    for p1, s in sorted(res["stats"].items()):
        print("   %-8s rows %s" % (p1, s["rows"]))
    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump({"stats": res["stats"], "label_stats": res["label_stats"], "codes": res["codes"],
                   "head_tokens": res["head_tokens"], "problems": res["problems"][:200]}, f, indent=1)
    print("Chun-Li codes:", res["codes"])
    print("head tokens:", res["head_tokens"])
    print("label stats:", res["label_stats"])
    for p in res["problems"][:40]:
        print("PROBLEM:", p)
    if res["problems"]:
        print("%d problems" % len(res["problems"]))
    return 1 if res["problems"] else 0


if __name__ == "__main__":
    sys.exit(main())
