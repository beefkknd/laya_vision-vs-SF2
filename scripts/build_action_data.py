"""Build the three movement datasets (act / where / dist; sf2.data.action_data, docs/prereg_movement_data.md "Owner
decisions on the label") from the action collection (scripts/collect_actions.py):

    python scripts/build_action_data.py [--root rollouts/mv3] [--act test_data_act] [--where test_data_where]
                                        [--dist test_data_dist] [--workers 7]

Writes <out>/<opp>/{train,val,test_real}.jsonl, frames -> <root>/<opp>/images (symlink), <out>/build.json and
<out>/summary.json for each dataset. Exit 1 on any problem (a pair whose collector label the RAM does not give, a
missing image, a duplicate row id, a question longer than laya's head). Then: scripts/gate_action_data.py.
"""
import argparse
import json
import os
import sys

import _path  # noqa: F401

from sf2.data import action_data as D


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=os.path.join("rollouts", "mv3"))
    for ds in D.DATASETS:
        ap.add_argument("--" + ds, default=D.OUTS[ds])
    ap.add_argument("--opps", help="only these opponents (comma-separated)")
    ap.add_argument("--workers", type=int, default=7)
    ap.add_argument("--tokenizer", default=D.TOKENIZER, help="laya-vision processor dir ('none' skips the check)")
    args = ap.parse_args(argv)
    tok = None
    if args.tokenizer != "none":
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(args.tokenizer)
    outs = {ds: getattr(args, ds) for ds in D.DATASETS}
    res = D.build(args.root, outs, args.opps.split(",") if args.opps else None, args.workers, tokenizer=tok)
    for ds, out in outs.items():
        print("== %s (%s)" % (ds, out))
        for opp, s in sorted(res["stats"][ds].items()):
            print("   %-8s rows %s" % (opp, s["rows"]))
        with open(os.path.join(out, "summary.json"), "w") as f:
            json.dump({"stats": res["stats"][ds], "label_stats": res["label_stats"], "options": res["options"],
                       "head_tokens": res["head_tokens"], "problems": res["problems"][:200]}, f, indent=1)
    print("options per actor:", {a: len(c) for a, c in res["options"].items()})
    print("head tokens:", res["head_tokens"])
    print("label stats:", res["label_stats"])
    for p in res["problems"][:40]:
        print("PROBLEM:", p)
    if res["problems"]:
        print("%d problems" % len(res["problems"]))
    return 1 if res["problems"] else 0


if __name__ == "__main__":
    sys.exit(main())
