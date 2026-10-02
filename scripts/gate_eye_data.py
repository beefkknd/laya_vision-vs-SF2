"""The eye datasets' gates (sf2.data.eye_gate; docs/eye_questions_v1.md rules 5-6) and one contact sheet per question.
The exit code decides: 0 only when every gate of every dataset passes (incl. the shortcut check).

    python scripts/gate_eye_data.py [--questions q1,q3,q4,q5] [--root rollouts/pairs2p] [--sheets test_data_eye_contact]
    python scripts/gate_eye_data.py --questions q3v2,q4v2 --sheets test_data_eye_contact_v2     # questions v2

Writes <data>/gate.json per dataset and <sheets>/<q>_<name>.png.
"""
import argparse
import json
import os
import sys

import _path  # noqa: F401
from sf2.data import eye_data as E
from sf2.data import eye_gate as G
from sf2.data import eye_shortcut as SC


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--questions", default=",".join(E.V1), help="of %s" % ",".join(E.Q))
    ap.add_argument("--root", default="rollouts/pairs2p")
    ap.add_argument("--out-pattern", default=None, help="data dir per question, %%s = q (default %s)" % E.OUT)
    ap.add_argument("--margin", type=float, default=SC.MARGIN)
    ap.add_argument("--sample", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sheets", default="test_data_eye_contact", help="'none' skips them")
    args = ap.parse_args(argv)
    datas = {q: (args.out_pattern % q if args.out_pattern else E.out_of(q)) for q in args.questions.split(",")}
    rep = G.run_gates(datas, args.root, args.sample, args.seed, args.margin)
    print("%s splits %s" % ("PASS" if rep["splits"]["pass"] else "FAIL", json.dumps(rep["splits"])[:600]))
    for q, d in datas.items():
        r = rep["datasets"][q]
        with open(os.path.join(d, "gate.json"), "w") as fh:
            json.dump(dict(r, splits=rep["splits"]), fh, indent=1)
        for name in G.GATES:
            g = r["gates"][name]
            print("%s %s %s %s" % ("PASS" if g["pass"] else "FAIL", q, name,
                                   json.dumps({k: v for k, v in g.items() if k != "pass"})[:700]))
        if args.sheets != "none":
            print("contact sheet:", G.contact_sheet(q, d, os.path.join(args.sheets, "%s_%s.png" % (q, E.Q[q]["name"])),
                                                    seed=args.seed))
    print("eye gate: %s" % ("PASS" if rep["pass"] else "FAIL"))
    return 0 if rep["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
