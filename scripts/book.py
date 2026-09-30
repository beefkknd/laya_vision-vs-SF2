"""The book of verified players' tips (lessons/book.json), per opponent, for Qwen's lesson loop
(scripts/qwen_lessons.py --book; sf2.system2.lessons.from_book).

A tip is verified when its SINGLE-line arm in a pre-registered fixed-advice batch (docs/prereg_*_expert.md,
docs/prereg_new_opponents.md; scripts/fixed_arms_report.py over logs/ab/<opp>_expert) helps vs no advice: the run-level
95% interval above 0 (the seed as the unit). A multi-line set is not a tip; a line naming forward is left out (System 1
cannot follow it: text laya untrained, docs/component_boundaries.md). Each line keeps its A/B mean, interval, seeds
and batch; the single-line arms that did not help are listed with why. Deterministic: the same batches, the same
bytes (``check`` rebuilds and compares).

    python scripts/book.py build [--data DIR] [--out lessons/book.json]
    python scripts/book.py check [--data DIR] [--out lessons/book.json]

--data: the checkout the batches ran in (their logs name runs relative to it; default: here).
"""
import argparse
import json
import os
import sys
from typing import Dict, List

import _path  # noqa: F401
from fixed_arms_report import report, roots
from sf2.system1.advice import parse
from sf2.system1.system1 import choices
from sf2.system2 import lessons as L

ME = "chunli"
OUT = os.path.join("lessons", "book.json")
BATCHES = {o: "logs/ab/%s_expert" % o for o in ("ryu", "ken", "honda", "zangief", "guile", "dhalsim")}
KIND = {"soft": "use_more", "hard": "always", "neg": "avoid"}


def claim(line: str) -> Dict:
    """A line as a registry claim; it must render back to itself (the grammar text laya reads)."""
    les = parse(line, choices(ME))
    if les.polarity not in KIND:
        raise SystemExit("book line %r names none of her moves" % line)
    c = {"kind": KIND[les.polarity], "move": les.move, "range": les.where, "when": les.when}
    if L.render(c) != line:
        raise SystemExit("book line %r does not render back (%r)" % (line, L.render(c)))
    return c


def arm_lines(runs: List[str], opp: str, arm: str) -> List[str]:
    """The lines an arm played with, the same in every seed's run (memory_<opp>_<arm>.json)."""
    seen = set()
    for r in runs:
        with open(os.path.join(r, "memory_%s_%s.json" % (opp, arm))) as f:
            seen.add(tuple(x["text"] for x in json.load(f)["lessons"]))
    if len(seen) != 1:
        raise SystemExit("%s %s: the arm's lines differ between seeds: %s" % (opp, arm, sorted(seen)))
    return list(seen.pop())


def seeds(runs: List[str]) -> List[int]:
    out = []
    for r in runs:
        with open(os.path.join(r, "run.json")) as f:
            out.append(json.load(f)["seed"])
    return sorted(out)


def _why(v: Dict) -> str:
    return {"HURTS": "hurts", "NOT SHOWN": "not shown"}.get(v["verdict"], v["verdict"].lower())


def opponent(data: str, opp: str, batch: str) -> Dict:
    runs = roots(batch, data)
    r = report(batch, opp, [], data)
    ss = seeds(runs)
    lines, rest = [], []
    for arm, v in sorted(r["vs_none"].items()):
        got = arm_lines(runs, opp, arm)
        ab = {"arm": arm, "mean": round(v["mean"], 2), "ci95": [round(x, 2) for x in v["ci95"]], "runs": v["runs"]}
        if len(got) != 1:
            rest.append(dict(ab, line="; ".join(got), why="%d lines: a set, not a tip" % len(got)))
            continue
        c = claim(got[0])
        if c["move"] in L.UNFOLLOWABLE:
            rest.append(dict(ab, line=got[0], why="names %s: %s" % (c["move"], L.UNFOLLOWABLE_WHY % c["move"])))
        elif v["verdict"] == "HELPS" and v["ci95"][0] > 0:
            lines.append(dict(ab, line=got[0], claim=c, seeds=ss, batch=batch))
        else:
            rest.append(dict(ab, line=got[0], why="%s vs no advice" % _why(v)))
    key = lambda x: (-x["mean"], x["arm"])                         # noqa: E731
    return {"batch": batch, "seeds": ss, "rounds_per_arm": r["rounds_per_arm"], "lines": sorted(lines, key=key),
            "not_verified": sorted(rest, key=key)}


def build(data: str, batches: Dict[str, str] = None) -> Dict:
    batches = BATCHES if batches is None else batches
    return {"me": ME, "made_by": "scripts/book.py build",
            "rule": "a single-line fixed-advice arm that helps vs no advice: run-level 95% interval above 0, the seed "
                    "as the unit (scripts/fixed_arms_report.py); lines naming forward left out",
            "opponents": {o: opponent(data, o, b) for o, b in sorted(batches.items())}}


def dumps(doc: Dict) -> str:
    return json.dumps(doc, indent=1, sort_keys=True) + "\n"


def write(doc: Dict, out: str) -> None:
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out + ".tmp", "w") as f:
        f.write(dumps(doc))
    os.replace(out + ".tmp", out)


def check(data: str, batches: Dict[str, str], out: str) -> List[str]:
    """What differs between ``out`` and a rebuild (empty: the file is exactly what the command builds)."""
    with open(out) as f:
        have = f.read()
    want = dumps(build(data, batches))
    if have == want:
        return []
    old, new = json.loads(have), json.loads(want)
    diff = [o for o in sorted(set(old.get("opponents", {})) | set(new["opponents"]))
            if old.get("opponents", {}).get(o) != new["opponents"].get(o)]
    return ["%s: differs from the rebuild" % o for o in diff] or ["the file's bytes differ from the rebuild"]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=("build", "check"))
    ap.add_argument("--data", default=".", help="the checkout the batches ran in")
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()
    if args.cmd == "check":
        bad = check(args.data, BATCHES, args.out)
        print("\n".join(bad) or "%s is what scripts/book.py build makes" % args.out)
        return 1 if bad else 0
    doc = build(args.data)
    write(doc, args.out)
    for o, d in doc["opponents"].items():
        print("%-8s %s" % (o, "; ".join("%s (%+.1f [%+.1f, %+.1f])" % (x["line"], x["mean"], *x["ci95"])
                                        for x in d["lines"]) or "(none)"))
    print("saved", args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
