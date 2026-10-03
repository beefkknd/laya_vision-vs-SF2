"""Score a book / loop-verdict's rules against the value lookup table (README G5), OFFLINE and OUTSIDE play.

    python scripts/score_rules.py lessons/book.json            # the verified players'-tip book, per opponent
    python scripts/score_rules.py runs/<run>/verdict.json      # a loop run's end registry (or in_play lines)
    python scripts/score_rules.py <file> --table lessons/value_oracle_v1.json --out /tmp/scores.json

The table is TESTING-only (``scripts/hard_gate.py`` keeps it out of the play path); this CLI is analysis, never
imported by a runner. Exit 0 always (it reports; it does not gate).
"""
import argparse
import json
import os
import sys
from dataclasses import asdict
from typing import Dict, List, Tuple

import _path  # noqa: F401
from sf2.config import REPO
from sf2.data.value_oracle import load as load_table
from sf2.eval.rule_score import CHAR, Rule, rule_from_claim, rule_from_line, score_rule, summarize

DEFAULT_TABLE = os.path.join("lessons", "value_oracle_v1.json")


def rules_of(doc: Dict) -> List[Tuple[str, Rule]]:
    """(group, Rule) pairs from a book.json (grouped per opponent) or a verdict.json (registry or in_play)."""
    if isinstance(doc, dict) and "opponents" in doc:
        return [(opp, rule_from_claim(line["claim"], source="book", opp=None))
                for opp, o in doc["opponents"].items() for line in o.get("lines", [])]
    if isinstance(doc, dict) and "registry_end" in doc:
        live = {"verified", "testing", "registered"}
        return [("registry", rule_from_claim(e["claim"], source="registry:" + e.get("state", "")))
                for e in doc["registry_end"] if e.get("state") in live and e.get("claim")]
    if isinstance(doc, dict) and "in_play_end" in doc:
        return [("in_play", rule_from_line(line, source="in_play")) for line in doc["in_play_end"]]
    raise SystemExit("unrecognised file: expected a book.json (opponents) or a verdict.json "
                     "(registry_end / in_play_end)")


def _cells_str(sc) -> str:
    if not sc.cells:
        return "-"
    return "; ".join("%s att=%d air=%d: %s=%+.2f (rank %d/%d, best %s=%+.2f, walk %+.2f)"
                     % (c.cell[1], c.cell[2], c.cell[3], c.move, c.value, c.rank, c.n_moves,
                        c.best_move, c.best_value, c.forward) for c in sc.cells)


def report(scored: List[Tuple[str, object]]) -> Dict:
    lines: List[str] = []
    by_group: Dict[str, List[object]] = {}
    for group, sc in scored:
        by_group.setdefault(group, []).append(sc)
        tag = "NOT_SCORABLE" if not sc.scorable else sc.verdict.upper()
        head = "[%-12s] %-9s %s" % (group, tag, sc.rule.line or sc.rule.move)
        lines.append(head)
        lines.append("    " + (sc.reason if not sc.scorable else _cells_str(sc)))
    lines.append("")
    lines.append("=== summary ===")
    groups = {}
    for group, items in by_group.items():
        s = summarize(items)
        groups[group] = s
        lines.append("%-12s rules=%d  good=%d ok=%d bad=%d not_scorable=%d"
                     % (group, s["rules"], s["good"], s["ok"], s["bad"], s["not_scorable"]))
    total = summarize([sc for _, sc in scored])
    lines.append("%-12s rules=%d  good=%d ok=%d bad=%d not_scorable=%d"
                 % ("TOTAL", total["rules"], total["good"], total["ok"], total["bad"], total["not_scorable"]))
    print("\n".join(lines))
    return {"groups": groups, "total": total,
            "rules": [{"group": g, **asdict(sc)} for g, sc in scored]}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("file", help="a lessons/book.json or a loop run's verdict.json")
    ap.add_argument("--table", default=DEFAULT_TABLE, help="the value table (default: %s)" % DEFAULT_TABLE)
    ap.add_argument("--char", default=CHAR, help="the acting character the table is keyed on (default: %s)" % CHAR)
    ap.add_argument("--out", default=None, help="write the per-rule scores + summary as JSON here")
    args = ap.parse_args(argv)

    table_path = args.table if os.path.isabs(args.table) else os.path.join(REPO, args.table)
    if not os.path.isfile(args.file):
        print("score_rules: no such file %s" % args.file, file=sys.stderr)
        return 2
    if not os.path.isfile(table_path):
        print("score_rules: no such table %s" % table_path, file=sys.stderr)
        return 2
    table = load_table(table_path)
    doc = json.load(open(args.file))
    scored = [(group, score_rule(rule, table, args.char)) for group, rule in rules_of(doc)]
    out = report(scored)
    if args.out:
        with open(args.out, "w") as f:
            json.dump(out, f, indent=1, default=str)
        print("\nwrote %s" % args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
