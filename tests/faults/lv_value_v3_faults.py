"""Seeded faults for the value fine-tune's second-run flags (2026-09-30: --no-cap, --forward-value-cap, --balance
sampling, --select nll, the value xent log, --resume-guard): each mutates one line of the code, runs the tests that
must catch it, and restores the file. Every fault must turn the tests red; exit 1 if one stays green.

    python tests/faults/lv_value_v3_faults.py
"""
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TESTS = ["tests/test_value_data_v3.py", "tests/test_value_data.py", "tests/test_train_v3.py",
         "tests/test_train_coverage.py"]
VD, TD, TS, TR = "sf2/data/value_data.py", "sf2/data/train_data.py", "sf2/data/train_select.py", "scripts/train.py"
FAULTS = [
    ("default changed: the build runs without the cap", VD,
     "min_value_train: int = MIN_VALUE_TRAIN, no_cap: bool = False,",
     "min_value_train: int = MIN_VALUE_TRAIN, no_cap: bool = True,"),
    ("--no-cap ignored: only the capped games kept", VD,
     "keep = set(games) if no_cap else keep_games(games, cap) if cap is not None else set()",
     "keep = keep_games(games, cap) if cap is not None else set()"),
    ("--no-cap still writes test_extra", VD,
     "OLD_FILES + (() if no_cap else (EXTRA_FILE,))", "OLD_FILES + (EXTRA_FILE,)"),
    ("forward cap: the mirror keeps its value row", VD,
     'files["train"] += [r, m] if r["id"] in drop_value', 'files["train"] += [r, m, value_row(m)] if r["id"] in drop_value'),
    ("forward cap: outcome rows dropped too", VD,
     'files["train"] += [r, m] if r["id"] in drop_value', 'files["train"] += [] if r["id"] in drop_value'),
    ("forward cap: median over attacks that have rows only", VD,
     "[counts.get(a, 0) for a in attacks]", "[counts[a] for a in attacks if counts.get(a)]"),
    ("forward cap: random subsample instead of the id hash", VD,
     'ranked = sorted(ids, key=lambda i: hashlib.sha256((FORWARD_HASH_SALT + i).encode()).hexdigest())',
     'ranked = __import__("random").sample(list(ids), len(ids))'),
    ("sampling check ignores the trainer's weights", TD,
     "return vt.mix_probabilities(groups, MIX_WEIGHTS, MIX_ALPHA)", "return vt.mix_probabilities(groups)"),
    ("sampling mode still applies the row-count share check", TD,
     "if share_check and present and", "if present and"),
    ("train.py passes the trainer another mix than the one checked", TR,
     "mix_alpha=TD.MIX_ALPHA)", "mix_alpha=0.5)"),
    ("select nll keeps the highest NLL", TS,
     "else value < best[select]", "else value > best[select]"),
    ("train.py ignores --select", TR,
     "Selection(evaluate, save_best, args.select, args.patience)",
     'Selection(evaluate, save_best, "acc", args.patience)'),
    ("value xent logged without its prior", TS,
     '"prior_xent": v.get("prior_xent")}', '"prior_xent": None}'),
    ("resume guard off by default", TR,
     "action=argparse.BooleanOptionalAction, default=True,", "action=argparse.BooleanOptionalAction, default=False,"),
    ("resume guard never refuses", TS,
     "if guard and os.path.exists(out):", "if guard and False:"),
]


def run(fault) -> bool:
    name, path, old, new = fault
    full = os.path.join(ROOT, path)
    with open(full) as f:
        src = f.read()
    if src.count(old) != 1:
        raise SystemExit("fault %r: the line to mutate is not found once in %s" % (name, path))
    try:
        with open(full, "w") as f:
            f.write(src.replace(old, new))
        r = subprocess.run([sys.executable, "-m", "pytest", "-q", "-x", "-p", "no:cacheprovider"] + TESTS, cwd=ROOT,
                           capture_output=True, text=True)
    finally:
        with open(full, "w") as f:
            f.write(src)
    return r.returncode != 0


def main() -> int:
    missed = 0
    for fault in FAULTS:
        red = run(fault)
        missed += not red
        print("%-5s %s" % ("RED" if red else "GREEN", fault[0]))
    print("%d of %d faults caught" % (len(FAULTS) - missed, len(FAULTS)))
    return 1 if missed else 0


if __name__ == "__main__":
    sys.exit(main())
