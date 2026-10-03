"""Outcome-driven loop driver (step C). Wires the tested pure cores to real games.

Qwen writes a situational playbook -> text-laya plays it (routing on) -> the engine keeps the PLAYBOOK
only if it beats the incumbent on a DEV block AND replicates on a FRESH held-out block (effective
coverage = fire x follows x hp-delta). Noise control: a few candidates per round, each over enough
seeds; dev/held pools are disjoint and rotate; the TERMINAL block is reserved for one untouched test at
the end. Every candidate's result lands in the evidence ledger (out/loop/<opp>_<ts>/ledger.jsonl) - the
system-of-record, not prose.

The Coach proposer (Qwen) is pluggable; for now candidates come from a JSON file (--candidates). Use
--selfcheck to wire a full session with a FAKE measurer (no emulator) and prove the driver end-to-end.

Examples:
  # hermetic wiring check, no games:
  python scripts/outcome_loop.py --opp ken --selfcheck --candidates cand.json --dev 0-1 --held 2-3 --terminal 90-91
  # real smoke (2+2 seeds, one round, one candidate):
  python scripts/outcome_loop.py --opp ken --candidates cand.json --dev 0-1 --held 2-3 --terminal 90-91 --rounds 1
cand.json: a JSON list of {"id","rules":[...]}, or {"<round>": [ {...}, ... ]}.
"""
import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sf2.config import REPO  # noqa: E402
from sf2.system1.advice import char_menu_moves  # noqa: E402
from sf2.system2 import seed_rules  # noqa: E402
from sf2.system2.measurer import assemble  # noqa: E402
from sf2.system2.outcome_loop import Playbook, SeedBlocks, blocks_for_round, run_round, run_session  # noqa: E402
from sf2.system2.promotion import BlockStat, Cfg  # noqa: E402
from sf2.system2.rule_entry import carry_entries  # noqa: E402
from sf2.system2.screen_evidence import read_decisions  # noqa: E402

PY = os.path.join(REPO, ".venv", "bin", "python")
ROM = os.environ.get("SF2_ROM") or os.path.join(REPO, "roms", "Street Fighter II (USA).sfc")
PORTS = list(range(48901, 48917))
BOOK = os.path.join("lessons", "book.json")


def _seeds(spec):
    if "-" in spec and "," not in spec:
        a, b = spec.split("-")
        return tuple(range(int(a), int(b) + 1))
    return tuple(int(x) for x in spec.split(","))


def book_incumbent(opp, me):
    """The loop's starting playbook: the verified book-seed lines for this opponent."""
    lines = tuple(e["line"] for e in seed_rules.seed_lessons(BOOK, opp, me))
    return Playbook("book", lines)


def load_candidates(path, cap):
    """JSON list (round-0 candidates) or {round: [...]}. Returns a proposer(opp, incumbent, k)."""
    blob = json.load(open(path))
    by_round = blob if isinstance(blob, dict) else {"0": blob}

    def mk(d):
        return Playbook(d["id"], tuple(d["rules"]))

    def proposer(opp, incumbent, k):
        return [mk(d) for d in by_round.get(str(k), [])][:cap]
    return proposer


def make_measure(opp, me, cat, move, out_root, moves):
    """Real measurer: runs both arms over the seeds in parallel and assembles a BlockStat."""
    os.makedirs(out_root, exist_ok=True)

    def run_arm(rules, seed, port):
        tag = "incumbent" if rules == () else ("r%08x" % (abs(hash(rules)) & 0xFFFFFFFF))
        carry = os.path.join(out_root, "carry_%s.json" % tag)
        if not os.path.exists(carry):
            json.dump(carry_entries(list(rules), moves), open(carry, "w"), indent=1)
        out = os.path.join(out_root, "run_%s_s%d" % (tag, seed))
        cmd = [PY, "scripts/play_loop_screen.py", "--me", me, "--opp", opp, "--games", "1", "--rounds", "1",
               "--seed", str(seed), "--qwen-mode", "one", "--cat-advisor", cat, "--move-advisor", move,
               "--no-score", "--rom", ROM, "--port", str(port), "--carry", carry, "--out", out]
        subprocess.run(cmd, cwd=REPO, stdout=open(out + ".log", "w"), stderr=subprocess.STDOUT, timeout=600)
        v = json.load(open(os.path.join(out, "verdict.json")))
        g = v["games"][0]
        hp = v["game_hp"][0]
        win = 1 if g.get("won", 0) > g.get("lost", 0) else 0
        dec = read_decisions(os.path.join(out, "g00_r0"))
        return hp, win, dec

    def measure(incumbent, candidate, seeds):
        seeds = list(seeds)
        jobs = ([("base", incumbent.rules, s) for s in seeds]
                + [("cand", candidate.rules, s) for s in seeds])
        res = {}
        with ThreadPoolExecutor(max_workers=min(8, len(PORTS))) as ex:
            futs = {ex.submit(run_arm, rules, s, PORTS[i % len(PORTS)]): (arm, s)
                    for i, (arm, rules, s) in enumerate(jobs)}
            for f in futs:
                res[futs[f]] = f.result()
        base = [res[("base", s)] for s in seeds]
        cand = [res[("cand", s)] for s in seeds]
        return assemble(base, cand)
    return measure


def fake_measure(incumbent, candidate, seeds):
    """Deterministic FAKE for --selfcheck: a candidate whose id contains 'win' beats the incumbent and
    replicates; anything else is inconclusive. No emulator, no Qwen - proves the driver wiring only."""
    n = len(list(seeds))
    if "win" in candidate.id:
        return BlockStat(delta=60.0, lo=30.0, hi=90.0, cand_wins=n, n=n, fire_rate=0.9, follows=0.95)
    return BlockStat(delta=-5.0, lo=-40.0, hi=30.0, cand_wins=n // 2, n=n, fire_rate=0.6, follows=0.9)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--opp", required=True)
    ap.add_argument("--me", default="chunli")
    ap.add_argument("--candidates", required=True, help="JSON list or {round:[...]} of {id,rules}")
    ap.add_argument("--dev", default="0-5")
    ap.add_argument("--held", default="6-11")
    ap.add_argument("--terminal", default="90-95")
    ap.add_argument("--rounds", type=int, default=1)
    ap.add_argument("--max-candidates", type=int, default=3, help="noise-control cap per round")
    ap.add_argument("--selfcheck", action="store_true", help="use the fake measurer (no games)")
    ap.add_argument("--cat-advisor", default=os.path.join("runs", "text_laya", "cat_v3"))
    ap.add_argument("--move-advisor", default=os.path.join("runs", "text_laya", "move_v2"))
    args = ap.parse_args()

    # one dev block, one held block, one terminal block (rotation kicks in with more rounds/blocks)
    seeds = SeedBlocks(dev=(_seeds(args.dev),), held=(_seeds(args.held),), terminal=_seeds(args.terminal))
    moves = char_menu_moves(args.me)
    out_root = os.path.join(REPO, "out", "loop", "%s_%d" % (args.opp, int(time.time())))
    os.makedirs(out_root, exist_ok=True)
    proposer = load_candidates(args.candidates, args.max_candidates)
    measure = fake_measure if args.selfcheck else make_measure(
        args.opp, args.me, args.cat_advisor, args.move_advisor, out_root, moves)

    ledger_path = os.path.join(out_root, "ledger.jsonl")
    lf = open(ledger_path, "w")

    def ledger_write(row):
        lf.write(json.dumps(row) + "\n")
        lf.flush()

    incumbent0 = book_incumbent(args.opp, args.me)
    print("out: %s" % out_root)
    print("incumbent0 (book): %s" % list(incumbent0.rules))
    print("seeds dev=%s held=%s terminal=%s  rounds=%d  mode=%s\n"
          % (args.dev, args.held, args.terminal, args.rounds, "SELFCHECK" if args.selfcheck else "REAL"))

    final = run_session(args.opp, incumbent0, proposer, measure, seeds, args.rounds, ledger_write)
    print("\nfinal incumbent: %s\n  rules: %s" % (final.id, list(final.rules)))

    # TERMINAL untouched test: final playbook vs the book incumbent on the reserved block (once)
    if final.id != incumbent0.id:
        print("\nTERMINAL untouched test (final vs book) on seeds %s" % args.terminal)
        bs = measure(incumbent0, final, seeds.terminal)
        term = {"round": "terminal", "opp": args.opp, "candidate": final.id,
                "terminal_delta": bs.delta, "terminal_ci": [bs.lo, bs.hi],
                "terminal_wins": bs.cand_wins, "terminal_n": bs.n,
                "terminal_fire": bs.fire_rate, "terminal_follows": bs.follows}
        ledger_write(term)
        print("  delta %+.1f CI[%+.1f,%+.1f]  wins %d/%d  fire %.0f%% follows %.0f%%"
              % (bs.delta, bs.lo, bs.hi, bs.cand_wins, bs.n, 100 * bs.fire_rate, 100 * bs.follows))
    else:
        print("\n(no promotion this session; terminal test skipped)")
    lf.close()
    print("\nledger: %s" % ledger_path)


if __name__ == "__main__":
    main()
