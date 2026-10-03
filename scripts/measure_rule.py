"""Outcome measurement engine (Step 2, 2026-10-03). Does a candidate rule / playbook change in-game outcome?

INDEPENDENT sampling across seeds (pairing was shown not to reduce variance - the arms' per-seed correlation went
negative), hp-margin as the primary metric with win-rate reported alongside, and a Welch two-sample 95% CI on the
means. Two uses:
  * HYPOTHESIS mode (default): a rule is interesting only if its hp-margin delta CI excludes 0 on DEV seeds, THEN
    replicates on untouched HELD-OUT seeds. This guards against adaptive overfitting (a dev-significant effect that
    does not survive held-out is noise - seen with "always walk_forward" vs Honda).
  * TRUSTED mode (--trusted): the candidate comes from a game guide / frame data, so we do NOT need to prove the
    tactic (it is known-good); we measure it to check our PIPELINE can EXECUTE it. A known-good move that shows no
    effect exposes a blocked channel quickly (how walk_back exposed the movement gap). No held-out; just the mean.

No Qwen, no RAM: games run through scripts/play_loop_screen.py with --qwen-mode one --games 1 --rounds 1 (fixed policy,
no learning) and --no-score (round result + hp from the health bars). One savestate => the result is conditional on
that start.

Examples:
  # single added rule, hypothesis test (dev -> held-out):
  python scripts/measure_rule.py --opp honda --add "use more c.mk far away when he jumps" --dev 0-7 --held 8-15
  # a trusted game-guide playbook (replaces the book seed), just report the mean:
  python scripts/measure_rule.py --opp honda --trusted --rules \
    "avoid forward at mid range" "use more block_low at mid range when he attacks" \
    "use more c.mk far away when he jumps" --dev 0-11
"""
import argparse
import hashlib
import json
import math
import os
import statistics as st
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sf2.config import REPO  # noqa: E402
from sf2.system1.advice import char_menu_moves  # noqa: E402
from sf2.system2 import lessons as L, seed_rules  # noqa: E402
from sf2.system2.rule_entry import claim_of, entry as _entry  # noqa: E402  (shared carry shape)

PY = os.path.join(REPO, ".venv", "bin", "python")
ROM = os.environ.get("SF2_ROM") or os.path.join(REPO, "roms", "Street Fighter II (USA).sfc")
OUT = os.path.join(REPO, "out", "measure")
PORTS = list(range(48901, 48917))
BOOK = os.path.join("lessons", "book.json")


def build_carries(opp, me, rules, add):
    moves = char_menu_moves(me)
    base = list(seed_rules.seed_lessons(BOOK, opp, me))
    cand = (list(base) if add else []) + [_entry(claim_of(r, moves)) for r in rules]
    tag = hashlib.sha1((("add:" if add else "repl:") + ";".join(rules)).encode()).hexdigest()[:8]  # per-candidate dir (no overwrite)
    d = os.path.join(OUT, "%s_%s_%s" % (me, opp, tag))
    os.makedirs(d, exist_ok=True)
    bp, cp = os.path.join(d, "baseline.json"), os.path.join(d, "candidate.json")
    json.dump(base, open(bp, "w"), indent=1)
    json.dump(cand, open(cp, "w"), indent=1)
    return bp, cp, L.in_play(base), L.in_play(cand), d


def _run(opp, me, seed, carry, port, tag, out_dir, cat, move):
    out = os.path.join(out_dir, "%s_s%d" % (tag, seed))
    cmd = [PY, "scripts/play_loop_screen.py", "--me", me, "--opp", opp, "--games", "1", "--rounds", "1",
           "--seed", str(seed), "--qwen-mode", "one", "--cat-advisor", cat, "--move-advisor", move,
           "--no-score", "--rom", ROM, "--port", str(port), "--carry", carry, "--out", out]
    subprocess.run(cmd, cwd=REPO, stdout=open(out + ".log", "w"), stderr=subprocess.STDOUT, timeout=600)
    v = json.load(open(os.path.join(out, "verdict.json")))
    g = v["games"][0]
    return v["game_hp"][0], (1 if g.get("won", 0) > g.get("lost", 0) else 0)


def sample(opp, me, bp, cp, seeds, out_dir, cat, move):
    jobs = []
    for i, s in enumerate(seeds):
        jobs.append((s, "base", bp, PORTS[(2 * i) % len(PORTS)]))
        jobs.append((s, "cand", cp, PORTS[(2 * i + 1) % len(PORTS)]))
    res = {}
    with ThreadPoolExecutor(max_workers=8) as ex:
        futs = {ex.submit(_run, opp, me, s, carry, port, tag, out_dir, cat, move): (s, tag)
                for (s, tag, carry, port) in jobs}
        for f in futs:
            res[futs[f]] = f.result()
    bh = [res[(s, "base")][0] for s in seeds]
    ch = [res[(s, "cand")][0] for s in seeds]
    bw = sum(res[(s, "base")][1] for s in seeds)
    cw = sum(res[(s, "cand")][1] for s in seeds)
    return bh, ch, bw, cw


def welch(a, b):
    na, nb = len(a), len(b)
    ma, mb = st.mean(a), st.mean(b)
    va, vb = (st.variance(a) if na > 1 else 0.0), (st.variance(b) if nb > 1 else 0.0)
    se = math.sqrt(va / na + vb / nb)
    d = mb - ma
    return d, d - 1.96 * se, d + 1.96 * se


def report(tag, opp, bh, ch, bw, cw):
    n = len(bh)
    d, lo, hi = welch(bh, ch)
    sig = lo > 0 or hi < 0
    print("[%s] %s  n=%d  baseline: win %d/%d hp %+.1f   candidate: win %d/%d hp %+.1f"
          % (tag, opp, n, bw, n, st.mean(bh), cw, n, st.mean(ch)))
    print("       hp delta (cand-base) = %+.1f  95%%CI [%+.1f, %+.1f]  %s"
          % (d, lo, hi, "SIGNIFICANT" if sig else "inconclusive (CI spans 0)"))
    return d, lo, hi, sig


def _seeds(spec):
    if "-" in spec and "," not in spec:
        a, b = spec.split("-")
        return list(range(int(a), int(b) + 1))
    return [int(x) for x in spec.split(",")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--opp", required=True)
    ap.add_argument("--me", default="chunli")
    ap.add_argument("--rules", nargs="+", help="candidate in-play advice lines (replace the book seed unless --add)")
    ap.add_argument("--add", action="store_true", help="candidate = book seed PLUS --rules (default: --rules replace it)")
    ap.add_argument("--dev", default="0-7", help="seed set, e.g. 0-7 or 0,1,2,3")
    ap.add_argument("--held", default=None, help="held-out seed set; confirmed only if dev is significant")
    ap.add_argument("--trusted", action="store_true", help="game-guide candidate: skip held-out, just report the mean")
    ap.add_argument("--cat-advisor", default=os.path.join("runs", "text_laya", "cat_v3"))
    ap.add_argument("--move-advisor", default=os.path.join("runs", "text_laya", "move_v2"))
    args = ap.parse_args()
    if not args.rules:
        raise SystemExit("give --rules (one or more advice lines)")
    bp, cp, bl, cl, out_dir = build_carries(args.opp, args.me, args.rules, args.add)
    print("baseline in_play : %s" % bl)
    print("candidate in_play: %s" % cl)
    print("mode: %s\n" % ("TRUSTED (game-guide; pipeline/channel check)" if args.trusted else "HYPOTHESIS (dev -> held-out)"))
    dev = _seeds(args.dev)
    bh, ch, bw, cw = sample(args.opp, args.me, bp, cp, dev, out_dir, args.cat_advisor, args.move_advisor)
    d, lo, hi, sig = report("DEV", args.opp, bh, ch, bw, cw)
    if args.trusted:
        print("\n-> trusted candidate: effect above is the measurement (no held-out). "
              "A known-good move with ~0 effect suggests a blocked channel.")
    elif args.held and sig:
        print("\n-> DEV significant; confirming on HELD-OUT")
        hb, hc, hbw, hcw = sample(args.opp, args.me, bp, cp, _seeds(args.held), out_dir, args.cat_advisor, args.move_advisor)
        report("HELD", args.opp, hb, hc, hbw, hcw)
    elif args.held:
        print("\n-> DEV inconclusive; skipping held-out (would be fishing)")


if __name__ == "__main__":
    main()
