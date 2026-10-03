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
from sf2.system2.coverage import coverage  # noqa: E402
from sf2.system2.measurer import assemble  # noqa: E402
from sf2.system2.outcome_loop import Playbook, SeedBlocks, blocks_for_round, run_round, run_session  # noqa: E402
from sf2.system2.promotion import BlockStat, Cfg  # noqa: E402
from sf2.system2.seq_loop import SeqSeeds, run_opponent  # noqa: E402
from sf2.system2.sequential import Block, SeqCfg  # noqa: E402
from sf2.system2.rule_entry import candidate_rules, carry_entries  # noqa: E402
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


def make_coach_proposer(opp, me, cat, move, out_root, moves, scout_games=2, scout_seed0=700, cap=1):
    """The live Coach as the loop's proposer: play a couple of SCOUT games with the current incumbent
    (routing on, qwen-mode two = Scout+Coach) on seeds DISJOINT from dev/held/terminal, then read the
    Coach's kept claims from the trace and build ONE candidate playbook = incumbent + those claims
    (noise control: cap candidates). The scout games are research only - the outcome engine still owns
    acceptance (dev + held + decide). Qwen server via $SF2_QWEN_URL."""
    def proposer(opp_, incumbent, k):
        carry = os.path.join(out_root, "coach_incumbent_r%d.json" % k)
        json.dump(carry_entries(list(incumbent.rules), moves), open(carry, "w"), indent=1)
        out = os.path.join(out_root, "scout_r%d" % k)
        seed = scout_seed0 + k  # research seeds, kept away from dev/held/terminal
        cmd = [PY, "scripts/play_loop_screen.py", "--me", me, "--opp", opp, "--games", str(scout_games),
               "--rounds", "1", "--seed", str(seed), "--qwen-mode", "two", "--cat-advisor", cat,
               "--move-advisor", move, "--no-score", "--rom", ROM, "--port", str(PORTS[-1]),
               "--carry", carry, "--out", out]
        subprocess.run(cmd, cwd=REPO, stdout=open(out + ".log", "w"), stderr=subprocess.STDOUT, timeout=1200)
        rows = [json.loads(l) for l in open(os.path.join(out, "trace.jsonl"))]
        qwen_events = [r for r in rows if r.get("event") == "qwen" and (r.get("claims"))]
        if not qwen_events:
            return []
        claims = qwen_events[-1]["claims"]  # the last (most-informed) game's kept claims
        cand = candidate_rules(incumbent.rules, claims, moves)
        if cand == incumbent.rules:
            return []
        label = "coach_r%d" % k
        print("  [coach] r%d proposed %d claim(s): %s" % (k, len(claims), [c.get("move") for c in claims]))
        return [Playbook(label, cand)][:cap]
    return proposer


def _make_run_arm(opp, me, cat, move, out_root, moves):
    """Returns run_arm(rules, seed, port) -> (hp, win, decisions): one game, routing on, no Qwen
    (qwen-mode one). Shared by every real measurer below."""
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
    return run_arm


def _parallel(run_arm, jobs, workers):
    """jobs: [(key, rules, seed)] -> {key: (hp, win, decisions)}, run <= workers at a time."""
    res = {}
    with ThreadPoolExecutor(max_workers=min(workers, len(PORTS))) as ex:
        futs = {ex.submit(run_arm, rules, s, PORTS[i % len(PORTS)]): key
                for i, (key, rules, s) in enumerate(jobs)}
        for f in futs:
            res[futs[f]] = f.result()
    return res


def make_measure(opp, me, cat, move, out_root, moves, workers=6):
    """BlockStat measurer for the decide-engine: both arms over the seeds, assembled."""
    run_arm = _make_run_arm(opp, me, cat, move, out_root, moves)

    def measure(incumbent, candidate, seeds):
        seeds = list(seeds)
        jobs = ([(("base", s), incumbent.rules, s) for s in seeds]
                + [(("cand", s), candidate.rules, s) for s in seeds])
        res = _parallel(run_arm, jobs, workers)
        base = [res[("base", s)] for s in seeds]
        cand = [res[("cand", s)] for s in seeds]
        return assemble(base, cand)
    return measure


def make_measure_block(opp, me, cat, move, out_root, moves, workers=6):
    """Option-D measurer: both arms over the seeds -> a sequential.Block (per-seed hp arrays + wins +
    the candidate's pooled fire/follows), so confirm blocks can be pooled."""
    run_arm = _make_run_arm(opp, me, cat, move, out_root, moves)

    def measure_block(incumbent, candidate, seeds):
        seeds = list(seeds)
        jobs = ([(("base", s), incumbent.rules, s) for s in seeds]
                + [(("cand", s), candidate.rules, s) for s in seeds])
        res = _parallel(run_arm, jobs, workers)
        base = [res[("base", s)] for s in seeds]
        cand = [res[("cand", s)] for s in seeds]
        fire, follows = coverage([d for c in cand for d in c[2]])
        return Block(inc_hp=tuple(b[0] for b in base), cand_hp=tuple(c[0] for c in cand),
                     inc_wins=sum(b[1] for b in base), cand_wins=sum(c[1] for c in cand),
                     fire_rate=fire, follows=follows)
    return measure_block


def make_measure_wins(opp, me, cat, move, out_root, moves, workers=6):
    """Measure one playbook's own round win-rate over the seeds (the solved/freeze gate)."""
    run_arm = _make_run_arm(opp, me, cat, move, out_root, moves)

    def measure_wins(playbook, seeds):
        seeds = list(seeds)
        res = _parallel(run_arm, [((s,), playbook.rules, s) for s in seeds], workers)
        return sum(res[(s,)][1] for s in seeds), len(seeds)
    return measure_wins


def fake_measure(incumbent, candidate, seeds):
    """Deterministic FAKE for --selfcheck: a candidate whose id contains 'win' beats the incumbent and
    replicates; anything else is inconclusive. No emulator, no Qwen - proves the driver wiring only."""
    n = len(list(seeds))
    if "win" in candidate.id:
        return BlockStat(delta=60.0, lo=30.0, hi=90.0, cand_wins=n, n=n, fire_rate=0.9, follows=0.95)
    return BlockStat(delta=-5.0, lo=-40.0, hi=30.0, cand_wins=n // 2, n=n, fire_rate=0.6, follows=0.9)


def _blocks(spec, size):
    """A seed spec sliced into fresh blocks of `size`: '0-23' size 12 -> ((0..11),(12..23))."""
    xs = list(_seeds(spec))
    return tuple(tuple(xs[i:i + size]) for i in range(0, len(xs), size) if len(xs[i:i + size]) == size)


def _ledger(out_root):
    lf = open(os.path.join(out_root, "ledger.jsonl"), "w")

    def write(row):
        lf.write(json.dumps(row) + "\n")
        lf.flush()
    return lf, write


def run_decide(args, out_root, moves, proposer, incumbent0):
    """The original dev + held-out decide engine (kept for comparison)."""
    seeds = SeedBlocks(dev=(_seeds(args.dev),), held=(_seeds(args.held),), terminal=_seeds(args.terminal))
    measure = fake_measure if args.selfcheck else make_measure(
        args.opp, args.me, args.cat_advisor, args.move_advisor, out_root, moves, workers=args.workers)
    lf, ledger_write = _ledger(out_root)
    print("engine=decide  dev=%s held=%s terminal=%s rounds=%d\n" % (args.dev, args.held, args.terminal, args.rounds))
    final = run_session(args.opp, incumbent0, proposer, measure, seeds, args.rounds, ledger_write)
    print("\nfinal incumbent: %s\n  rules: %s" % (final.id, list(final.rules)))
    if final.id != incumbent0.id:
        bs = measure(incumbent0, final, seeds.terminal)
        ledger_write({"round": "terminal", "opp": args.opp, "candidate": final.id,
                      "terminal_delta": bs.delta, "terminal_ci": [bs.lo, bs.hi], "terminal_wins": bs.cand_wins,
                      "terminal_n": bs.n, "terminal_fire": bs.fire_rate, "terminal_follows": bs.follows})
        print("TERMINAL delta %+.1f CI[%+.1f,%+.1f] wins %d/%d" % (bs.delta, bs.lo, bs.hi, bs.cand_wins, bs.n))
    else:
        print("\n(no promotion; terminal skipped)")
    lf.close()


def run_seq(args, out_root, moves, proposer, incumbent0):
    """Option D: win-based solved stop + screen -> sequential powered confirm."""
    pool = _blocks(args.pool, args.block_size)
    seeds = SeqSeeds(solved=_seeds(args.solved), pool=pool, terminal=_seeds(args.terminal))
    cfg = SeqCfg(win_target=args.win_target, max_looks=args.max_looks)
    mb = make_measure_block(args.opp, args.me, args.cat_advisor, args.move_advisor, out_root, moves, workers=args.workers)
    mw = make_measure_wins(args.opp, args.me, args.cat_advisor, args.move_advisor, out_root, moves, workers=args.workers)
    lf, ledger_write = _ledger(out_root)
    print("engine=seq  solved=%s pool=%d blocks of %d  win_target=%.2f max_looks=%d rounds=%d\n"
          % (args.solved, len(pool), args.block_size, cfg.win_target, cfg.max_looks, args.rounds))
    res = run_opponent(args.opp, incumbent0, proposer, mb, mw, seeds, args.rounds, cfg)
    for row in res.rows:
        ledger_write(row)
    print("\nstatus: %s   final incumbent: %s\n  rules: %s\n  confirm looks used: %d"
          % (res.status, res.final.id, list(res.final.rules), res.looks_used))
    # terminal rollback check: final vs book on the reserved untouched block (report only)
    if res.final.id != incumbent0.id:
        tb = mb(incumbent0, res.final, seeds.terminal)
        from sf2.system2.sequential import _welch
        d, se = _welch(tb.inc_hp, tb.cand_hp)
        z = d / se if se else 0.0
        ledger_write({"stage": "terminal", "opp": args.opp, "candidate": res.final.id,
                      "terminal_delta": round(d, 1), "terminal_z": round(z, 2),
                      "terminal_cand_wins": tb.cand_wins, "terminal_inc_wins": tb.inc_wins, "n": len(tb.cand_hp)})
        print("TERMINAL (rollback check) delta %+.1f z %.2f wins cand %d vs inc %d%s"
              % (d, z, tb.cand_wins, tb.inc_wins, "  <- ROLLBACK (z<=-2)" if z <= -2 else ""))
    lf.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--opp", required=True)
    ap.add_argument("--me", default="chunli")
    ap.add_argument("--engine", choices=("seq", "decide"), default="seq", help="seq = option D (default)")
    ap.add_argument("--candidates", help="JSON list or {round:[...]} of {id,rules} (offline/stub proposer)")
    ap.add_argument("--coach", action="store_true", help="use the live Qwen Coach as the proposer (needs $SF2_QWEN_URL)")
    ap.add_argument("--scout-games", type=int, default=2, help="scout games per round for the Coach proposer")
    ap.add_argument("--workers", type=int, default=6, help="parallel emulator+MLX jobs (GPU/memory headroom)")
    # seq engine seeds
    ap.add_argument("--solved", default="200-211", help="seq: block to measure the incumbent's win-rate")
    ap.add_argument("--pool", default="0-95", help="seq: fresh seeds, sliced into blocks for dev+confirm")
    ap.add_argument("--block-size", type=int, default=12)
    ap.add_argument("--win-target", type=float, default=0.60, help="seq: round win-rate that counts as SOLVED")
    ap.add_argument("--max-looks", type=int, default=3, help="seq: max confirm blocks per candidate")
    # decide engine seeds
    ap.add_argument("--dev", default="0-5")
    ap.add_argument("--held", default="6-11")
    ap.add_argument("--terminal", default="90-95")
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--max-candidates", type=int, default=1, help="noise-control cap per round")
    ap.add_argument("--selfcheck", action="store_true", help="decide engine: fake measurer (no games)")
    ap.add_argument("--cat-advisor", default=os.path.join("runs", "text_laya", "cat_v3"))
    ap.add_argument("--move-advisor", default=os.path.join("runs", "text_laya", "move_v2"))
    args = ap.parse_args()

    moves = char_menu_moves(args.me)
    out_root = os.path.join(REPO, "out", "loop", "%s_%s_%d" % (args.opp, args.engine, int(time.time())))
    os.makedirs(out_root, exist_ok=True)
    if not args.coach and not args.candidates:
        raise SystemExit("give --coach (live Qwen proposer) or --candidates <file> (offline stub)")
    if args.coach:
        proposer = make_coach_proposer(args.opp, args.me, args.cat_advisor, args.move_advisor,
                                       out_root, moves, scout_games=args.scout_games, cap=args.max_candidates)
    else:
        proposer = load_candidates(args.candidates, args.max_candidates)

    incumbent0 = book_incumbent(args.opp, args.me)
    print("out: %s" % out_root)
    print("incumbent0 (book): %s" % list(incumbent0.rules))
    (run_seq if args.engine == "seq" else run_decide)(args, out_root, moves, proposer, incumbent0)
    print("\nledger: %s" % os.path.join(out_root, "ledger.jsonl"))


if __name__ == "__main__":
    main()
