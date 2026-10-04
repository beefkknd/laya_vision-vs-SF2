"""Choice 3: tune the bee quorum's numbers by evolution, scored on real play (docs/plan_bee_quorum.md).

A (1 + lambda) evolution strategy over sf2/quorum/genome.py's genes:
  * each generation: the parent and ``--lam`` mutated children play the SAME seeds (common random numbers), so a
    score difference comes from the genes, not the CPU's dice; the parent is re-scored every generation (an old
    lucky score is never trusted);
  * every candidate starts from the same frozen ``--carry-table`` / ``--carry-quorum`` snapshot and saves its own
    copy, so fitness includes how fast its settings learn, not luck carried over;
  * fitness = mean net hp per round + win weight x round win rate - Qwen weight x escalation above target
    (genome.fitness), read from each run's trace.jsonl rounds and verdict.json;
  * every ``--holdout-every`` generations the best genome also plays ``--holdout-opp``, never used for selection.

Each evaluation is one command (``--cmd``, a template) run in a subprocess; ``--workers`` run at once, each with its
own ports. The default template is scripts/play_loop_screen.py --policy quorum --quorum-mode vote. Example:

    python scripts/quorum_evolve.py --out runs/evolve1 --opp dhalsim --generations 15 --lam 6 --workers 3 \\
        --games 2 --rounds 3 --carry-table runs/table.json --extra "--no-score --me chunli"
    python scripts/quorum_evolve.py --out /tmp/evo --dry-run          # the loop alone, with a fake fitness
"""
import argparse
import json
import os
import random
import shlex
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional

import _path  # noqa: F401
from sf2.config import PORTS
from sf2.quorum import genome as G
from sf2.quorum.config import QuorumConfig

HERE = os.path.dirname(os.path.abspath(__file__))
CMD = ("{python} " + os.path.join(HERE, "play_loop_screen.py") + " --policy quorum --quorum-mode vote "
       "--quorum-config {config} --out {out} --seed {seed} --opp {opp} --games {games} --rounds {rounds} "
       "--port {port} --replay-port {replay_port} --save-table {out}/table.json --save-quorum {out}/quorum.json "
       "{carry} {extra}")


def read_run(out: str) -> Dict:
    """(rounds, escalation rate) from one finished run directory."""
    rounds = []
    trace = os.path.join(out, "trace.jsonl")
    if os.path.exists(trace):
        with open(trace) as f:
            for line in f:
                e = json.loads(line)
                if e.get("event") == "round":
                    rounds.append(e)
    esc = 0.0
    vpath = os.path.join(out, "verdict.json")
    if os.path.exists(vpath):
        with open(vpath) as f:
            esc = (json.load(f).get("quorum") or {}).get("escalation_rate", 0.0)
    return {"rounds": rounds, "escalation_rate": esc}


def fake_fitness(genes: Dict[str, float], seed: int) -> float:
    """--dry-run: a smooth bowl with its peak at theta 0.6, beta 1.2, gamma 0.8, plus seed noise."""
    target = {"theta": 0.6, "beta": 1.2, "gamma": 0.8}
    return -sum((genes[k] - v) ** 2 for k, v in target.items()) * 100 + random.Random(seed).gauss(0, 0.5)


class Evaluator:
    def __init__(self, args):
        self.args = args
        self.slots = list(range(args.workers))

    def run(self, genes: Dict[str, float], seeds: List[int], tag: str, opp: str, slot: int) -> float:
        a = self.args
        if a.dry_run:
            return sum(fake_fitness(genes, s) for s in seeds) / len(seeds)
        cfg = G.decode(genes, a.base)
        cdir = os.path.join(a.out, tag)
        os.makedirs(cdir, exist_ok=True)
        cpath = os.path.join(cdir, "config.json")
        cfg.save(cpath)
        rounds, escs = [], []
        for s in seeds:
            out = os.path.join(cdir, "%s_seed%d" % (opp, s))
            carry = ""
            if a.carry_table:
                carry += " --carry-table " + shlex.quote(a.carry_table)
            if a.carry_quorum:
                carry += " --carry-quorum " + shlex.quote(a.carry_quorum)
            cmd = a.cmd.format(python=shlex.quote(sys.executable), config=shlex.quote(cpath), out=shlex.quote(out),
                               seed=s, opp=opp, games=a.games, rounds=a.rounds, port=a.base_port + slot,
                               replay_port=a.base_replay_port + slot, carry=carry, extra=a.extra)
            with open(out + ".log", "w") as log:
                code = subprocess.call(shlex.split(cmd), stdout=log, stderr=subprocess.STDOUT)
            if code != 0:
                print("  %s seed %d failed (exit %d), see %s.log" % (tag, s, code, out), flush=True)
                continue
            r = read_run(out)
            rounds += r["rounds"]
            escs.append(r["escalation_rate"])
        return G.fitness(rounds, sum(escs) / len(escs) if escs else 1.0, a.win_weight, a.qwen_weight, a.qwen_target)

    def many(self, jobs) -> List[float]:
        """jobs: [(genes, seeds, tag, opp)] -> fitness each, ``--workers`` at a time, one port slot per worker."""
        free = list(self.slots)
        results: List[Optional[float]] = [None] * len(jobs)

        def one(i):
            slot = free.pop()
            try:
                results[i] = self.run(*jobs[i], slot=slot)
            finally:
                free.append(slot)
        with ThreadPoolExecutor(max_workers=max(1, self.args.workers)) as pool:
            list(pool.map(one, range(len(jobs))))
        return results


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--base-config", default=None, help="starting QuorumConfig JSON (default: the defaults, mode vote)")
    ap.add_argument("--generations", type=int, default=15)
    ap.add_argument("--lam", type=int, default=6, help="children per generation")
    ap.add_argument("--sigma", type=float, default=0.15, help="mutation step, as a fraction of each gene's range")
    ap.add_argument("--seeds-per-gen", type=int, default=2, help="seeds every candidate plays each generation")
    ap.add_argument("--opp", default="dhalsim")
    ap.add_argument("--holdout-opp", default=None, help="scored every --holdout-every generations, never selected on")
    ap.add_argument("--holdout-every", type=int, default=5)
    ap.add_argument("--games", type=int, default=2)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--base-port", type=int, default=PORTS["system1"][0],
                    help="worker i plays on base-port + i (default: the start of PORTS['system1'])")
    ap.add_argument("--base-replay-port", type=int, default=PORTS["replay"][0],
                    help="worker i scores on base-replay-port + i (default: the start of PORTS['replay'])")
    ap.add_argument("--carry-table", default=None, help="the frozen table snapshot every candidate starts from")
    ap.add_argument("--carry-quorum", default=None, help="the frozen reliability snapshot every candidate starts from")
    ap.add_argument("--cmd", default=CMD, help="evaluation command template (see CMD)")
    ap.add_argument("--extra", default="", help="extra arguments appended to every evaluation command")
    ap.add_argument("--win-weight", type=float, default=20.0)
    ap.add_argument("--qwen-weight", type=float, default=50.0)
    ap.add_argument("--qwen-target", type=float, default=0.10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true", help="fake fitness, no emulator: test the loop itself")
    args = ap.parse_args(argv)
    args.base = QuorumConfig.load(args.base_config) if args.base_config else QuorumConfig(mode="vote")
    os.makedirs(args.out, exist_ok=True)
    rng = random.Random(args.seed)
    ev = Evaluator(args)
    parent = G.encode(args.base)
    log = open(os.path.join(args.out, "log.jsonl"), "a")
    best_ever = (float("-inf"), parent)
    t0 = time.time()
    for gen in range(args.generations):
        seeds = [args.seed * 100000 + gen * 100 + i for i in range(args.seeds_per_gen)]   # shared this generation
        kids = [G.mutate(parent, rng, args.sigma) for _ in range(args.lam)]
        cands = [parent] + kids
        jobs = [(c, seeds, "g%03d_c%02d" % (gen, i), args.opp) for i, c in enumerate(cands)]
        fits = ev.many(jobs)
        for i, (c, f) in enumerate(zip(cands, fits)):
            log.write(json.dumps({"gen": gen, "cand": i, "parent": i == 0, "fitness": f, "seeds": seeds,
                                  "genes": c}) + "\n")
        log.flush()
        best_i = max(range(len(cands)), key=lambda i: fits[i])
        parent = cands[best_i]
        if fits[best_i] > best_ever[0]:
            best_ever = (fits[best_i], parent)
            G.decode(parent, args.base).save(os.path.join(args.out, "best_config.json"))
        print("gen %d: parent %.2f, best %.2f (cand %d), %.0fs" % (gen, fits[0], fits[best_i], best_i,
                                                                  time.time() - t0), flush=True)
        if args.holdout_opp and (gen + 1) % args.holdout_every == 0:
            h = ev.many([(parent, seeds, "g%03d_holdout" % gen, args.holdout_opp)])[0]
            log.write(json.dumps({"gen": gen, "holdout": args.holdout_opp, "fitness": h, "genes": parent}) + "\n")
            log.flush()
            print("  holdout %s: %.2f" % (args.holdout_opp, h), flush=True)
    log.close()
    print("best fitness %.2f -> %s" % (best_ever[0], os.path.join(args.out, "best_config.json")))
    return 0


if __name__ == "__main__":
    sys.exit(main())
