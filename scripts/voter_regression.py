"""Cross-character regression gate for SHARED quorum-voter changes.

The Honda crater (2026-10-07): a margin-drop in `tally.table_proposal` was validated on ONE character
(Zangief 75->92%) and committed; it took honda 92%->18.8% because honda's one-step "least-bad" moves
are traps that a strong table vote then played. A single-character green is NOT sufficient for a
change to the SHARED exploit voter (tally.py) or its config (config.py priors/formula). This gate
re-measures the whole trained roster frozen bees-off vs Ryu and FAILS if any character drops below
its floor. Mechanical: match% from each run's verdict, compared to a predeclared floor. Run it BEFORE
committing any shared-voter change.

    python scripts/voter_regression.py            # 8 seeds x 8 games per char; exit 1 if any regress
"""
import argparse, json, os, subprocess, sys, tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = os.path.join(REPO, ".venv", "bin", "python")
ROM = os.environ.get("SF2_ROM", os.path.join(REPO, "roms", "Street Fighter II (USA).sfc"))
# (character, canonical table, floor match%) -- a drop below the floor is a regression, not noise.
ROSTER = [("honda", "runs/tables/honda.json", 80.0), ("chunli", "runs/tables/chunli.json", 80.0),
          ("ken", "runs/tables/ken.json", 68.0), ("ryu", "runs/tables/ryu.json", 78.0)]


def _frozen_config():
    """A frozen eval config written to a temp file: laya+table only (no explore bee, no counter, eps 0)."""
    sys.path.insert(0, REPO)
    from sf2.quorum.config import QuorumConfig
    cfg = QuorumConfig.for_eval(QuorumConfig())
    fd, path = tempfile.mkstemp(suffix=".json", prefix="frozen_")
    json.dump(cfg.to_dict(), os.fdopen(fd, "w"))
    return path


def _measure(char, table, qcfg, opp, seeds, games, rounds, base_port):
    def one(seed, idx):
        out = tempfile.mkdtemp(prefix="reg_%s_%d_" % (char, seed))
        cmd = [PY, "scripts/play_loop_screen.py", "--me", char, "--opp", opp, "--policy", "quorum",
               "--quorum-mode", "vote", "--cat-advisor", "runs/text_laya/cat_v4",
               "--move-advisor", "runs/text_laya/move_v3", "--carry-table", table,
               "--quorum-config", qcfg, "--no-learn", "--explore", "0", "--no-score",
               "--games", str(games), "--rounds", str(rounds), "--seed", str(seed), "--out", out,
               "--shared-text-laya", "--port", str(base_port + idx), "--replay-port", str(base_port + 500 + idx)]
        subprocess.run(cmd, cwd=REPO, env=dict(os.environ, SF2_ROM=ROM),
                       stdout=open(os.path.join(out, "log"), "w"), stderr=subprocess.STDOUT)
        v = os.path.join(out, "verdict.json")
        if not os.path.exists(v):
            return (0, 0)
        g = json.load(open(v))["games"]
        return (sum(x["won"] > x["lost"] for x in g), len(g))
    won = tot = 0
    with ThreadPoolExecutor(max_workers=len(seeds)) as ex:
        for w, t in ex.map(lambda st: one(st[1], st[0]), list(enumerate(seeds))):
            won += w; tot += t
    return 100.0 * won / max(1, tot), won, tot


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--opp", default="ryu")
    ap.add_argument("--seeds", default="0,1,2,3,4,5,6,7")
    ap.add_argument("--games", type=int, default=8)
    ap.add_argument("--rounds", type=int, default=3)
    a = ap.parse_args(argv)
    seeds = [int(s) for s in a.seeds.split(",")]
    qcfg = _frozen_config()
    print("CROSS-CHARACTER VOTER REGRESSION (frozen bees-off vs %s, %d seeds x %d games)" % (a.opp, len(seeds), a.games))
    failed = []
    for i, (char, table, floor) in enumerate(ROSTER):
        if not os.path.exists(os.path.join(REPO, table)):
            print("  SKIP %-7s (no table %s)" % (char, table)); continue
        pct, won, tot = _measure(char, table, qcfg, a.opp, seeds, a.games, a.rounds, 48900 + i * 20)
        ok = pct >= floor
        print("  %-5s %-7s match %.1f%% (%d/%d)  floor %.0f%%" % ("PASS" if ok else "FAIL", char, pct, won, tot, floor))
        if not ok:
            failed.append(char)
    print("REGRESSION GATE: %s" % ("PASS" if not failed else "FAIL (" + ", ".join(failed) + ")"))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
