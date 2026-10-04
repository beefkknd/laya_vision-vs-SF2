#!/usr/bin/env python3
"""Continuous CAREER / arcade ladder for text-laya: start BLANK (empty short memory) and just keep
playing. Each opponent is played in blocks of games with carryover, so the Coach builds a playbook from
scratch while text-laya plays; a loss is simply the next match (it replays); once the opponent is BEATEN
(recent win-rate >= target) it MOVES ON to the next fighter. The ladder loops forever (Ctrl-C to stop).

It writes a SESSION dir (rollouts/career/<name>/round_<NN>_<opp>/) so the monitor TUI can watch the whole
career live:  python scripts/monitor_tui.py --session rollouts/career/<name>    (trend climbs, pulse fires)
or in one command:  python scripts/monitor_tui.py --career "chunli"

Needs the Qwen server (SF2_QWEN_URL) and the ROM (SF2_ROM or the repo default). qwen-mode two = Scout+Coach.

    python scripts/play_career.py                       # chunli vs the 7-fighter ladder, blank, forever
    python scripts/play_career.py --opps honda,guile    # just these, in order
    python scripts/play_career.py --dry-run             # print the plan, play nothing
"""
import argparse
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sf2.config import QWEN_URL, REPO  # noqa: E402
from sf2.system1.advice import char_menu_moves  # noqa: E402
from sf2.system2 import seed_rules  # noqa: E402
from sf2.system2.rule_entry import default_kit  # noqa: E402
from sf2.vocab import FIGHTERS  # noqa: E402

PY = os.path.join(REPO, ".venv", "bin", "python")
ROM = os.environ.get("SF2_ROM") or os.path.join(REPO, "roms", "Street Fighter II (USA).sfc")


def qwen_alive(url=None, timeout=8):
    """True if the Qwen server answers at all (any HTTP status = alive; a connection error = down).
    The Coach needs it; without it every block fails silently and the session dir is just text logs."""
    url = url or QWEN_URL
    try:
        urllib.request.urlopen(urllib.request.Request(url, method="GET"), timeout=timeout)
        return True
    except urllib.error.HTTPError:
        return True          # it responded (401/405/...) -> up
    except Exception:
        return False         # connection refused / DNS / timeout -> down


# Forced exploration now lives in the LIVE per-round policy (sf2.system2.short_memory + explore_pool): while
# she is losing, the loop swaps one short-memory line WITHIN the block, so the career driver no longer injects
# rules between blocks. The old between-block trigger (loss_window / needs_intervention / pick_forced_rule /
# EXPLORE_POOL) is gone -- one policy, one place (docs/plan_simple_learning.md).


def round_results(out_dir):
    """The per-round win/loss list from a finished block's trace.jsonl, in order."""
    try:
        rows = [json.loads(l) for l in open(os.path.join(out_dir, "trace.jsonl"))]
    except OSError:
        return []
    return [r.get("result") for r in rows if r.get("event") == "round"]


def resolve_playbook(pb):
    """`--playbook` -> (folder, playbook.json path). A '*.json' is used as the file (its dir is the home);
    anything else is a FOLDER name: absolute as given, else under <repo>/playbooks/. The folder holds the
    playbook AND the play log, and the TUI watches it."""
    if pb.endswith(".json"):
        return (os.path.dirname(os.path.abspath(pb)) or REPO), os.path.abspath(pb)
    folder = pb if os.path.isabs(pb) else os.path.join(REPO, "playbooks", pb)
    return folder, os.path.join(folder, "playbook.json")


def ladder_for(me, opps):
    """The opponent order: an explicit --opps list, else every fighter except `me`."""
    if opps:
        return [o.strip() for o in opps.split(",") if o.strip()]
    return [f for f in FIGHTERS if f != me]


def block_winrate(out_dir):
    """Recent win-rate from a finished block's verdict.json: fraction of games won. None if unreadable."""
    try:
        v = json.load(open(os.path.join(out_dir, "verdict.json")))
        games = v.get("games") or []
        if not games:
            return None
        wins = sum(1 for g in games if g.get("won", 0) > g.get("lost", 0))
        return wins / len(games)
    except (OSError, ValueError, KeyError):
        return None


def plan(me, ladder, block, cap, laps):
    """The sequence of (lap, opp, block_index) the driver WOULD play (for --dry-run / tests)."""
    out = []
    lap = 0
    while laps == 0 or lap < laps:
        for opp in ladder:
            for b in range(cap):
                out.append((lap, opp, b))
        lap += 1
        if laps == 0 and lap >= 1:        # dry-run shows one lap when endless
            break
    return out


def run_block(me, opp, nn, session_dir, carry, save_reg, block, rounds, cat, move,
              qmode="two", watch=False, speed=100, policy="rules", table_path=None):
    out = os.path.join(session_dir, "round_%02d_%s" % (nn, opp))
    cmd = [PY, "scripts/play_loop_screen.py", "--me", me, "--opp", opp,
           "--games", str(block), "--rounds", str(rounds), "--no-score", "--rom", ROM, "--out", out]
    if watch:
        cmd += ["--watch", "--speed", str(speed)]      # open a visible Mesen window for the match
    if policy in ("rules", "hybrid"):                  # text-laya plays (hybrid: the table overrides on top)
        cmd += ["--qwen-mode", qmode, "--cat-advisor", cat, "--move-advisor", move]
        if carry:
            cmd += ["--carry", carry]
        if save_reg:
            cmd += ["--save-registry", save_reg]
    if policy in ("table", "hybrid"):                  # the value table (alone, or as the hybrid override)
        cmd += ["--policy", policy]
        if table_path:
            cmd += ["--carry-table", table_path, "--save-table", table_path]   # grows in place across blocks
    subprocess.run(cmd, cwd=REPO, stdout=open(out + ".log", "w"), stderr=subprocess.STDOUT, timeout=3600)
    return out


def main():
    ap = argparse.ArgumentParser(description="Continuous blank-start career / arcade ladder.")
    ap.add_argument("--me", default="chunli")
    ap.add_argument("--opps", default=None, help="comma list; default = every fighter except --me")
    ap.add_argument("--policy", default="rules", choices=("rules", "table", "hybrid"),
                    help="rules (default): text-laya + short memory + Qwen; table: the self-learning value table "
                         "alone (no Qwen/text-laya); hybrid: text-laya plays + the table overrides where confident "
                         "(the owner's pick after the pure table lost)")
    ap.add_argument("--block", type=int, default=4, help="matches per block (a loss just replays next match)")
    ap.add_argument("--rounds", type=int, default=3, help="rounds per match (a match is best-of-3 = 2-3 rounds)")
    ap.add_argument("--win-target", type=float, default=0.60, help="recent win-rate that counts as BEATEN")
    ap.add_argument("--cap", type=int, default=6, help="max blocks on one opponent before moving on anyway")
    ap.add_argument("--laps", type=int, default=0, help="times through the ladder; 0 = forever (Ctrl-C)")
    ap.add_argument("--carry-forward", action="store_true", help="keep the learned memory into the next opponent")
    ap.add_argument("--book", default=None, help="seed a NEW playbook from this book instead of the blank kit")
    ap.add_argument("--playbook", default=None, help="a playbook FILE or FOLDER to START from and keep GROWING; "
                    "it persists across runs, so pass the same one again to RESUME. Created + seeded if missing.")
    ap.add_argument("--watch", action="store_true", help="open a VISIBLE Mesen window for each match (see the SNES game)")
    ap.add_argument("--speed", type=int, default=100, help="--watch emulation speed percent")
    ap.add_argument("--no-qwen-check", action="store_true", help="skip the Qwen server preflight")
    ap.add_argument("--name", default=None)
    ap.add_argument("--cat-advisor", default=os.path.join("runs", "text_laya", "cat_v3"))
    ap.add_argument("--move-advisor", default=os.path.join("runs", "text_laya", "move_v2"))
    ap.add_argument("--dry-run", action="store_true", help="print the plan and exit (no games)")
    args = ap.parse_args()

    ladder = ladder_for(args.me, args.opps)
    name = args.name or ("career_%s_%d" % (args.me, int(time.time())))
    # A --playbook makes ONE named folder the home for the growing playbook AND the play log (and what the
    # TUI watches): `--playbook foo` -> playbooks/foo/ with playbook.json inside; pass it again to RESUME.
    if args.playbook:
        session_dir, career_reg = resolve_playbook(args.playbook)
    else:
        session_dir = os.path.join(REPO, "rollouts", "career", name)
        career_reg = os.path.join(session_dir, "career_reg.json")

    if args.dry_run:
        print("me=%s ladder=%s block=%d win_target=%.2f cap=%d laps=%s start=%s carry_forward=%s"
              % (args.me, ladder, args.block, args.win_target, args.cap,
                 args.laps or "forever", ("book" if args.book else "BLANK"), args.carry_forward))
        print("session_dir=%s" % session_dir)
        for lap, opp, b in plan(args.me, ladder, args.block, args.cap, args.laps):
            print("  lap %d  vs %-8s  block %d" % (lap, opp, b))
        return 0

    if args.policy in ("rules", "hybrid") and not args.no_qwen_check and not qwen_alive():
        print("Qwen server not reachable at %s\n"
              "The Coach needs it. Set SF2_QWEN_URL to a live server, e.g.\n"
              "  export SF2_QWEN_URL=http://100.66.12.33:8080/v1/chat/completions\n"
              "then rerun (or pass --no-qwen-check to skip). Aborting so the session isn't just empty logs."
              % QWEN_URL, file=sys.stderr)
        return 2

    os.makedirs(session_dir, exist_ok=True)
    print("CAREER %s  ladder: %s" % (args.me, " -> ".join(ladder)))
    print("session_dir: %s\nwatch:  python scripts/monitor_tui.py --session %s\n" % (session_dir, session_dir))

    moves = char_menu_moves(args.me)
    table_path = os.path.join(session_dir, "table.json")        # the value table's growing home (policy table)

    def _reg_lines():
        try:
            return [e.get("line") for e in json.load(open(career_reg))]
        except (OSError, ValueError):
            return []

    def _table_cells():
        try:
            return len(json.load(open(table_path))["cells"])
        except (OSError, ValueError, KeyError):
            return 0

    if args.policy in ("table", "hybrid"):
        # the value table grows in table.json across blocks (resume = the file persists)
        print("RESUMING table %s (%d cells)" % (table_path, _table_cells()) if os.path.exists(table_path)
              else "start: BLANK value table")
    if args.policy in ("rules", "hybrid"):
        # ONE growing playbook for text-laya's short memory: carried into EVERY block, only replaced on a valid save.
        # If the playbook file already exists, RESUME from it; otherwise seed it.
        if os.path.exists(career_reg) and _reg_lines():
            print("RESUMING playbook %s (%d rules)" % (career_reg, len(_reg_lines())))
        else:
            seed_entries = (list(seed_rules.seed_lessons(args.book, ladder[0], args.me)) or default_kit(moves)
                            if args.book else default_kit(moves))
            json.dump(seed_entries, open(career_reg, "w"), indent=1)
            print("start kit: %s" % [e["line"] for e in seed_entries])

    # continue the play-log numbering so a resumed playbook APPENDS its history instead of overwriting it
    existing = [n for n in os.listdir(session_dir) if n.startswith("round_") and os.path.isdir(os.path.join(session_dir, n))]
    nn = (max((int(n.split("_")[1]) for n in existing), default=-1) + 1) if existing else 0
    lap = 0
    try:
        while args.laps == 0 or lap < args.laps:
            for opp in ladder:
                beaten = False
                for b in range(args.cap):
                    if args.policy == "table":
                        out = run_block(args.me, opp, nn, session_dir, None, None, args.block, args.rounds,
                                        args.cat_advisor, args.move_advisor, watch=args.watch, speed=args.speed,
                                        policy="table", table_path=table_path)
                        state_desc = "cells=%d" % _table_cells()
                    else:                                        # rules OR hybrid: grow text-laya's registry; hybrid also the table
                        tmp = os.path.join(session_dir, "_block_save.json")
                        if os.path.exists(tmp):
                            os.remove(tmp)
                        out = run_block(args.me, opp, nn, session_dir, career_reg, tmp,
                                        args.block, args.rounds, args.cat_advisor, args.move_advisor,
                                        watch=args.watch, speed=args.speed, policy=args.policy,
                                        table_path=(table_path if args.policy == "hybrid" else None))
                        # GROW: promote the block's save into the career registry ONLY if valid, so a crashed/
                        # restarted block keeps the previous playbook instead of losing it.
                        try:
                            saved = json.load(open(tmp))
                            if isinstance(saved, list) and saved:
                                os.replace(tmp, career_reg)
                        except (OSError, ValueError):
                            print("  (block produced no registry -- keeping the current playbook, %d rules)" % len(_reg_lines()))
                        state_desc = "rules=%d" % len(_reg_lines())
                        if args.policy == "hybrid":
                            state_desc += " cells=%d" % _table_cells()
                    wr = block_winrate(out)
                    nn += 1
                    rr = round_results(out)
                    print("  vs %-8s block %d: win-rate %s  %s  rounds=%s  (%s)"
                          % (opp, b, ("%.0f%%" % (100 * wr)) if wr is not None else "n/a",
                             state_desc, "".join("W" if r == "win" else "L" for r in rr),
                             os.path.basename(out)))

                    if wr is not None and wr >= args.win_target:
                        beaten = True
                        print("  -> BEATEN %s (>= %.0f%%), moving on (playbook kept, %d rules)"
                              % (opp, 100 * args.win_target, len(_reg_lines())))
                        break
                if not beaten:
                    print("  -> moved on from %s after %d blocks (playbook kept, %d rules)"
                          % (opp, args.cap, len(_reg_lines())))
            lap += 1
            print("=== lap %d complete (playbook: %d rules) ===" % (lap, len(_reg_lines())))
    except KeyboardInterrupt:
        print("\n(stopped)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
