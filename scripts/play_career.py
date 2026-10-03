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

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sf2.config import REPO  # noqa: E402
from sf2.vocab import FIGHTERS  # noqa: E402

PY = os.path.join(REPO, ".venv", "bin", "python")
ROM = os.environ.get("SF2_ROM") or os.path.join(REPO, "roms", "Street Fighter II (USA).sfc")


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


def run_block(me, opp, nn, session_dir, carry, save_reg, block, rounds, cat, move, qmode="two"):
    out = os.path.join(session_dir, "round_%02d_%s" % (nn, opp))
    cmd = [PY, "scripts/play_loop_screen.py", "--me", me, "--opp", opp,
           "--games", str(block), "--rounds", str(rounds), "--qwen-mode", qmode,
           "--cat-advisor", cat, "--move-advisor", move, "--no-score", "--rom", ROM, "--out", out]
    if carry:
        cmd += ["--carry", carry]
    if save_reg:
        cmd += ["--save-registry", save_reg]
    subprocess.run(cmd, cwd=REPO, stdout=open(out + ".log", "w"), stderr=subprocess.STDOUT, timeout=3600)
    return out


def main():
    ap = argparse.ArgumentParser(description="Continuous blank-start career / arcade ladder.")
    ap.add_argument("--me", default="chunli")
    ap.add_argument("--opps", default=None, help="comma list; default = every fighter except --me")
    ap.add_argument("--block", type=int, default=4, help="games per block (a loss just replays next game)")
    ap.add_argument("--rounds", type=int, default=1, help="SF2 rounds per match")
    ap.add_argument("--win-target", type=float, default=0.60, help="recent win-rate that counts as BEATEN")
    ap.add_argument("--cap", type=int, default=6, help="max blocks on one opponent before moving on anyway")
    ap.add_argument("--laps", type=int, default=0, help="times through the ladder; 0 = forever (Ctrl-C)")
    ap.add_argument("--carry-forward", action="store_true", help="keep the learned memory into the next opponent")
    ap.add_argument("--book", default=None, help="start from this book instead of a BLANK playbook")
    ap.add_argument("--name", default=None)
    ap.add_argument("--cat-advisor", default=os.path.join("runs", "text_laya", "cat_v3"))
    ap.add_argument("--move-advisor", default=os.path.join("runs", "text_laya", "move_v2"))
    ap.add_argument("--dry-run", action="store_true", help="print the plan and exit (no games)")
    args = ap.parse_args()

    ladder = ladder_for(args.me, args.opps)
    name = args.name or ("career_%s_%d" % (args.me, int(time.time())))
    session_dir = os.path.join(REPO, "rollouts", "career", name)

    if args.dry_run:
        print("me=%s ladder=%s block=%d win_target=%.2f cap=%d laps=%s start=%s carry_forward=%s"
              % (args.me, ladder, args.block, args.win_target, args.cap,
                 args.laps or "forever", ("book" if args.book else "BLANK"), args.carry_forward))
        print("session_dir=%s" % session_dir)
        for lap, opp, b in plan(args.me, ladder, args.block, args.cap, args.laps):
            print("  lap %d  vs %-8s  block %d" % (lap, opp, b))
        return 0

    os.makedirs(session_dir, exist_ok=True)
    print("CAREER %s  ladder: %s" % (args.me, " -> ".join(ladder)))
    print("session_dir: %s\nwatch:  python scripts/monitor_tui.py --session %s\n" % (session_dir, session_dir))

    # the blank (or book) starting registry file carried into an opponent's FIRST block
    blank_path = os.path.join(session_dir, "_blank.json")
    with open(blank_path, "w") as f:
        f.write("[]")

    nn = 0
    carry_next = None                      # None -> blank/book; a path -> carry it
    lap = 0
    try:
        while args.laps == 0 or lap < args.laps:
            for opp in ladder:
                # per-opponent start: carry-forward from the last opponent, else blank (or book seed)
                reg = carry_next if (args.carry_forward and carry_next) else (None if args.book else blank_path)
                beaten = False
                for b in range(args.cap):
                    save_reg = os.path.join(session_dir, "reg_%s.json" % opp)
                    out = run_block(args.me, opp, nn, session_dir, reg, save_reg,
                                    args.block, args.rounds, args.cat_advisor, args.move_advisor)
                    wr = block_winrate(out)
                    reg = save_reg                       # learn ON TOP within this opponent
                    nn += 1
                    print("  vs %-8s block %d: win-rate %s  (%s)"
                          % (opp, b, ("%.0f%%" % (100 * wr)) if wr is not None else "n/a", os.path.basename(out)))
                    if wr is not None and wr >= args.win_target:
                        beaten = True
                        print("  -> BEATEN %s (>= %.0f%%), moving on" % (opp, 100 * args.win_target))
                        break
                if not beaten:
                    print("  -> moved on from %s after %d blocks (not beaten)" % (opp, args.cap))
                carry_next = os.path.join(session_dir, "reg_%s.json" % opp)
            lap += 1
            print("=== lap %d complete ===" % lap)
    except KeyboardInterrupt:
        print("\n(stopped)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
