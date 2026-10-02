"""Screen-only play (docs/laya_text_only_plan.md, step 3): arm S0 = T0 (the lookup table + text laya, no advice, no Qwen)
with every fact read from the screen (sf2/system1/screen_play.py). Nothing reads RAM while a game is played: Mesen
runs mesen/sf2_bridge_screen.lua and the emulator handle raises on any RAM read. Results (hp, wins) come only from the
offline replay: scripts/replay_score.py <out>.

    python scripts/play_screen.py --opp ryu --games 1 --oracle lessons/value_oracle_v1.json --shared-text-laya

Writes <out>/run.json and <out>/g<game>_r<round>/ (sf2/system1/screen_play.py lists the files).
A game = --rounds rounds, each from the start savestate states/p1_chunli_vs_<opp>.state with T0's random start delay
(random.Random(--seed): the same delays as a T0 run of the same seed).
"""
import argparse
import hashlib
import json
import os
import random
import subprocess
import sys
import time

import _path  # noqa: F401
from sf2.config import PORTS, REPO, TEXT_LAYA
from sf2.data import value_oracle
from sf2.system1.advisor import Advisor
from sf2.system1.screen_emu import SCREEN_BRIDGE, open_screen
from sf2.system1.screen_play import play_screen_round
from sf2.system1.system1 import System1

ME = "chunli"
DELAY_MIN, DELAY_SPAN = 4, 40          # T0: 4 + rng.randrange(40) idle frames after loading the state


def sha256_file(path: str) -> str:
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def code_state() -> dict:
    """The commit, and the sha256 of every reader / screen-play file (the reader may be uncommitted work)."""
    r = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=REPO)
    files = sorted(os.path.join(d, f) for d in ("sf2/screen", "sf2/system1") for f in os.listdir(os.path.join(REPO, d))
                   if f.endswith(".py") and (d == "sf2/screen" or f.startswith("screen_")))
    return {"commit": r.stdout.strip(), "files": {p: sha256_file(os.path.join(REPO, p)) for p in files},
            "bridge": sha256_file(SCREEN_BRIDGE)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--opp", default="ryu")
    ap.add_argument("--games", type=int, default=1)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--oracle", default=os.path.join("lessons", "value_oracle_v1.json"))
    ap.add_argument("--advisor", default=TEXT_LAYA)
    ap.add_argument("--shared-text-laya", action="store_true", help="text laya from the one shared server")
    ap.add_argument("--port", type=int, default=PORTS["system1"][0] + PORTS["system1"][1] - 1)
    ap.add_argument("--rom", default=os.environ.get("SF2_ROM"))
    ap.add_argument("--out", default=None, help="default rollouts/screen_play/<stamp>_<opp>")
    args = ap.parse_args()
    state_path = os.path.join("states", "p1_%s_vs_%s.state" % (ME, args.opp))
    if not os.path.exists(state_path):
        raise SystemExit("no savestate %s" % state_path)
    with open(state_path, "rb") as f:
        state = f.read()
    state_id = {"path": state_path, "sha256": hashlib.sha256(state).hexdigest()}
    out = args.out or os.path.join("rollouts", "screen_play", "%s_%s" % (time.strftime("%Y%m%d_%H%M%S"), args.opp))
    if os.path.exists(os.path.join(out, "run.json")):
        raise SystemExit("%s already holds a run" % out)
    os.makedirs(out, exist_ok=True)
    table = value_oracle.load(args.oracle)
    run = {"arm": "S0", "me": ME, "opp": args.opp, "games": args.games, "rounds": args.rounds, "seed": args.seed,
           "state": state_id, "oracle": args.oracle, "oracle_sha256": sha256_file(args.oracle),
           "advisor": args.advisor, "shared_text_laya": args.shared_text_laya, "advice": "none", "qwen": "off",
           "code": code_state(), "started": time.strftime("%Y-%m-%d %H:%M:%S")}
    with open(os.path.join(out, "run.json"), "w") as f:
        json.dump(run, f, indent=1)
    rng = random.Random(args.seed)
    shared = {"shared": True} if args.shared_text_laya else {}
    with Advisor(args.advisor, **shared) as advisor, open_screen(args.port, args.rom) as emu:
        s1 = System1(None, ME, advisor=advisor, oracle=table)
        s1.advice_on = False                        # T0 = the no-advice arm: text laya is told "Advice: none"
        s1.short = {"me": ME, "opp": args.opp, "lessons": []}
        for g in range(args.games):
            for r in range(args.rounds):
                rd = os.path.join(out, "g%02d_r%d" % (g, r))
                delay = DELAY_MIN + rng.randrange(DELAY_SPAN)
                s = play_screen_round(emu, s1, ME, args.opp, state, state_id, delay, rd)
                emu = emu.new_round()
                print("game %d round %d: %s at frame %d, %d decisions, %d unknown, %.1f ms/decision "
                      "(read %.1f, decide %.1f)" % (g, r, s["end"], s["end_frame"], s["decisions"],
                                                     s["unknown_fighters_at_decisions"], s["decision_ms_mean"],
                                                     s["read_ms_mean"], s["decide_ms_mean"]), flush=True)
    print("done: %s (score it: python scripts/replay_score.py %s)" % (out, out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
