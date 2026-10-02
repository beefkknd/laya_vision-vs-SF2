"""Score a screen-only run OFFLINE (docs/laya_text_only_plan.md: "after - scoring: replay the saved savestate + buttons
offline and read RAM truth there"): every round of <run> (scripts/play_screen.py) is replayed in a fresh headless Mesen
WITH RAM (sf2/system1/screen_replay.py). A round whose replayed frames differ from the saved ones in any pixel is
REFUSED (exit 1, nothing of it scored). Otherwise writes, per round, <round>/replay.json (result, hp, and at every
decision the reader's facts and words next to RAM's), and <run>/score.json (all rounds).

    python scripts/replay_score.py rollouts/screen_play/<run>
    python scripts/replay_score.py <run> --fault drop:g00_r0:120     # seeded fault: the move pressed at the first decision >= frame 120 dropped
"""
import argparse
import hashlib
import glob
import json
import os
import shutil
import sys
import tempfile

import _path  # noqa: F401
from sf2.config import PORTS
from sf2.eval.runner import open_fight
from sf2.system1.screen_replay import FIELDS, ReplayMismatch, score_round


def drop_press(rec_dir: str, k: int) -> int:
    """Seeded fault: the move pressed at the first decision at frame >= k is dropped (its frames get no buttons).
    (Dropping one frame inside a move often changes nothing on the screen - the game ignores it - and such a replay is
    rightly identical; a dropped move is a real difference.) Returns the decision's frame."""
    with open(os.path.join(rec_dir, "decisions.jsonl")) as f:
        d = next(d for d in map(json.loads, f) if d["k"] >= k and any(p != "-" for p in d["pressed"]))
    path = os.path.join(rec_dir, "inputs.json")
    with open(path) as f:
        inp = json.load(f)
    n = len(d["pressed"])
    if inp["inputs"][d["k"]:d["k"] + n] != d["pressed"]:
        raise SystemExit("%s: the recorded inputs at frame %d are not decision %d's press" % (rec_dir, d["k"], d["i"]))
    inp["inputs"] = inp["inputs"][:d["k"]] + ["-"] * n + inp["inputs"][d["k"] + n:]
    with open(path, "w") as f:
        json.dump(inp, f)
    return d["k"]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run")
    ap.add_argument("--port", type=int, default=PORTS["replay"][0] + PORTS["replay"][1] - 1)
    ap.add_argument("--rom", default=os.environ.get("SF2_ROM"))
    ap.add_argument("--fault", help="drop:<round dir name>:<frame> - score a copy with one press dropped (must be "
                                    "refused)")
    args = ap.parse_args()
    run_dir = args.run
    with open(os.path.join(run_dir, "run.json")) as f:
        run = json.load(f)
    tmp = None
    if args.fault:
        kind, rnd, k = args.fault.split(":")
        if kind != "drop":
            raise SystemExit("unknown fault %r" % kind)
        tmp = tempfile.mkdtemp(prefix="replay_fault_")
        shutil.copytree(run_dir, os.path.join(tmp, "run"))
        run_dir = os.path.join(tmp, "run")
        print("seeded fault: %s, the move pressed at frame %d dropped" % (rnd, drop_press(os.path.join(run_dir, rnd), int(k))))
    rounds = sorted(d for d in glob.glob(os.path.join(run_dir, "g*_r*")) if os.path.isdir(d))
    if not rounds:
        raise SystemExit("no rounds in %s" % run_dir)
    out, refused = [], []
    try:
        with open_fight(run["me"], run["opp"], args.port, args.rom, state=run["state"]["path"]) as (b, state):
            if hashlib.sha256(state).hexdigest() != run["state"]["sha256"]:
                raise SystemExit("%s changed since the run (sha256 differs)" % run["state"]["path"])
            for rd in rounds:
                try:
                    s = score_round(b, state, rd, run["me"], run["opp"])
                except ReplayMismatch as e:
                    refused.append(str(e))
                    print("REFUSED", e, flush=True)
                    continue
                with open(os.path.join(rd, "replay.json"), "w") as f:
                    json.dump(s, f, indent=1)
                out.append({k: v for k, v in s.items() if k != "per_decision"})
                print("%s: %s hp %+d (dealt %d taken %d) result at frame %s, screen ended %s; %d decisions; "
                      "replay identical (%d frames)" % (os.path.basename(rd), s["result"], s["hp"], s["dealt"],
                                                        s["taken"], s["result_frame"], s["screen_end_frame"],
                                                        s["decisions"], s["captures"]), flush=True)
                print("   reader = RAM at decisions: " + " ".join(
                    "%s %.0f%%" % (fld, 100 * s["agreement"][fld]) for fld in FIELDS if s["agreement"][fld] is not None))
    finally:
        if tmp:
            shutil.rmtree(tmp, ignore_errors=True)
    if not args.fault:
        with open(os.path.join(run_dir, "score.json"), "w") as f:
            json.dump({"rounds": out, "refused": refused}, f, indent=1)
    if refused:
        print("%d of %d rounds refused: not scored" % (len(refused), len(rounds)))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
