"""Offline gates of the laya-vision value fine-tune (docs/prereg_lv_value.md) on held-out live decisions of
test_data_v2 (built by scripts/build_value_data.py): every choice is asked at every decision, as System 1 asks it.

    python scripts/eval_value.py --model runs/lv_value/best           # the value checkpoint (note v2, values)
    python scripts/eval_value.py --model runs/all8/best               # the baseline: P(hit) / P(blocked) ranking, note v1

Per file and character: outcome accuracy of the move taken, the ranking's top-3 share of the throw up close and of a
block when he attacks vs when he does not, and (value checkpoint) calibration of the predicted expected net against
the real net. Writes <model>/eval_value.json; exit 1 if a pre-registered gate fails (a value checkpoint only).
"""
import argparse
import json
import os
import sys
from typing import Dict, List

import _path  # noqa: F401
from sf2.data.value import expected_net, note_version, value_probs, value_question
from sf2.data.vs_sweep import outcome_question
from sf2.eval.value_eval import calibration, calibration_gate, top3_share
from sf2.system1.system1 import VALUE_KEY, choices

FILES = ("test_real_left", "test_real_right", "test_extra", "test_heldout_guile")
MIN_SPREAD = 15.0          # gate 2: top predicted-net bin's real net above the bottom's (life points)
THROW_TOP3_CLOSE = 0.5     # gate 3: Chun-Li's throw in the top 3 up close
BLOCKS = ("block_high", "block_low")


def live_decisions(root: str, char: str, name: str) -> List[Dict]:
    path = os.path.join(root, char, name + ".jsonl")
    if not os.path.exists(path):
        return []
    rows = [json.loads(line) for line in open(path)]
    return [r for r in rows if r.get("kind") == "live" and r.get("task") != "value" and not r.get("mirrored")]


def _state(root: str, char: str, r: Dict, version: int) -> Dict:
    from PIL import Image

    text = r["state_text"] if version == 2 else r["state_text"].rsplit(" opp_attacking=", 1)[0]
    return {"images": [Image.open(os.path.join(root, char, p)).convert("RGB") for p in r["images"]], "context": text}


def ask(agent, root: str, char: str, r: Dict, value: bool, version: int) -> Dict:
    """One decision: every choice's ranking score (expected net, or P(hit) / P(blocked)) and the taken move's outcome."""
    moves = choices(char)
    rated = [m for m in moves if m != "forward"]
    qs = {m: outcome_question(m) for m in rated}
    if value:
        qs.update({VALUE_KEY + m: value_question(m) for m in moves})
    ans = agent.predict(_state(root, char, r, version), qs)["answers"]
    if value:
        rank = {m: expected_net(value_probs(ans[VALUE_KEY + m]["probabilities"])) for m in moves}
    else:
        rank = {m: ans[m]["probabilities"]["blocked" if m in BLOCKS else "hit"] for m in rated}
    taken = r["action"]
    pred = max(ans[taken]["probabilities"], key=ans[taken]["probabilities"].get) if taken in ans else None
    return {"values": rank, "range": r["range"], "opp_state": r.get("opp_state"), "action": taken,
            "outcome": r["outcome"], "pred_outcome": pred, "net": r["dealt"] - r["taken"],
            "pred_net": rank.get(taken) if value else None}


def summarize(ds: List[Dict], char: str, value: bool) -> Dict:
    att = lambda d: d["opp_state"] in ("attack", "special")  # noqa: E731
    out = {"n": len(ds),
           "outcome_acc": (sum(d["pred_outcome"] == d["outcome"] for d in ds if d["pred_outcome"])
                           / max(1, sum(bool(d["pred_outcome"]) for d in ds))),
           "block_top3_attacking": {b: top3_share(ds, b, att) for b in BLOCKS},
           "block_top3_not_attacking": {b: top3_share(ds, b, lambda d: not att(d)) for b in BLOCKS}}
    if "throw" in choices(char):
        out["throw_top3_close"] = top3_share(ds, "throw", lambda d: d["range"] == "close")
    if value and len(ds) >= 25:
        cal = calibration([(d["pred_net"], d["net"]) for d in ds], k=5)
        out["calibration"], out["calibration_gate"] = cal, calibration_gate(cal, MIN_SPREAD)
    return out


def gates(report: Dict) -> List[str]:
    """Pre-registered gates 2-4 (gate 1, outcome accuracy on all test rows, is scripts/eval_outcome.py's)."""
    fails = []
    for name, chars in report.items():
        for char, s in chars.items():
            g = s.get("calibration_gate")
            if g and not g["pass"] and name != "test_extra":
                fails.append("%s %s calibration %s" % (name, char, g))
    for name in ("test_real", "test_heldout_guile"):
        t = report.get(name, {}).get("chunli", {}).get("throw_top3_close", {})
        if t.get("share") is not None and t["share"] < THROW_TOP3_CLOSE:
            fails.append("%s chunli throw top-3 up close %.2f < %.2f" % (name, t["share"], THROW_TOP3_CLOSE))
    return fails


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--root", default="test_data_v2")
    ap.add_argument("--chars", default="chunli,ryu,ken,guile,honda,blanka,zangief,dhalsim")
    ap.add_argument("--limit", type=int, default=0, help="decisions per file and character (0: all)")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    import laya

    agent = laya.load_vlm(args.model, device=args.device)
    version, value = note_version(agent.cfg), bool(agent.cfg.get("value_questions"))
    print("%s: note v%d, %s" % (args.model, version, "value ranking" if value else "P(hit) ranking"), flush=True)
    report: Dict[str, Dict] = {}
    for char in args.chars.split(","):
        groups = {"test_real": live_decisions(args.root, char, "test_real_left")
                  + live_decisions(args.root, char, "test_real_right"),
                  "test_extra": live_decisions(args.root, char, "test_extra"),
                  "test_heldout_guile": live_decisions(args.root, char, "test_heldout_guile")}
        for name, rows in groups.items():
            rows = rows[:args.limit] if args.limit else rows
            if not rows:
                continue
            ds = [ask(agent, args.root, char, r, value, version) for r in rows]
            s = summarize(ds, char, value)
            report.setdefault(name, {})[char] = s
            print("%-18s %-8s n=%5d outcome %.3f throw-close %s cal %s" % (
                name, char, s["n"], s["outcome_acc"], s.get("throw_top3_close"),
                s.get("calibration_gate")), flush=True)
    fails = gates(report) if value else []
    with open(os.path.join(args.model, "eval_value.json"), "w") as f:
        json.dump({"model": args.model, "note_version": version, "value": value, "report": report,
                   "gate_failures": fails}, f, indent=1)
    for x in fails:
        print("GATE FAIL", x)
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
