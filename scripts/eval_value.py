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
import random
import sys
from typing import Dict, List

import _path  # noqa: F401
from sf2.data import value_oracle
from sf2.data.value import expected_net, note_version, value_probs, value_question
from sf2.data.vs_sweep import outcome_question
from sf2.eval.value_eval import calibration, calibration_gate, top3_share
from sf2.system1.system1 import VALUE_KEY, choices

FILES = ("test_real_left", "test_real_right", "test_extra", "test_heldout_guile")
MIN_SPREAD = 15.0          # gate 2: top predicted-net bin's real net above the bottom's (life points)
THROW_TOP3_CLOSE = 0.5     # gate 3: Chun-Li's throw in the top 3 up close
BLOCKS = ("block_high", "block_low")
MIN_SPREAD_V2 = 10.0       # run 2: explored calibration spread >= max(the table's spread, this)
THROW_MARGIN = 0.10        # run 2: throw top-3 share where it pays >= the table's share minus this


def sample(rows: List[Dict], n: int, seed: int = 0) -> List[Dict]:
    """A seeded sample of ``n`` rows in their file order (all of them when n is 0 or not smaller)."""
    if not n or n >= len(rows):
        return rows
    keep = sorted(random.Random(seed).sample(range(len(rows)), n))
    return [rows[i] for i in keep]


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
            "explored": r.get("explored"), "opp_air": bool(r.get("opp_air")),
            "outcome": r["outcome"], "pred_outcome": pred, "net": r["damage"] - r["damage_taken"],     # every live row: old ones have no dealt/taken
           
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
    # calibration over the decisions whose move System 1 can pick (old live rows also hold back / jump / idle ...)
    pairs = [(d["pred_net"], d["net"]) for d in ds if d["pred_net"] is not None]
    if value and len(pairs) >= 25:
        cal = calibration(pairs, k=5)
        out["calibration"], out["calibration_gate"] = cal, calibration_gate(cal, MIN_SPREAD)
        out["calibration_n"] = len(pairs)
    return out


def with_oracle(ds: List[Dict], char: str, table: Dict) -> List[Dict]:
    """Each decision with the lookup table's ranking (sf2.data.value_oracle) of the same moves: the reference the
    re-anchored gates compare with (docs/reviews/2026-09-30_dr_fable_lv_value.md)."""
    out = []
    for d in ds:
        row = table.get((char, d["range"], int(d["opp_state"] in ("attack", "special")), int(d["opp_air"])), {})
        ov = {m: float(row.get(m, 0.0)) for m in d["values"]} if d["values"] else {}
        out.append(dict(d, oracle_values=ov, oracle_net=ov.get(d["action"]) if d["pred_net"] is not None else None))
    return out


def sweet(d: Dict) -> bool:
    """Up close, he is not attacking and on the ground: where the labels say the throw is her best move."""
    return d["range"] == "close" and d["opp_state"] not in ("attack", "special") and not d["opp_air"]


def summarize_v2(ds: List[Dict], char: str) -> Dict:
    """Run 2's re-anchored numbers: the throw where it pays, and calibration on explored (unbiased) decisions, each
    next to the lookup table's on the same decisions."""
    out = {}
    if "throw" in choices(char):
        out["throw_top3_sweet"] = top3_share(ds, "throw", sweet)
        out["oracle_throw_top3_sweet"] = top3_share(ds, "throw", sweet, key="oracle_values")
    exp = [d for d in ds if d.get("explored") and d["pred_net"] is not None]
    if len(exp) >= 25:
        cal = calibration([(d["pred_net"], d["net"]) for d in exp], k=5)
        ocal = calibration([(d["oracle_net"], d["net"]) for d in exp], k=5)
        need = max(calibration_gate(ocal, 0)["spread"], MIN_SPREAD_V2)
        out.update(calibration_explored=cal, oracle_calibration_explored=ocal, calibration_explored_n=len(exp),
                   calibration_v2_gate=calibration_gate(cal, need))
    return out


def gates_v2(report: Dict) -> List[str]:
    """Run 2's gates (docs/prereg_lv_value_run2.md): calibration on explored decisions at least as wide as the table's
    (and >= MIN_SPREAD_V2) in every file but test_extra; Chun-Li's throw in the top 3 where it pays at least the
    table's share minus THROW_MARGIN, on test_real and held-out Guile."""
    fails = []
    for name, chars in report.items():
        for char, s in chars.items():
            g = s.get("v2", {}).get("calibration_v2_gate")
            if g and not g["pass"] and name != "test_extra":
                fails.append("%s %s explored calibration %s" % (name, char, g))
    for name in ("test_real", "test_heldout_guile"):
        v2 = report.get(name, {}).get("chunli", {}).get("v2", {})
        t, o = v2.get("throw_top3_sweet", {}), v2.get("oracle_throw_top3_sweet", {})
        if t.get("share") is not None and t["share"] < o["share"] - THROW_MARGIN:
            fails.append("%s chunli throw top-3 where it pays %.2f < table %.2f - %.2f" % (
                name, t["share"], o["share"], THROW_MARGIN))
    return fails


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
    ap.add_argument("--sample", type=int, default=0,
                    help="a seeded sample of this many decisions per file and character (0: all)")
    ap.add_argument("--gates", choices=("v1", "v2"), default="v1",
                    help="v1: the first run's pre-registered gates; v2: run 2's, anchored to the lookup table")
    ap.add_argument("--oracle", default=os.path.join("lessons", "value_oracle_v1.json"))
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    table = value_oracle.load(args.oracle) if args.gates == "v2" else None

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
            rows = sample(rows, args.sample)
            if not rows:
                continue
            ds = [ask(agent, args.root, char, r, value, version) for r in rows]
            s = summarize(ds, char, value)
            if table is not None:
                s = dict(s, v2=summarize_v2(with_oracle(ds, char, table), char))
            report.setdefault(name, {})[char] = s
            print("%-18s %-8s n=%5d outcome %.3f throw-close %s cal %s" % (
                name, char, s["n"], s["outcome_acc"], s.get("throw_top3_close"),
                s.get("calibration_gate")), flush=True)
    fails = ((gates_v2 if args.gates == "v2" else gates)(report)) if value else []
    name = "eval_value.json" if args.gates == "v1" else "eval_value_v2.json"
    with open(os.path.join(args.model, name), "w") as f:
        json.dump({"model": args.model, "note_version": version, "value": value, "gates": args.gates,
                   "sample": args.sample, "oracle": args.oracle if table is not None else None, "report": report,
                   "gate_failures": fails}, f, indent=1)
    for x in fails:
        print("GATE FAIL", x)
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
