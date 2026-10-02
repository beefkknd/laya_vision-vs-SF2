"""Scoring a screen-only game OFFLINE (docs/laya_text_only_plan.md: RAM only outside play). Replays the round's start
savestate + recorded buttons in a fresh emulator WITH RAM, proves every captured frame identical (sha256 of the RGB
bytes, and the saved decision PNGs pixel for pixel) and only then reads RAM truth: the round's result and hp, and the
T0 words / facts at every decision next to what the screen reader gave. A replay with any mismatch is refused: nothing
is scored from a drifted game.

Never imported by a play module (tests/test_screen_play.py scans them).
"""
import json
import os
from collections import Counter
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

from ..data.value_oracle import cell
from ..data.vs_sweep import MOVEMENT, note
from ..emu.vs import GROUND_Y, NAMES, view
from ..vocab import FULL_LIFE
from .screen_emu import frame_hash
from .system1 import RESULT, situation

CHUNK = 512                 # frames per RUN while replaying (a RUN boundary does not change the emulation)
FIELDS = ("table_cell", "can_act", "side", "range", "opp_airborne", "opp_crouch", "opp_attacking", "doing", "my_bar", "his_bar",
          "facing", "note", "situation")


class ReplayMismatch(RuntimeError):
    pass


def _life(v: int) -> int:
    return v if v < 200 else 0


def load_inputs(rec_dir: str) -> Tuple[Dict, List[List[str]], Dict[int, str]]:
    with open(os.path.join(rec_dir, "inputs.json")) as f:
        inp = json.load(f)
    with open(os.path.join(rec_dir, "captures.json")) as f:
        caps = {int(k): v for k, v in json.load(f).items()}
    frames = [[] if b == "-" else b.split("+") for b in inp["inputs"]]
    if len(frames) != inp["frames"]:
        raise ValueError("%s: %d inputs for %d frames" % (rec_dir, len(frames), inp["frames"]))
    return inp, frames, caps


def replay(bridge, state: bytes, frames: List[List[str]], caps: Dict[int, str],
           keep: Optional[set] = None) -> Tuple[List[Dict[str, int]], List[int], Dict[int, np.ndarray]]:
    """(RAM row before every frame 0..N, mismatching capture indices, the frames in ``keep``)."""
    obs = bridge.load_state(state)
    rows = [dict(zip(NAMES, obs.rams[-1]))]
    bad, kept = [], {}

    def check(k, img):
        if k in caps and frame_hash(img) != caps[k]:
            bad.append(k)
        if keep and k in keep:
            kept[k] = img

    check(0, obs.images[0])
    k0 = 0
    while k0 < len(frames):
        chunk = frames[k0:k0 + CHUNK]
        want = [k - k0 for k in caps if k0 < k <= k0 + len(chunk)]
        o = bridge.run(chunk, caps=want)
        rows.extend(dict(zip(NAMES, r)) for r in o.rams[1:])
        for c, img in o.images.items():
            check(k0 + c, img)
        k0 += len(chunk)
    if len(rows) != len(frames) + 1:
        raise RuntimeError("replay gave %d rows for %d frames" % (len(rows), len(frames)))
    return rows, sorted(bad), kept


def truth(me: str, opp: str, row: Dict[str, int], action: Optional[str]) -> Dict:
    """T0's facts and words from one RAM row (what the T0 path would have read at that frame)."""
    side = "left" if row["p1_x"] < row["p2_x"] else "right"
    text = note(me, opp, view(row, 1), side, version=2)
    rng, doing, my_bar, his_bar = situation(row)
    f = dict(kv.split("=") for kv in text.split()[1:])
    facing = None
    if action is not None:
        facing = (row["p1_x"] < row["p2_x"]) if action in MOVEMENT else row["p1_facing"] == 0x40
    return {"table_cell": list(cell(text)), "can_act": row["p1_state"] in (0, 2) and row["p1_y"] == GROUND_Y,
            "side": side, "range": rng,
            "opp_airborne": int(f["opp_airborne"]), "opp_crouch": int(f["opp_crouch"]),
            "opp_attacking": int(f["opp_attacking"]), "doing": doing, "my_bar": my_bar, "his_bar": his_bar,
            "facing": facing, "note": text, "situation": [rng, doing, my_bar, his_bar],
            "dx": row["p2_x"] - row["p1_x"], "my_life": _life(row["p1_life"]), "his_life": _life(row["p2_life"]),
            "p1_state": row["p1_state"], "p2_state": row["p2_state"]}


def seen(d: Dict) -> Dict:
    """The same fields from what the screen reader gave at the decision (decisions.jsonl)."""
    m = d["moment"]
    f = dict(kv.split("=") for kv in d["note"].split()[1:])
    movement = d.get("action") in MOVEMENT
    facing = (m["my_x"] < m["his_x"]) if (movement or m["my_facing"] is None) else m["my_facing"] == "right"
    return {"table_cell": list(cell(d["note"])), "can_act": m["can_act"], "side": m["side"], "range": d["situation"][0],
            "opp_airborne": int(f["opp_airborne"]), "opp_crouch": int(f["opp_crouch"]),
            "opp_attacking": int(f["opp_attacking"]), "doing": d["situation"][1], "my_bar": d["situation"][2],
            "his_bar": d["situation"][3], "facing": facing, "note": d["note"], "situation": d["situation"],
            "dx": m["dx"], "my_life": m["my_life"], "his_life": m["his_life"]}


def result_of(rows: List[Dict[str, int]]) -> Tuple[str, Optional[int]]:
    for k, r in enumerate(rows):
        if r["result"]:
            return RESULT.get(r["result"], "result_%d" % r["result"]), k
    return "unfinished", None


def drops(rows: List[Dict[str, int]], key: str) -> int:
    return sum(max(0, _life(a[key]) - _life(b[key])) for a, b in zip(rows, rows[1:]))


def score_round(bridge, state: bytes, rec_dir: str, me: str, opp: str) -> Dict:
    """Replay one recorded round and score it; raises ReplayMismatch (nothing scored) on any frame difference."""
    inp, frames, caps = load_inputs(rec_dir)
    with open(os.path.join(rec_dir, "decisions.jsonl")) as f:
        decs = [json.loads(line) for line in f]
    pngs = {d["k"] for d in decs} | {d["k_prev"] for d in decs}
    rows, bad, kept = replay(bridge, state, frames, caps, keep=pngs)
    for k in sorted(pngs):
        path = os.path.join(rec_dir, "frames", "%05d.png" % k)
        if os.path.exists(path):
            with Image.open(path) as im:
                if k not in kept or not np.array_equal(np.asarray(im.convert("RGB")), kept[k]):
                    bad.append(k)
    if bad:
        raise ReplayMismatch("%s: %d of %d captured frames differ in the replay (first: %s)"
                             % (rec_dir, len(set(bad)), len(caps), sorted(set(bad))[:10]))
    res, rk = result_of(rows)
    upto = rows[:rk + 1] if rk is not None else rows
    per = []
    agree = Counter()
    for d in decs:
        t, s = truth(me, opp, rows[d["k"]], d.get("action")), seen(d)
        same = {fld: t[fld] == s[fld] for fld in FIELDS}
        agree.update(fld for fld, ok in same.items() if ok)
        per.append({"i": d["i"], "k": d["k"], "action": d.get("action"), "ram": t, "screen": s, "same": same,
                    "dx_err": s["dx"] - t["dx"], "after_result": rk is not None and d["k"] > rk})
    reads_path = os.path.join(rec_dir, "reads.jsonl")
    reads = [json.loads(line) for line in open(reads_path)] if os.path.exists(reads_path) else []
    can = Counter()                        # (screen can act, RAM can act) over every frame read before the result
    for r in reads:
        if rk is None or r["k"] <= rk:
            row = rows[r["k"]]
            can[(bool(r["can_act"]), row["p1_state"] in (0, 2) and row["p1_y"] == GROUND_Y)] += 1
    n = len(decs)
    live = [p for p in per if not p["after_result"]]
    return {"rec": rec_dir, "frames": len(frames), "captures": len(caps), "replay_match": True,
            "result": res, "result_frame": rk, "screen_end_frame": inp["frames"],
            "dealt": drops(upto, "p2_life"), "taken": drops(upto, "p1_life"),
            "hp": drops(upto, "p2_life") - drops(upto, "p1_life"),
            "my_life_end": _life(upto[-1]["p1_life"]), "opp_life_end": _life(upto[-1]["p2_life"]),
            "decisions": n, "decisions_before_result": len(live),
            "agreement": {fld: round(agree[fld] / n, 4) if n else None for fld in FIELDS},
            "agreement_before_result": {fld: round(sum(p["same"][fld] for p in live) / len(live), 4) if live else None
                                        for fld in FIELDS},
            "dx_abs_err_median": float(np.median([abs(p["dx_err"]) for p in per])) if per else None,
            "dx_within_4px": round(sum(abs(p["dx_err"]) <= 4 for p in per) / n, 4) if n else None,
            "can_act_reads": {"both": can[(True, True)], "screen_only": can[(True, False)],
                              "ram_only": can[(False, True)], "neither": can[(False, False)]},
            "full_life": FULL_LIFE, "per_decision": per}

