"""Stage-1 dataset (still opponent) per character, into test_data/<char>/. See sf2/vs_sweep.py for the design.

    # 1. collect shards (each its own headless Mesen; run them in parallel on different ports)
    python scripts/vs_dataset.py collect --p1 ryu --p2 chunli --who 1 --range close --port 48001   # Ryu, left
    python scripts/vs_dataset.py collect --p1 ryu --p2 chunli --who 2 --port 48004                  # Chun-Li, right
    # 2. merge, mirror, write per character
    python scripts/vs_dataset.py build
    # or both steps, all shards in parallel (the usual way):
    python scripts/vs_dataset.py run --chars ryu,chunli              # needs $SF2_ROM
    python scripts/vs_dataset.py run --chars ryu,chunli --no-right-test  # skip the real right-side check set
    python scripts/vs_dataset.py run --pairs ken:guile,honda:blanka,zangief:dhalsim   # 3 pairs, 36 jobs at once

A fighter on the left (``--who 1``) gives train + test; on the right (``--who 2``) test only (real frames for the
mirroring check). ``build`` exits 1 if any (action, range) combination is short of examples, or if an attack
never came out (a harness bug, not a data point).
"""
import argparse
import collections
import json
import os
import sys
from typing import Dict, List

import numpy as np

import _path  # noqa: F401
from sf2.config import PAD
from sf2.dataset import read, save_png, write_jsonl
from sf2.headless import launch_argv
from sf2.mesen import MesenBridge
from sf2.vs import boot_vs, gap_state, record, view
from sf2.vs_moves import CONDS
from sf2.frames import HUD_ROWS, mirror_frame, model_frame
from sf2 import vs_defense as D
from sf2.vocab import RANGES, range_of
from sf2.vs_sweep import (GAPS, LEAD, MOVEMENT, OUTCOMES, POSTURES, PREV_GAP, STAGE1_POSTURES, actions,
                          mirror_record, note, outcome, outcome_question, split_of, static_actions, current_note)

ROOT = "test_data"
MIN_TRAIN, MIN_TEST = 14, 6   # per (action, range): 7 gaps x 2 postures train, 3 x 2 test
MIN_DEF_TRAIN, MIN_DEF_TEST = 18, 8   # blocks, per (answer, probe): 7 gaps x 3 ranges train, 3 x 3 test
MIN_LIVE = 20                  # live play: training rows per move


def collect(args) -> int:
    me, opp = (args.p1, args.p2) if args.who == 1 else (args.p2, args.p1)
    side = "left" if args.who == 1 else "right"
    splits = {"train", "test"} if args.who == 1 else {"test"}
    ranges = [args.range] if args.range else list(RANGES)
    img_dir = os.path.join(ROOT, me, "images")
    shard_dir = os.path.join(ROOT, "_shards")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(shard_dir, exist_ok=True)
    shard = os.path.join(shard_dir, "%s_%s_%s.jsonl" % (me, side, "_".join(ranges)))
    acts = static_actions(me)

    b = MesenBridge(args.port, launch=launch_argv(args.port, args.rom, args.mesen))
    recs: List[Dict] = []
    skipped = []
    try:
        b.set_capture("raw")
        start = boot_vs(b, args.p1, args.p2)
        # power-on is not bit-exact (the reset's sub-frame timing leaves the RNG and animation counters differing:
        # damage and some frames vary between boots), so every record names the boot it came from, for replay
        boot = os.path.splitext(os.path.basename(shard))[0] + ".start.state"
        with open(os.path.join(shard_dir, boot), "wb") as f:
            f.write(start)
        for rng in ranges:
            for gi, target in enumerate(GAPS[rng]):
                split = split_of(gi)
                if split not in splits:
                    continue
                state, got = gap_state(b, start, target)
                if range_of(got) != rng:
                    skipped.append((rng, target, got))
                    continue
                for posture, hold in POSTURES.items():
                    key = "%s_%s_%d_%s" % (side, rng, gi, posture)
                    paths = ["images/%s_prev.png" % key, "images/%s_now.png" % key]
                    for n, (action, steps) in enumerate(acts.items()):
                        b.load_state(state)
                        take = record(b, args.who, (((), LEAD),) + steps, (), 240, CONDS, PAD,
                                      shots={LEAD - PREV_GAP, LEAD} if n == 0 else set(), d_hold=hold)
                        if n == 0:
                            for p, k in zip(paths, (LEAD - PREV_GAP, LEAD)):
                                save_png(take.images[k], os.path.join(ROOT, me, p))
                        rows = [view(r, args.who) for r in take.rows]
                        now = rows[LEAD]
                        pin = take.p1 if args.who == 1 else take.p2
                        nframes = sum(s[1] for s in steps)
                        rec = {"id": "%s-%s-%s" % (me, key, action), "char": me, "opp": opp, "side": side,
                               "facing": "right" if side == "left" else "left", "range": rng, "gap": abs(now["d_x"] -
                               now["a_x"]), "gap_index": gi, "dx": now["d_x"] - now["a_x"], "posture": posture,
                               "action": action, "kind": "movement" if action in MOVEMENT else "attack",
                               "buttons": pin[LEAD + 1: LEAD + 1 + nframes], "images": paths,
                               "state_text": note(me, opp, now, side),
                               "split": split, "mirrored": False, "source": "real", "boot": "_shards/" + boot}
                        rec.update(outcome(rows[LEAD:], action))
                        recs.append(rec)
                print("%s %s %-5s gap %3d (target %3d) %-5s: %d examples" % (me, side, rng, got, target, split,
                                                                               len(recs)), flush=True)
    finally:
        b.close()
    write_jsonl(shard, recs)
    print("wrote %d examples to %s; gaps skipped (out of range): %s" % (len(recs), shard, skipped or "none"))
    return 0


def collect_defense(args) -> int:
    """Block data (sf2/vs_defense.py): at every gap, the other fighter throws each probe attack and the character answers
    with block_high, block_low or nothing; RAM says what happened to its health."""
    me, opp = (args.p1, args.p2) if args.who == 1 else (args.p2, args.p1)
    side = "left" if args.who == 1 else "right"
    attacker = 3 - args.who
    splits = {"train", "test"} if args.who == 1 else {"test"}
    ranges = [args.range] if args.range else list(RANGES)
    shard_dir = os.path.join(ROOT, "_shards")
    os.makedirs(os.path.join(ROOT, me, "images"), exist_ok=True)
    os.makedirs(shard_dir, exist_ok=True)
    shard = os.path.join(shard_dir, "%s_%s_%s_defense.jsonl" % (me, side, "_".join(ranges)))
    probes = D.probes(static_actions(opp))
    b = MesenBridge(args.port, launch=launch_argv(args.port, args.rom, args.mesen))
    recs: List[Dict] = []
    try:
        b.set_capture("raw")
        start = boot_vs(b, args.p1, args.p2)
        boot = os.path.splitext(os.path.basename(shard))[0] + ".start.state"
        with open(os.path.join(shard_dir, boot), "wb") as f:
            f.write(start)
        for rng in ranges:
            for gi, target in enumerate(GAPS[rng]):
                split = split_of(gi)
                if split not in splits:
                    continue
                state, got = gap_state(b, start, target)
                if range_of(got) != rng:
                    continue
                for name, p in probes.items():
                    recs += _defense_rows(b, state, args.who, attacker, me, opp, side, rng, gi, split, name, p, boot)
                print("%s %s defense %-5s gap %3d %-5s: %d examples" % (me, side, rng, got, split, len(recs)),
                      flush=True)
    finally:
        b.close()
    write_jsonl(shard, recs)
    print("wrote %d examples to %s" % (len(recs), shard))
    return 0


def _defense_take(b, state, who, attacker, p, answer, conds, shots):
    b.load_state(state)
    return record(b, attacker, p["attacker"], p["lead"] + ((D.ANSWERS[answer], D.HOLD),), 60, conds, PAD,
                  shots=shots, track=(who,))     # a block holds away from the attacker, even after a cross-up


def _defense_rows(b, state, who, attacker, me, opp, side, rng, gi, split, name, p, boot) -> List[Dict]:
    """One probe at one gap: find the take (for the jump-in, the kick height that lands on a character who does not
    block), save its decision frames, then every answer's outcome."""
    heights = D.KICK_HEIGHTS if name == "jump_in" else (D.DESCEND_Y,)
    for kh in heights:
        conds = D.conds(kh)
        idle = _defense_take(b, state, who, attacker, p, "idle", conds, set(range(0, 200)))
        rows = [view(r, attacker) for r in idle.rows]
        k = D.decision_frame(rows, name)
        if D.outcome(rows[k:])["outcome"] == "got_hit":
            break
    key = "%s_%s_%d_def_%s" % (side, rng, gi, name.replace(".", ""))
    paths = ["images/%s_prev.png" % key, "images/%s_now.png" % key]
    for path, f in zip(paths, (k - PREV_GAP, k)):
        save_png(idle.images[f], os.path.join(ROOT, me, path))
    mine = view(idle.rows[k], who)
    base = D.outcome(rows[k:])
    out = []
    for answer in D.ANSWERS:
        take = idle if answer == "idle" else _defense_take(b, state, who, attacker, p, answer, conds, set())
        o = D.outcome([view(r, attacker) for r in take.rows][k:])
        pin = take.p1 if who == 1 else take.p2
        out.append({
            "id": "%s-%s-%s" % (me, key, answer), "char": me, "opp": opp, "side": side,
            "facing": "right" if side == "left" else "left", "range": range_of(abs(mine["d_x"] - mine["a_x"])),
            "gap": abs(mine["d_x"] - mine["a_x"]),
            "gap_index": gi, "dx": mine["d_x"] - mine["a_x"], "posture": "stand", "action": answer,
            "kind": "defense", "probe": name, "probe_height": p["height"], "kick_y": kh if name == "jump_in" else None,
            "buttons": pin[k + 1: k + 1 + (D.HOLD if answer != "idle" else 0)], "images": paths,
            "state_text": note(me, opp, mine, side), "split": split, "mirrored": False, "source": "real",
            "boot": "_shards/" + boot, "outcome": o["outcome"], "damage_taken": o["damage_taken"],
            "damage_saved": base["damage_taken"] - o["damage_taken"], "damage": 0, "thrown": False,
            "executed": True, "attacked": False, "busy_frames": 0, "travel": 0})
    return out


def import_live(args) -> int:
    """Live play (scripts/play_system1.py game logs, e.g. the explorer vs the CPU Dhalsim) -> shards of laya rows: the
    decision's two frames and note, the move, and what really happened (RAM). Held out by game: games whose number ends
    in TEST_INDEX go to test."""
    import shutil
    total = 0
    for char in sorted(os.listdir(args.log)):
        path = os.path.join(args.log, char, "actions.jsonl")
        if not os.path.exists(path):
            continue
        os.makedirs(os.path.join(ROOT, char, "images"), exist_ok=True)
        recs = []
        for line in open(path):
            a = json.loads(line)
            key = "live_%s_g%02d_f%05d" % (os.path.basename(os.path.normpath(args.log)), a["game"], a["frame"])
            paths = ["images/%s_prev.png" % key, "images/%s_now.png" % key]
            for src, dst in zip(a["images"], paths):
                shutil.copy(os.path.join(args.log, char, "images", src), os.path.join(ROOT, char, dst))
            dx = a["gap"] if a["side"] == "left" else -a["gap"]
            recs.append({
                "id": "%s-%s-%s" % (char, key, a["action"]), "char": char, "opp": a["opp"], "side": a["side"],
                "facing": "right" if a["side"] == "left" else "left", "range": a["range"], "gap": a["gap"],
                "gap_index": a["game"] % 10, "game": a["game"], "dx": dx, "posture": "live", "action": a["action"],
                "kind": "live", "move_kind": a["kind"], "buttons": [], "images": paths,
                "state_text": a["prompt"].split("\nmemory")[0], "split": split_of(a["game"] % 10), "mirrored": False,
                "source": "live:" + args.log, "boot": None, "outcome": a["actual"], "damage": a["dealt"],
                "damage_taken": a["taken"], "thrown": False, "executed": True, "attacked": a["kind"] == "attack",
                "busy_frames": a["frames"], "travel": 0, "my_life": a["my_life"], "opp_life": a["opp_life"],
                "opp_air": a["opp_air"]})
        write_jsonl(os.path.join(ROOT, "_shards", "%s_live.jsonl" % char), recs)
        total += len(recs)
        print("%-8s %d live rows (%d train, %d test)" % (char, len(recs), sum(r["split"] == "train" for r in recs),
                                                         sum(r["split"] == "test" for r in recs)))
    print("imported %d live rows from %s" % (total, args.log))
    return 0


def _load(path: str) -> np.ndarray:
    from PIL import Image

    with Image.open(path) as im:
        return np.asarray(im.convert("RGB")).copy()


def _write_frames(base: str, recs: List[Dict], mirrored: List[Dict]) -> List[str]:
    """frames/<x>.png = model_frame(images/<x>.png); frames/mirror_<x>.png = its whole-frame flip. Checked: every
    referenced frame exists, its HUD rows are black, and each mirror is exactly the flip of its source."""
    os.makedirs(os.path.join(base, "frames"), exist_ok=True)
    problems = []
    for r in recs:
        for p in r["images"]:
            raw = _load(os.path.join(base, p.replace("frames/", "images/")))
            save_png(model_frame(raw), os.path.join(base, p))
    for m in mirrored:
        for p in m["images"]:
            src = p.replace("frames/mirror_", "frames/")
            save_png(mirror_frame(_load(os.path.join(base, src.replace("frames/", "images/")))), os.path.join(base, p))
    checked = set()
    for r in recs + mirrored:
        for p in r["images"]:
            if p in checked:
                continue
            checked.add(p)
            img = _load(os.path.join(base, p))
            if img[:HUD_ROWS].any():
                problems.append("%s: HUD not blank" % p)
            if "mirror_" in p and not np.array_equal(img, _load(os.path.join(base, p.replace("mirror_", "")))[:, ::-1]):
                problems.append("%s: not the exact flip of its source" % p)
    return problems


def _laya(recs: List[Dict]) -> List[Dict]:
    return [dict(r, question=outcome_question(r["action"]), label=OUTCOMES.index(r["outcome"])) for r in recs]


def _file_side(r: Dict) -> str:
    """The side a row was COLLECTED on (which file it belongs in); a cross-up block row's own side is where the
    character ended up at the decision frame."""
    return r.get("collected_side", r["side"])


def build(args) -> int:
    shards = [os.path.join(ROOT, "_shards", f) for f in sorted(os.listdir(os.path.join(ROOT, "_shards")))
              if f.endswith(".jsonl")]
    by_char: Dict[str, List[Dict]] = collections.defaultdict(list)
    for s in shards:
        for r in read(s):
            if r["kind"] == "live" and r["dx"] == 0:
                continue        # the fighters at the same x (crossing): no left or right, a mirror would contradict it
            if r["posture"] in STAGE1_POSTURES or r["kind"] == "live":
                r["state_text"] = current_note(r)    # today's note: no opponent name, general bars, no constants
                if r["kind"] == "defense":     # what laya SEES at the decision frame (a jump-in has flown in, and
                    r["range"] = range_of(r["gap"])    # may have crossed over: then the character is on the other side)
                    r["collected_side"] = r.get("collected_side", r["side"])
                    now = "left" if r["dx"] > 0 else "right"
                    if now != r["side"]:
                        r["state_text"] = r["state_text"].replace("side=" + r["side"], "side=" + now)
                        r["side"], r["facing"] = now, "right" if now == "left" else "left"
                by_char[r["char"]].append(r)
    problems = []
    # live play: the same number of training rows for every character, and never more than its other data (the
    # still-opponent + block rows), so live play cannot dominate laya (System 1 stays general); earliest games first
    live_n = {c: sum(r["kind"] == "live" and r["split"] == "train" for r in rs) for c, rs in by_char.items()}
    other_n = {c: sum(r["kind"] != "live" and r["split"] == "train" and _file_side(r) == "left" for r in rs)
               for c, rs in by_char.items()}
    cap = min([n for n in live_n.values() if n] + list(other_n.values())) if any(live_n.values()) else 0
    for c, rs in by_char.items():
        live_train = sorted((r for r in rs if r["kind"] == "live" and r["split"] == "train"),
                            key=lambda r: (r["game"], r["id"]))
        drop = {id(r) for r in live_train[cap:]}
        by_char[c] = [r for r in rs if id(r) not in drop]
    if cap:
        print("live training rows per character: %d (capped; before %s)" % (cap, live_n))
    for char, recs in sorted(by_char.items()):
        base = os.path.join(ROOT, char)
        # model frames: raw captures (images/) with the HUD blanked, in frames/; mirrored ones flipped whole
        recs = [dict(r, images=[p.replace("images/", "frames/") for p in r["images"]]) for r in recs]
        # training: the left-side rows (their mirror is the right side) and every live row (live play has both sides)
        train = [r for r in recs if r["split"] == "train" and (_file_side(r) == "left" or r["kind"] == "live")]
        mirrored = [dict(m, images=[p.replace("frames/", "frames/mirror_") for p in m["images"]])
                    for m in map(mirror_record, train)]
        problems += _write_frames(base, recs, mirrored)
        tests = {s: [r for r in recs if r["split"] == "test" and _file_side(r) == s] for s in ("left", "right")}
        # laya-vision's loader (laya.vlm_train.jsonl_example) needs a question dict and an int label
        train, mirrored = _laya(train), _laya(mirrored)
        tests = {s: _laya(rs) for s, rs in tests.items()}
        write_jsonl(os.path.join(base, "train.jsonl"), train + mirrored)   # what scripts/train.py reads
        write_jsonl(os.path.join(base, "train_real.jsonl"), train)
        write_jsonl(os.path.join(base, "train_mirrored.jsonl"), mirrored)
        for s, rs in tests.items():
            write_jsonl(os.path.join(base, "test_real_%s.jsonl" % s), rs)
        # gates: every (action, range) combination filled; every attack came out
        count = collections.Counter((r["action"], r["range"], r["split"], r["side"]) for r in recs
                                    if r["kind"] != "defense")
        for action in static_actions(char):
            for rng in RANGES:
                need_sides = (("train", "left", MIN_TRAIN), ("test", "left", MIN_TEST)) + (
                    (("test", "right", MIN_TEST),) if args.right_test else ())
                for split, side, need in need_sides:
                    if count[(action, rng, split, side)] < need:
                        problems.append("%s %s %s %s/%s: %d < %d" % (char, action, rng, split, side,
                                                                       count[(action, rng, split, side)], need))
        problems += ["%s %s never came out (%s)" % (char, r["action"], r["id"]) for r in recs if not r["executed"]]
        live = [r for r in recs if r["kind"] == "live"]
        if live:     # live play: every move tried in training, and a held-out share
            lc = collections.Counter(r["action"] for r in live if r["split"] == "train")
            problems += ["%s live: %s has %d training rows < %d" % (char, a, lc[a], MIN_LIVE) for a in actions(char)
                         if lc[a] < MIN_LIVE]
            if not [r for r in live if r["split"] == "test"]:
                problems.append("%s live: no held-out games" % char)
        # blocks: every answer to every probe, on the training side and on both test sides
        dcount = collections.Counter((r["action"], r["probe"], r["split"], _file_side(r)) for r in recs
                                     if r["kind"] == "defense")
        for answer in D.ANSWERS:
            for probe in ("s.hk", "c.mk", "sweep", "jump_in"):
                for split, side, need in (("train", "left", MIN_DEF_TRAIN), ("test", "left", MIN_DEF_TEST)) + (
                        (("test", "right", MIN_DEF_TEST),) if args.right_test else ()):
                    if dcount[(answer, probe, split, side)] < need:
                        problems.append("%s %s vs %s %s/%s: %d < %d" % (char, answer, probe, split, side,
                                                                      dcount[(answer, probe, split, side)], need))
        table = collections.defaultdict(collections.Counter)
        for r in train:
            table[(r["action"], r["range"])][r["outcome"]] += 1
        stats = {"train": len(train), "train_mirrored": len(mirrored),
                 "test_real_left": len(tests["left"]), "test_real_right": len(tests["right"]),
                 "outcomes_train": {"%s@%s" % k: dict(v) for k, v in sorted(table.items())}}
        with open(os.path.join(base, "stats.json"), "w") as f:
            json.dump(stats, f, indent=1)
        print("%-7s train %d (+%d mirrored)  test real left %d / right %d" % (
            char, len(train), len(mirrored), len(tests["left"]), len(tests["right"])))
    for p in problems:
        print("PROBLEM", p)
    return 1 if problems else 0


def run(args) -> int:
    """Everything, in parallel: per character, one collect per range on the left (train + test) and, with
    --right-test, one per range on the right (real test frames, the mirroring check); then build. Each character's
    still opponent is --dummy, or else the next character in --chars (the last one's is the first)."""
    import subprocess

    if args.pairs:   # "a:b,c:d": a's dummy is b and b's is a
        who_vs = [(x, y) for pair in args.pairs.split(",") for a, b in [pair.split(":")] for x, y in ((a, b), (b, a))]
    else:
        chars = args.chars.split(",")
        who_vs = [(me, args.dummy or chars[(i + 1) % len(chars)]) for i, me in enumerate(chars)]
    jobs, port = [], args.base_port
    os.makedirs(os.path.join("logs", "dataset"), exist_ok=True)
    for me, dummy in who_vs:
        if dummy == me:
            raise SystemExit("%s needs a different dummy (--dummy, or two or more --chars)" % me)
        for who in (1, 2) if args.right_test else (1,):
            p1, p2 = (me, dummy) if who == 1 else (dummy, me)
            for rng in RANGES:
                for kind in args.kinds.split(","):
                    log = os.path.join("logs", "dataset", "%s_%s_%s_%s.log" % (
                        me, "left" if who == 1 else "right", rng, kind))
                    cmd = [sys.executable, os.path.abspath(__file__), "collect" if kind == "static" else
                           "collect-defense", "--p1", p1, "--p2", p2, "--who", str(who), "--range", rng,
                           "--port", str(port)] + (["--rom", args.rom] if args.rom else [])
                    jobs.append((log, subprocess.Popen(cmd, stdout=open(log, "w"), stderr=subprocess.STDOUT)))
                    port += 1
    print("%d collect jobs running (logs in logs/dataset/)" % len(jobs), flush=True)
    failed = [log for log, proc in jobs if proc.wait() != 0]
    for log in failed:
        print("FAILED collect, see", log)
    if failed:
        return 1
    return build(args)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name, text in (("collect", "still-opponent data"), ("collect-defense", "block data (sf2/vs_defense.py)")):
        c = sub.add_parser(name, help=text)
        c.add_argument("--p1", required=True)
        c.add_argument("--p2", required=True)
        c.add_argument("--who", type=int, choices=(1, 2), required=True)
        c.add_argument("--range", choices=RANGES)
        c.add_argument("--port", type=int, required=True)
        c.add_argument("--rom", default=os.environ.get("SF2_ROM"))
        c.add_argument("--mesen", default=os.environ.get("SF2_MESEN"))
    il = sub.add_parser("import-live", help="game logs of live play -> shards (scripts/play_system1.py output)")
    il.add_argument("--log", required=True, help="e.g. rollouts/live_dhalsim")
    bd = sub.add_parser("build")
    r = sub.add_parser("run", help="collect every shard in parallel, then build")
    r.add_argument("--kinds", default="static,defense", help="static (still opponent), defense (blocks), or both")
    r.add_argument("--chars", default="ryu,chunli", help="comma-separated; each gets its own test_data/<char>/")
    r.add_argument("--pairs", help="a:b,c:d - each pair are each other's still opponent (overrides --chars)")
    r.add_argument("--dummy", help="the still opponent for every character (default: the next in --chars)")
    r.add_argument("--base-port", type=int, default=48001)
    r.add_argument("--rom", default=os.environ.get("SF2_ROM"))
    for p in (bd, r):
        p.add_argument("--no-right-test", dest="right_test", action="store_false",
                       help="skip the real right-side test set (the one-time mirroring check)")
    args = ap.parse_args()
    return {"collect": collect, "collect-defense": collect_defense, "import-live": import_live, "build": build,
            "run": run}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
