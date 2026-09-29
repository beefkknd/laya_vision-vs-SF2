"""Collecting laya-vision's stage-1 data into shards (test_data/_shards/, scripts/vs_dataset.py collect /
collect-defense / import-live): still-opponent exchanges at set gaps, block exchanges, and rows from live play.
See sf2/data/vs_sweep.py for the design."""
import json
import os
from typing import Dict, List


from ..config import PAD, TEST_DATA
from .dataset import save_png, write_jsonl
from ..emu.headless import launch_argv
from ..emu.mesen import MesenBridge
from ..emu.vs import boot_vs, gap_state, record, view
from .vs_moves import CONDS
from . import vs_defense as D
from ..vocab import RANGES, range_of
from .vs_sweep import (GAPS, LEAD, MOVEMENT, POSTURES, PREV_GAP, note, outcome, split_of, static_actions)


ROOT = TEST_DATA


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
    """Block data (sf2/data/vs_defense.py): at every gap, the other fighter throws each probe attack and the character answers
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
