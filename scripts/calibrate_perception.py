"""Calibrate the U arm's perception thresholds from logged data (docs/prereg_u_perception.md), before any training,
and build question 8's soft targets.

    python scripts/calibrate_perception.py \\
        --rows <his_moves probe>/*/rows.jsonl --lag-probe <his_moves probe>/ryu ... \\
        --ram <smoke>/chunli/ram.jsonl

From the decision logs (--logs, default the value collection and the random play vs Dhalsim):
  throw band  per character: the gap cut that best separates throws that connected (he went to "thrown") from throws
              that did not (a decision stump: no free threshold; opponent on the ground at the decision)
  poke band   the same for ground normals that connected (hit or blocked) vs whiffed
From per-frame RAM rows (--rows: the his_moves probe, or ram.jsonl sidecars):
  walls       per character the x it piles up at near each stage edge; the walls are the outermost; corner D = the
              widest character's offset from them (a wide fighter's own wall is short of the stage's)
  trend eps   half the smallest 4-frame displacement of a deliberate walk (rows that carry the inputs, "inp")
From --ram sidecars with their actions.jsonl:
  k           the median frames from my decision to contact of my fastest connecting move (>= MIN_CONTACTS)
From --lag-probe dirs (rows.jsonl every frame + img/sSS_tTTTT.png every 3rd): the display lag of the HUD clock.
Not decided by data (stated in "undecided"): mid_max (vocab.MID, the table's mid/far cut), near_px (= mid_max), and
any of the above whose source is missing.

Writes lessons/perception_thresholds_v1.json and (unless --no-q8) lessons/perception_q8_targets_v1.json. Every input is
recorded with its sha256 ("sources").

v1 (2026-10-01; the probe and the smoke game copied from the session scratchpad to logs/perception_probe/):
    P=logs/perception_probe
    python scripts/calibrate_perception.py --rows $P/{ryu,ken,honda}/rows.jsonl --lag-probe $P/{ryu,ken,honda} \
        --ram $P/smoke/chunli/ram.jsonl --no-q8
v2 (after the U collection, play_system1 --ram-log into rollouts/u_perception/<opp>/<me>/): k, walls and corner D from
the collection (all 8 characters' stages), the probe kept for the trend and the lag:
    python scripts/calibrate_perception.py --rows $P/{ryu,ken,honda}/rows.jsonl --lag-probe $P/{ryu,ken,honda} \
        --ram-root rollouts/u_perception --out lessons/perception_thresholds_v2.json --no-q8
"""
import argparse
import collections
import glob
import itertools
import json
import os
import statistics
import hashlib
import sys
from typing import Dict, Iterable, Iterator, List, Optional, Sequence

import _path  # noqa: F401
from sf2.data.perception import LAG, Q8_ANSWERS, STAGE_X, STUN, decode, q8_targets
from sf2.vocab import CHARACTERS, MID

NORMALS = ("lp", "mp", "hp", "lk", "mk", "hk", "c.lk", "c.mk", "sweep", "c.hp")
MIN_POSITIVES = 10       # a character's own band needs this many connects, else it uses the pooled one
MIN_CONTACTS = 5         # a move's frames-to-contact median needs this many contacts
MIN_PILE = 20            # frames at one x for it to be a character's wall (a pile-up, not a transient)
EDGE = 20                # px from the outermost x seen where a character's wall pile-up is looked for
DEFAULT_LOGS = ("rollouts/lv_value/*/*/actions.jsonl", "rollouts/live_dhalsim/*/actions.jsonl")


def read_jsonl(patterns: Iterable[str]) -> List[Dict]:
    paths = sorted(p for pat in patterns for p in glob.glob(pat))
    return [json.loads(line) for p in paths for line in open(p) if line.strip()]


# ---- bands ------------------------------------------------------------------------------------------------------

def stump(pos: Sequence[int], neg: Sequence[int]) -> Optional[int]:
    """The cut c (a gap seen in the data) that classifies the most entries right as "connects iff gap <= c"; the
    smallest of equally good cuts. None without positives."""
    if not pos:
        return None
    best, best_c = -1, None
    for c in sorted(set(pos) | set(neg)):
        right = sum(g <= c for g in pos) + sum(g > c for g in neg)
        if right > best:
            best, best_c = right, c
    return best_c


def _band(entries: Iterable[Dict], pick, connected) -> Dict:
    groups: Dict[str, List[List[int]]] = collections.defaultdict(lambda: [[], []])
    for e in entries:
        if pick(e) and not e["opp_air"]:
            groups[e["me"]][0 if connected(e) else 1].append(e["gap"])
            groups["all"][0 if connected(e) else 1].append(e["gap"])
    out: Dict = {"n": {}}
    for me, (pos, neg) in sorted(groups.items()):
        out["n"][me] = (len(pos), len(neg))
        if me == "all" or len(pos) >= MIN_POSITIVES:
            cut = stump(pos, neg)
            if cut is not None:
                out[me] = cut
    return out


def throw_band(entries: Iterable[Dict]) -> Dict:
    return _band(entries, lambda e: e["action"] == "throw", lambda e: "thrown" in e["opp_reaction"])


def poke_band(entries: Iterable[Dict]) -> Dict:
    return _band(entries, lambda e: e["action"] in NORMALS, lambda e: e["actual"] in ("hit", "blocked"))


# ---- walls, corner, trend ---------------------------------------------------------------------------------------

def walls(rows: Iterable[Dict]) -> Dict:
    """Streams ``rows`` once, keeping only x counts per character (a whole collection fits). Impossible x (outside
    0..STAGE_X: a wrapped negative) are dropped; a character's wall is the most frequent x within EDGE of the
    outermost x seen, and only with >= MIN_PILE frames there (a pile-up, not a transient). No wall on a side, or an
    impossible one: SystemExit."""
    xs: Dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    dropped = 0
    for r in rows:
        for p in (1, 2):
            x = r["p%d_x" % p]
            if not 0 <= x <= STAGE_X:
                dropped += 1
                continue
            xs[CHARACTERS.get(r["p%d_char" % p], str(r["p%d_char" % p]))][x] += 1
    lo_seen = min(x for v in xs.values() for x in v)
    hi_seen = max(x for v in xs.values() for x in v)
    by_char: Dict[str, Dict[str, int]] = {}
    for c, v in sorted(xs.items()):
        w = {}
        for side, near in (("lo", [x for x in sorted(v) if x <= lo_seen + EDGE]),
                           ("hi", [x for x in sorted(v) if x >= hi_seen - EDGE])):
            mode = max(near, key=lambda x: v[x]) if near else None      # the most frequent x near the edge
            if mode is not None and v[mode] >= MIN_PILE:
                w[side] = mode
        by_char[c] = w
    if not any("lo" in w for w in by_char.values()) or not any("hi" in w for w in by_char.values()):
        raise SystemExit("no wall pile-up (>= %d frames) on both sides: %s" % (MIN_PILE, by_char))
    lo = min(w["lo"] for w in by_char.values() if "lo" in w)
    hi = max(w["hi"] for w in by_char.values() if "hi" in w)
    d = max([w["lo"] - lo for w in by_char.values() if "lo" in w] + [hi - w["hi"] for w in by_char.values() if "hi" in w])
    if not 0 <= lo < hi <= STAGE_X:
        raise SystemExit("impossible walls %s" % [lo, hi])
    return {"walls": [lo, hi], "corner_d": d, "by_char": by_char, "x_seen": [lo_seen, hi_seen],
            "dropped_impossible_x": dropped}


def trend_eps(rows: Sequence[Dict]) -> Optional[Dict]:
    """Half the 5th percentile of player 1's 4-frame |dx| over runs where it held one walk direction, stood
    (state 0) and moved; None when the rows carry no inputs."""
    moves = []
    for i in range(len(rows) - 4):
        w = rows[i:i + 5]
        if "inp" not in w[0] or any(r.get("seg") != w[0].get("seg") for r in w):
            continue
        if all(r["inp"] == w[0]["inp"] and r["inp"] in (["left"], ["right"]) and r["p1_state"] == 0 for r in w[:4]):
            dx = abs(w[4]["p1_x"] - w[0]["p1_x"])
            if dx:
                moves.append(dx)
    if not moves:
        return None
    p5 = sorted(moves)[len(moves) // 20]
    return {"eps": p5 // 2, "walk_4f_p5": p5, "walk_4f_median": statistics.median(moves), "n": len(moves)}


# ---- k ----------------------------------------------------------------------------------------------------------

def _contact(rows: List[Dict], n: int) -> Optional[int]:
    """Frames from the decision row n to his first contact (entering hit / block stun or thrown, or a life drop; NOT
    his guard stance 0x08, which the CPU raises as my attack starts); None if he is already stunned at n or there is
    none."""
    if rows[n]["p2_state"] in STUN:
        return None
    for i in range(n + 1, len(rows)):
        if rows[i]["p2_state"] in STUN or rows[i]["p2_life"] < rows[i - 1]["p2_life"]:
            return i - n
    return None


def _new_seen() -> Dict[str, Dict[str, List[int]]]:
    return collections.defaultdict(lambda: collections.defaultdict(list))


def _add_contacts(seen, actions: Iterable[Dict], ram: Iterable[Dict]) -> None:
    """One log directory: its actions joined to its ram records by (game, frame) (keys repeat across directories);
    the ram records are streamed, never all held."""
    by_key = {(a["game"], a["frame"]): a for a in actions if a["actual"] in ("hit", "blocked")}
    for rec in ram:
        a = by_key.get((rec["game"], rec["frame"]))
        if a is None:
            continue
        rows, n = decode(rec)
        s = _contact(rows, n)
        if s is not None:
            seen[a["me"]][a["action"]].append(s)


def k_frames(actions: Iterable[Dict], ram: Iterable[Dict]) -> Dict:
    seen = _new_seen()
    _add_contacts(seen, actions, ram)
    return _k_summary(seen)


def _stream(path: str) -> Iterator[Dict]:
    with open(path) as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def ram_dirs(root: str) -> List[str]:
    """The collection's log directories: <root>/<opp>/<me>/ with a ram.jsonl (play_system1 --ram-log)."""
    return sorted(os.path.dirname(p) for p in glob.glob(os.path.join(root, "*", "*", "ram.jsonl")))


def k_from_root(root: str) -> Dict:
    seen = _new_seen()
    for d in ram_dirs(root):
        _add_contacts(seen, read_jsonl([os.path.join(d, "actions.jsonl")]), _stream(os.path.join(d, "ram.jsonl")))
    return _k_summary(seen)


def frames_of(recs: Iterable[Dict]) -> Iterator[Dict]:
    """The per-frame rows of a ram.jsonl's records (in file order), each frame once although the decision windows
    overlap: a row's frame is the record's frame + (its index - n)."""
    last: Dict[int, int] = {}
    for rec in recs:
        rows, n = decode(rec)
        for i, r in enumerate(rows):
            f = rec["frame"] + i - n
            if f > last.get(rec["game"], -1 << 30):
                last[rec["game"]] = f
                yield r


def _k_summary(seen) -> Dict:
    by_move = {me: {m: [float(statistics.median(v)), len(v)] for m, v in sorted(ms.items()) if len(v) >= MIN_CONTACTS}
               for me, ms in sorted(seen.items())}
    meds = [v[0] for ms in by_move.values() for v in ms.values()]
    every = {(me, m): statistics.median(v) for me, ms in seen.items() for m, v in ms.items()}
    return {"k": int(min(meds)) if meds else None, "by_move": by_move,
            "provisional": int(min(every.values())) if every else None,
            "contacts": sum(len(v) for ms in seen.values() for v in ms.values()),
            "all_moves": {"%s %s" % km: [float(med), len(seen[km[0]][km[1]])] for km, med in sorted(every.items())}}


# ---- display lag ------------------------------------------------------------------------------------------------

def display_lag(dirs: Sequence[str], lags=range(-1, 4)) -> Dict:
    """Agreement, per lag L, between "the HUD clock digits changed between images t and t+3" and "the RAM clock changed
    between rows t-L and t+3-L" (probe dirs: rows.jsonl every frame, img/sSS_tTTTT.png)."""
    import numpy as np
    from PIL import Image

    agree = {L: [0, 0] for L in lags}
    for d in dirs:
        rows = {(r["seg"], r["t"]): r for r in read_jsonl([os.path.join(d, "rows.jsonl")])}
        imgs = {(int(os.path.basename(f)[1:3]), int(os.path.basename(f)[5:9])): f
                for f in glob.glob(os.path.join(d, "img", "s*_t*.png"))}
        digits = {}
        for k, f in imgs.items():
            im = np.asarray(Image.open(f)).astype(int)[44:62, 112:144]
            digits[k] = (im[..., 2] > 150) & (im[..., 0] < 60)        # the clock's blue outline
        for (s, t) in imgs:
            if (s, t + 3) not in imgs:
                continue
            changed = bool((digits[(s, t)] != digits[(s, t + 3)]).any())
            for L in lags:
                a, b = rows.get((s, t - L)), rows.get((s, t + 3 - L))
                if a and b:
                    agree[L][0] += (a["timer"] != b["timer"]) == changed
                    agree[L][1] += 1
    share = {L: round(a / n, 4) for L, (a, n) in agree.items() if n}
    return {"lag": max(share, key=share.get) if share else None, "agreement": share,
            "pairs": max(n for _, n in agree.values()) if agree else 0}


# ---- main -------------------------------------------------------------------------------------------------------

def source(paths: Iterable[str]) -> List[Dict]:
    """Each input as {path, sha256, bytes} (a file) or {path, sha256, files} (a directory: its files' relative paths
    and contents, in sorted order), so the json names exactly what it was calibrated on."""
    out = []
    for p in paths:
        h = hashlib.sha256()
        if os.path.isdir(p):
            files = sorted(os.path.relpath(os.path.join(dp, f), p) for dp, _, fs in os.walk(p) for f in fs)
            for rel in files:
                h.update(rel.encode() + b"\0")
                with open(os.path.join(p, rel), "rb") as f:
                    h.update(f.read())
            out.append({"path": p, "sha256": h.hexdigest(), "files": len(files)})
        else:
            with open(p, "rb") as f:
                data = f.read()
            out.append({"path": p, "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)})
    return out


def _root_source(root: str) -> Dict:
    h = hashlib.sha256()
    files = [os.path.join(d, n) for d in ram_dirs(root) for n in ("ram.jsonl", "actions.jsonl")]
    for s in source(files):
        h.update((s["path"] + s["sha256"]).encode())
    return {"path": root, "sha256": h.hexdigest(), "files": len(files),
            "note": "sha256 over the ram.jsonl and actions.jsonl files (path + sha256 each)"}


def thresholds(args) -> Dict:
    entries = read_jsonl(args.logs)
    if not entries:
        raise SystemExit("no decision logs matched %s" % (args.logs,))
    th: Dict = {"undecided": {}, "sources": {"logs": list(args.logs), "entries": len(entries),
                                             "rows": source(args.rows), "ram": source(args.ram),
                                             "lag_probe": source(args.lag_probe)}}
    if args.ram_root:
        th["sources"]["ram_root"] = _root_source(args.ram_root)
    tb, pb = throw_band(entries), poke_band(entries)
    th["throw_max"] = {k: v for k, v in tb.items() if k != "n"}
    th["poke_max"] = {k: v for k, v in pb.items() if k != "n"}
    th["evidence"] = {"throw_n_connected_missed": tb["n"], "poke_n_connected_missed": pb["n"]}
    th["mid_max"] = MID
    th["undecided"]["mid_max"] = "vocab.MID (the table's mid/far cut), kept for the table's cells"
    th["near_px"] = MID
    th["undecided"]["near_px"] = "= mid_max: a projectile inside mid range is near (a judgment, not data)"
    rows = read_jsonl(args.rows)
    collected = (r for d in (ram_dirs(args.ram_root) if args.ram_root else [])
                 for r in frames_of(_stream(os.path.join(d, "ram.jsonl"))))
    if rows or args.ram_root:
        w = walls(itertools.chain(rows, collected))
        th.update(walls=w["walls"], corner_d=w["corner_d"])
        th["evidence"]["walls"] = w
        te = trend_eps(rows)
    else:
        th.update(walls=None, corner_d=None)
        th["undecided"]["walls"] = th["undecided"]["corner_d"] = "no per-frame rows (--rows)"
        te = None
    if te:
        th["trend_eps"] = te["eps"]
        th["evidence"]["trend"] = te
    else:
        th["trend_eps"] = 2
        th["undecided"]["trend_eps"] = "no rows with inputs: 2 px (stated, not measured)"
    seen = _new_seen()
    for p in args.ram:
        _add_contacts(seen, read_jsonl([os.path.join(os.path.dirname(p), "actions.jsonl")]), _stream(p))
    for d in ram_dirs(args.ram_root) if args.ram_root else []:
        _add_contacts(seen, read_jsonl([os.path.join(d, "actions.jsonl")]), _stream(os.path.join(d, "ram.jsonl")))
    k = _k_summary(seen)
    th["k"] = k["k"] if k["k"] is not None else k["provisional"]
    th["evidence"]["k"] = k
    if k["k"] is None:
        th["undecided"]["k"] = ("no move with >= %d contacts in --ram (%d contacts): k = %s is PROVISIONAL (the fastest "
                                "median seen); recompute on the collection's ram.jsonl" % (MIN_CONTACTS, k["contacts"],
                                                                                         k["provisional"]))
    if args.lag_probe:
        lag = display_lag(args.lag_probe)
        th["lag"] = lag["lag"]
        th["evidence"]["lag"] = lag
    else:
        th["lag"] = LAG
        th["undecided"]["lag"] = "not re-measured here (--lag-probe): sf2.data.perception.LAG"
    return th


def summary(th: Dict) -> str:
    lines = ["throw band (gap <=): %s" % th["throw_max"], "poke band  (gap <=): %s" % th["poke_max"],
             "mid/far at %s, near projectile < %s px" % (th["mid_max"], th["near_px"]),
             "walls %s, corner D %s, trend eps %s, k %s, lag %s" % (th["walls"], th["corner_d"], th["trend_eps"],
                                                                     th["k"], th["lag"])]
    lines += ["undecided: %s - %s" % kv for kv in th["undecided"].items()]
    return "\n".join(lines)


def q8_file(root: str, resamples: int, seed: int) -> List[Dict]:
    entries = read_jsonl([os.path.join(root, "*", "*", "actions.jsonl")])
    if not entries:
        raise SystemExit("no value-collection logs under %s" % root)
    t = q8_targets(entries, resamples=resamples, seed=seed)
    return [dict(cell=list(c), move=m, **v) for c, row in sorted(t.items()) for m, v in sorted(row.items())]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--logs", nargs="+", default=list(DEFAULT_LOGS))
    ap.add_argument("--rows", nargs="*", default=[], help="per-frame RAM rows jsonl (the his_moves probe)")
    ap.add_argument("--ram", nargs="*", default=[], help="ram.jsonl sidecars (play_system1 --ram-log)")
    ap.add_argument("--ram-root", default=None,
                    help="a --ram-log collection: <root>/<opp>/<me>/{ram,actions}.jsonl (walls, corner D and k)")
    ap.add_argument("--lag-probe", nargs="*", default=[], help="probe dirs with rows.jsonl and img/")
    ap.add_argument("--out", default="lessons/perception_thresholds_v1.json")
    ap.add_argument("--q8-root", default="rollouts/lv_value", help="the value collection behind the table")
    ap.add_argument("--q8-out", default="lessons/perception_q8_targets_v1.json")
    ap.add_argument("--resamples", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-q8", action="store_true")
    args = ap.parse_args()
    th = thresholds(args)
    with open(args.out, "w") as f:
        json.dump(th, f, indent=1, sort_keys=True)
    print(summary(th))
    print("wrote", args.out)
    if not args.no_q8:
        rows = q8_file(args.q8_root, args.resamples, args.seed)
        with open(args.q8_out, "w") as f:
            json.dump({"answers": list(Q8_ANSWERS), "resamples": args.resamples, "seed": args.seed,
                       "source": args.q8_root, "targets": rows}, f, indent=0)
        conf = sum(max(r["p"].values()) >= 0.9 for r in rows)
        print("q8: %d (cell, move) targets, %d confident (>= 0.9 on one answer), %d spread; wrote %s" % (
            len(rows), conf, len(rows) - conf, args.q8_out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
