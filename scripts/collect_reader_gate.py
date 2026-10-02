"""Held-out gate data for the screen reader (docs/laya_text_only_plan.md, "Step 1-2 pre-registration"). Collection
time: RAM and the OAM snapshot are recorded here as the REFEREE only, saved next to the frames; the reader
(sf2/screen) never sees them.

    .venv/bin/python scripts/collect_reader_gate.py                 # both sets, many Mesens at once
    .venv/bin/python scripts/collect_reader_gate.py --sets a --limit 1   # a smoke
    .venv/bin/python scripts/collect_reader_gate.py --anchors          # out/screen_reader/anchors.json (catalog games)
    .venv/bin/python scripts/collect_reader_gate.py --calib-hud        # out/screen_reader/hud_digits.npz
    .venv/bin/python scripts/collect_reader_gate.py --fixtures         # tests/fixtures/screen/

Sets:
  a  2P versus on Ryu's stage (states/vs_<A>_vs_<B>.state, both controllers ours, sf2.data.pairs_collect.play_both),
     16 ordered pairs, seed 101 (the catalog used seed 0): every character twice as player 1 and twice as player 2.
  b  Chun-Li (player 1, a scripted cycle of her move list, sf2.data.pairs_collect.play_directed) vs the arcade CPU,
     each of 6 opponents on his own stage (states/p1_chunli_vs_<opp>.state), ROUNDS rounds each (another random start).
Every frame is stepped one at a time: the raw screen (RUN caps=[1]) and the sprite snapshot (VIDEO). After the ROM's
round result, TAIL more idle frames (so "round over" can be checked after it).

<out>/<set>/<name>/
  frames.npz   frames (n, 224, 256, 3) uint8; frames[k] = the screen captured at stream row k (row 0: none, zeros).
               It shows RAM row k - LAG (sf2.data.perception.LAG = 1) for the state labels.
  rows.npz     rows (n, m) int32, names: sf2.emu.vs.VARS + the extra label bytes + cam_x (0x19E7) / cam_y (0x19EA);
               pressed (2, n): the pressed class per slot (sf2.data.pairs_moves.pressed_class)
  oam.json.gz  per row k >= 1: {"k", "groups": {pal: [key, x0, y0, x1, y1, oam_facing]}} (pal 4 = p1, 6 = p2, 5 / 7)
  meta.json    set, pair, chars, stage, seed, result, result_k (first row with result != 0), moves, n
"""
import argparse
import gzip
import json
import os
import random
import sys
import time
from typing import Dict, List, Optional

import numpy as np

import _path  # noqa: F401
from sf2.config import REPO
from sf2.data import pairs_collect as PC
from sf2.data import pairs_moves as PM
from sf2.data.action_codes import EXTRA_NAMES, EXTRA_VARS
from sf2.emu.ram import Var
from sf2.emu.vs import NAMES, VARS
from sf2.eval.budget import Budget
from sf2.eval.runner import exit_on_sigterm, fan_out
from sf2.sprites.catalog import canonical, key_of
from sf2.sprites.collect import oam_facing
from sf2.sprites.emu import open_mesen, video
from sf2.sprites.oam import render_groups
from sf2.vocab import IDS

OUT = os.path.join(REPO, "out", "screen_gate", "data")
BASE_PORT = 53301
SEED = 101
TAIL = 120
ROUNDS = 4
CHARS = ("blanka", "chunli", "dhalsim", "guile", "honda", "ken", "ryu", "zangief")
OPPONENTS = ("ryu", "ken", "honda", "zangief", "guile", "dhalsim")
CAM_VARS = [Var("cam_x", 0x19E7, 2, False), Var("cam_y", 0x19EA, 1, True)]
ALL_VARS = VARS + EXTRA_VARS + CAM_VARS
ALL_NAMES = NAMES + EXTRA_NAMES + [v.name for v in CAM_VARS]
PALS = (4, 5, 6, 7)
H, W = 224, 256


def pairs_a() -> List[tuple]:
    """16 ordered pairs: (c[i], c[i+1]) and (c[i], c[i+3]) over the 8 characters (each twice per side)."""
    n = len(CHARS)
    return [(CHARS[i], CHARS[(i + d) % n]) for d in (1, 3) for i in range(n)]


def jobs() -> List[Dict]:
    a = [dict(set="a", name="%s_vs_%s" % (p1, p2), p1=p1, p2=p2, game=0,
              state=os.path.join("states", "vs_%s_vs_%s.state" % (p1, p2))) for p1, p2 in pairs_a()]
    b = [dict(set="b", name="chunli_vs_%s_r%d" % (o, r), p1="chunli", p2=o, game=r,
              state=os.path.join("states", "p1_chunli_vs_%s.state" % o)) for o in OPPONENTS for r in range(ROUNDS)]
    return a + b


class Recorder:
    """Steps the bridge one frame at a time (raw screen + sprite snapshot each) and records rows, frames, OAM truth.
    Looks like a bridge to play_both / play_directed (run / load_state)."""

    def __init__(self, bridge):
        self.inner = bridge
        self.rows: List[List[int]] = []
        self.frames: List[np.ndarray] = []
        self.oam: List[Dict] = []
        self.fresh = True

    def load_state(self, state: bytes):
        if self.rows:
            raise RuntimeError("one game per recorder")
        obs = self.inner.load_state(state)
        self.rows.append(list(obs.rams[-1]))
        self.frames.append(np.zeros((H, W, 3), np.uint8))
        self.oam.append({})
        return obs

    def run(self, frames, caps=(), p2=None):
        from sf2.emu.mesen import Obs
        rams = [self.rows[-1]]
        for j, f in enumerate(frames):
            obs = self.inner.run([f], caps=[1], p2=None if p2 is None else [p2[j]])
            img = obs.images[1]
            if img.shape != (H, W, 3):
                raise RuntimeError("frame shape %s" % (img.shape,))
            self.rows.append(list(obs.rams[1]))
            self.frames.append(img)
            self.oam.append(truth(video(self.inner)))
            rams.append(obs.rams[1])
        return Obs(rams)

    def __len__(self):
        return len(self.rows)


def truth(snap) -> Dict[str, list]:
    out = {}
    for pal, g in render_groups(snap, pals=set(PALS)).items():
        face = oam_facing(snap, pal)
        out[str(pal)] = [key_of(canonical(g.rgba, face)), *g.bbox, face]
    return out


def play(job: Dict, port: int, rom: Optional[str]) -> Dict:
    with open(os.path.join(REPO, job["state"]), "rb") as f:
        state = f.read()
    t0 = time.time()
    with open_mesen(port, rom) as br:
        br.set_capture("raw")
        br.set_vars(ALL_VARS)
        rec = Recorder(br)
        r0 = dict(zip(ALL_NAMES, br.load_state(state).rams[-1]))
        if (r0["p1_char"], r0["p2_char"]) != (IDS[job["p1"]], IDS[job["p2"]]):
            raise SystemExit("%s holds %s" % (job["state"], (r0["p1_char"], r0["p2_char"])))
        key = "%d:%s:%d" % (job.get("seed", SEED), job["name"], job["game"])
        start = random.Random("start:" + key)
        if job["set"] == "a":
            chars = {1: job["p1"], 2: job["p2"]}
            cycles = {p: PM.Cycle(list(PM.moves(c)), random.Random("cycle%d:%s" % (p, key))) for p, c in chars.items()}
            summary = PC.play_both(rec, ALL_NAMES, chars, cycles, state, start, rec.__len__)
        else:
            cyc = PM.Cycle(list(PM.moves("chunli")), random.Random("cycle1:" + key))
            summary = PC.play_directed(rec, ALL_NAMES, "chunli", cyc, state, start, rec.__len__)
            summary["moves"] = [m[:3] + [1] for m in summary["moves"]]
        rec.run([[]] * TAIL, p2=[[]] * TAIL if job["set"] == "a" else None)
    summary["seconds"] = round(time.time() - t0, 1)
    return write(job, rec, summary)


def write(job: Dict, rec: Recorder, summary: Dict) -> Dict:
    base = os.path.join(job["out"], job["set"], job["name"])
    os.makedirs(base, exist_ok=True)
    n = len(rec.rows)
    rows = np.array(rec.rows, np.int32)
    words = PM.pressed_words(summary["moves"], n)
    chars = {1: job["p1"], 2: job["p2"]}
    pressed = np.array([[PM.pressed_class(chars[p], w) or "" for w in words[p]] for p in (1, 2)])
    np.savez_compressed(os.path.join(base, "rows.npz"), rows=rows, names=np.array(ALL_NAMES), pressed=pressed)
    res = rows[:, ALL_NAMES.index("result")]
    nz = np.nonzero(res)[0]
    with gzip.open(os.path.join(base, "oam.json.gz"), "wt") as f:
        for k, g in enumerate(rec.oam):
            f.write(json.dumps(dict(k=k, groups=g)) + "\n")
    meta = dict(set=job["set"], name=job["name"], p1=job["p1"], p2=job["p2"], game=job["game"],
                seed=job.get("seed", SEED),
                stage="ryu" if job["set"] == "a" else job["p2"], state=job["state"], n=n,
                result=summary["result"], result_k=int(nz[0]) if len(nz) else None, tail=TAIL,
                moves=summary["moves"], seconds=summary["seconds"])
    np.savez_compressed(os.path.join(base, "frames.npz"), frames=np.stack(rec.frames))
    with open(os.path.join(base, "meta.json"), "w") as f:
        json.dump(meta, f)
    return meta


HUD_CALIB = os.path.join(REPO, "out", "screen_reader", "hud_calib.npz")
CALIB_STATE = os.path.join("states", "vs_ryu_vs_ken.state")     # Ryu's stage; NOT a gate game (no moves played)


def calib_hud(port: int, rom: Optional[str], path: str = HUD_CALIB) -> str:
    """HUD calibration frames (collection time, RAM = the label): player 1 walks back and forth (the camera moves, so
    the background behind the clock changes), nobody attacks, the clock runs 99 -> 00. Keeps up to 6 frames per clock
    value, each from a row whose clock byte equals that of the 3 rows before it (the screen shows a settled value).
    Writes frames (n, 224, 256, 3) and clock (n,) (the BCD byte as a number 0-99) to ``path``."""
    with open(os.path.join(REPO, CALIB_STATE), "rb") as f:
        state = f.read()
    keep, clock = [], []
    per: Dict[int, int] = {}
    with open_mesen(port, rom) as br:
        br.set_capture("raw")
        br.set_vars(ALL_VARS)
        br.load_state(state)
        hist: List[int] = []
        for k in range(12000):
            pad = ["left"] if (k // 90) % 2 == 0 else ["right"]
            obs = br.run([pad], caps=[1], p2=[[]])
            r = dict(zip(ALL_NAMES, obs.rams[1]))
            hist.append(r["timer"])
            value = (r["timer"] >> 4) * 10 + (r["timer"] & 15)
            if len(hist) >= 4 and len(set(hist[-4:])) == 1 and per.get(value, 0) < 6 and k % 7 == 0:
                per[value] = per.get(value, 0) + 1
                keep.append(obs.images[1])
                clock.append(value)
            if r["result"]:
                break
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez_compressed(path, frames=np.stack(keep), clock=np.array(clock))
    return path


ANCHORS = os.path.join(REPO, "out", "screen_reader", "anchors.json")
CATALOG = os.path.join(REPO, "out", "sprite_catalog")
START_CAM = (128, 0)       # the camera at a round's first rows (probe 2026-10-02: 0x19E7 = 128, 0x19EA = 0)
START_ROWS = 4


def build_anchors(catalog: str = CATALOG, path: str = ANCHORS) -> Dict:
    """Per catalog sprite and drawn facing, where RAM's (x, y) sits relative to the sprite's box (collection-time
    labelling from the catalog's own games, sf2.sprites.collect: bbox per row, RAM rows; lag 0 - the probe found the
    box and the camera of the same row agree exactly). The camera is not in those rows, but both fighters share it:
    off(u1) - off(u2) = (x0_1 - x_1) - (x0_2 - x_2) on every row; the first START_ROWS rows of each game fix the
    constant (camera START_CAM). Solved over the graph of sprites seen together, the most common difference per pair,
    a maximum spanning tree from the start rows. Writes {"<char>/<key>|<face>": [dx, n]}: screen x = x0 - dx; plus a
    consistency figure (rows whose two sprites agree with the table). Only x: the same check on the box bottom vs RAM y
    agreed on 56% of rows only (y is not a function of the sprite and the shared camera), so y is not anchored."""
    import glob
    import heapq
    from collections import Counter, defaultdict
    from sf2.sprites import build as B
    pair_d: Dict[tuple, Counter] = defaultdict(Counter)
    absolute: Dict[str, Counter] = defaultdict(Counter)
    rows_seen = []
    for pair_dir in sorted(glob.glob(os.path.join(catalog, "_pairs", "*_vs_*"))):
        a, b = os.path.basename(pair_dir).split("_vs_")
        chars = {"p1": a, "p2": b}
        for game, frames, rows, pressed in B.frames_of(pair_dir):
            by_k: Dict[int, Dict[str, tuple]] = defaultdict(dict)
            for fr in frames:
                if fr["group"] in chars and fr["oam_facing"] is not None:
                    p = 1 if fr["group"] == "p1" else 2
                    r = rows[fr["k"]]
                    u = "%s/%s|%d" % (chars[fr["group"]], fr["key"], fr["oam_facing"])
                    by_k[fr["k"]][fr["group"]] = (u, fr["bbox"][0] - r["p%d_x" % p])
            for k, g in by_k.items():
                for u, dx in g.values():
                    if k <= START_ROWS:
                        absolute[u][dx + START_CAM[0]] += 1
                if len(g) == 2:
                    (u1, dx1), (u2, dx2) = g["p1"], g["p2"]
                    pair_d[(u1, u2)][dx1 - dx2] += 1
                    rows_seen.append((u1, u2, dx1 - dx2))
    edges = defaultdict(list)
    for (u1, u2), c in pair_d.items():
        d, n = c.most_common(1)[0]
        edges[u1].append((n, u2, d))       # off(u2) = off(u1) - d
        edges[u2].append((n, u1, -d))
    off: Dict[str, tuple] = {}
    heap = []
    for u, c in absolute.items():
        dx, n = c.most_common(1)[0]
        heapq.heappush(heap, (-10 ** 9 - n, u, dx, n))
    while heap:
        neg, u, dx, n = heapq.heappop(heap)
        if u in off:
            continue
        off[u] = (dx, n)
        for m, v, d in edges[u]:
            if v not in off:
                heapq.heappush(heap, (-m, v, dx - d, m))
    ok = sum(1 for u1, u2, d in rows_seen if u1 in off and u2 in off and off[u1][0] - off[u2][0] == d)
    out = dict(anchors={u: list(v) for u, v in sorted(off.items())},
               consistency=dict(rows=len(rows_seen), agree=ok), start_cam=START_CAM)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(out, f)
    return out


FIXTURES = os.path.join(REPO, "tests", "fixtures", "screen")
FIXTURE_GAMES = ["a/ken_vs_ryu", "a/zangief_vs_dhalsim", "a/chunli_vs_dhalsim", "b/chunli_vs_honda_r0",
                 "b/chunli_vs_guile_r1", "b/chunli_vs_zangief_r3", "a/guile_vs_ryu", "a/blanka_vs_guile",
                 "b/chunli_vs_dhalsim_r1", "b/chunli_vs_ryu_r2", "a/dhalsim_vs_ken", "a/ryu_vs_chunli",
                 "b/chunli_vs_ken_r0"]
FIXTURE_PICKS = [(g, "start") for g in FIXTURE_GAMES] + [   # (set/game, what): the first frame meeting ``what``
    ("a/ken_vs_ryu", "apart"), ("a/chunli_vs_dhalsim", "apart"), ("b/chunli_vs_zangief_r3", "apart"),
    ("a/guile_vs_ryu", "p1_air"), ("a/blanka_vs_guile", "crossed"), ("b/chunli_vs_dhalsim_r1", "apart"),
    ("b/chunli_vs_ryu_r2", "shot"), ("a/dhalsim_vs_ken", "shot"),
    ("a/ryu_vs_chunli", "clock00"), ("b/chunli_vs_ken_r0", "bar_empty"),
]


def fixture_frames(data: str = OUT, out: str = FIXTURES) -> List[Dict]:
    """Frozen frames for the reader's unit tests (tests/test_screen_reader.py) with the referee's facts (RAM / OAM at
    collection time) as the expected values. Writes <out>/frames.npz (frames, prev) and <out>/expected.json."""
    from sf2.screen.assets import projectile_owners
    with open(os.path.join(CATALOG, "catalog.json")) as f:
        sprites = json.load(f)["sprites"]

    def majority(ck, field):
        c = {k: v for k, v in sprites.get(ck, {}).get("labels", {}).get(field, {}).items() if k != "unknown"}
        return max(sorted(c), key=c.get) if c else None
    shots = projectile_owners()
    frames, prevs, expected = [], [], []
    for game, what in FIXTURE_PICKS:
        base = os.path.join(data, game)
        with open(os.path.join(base, "meta.json")) as f:
            meta = json.load(f)
        z = np.load(os.path.join(base, "rows.npz"))
        names = [str(n) for n in z["names"]]
        rows = [dict(zip(names, map(int, r))) for r in z["rows"]]
        with gzip.open(os.path.join(base, "oam.json.gz"), "rt") as f:
            oam = [json.loads(line)["groups"] for line in f]
        chars = {1: meta["p1"], 2: meta["p2"]}

        def ok(k):
            r, g = rows[k], oam[k]
            both = "4" in g and "6" in g and all("%s/%s" % (chars[p], g[pal][0]) in sprites
                                                 for p, pal in ((1, "4"), (2, "6")))
            drawn = any(shots.get("_projectiles/" + v[0], ("?", False))[1] for p, v in g.items() if p in ("5", "7"))
            calm = all(rows[k - 1]["p%d_state" % p] not in (0x0E, 0x14) for p in (1, 2))   # not hit / thrown
            apart = both and calm and abs(r["p1_x"] - r["p2_x"]) >= 60 and min(g["4"][1], g["6"][1]) >= 0 and \
                max(g["4"][3], g["6"][3]) <= 256
            clock = (rows[k - 1]["timer"] >> 4) * 10 + (rows[k - 1]["timer"] & 15)
            return {"start": k == 1, "apart": apart and k > 300, "p1_air": apart and r["p1_y"] < 150,
                    "crossed": apart and r["p1_x"] > r["p2_x"] + 60, "shot": drawn and apart,
                    "clock00": clock == 0 and k > 5, "bar_empty": rows[k - 1]["p1_hp"] == 0 or rows[k - 1]["p2_hp"] == 0
                    }[what]
        k = next(k for k in range(1, len(rows)) if ok(k))
        fr = np.load(os.path.join(base, "frames.npz"))["frames"]
        frames.append(fr[k])
        prevs.append(fr[k - 1])
        r, rl, g = rows[k], rows[k - 1], oam[k]
        exp = dict(game=game, what=what, k=k, chars=[chars[1], chars[2]],
                   health=[rl["p1_hp"] / 176, rl["p2_hp"] / 176],
                   clock=(rl["timer"] >> 4) * 10 + (rl["timer"] & 15),
                   shot_drawn=any(shots.get("_projectiles/" + v[0], ("?", False))[1] for p, v in g.items()
                                  if p in ("5", "7")), p={})
        for p, pal in ((1, "4"), (2, "6")):
            ck = "%s/%s" % (chars[p], g[pal][0]) if pal in g else None
            exp["p"][str(p)] = dict(x=r["p%d_x" % p] - r["cam_x"], key=ck,
                                    facing={0x40: "right", 0x00: "left"}.get(g[pal][5]) if pal in g else None,
                                    action=majority(ck, "act2"), air_label=majority(ck, "air"),
                                    ram_air=rl["p%d_y" % p] != 192)
        expected.append(exp)
    os.makedirs(out, exist_ok=True)
    np.savez_compressed(os.path.join(out, "frames.npz"), frames=np.stack(frames), prev=np.stack(prevs))
    with open(os.path.join(out, "expected.json"), "w") as f:
        json.dump(expected, f, indent=1)
    return expected


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--sets", default="a,b")
    ap.add_argument("--limit", type=int, default=0, help="at most this many jobs per set (0: all)")
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--base-port", type=int, default=BASE_PORT)
    ap.add_argument("--rom", default=os.environ.get("SF2_ROM"))
    ap.add_argument("--calib-hud", action="store_true", help="write the HUD calibration frames (%s)" % HUD_CALIB)
    ap.add_argument("--anchors", action="store_true", help="write the sprite anchor table (%s)" % ANCHORS)
    ap.add_argument("--fixtures", action="store_true", help="write the reader's frozen test frames (%s)" % FIXTURES)
    ap.add_argument("--seed", type=int, default=SEED, help="the games' random starts / move cycles (catalog: 0)")
    ap.add_argument("--one", type=int, help=argparse.SUPPRESS)
    ap.add_argument("--port", type=int, help=argparse.SUPPRESS)
    args = ap.parse_args()
    exit_on_sigterm()
    if args.calib_hud:
        from sf2.screen.hud import DIGITS, build_digits
        if not os.path.exists(HUD_CALIB):
            calib_hud(args.base_port + 90, args.rom)
        z = np.load(HUD_CALIB)
        np.savez_compressed(DIGITS, digits=build_digits(z["frames"], z["clock"]))
        print("%s (%d calibration frames) -> %s" % (HUD_CALIB, len(z["clock"]), DIGITS))
        return 0
    if args.fixtures:
        for e in fixture_frames(args.out):
            print("%-26s %-9s k=%d" % (e["game"], e["what"], e["k"]))
        return 0
    if args.anchors:
        a = build_anchors()
        print("%d anchors; rows agreeing: %d / %d" % (len(a["anchors"]), a["consistency"]["agree"],
                                                       a["consistency"]["rows"]))
        return 0
    all_jobs = [dict(j, out=args.out, seed=args.seed) for j in jobs()]
    if args.one is not None:
        m = play(all_jobs[args.one], args.port, args.rom)
        print("%s: %s, %d rows, result at %s, %.0f s" % (m["name"], m["result"], m["n"], m["result_k"], m["seconds"]))
        return 0 if m["result_k"] is not None else 1
    sets = args.sets.split(",")
    todo, per = [], {}
    for i, j in enumerate(all_jobs):
        per[j["set"]] = per.get(j["set"], 0) + 1
        if j["set"] in sets and (not args.limit or per[j["set"]] <= args.limit) and \
                not os.path.exists(os.path.join(args.out, j["set"], j["name"], "meta.json")):
            todo.append((i, j))
    cmds = []
    for i, j in todo:
        cmd = [sys.executable, os.path.abspath(__file__), "--one", str(i), "--port", str(args.base_port + i),
               "--out", args.out, "--seed", str(args.seed)] + (["--rom", args.rom] if args.rom else [])
        cmds.append(((j["set"], j["name"]), cmd))
    print("%d games to play" % len(cmds), flush=True)
    t0 = time.time()
    failed = fan_out(cmds, os.path.join(REPO, "logs", "screen_gate"), job_gb=3.0,
                     budget=Budget(total_gb=3.0 * args.workers))
    print("done in %.0f s; failed: %s" % (time.time() - t0, failed), flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
