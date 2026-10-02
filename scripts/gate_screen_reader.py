"""Gate of the screen reader (docs/laya_text_only_plan.md, "Step 1-2 pre-registration: the screen reader and its
gate"). The reader (sf2.screen, image only) runs on every frame of the held-out games (scripts/collect_reader_gate.py);
RAM and the OAM snapshot saved at collection time are the referee. The exit code decides: 0 only if every bar passes
on both sets.

    .venv/bin/python scripts/gate_screen_reader.py [--data out/screen_gate/data] [--out out/screen_gate]

Frame k is the screen captured at stream row k. Alignment (measured, reported in gate.json "lag"): the drawn sprite
boxes, the camera and RAM x agree at row k (lag 0, as the anchor table); state labels use the repo's LAG = 1
(row k - 1: action, air, facing, life, result).

Bars (per set a / b, frames 1 .. the RAM result row unless said):
  identity      locked (player 1, player 2) characters == RAM's, 100% of rounds
  x             |reader x - (RAM x - camera x)| <= 4 px on >= 95% of frames with RAM gap >= 40 (non-overlap)
  health        |bar - drawn hp / 176| <= 0.03 on >= 99% of frames. Drawn hp = RAM 0x0D12 (+0x200 player 2), the
                game's displayed-hp field (it drains 1 hp a frame after a hit; RAM life 0x0C35 drops at once); verified
                vs the HUD on the seed-999 probe: bar px == hp / 2 on 100% of frames (life / 2: 81-89%). Gate
                amendment 2026-10-02.
  round_over    the reader's first "over" frame (minus LAG) no earlier than RAM's result row and no later than RAM's
                next-round-start row (meta next_k) + 300, on 100% of rounds (gate amendment 2026-10-02: round over = the
                next round visibly starting, or the time-over clock 00)
  air           reader in_air == (RAM y != 192), >= the air ceiling - 0.01 (ceiling: the catalog's air label of the
                true sprite vs RAM, same frames, per set; gate amendment 2026-10-02)
  facing        reader facing == RAM facing byte (0x40 right), >= the facing ceiling - 0.01 (ceiling: the drawn facing
                - OAM - vs RAM's byte, same frames, per set)
  action        accuracy vs RAM's 7-answer label (an unknown sprite answers the default "block", owner 2026-10-02)
                >= the catalog ceiling - 0.05 (ceiling: the catalog majority label of
                the TRUE sprite - the OAM key - on the same frames; a key not in the catalog counts as a miss)
  speed         median ms per frame (one process, sequential) < 20
Reported, not gated: unknown rate (not a failure by itself; logged under out/screen_reader_unknown/gate/), true sprites missing from the catalog, near contact (gap < 40), projectiles (vs the
RAM shot slots, and vs a shot sprite actually drawn - OAM), health vs RAM life (the old truth), per character / per
stage. The reader's unknown-sprite default action ("block", owner 2026-10-02) is written in gate.json.
"""
import argparse
import glob
import gzip
import json
import os
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from typing import Dict, List, Optional

import numpy as np

import _path  # noqa: F401
from sf2.config import REPO
from sf2.data.eye_v2 import act2
from sf2.emu.vs import GROUND_Y
from sf2.vocab import FULL_LIFE

DATA = os.path.join(REPO, "out", "screen_gate", "data")
OUT = os.path.join(REPO, "out", "screen_gate")
LAG = 1
NEAR = 40
BARS = dict(identity=1.0, x=0.95, health=0.99, round_over=1.0, ceiling_margin=0.01, action_margin=0.05,
            speed_ms=20.0)
X_TOL, HEALTH_TOL, NEXT_WINDOW = 4, 0.03, 300
HEALTH_TRUTH = "hp"          # the drawn hp (gate amendment 2026-10-02); "life" was the first gate's truth


def ceiling_bar(ceiling: Optional[float]) -> float:
    """Air / facing bar: the measured ceiling minus BARS["ceiling_margin"] (gate amendment 2026-10-02)."""
    return round((ceiling or 0.0) - BARS["ceiling_margin"], 4)


# ---------------------------------------------------------------------------------------------- truth (referee)
def load_truth(game_dir: str) -> Dict:
    """Per frame k and player p the referee's facts (RAM rows, OAM keys); the meta."""
    with open(os.path.join(game_dir, "meta.json")) as f:
        meta = json.load(f)
    z = np.load(os.path.join(game_dir, "rows.npz"))
    names = [str(n) for n in z["names"]]
    rows = [dict(zip(names, map(int, r))) for r in z["rows"]]
    pressed = {p: [c or None for c in z["pressed"][p - 1].tolist()] for p in (1, 2)}
    with gzip.open(os.path.join(game_dir, "oam.json.gz"), "rt") as f:
        oam = [json.loads(line)["groups"] for line in f]
    chars = {1: meta["p1"], 2: meta["p2"]}
    from sf2.screen.assets import projectile_owners       # which catalog projectile sprites are thrown shots
    shots = projectile_owners()
    frames = []
    for k in range(len(rows)):
        r, rl = rows[k], rows[max(0, k - LAG)]
        drawn = any(shots.get("_projectiles/" + g[0], ("?", False))[1]
                    for pal, g in oam[k].items() if pal in ("5", "7"))
        fr = dict(k=k, gap=abs(r["p1_x"] - r["p2_x"]), shot=bool(rl["shot1"] or rl["shot2"]),
                  shot0=bool(r["shot1"] or r["shot2"]), shot_drawn=drawn)
        for p in (1, 2):
            life = rl["p%d_life" % p]
            g = oam[k].get("4" if p == 1 else "6")
            fr[p] = dict(x=r["p%d_x" % p] - r["cam_x"], x_lag1=rl["p%d_x" % p] - rl["cam_x"],
                         air=rl["p%d_y" % p] != GROUND_Y, air0=r["p%d_y" % p] != GROUND_Y,
                         facing={0x40: "right", 0x00: "left"}.get(rl["p%d_facing" % p]),
                         facing0={0x40: "right", 0x00: "left"}.get(r["p%d_facing" % p]),
                         life=(life if life < 200 else 0) / FULL_LIFE, hp=rl["p%d_hp" % p] / FULL_LIFE,
                         hp0=r["p%d_hp" % p] / FULL_LIFE,
                         act=act2(rows, k - LAG, p, pressed[p][k - LAG]) if k >= LAG else None,
                         key="%s/%s" % (chars[p], g[0]) if g else None,
                         drawn={0x40: "right", 0x00: "left"}.get(g[5]) if g else None)
        frames.append(fr)
    return dict(meta=meta, frames=frames, result_k=meta["result_k"], next_k=meta.get("next_k"))


# ---------------------------------------------------------------------------------------------- the reader's output
def run_reader(game_dir: str) -> Dict:
    """The reader's facts per frame (k >= 1), as plain dicts, plus the lock."""
    from sf2.screen.reader import RoundReader
    from sf2.screen.unknown_log import UNKNOWN_DIR, UnknownLog
    frames = np.load(os.path.join(game_dir, "frames.npz"))["frames"]
    tail = os.path.relpath(game_dir, os.path.dirname(os.path.dirname(game_dir)))      # <data dir>/<set>/<game>
    rr = RoundReader(log=UnknownLog(os.path.join(UNKNOWN_DIR, "gate", os.path.basename(os.path.dirname(
        os.path.dirname(game_dir))), tail)))
    out, locks = [], []
    for k in range(1, len(frames)):
        f = rr.feed(frames[k])
        if not locks or locks[-1]["chars"] != list(rr.lock.chars):
            locks.append(dict(k=k, chars=list(rr.lock.chars)))
        per = {}
        for ff in (f.left, f.right):
            per[ff.player] = dict(found=ff.found, x=ff.x, air=ff.in_air, facing=ff.facing, sprite=ff.sprite,
                                  action=ff.action, unknown=ff.unknown, conf=ff.confidence, health=ff.health,
                                  side=ff.side)
        out.append(dict(k=k, p=per, over=f.round_state == "over", proj=len(f.projectiles),
                        proj_sides=[p.owner_side for p in f.projectiles], timer=f.hud.timer))
    return dict(frames=out, locks=locks)


# ---------------------------------------------------------------------------------------------- compare (pure)
def _catalog_majority(field: str = "act2") -> Dict[str, Optional[str]]:
    """Catalog sprite -> its majority label of ``field`` (act2 / air), "unknown" ignored."""
    with open(os.path.join(REPO, "out", "sprite_catalog", "catalog.json")) as f:
        sprites = json.load(f)["sprites"]
    out = {}
    for ck, e in sprites.items():
        c = {k: v for k, v in e.get("labels", {}).get(field, {}).items() if k != "unknown"}
        out[ck] = max(sorted(c), key=c.get) if c else None
    return out


def compare(truth: Dict, reader: Dict, majority: Dict[str, Optional[str]],
            air_major: Optional[Dict[str, Optional[str]]] = None) -> Dict:
    """Counters per (player, group) for one game. Pure: the gate's own logic, tested with known-bad readers."""
    meta, rk = truth["meta"], truth["result_k"]
    chars = {1: meta["p1"], 2: meta["p2"]}
    c: Dict[str, Counter] = defaultdict(Counter)
    lock = reader["locks"][0]["chars"] if reader["locks"] else None
    c["round"]["n"] += 1
    c["round"]["identity"] += lock == [chars[1], chars[2]]
    first_over = next((f["k"] for f in reader["frames"] if f["over"]), None)
    nk = truth.get("next_k")
    c["round"]["no_next_start"] += nk is None
    c["round"]["over_ok"] += first_over is not None and nk is not None and rk <= first_over - LAG <= nk + NEXT_WINDOW
    c["round"]["over_early"] += first_over is not None and first_over - LAG < rk
    for f in reader["frames"]:
        k = f["k"]
        if k > rk:
            continue
        tf = truth["frames"][k]
        c["proj"]["n"] += 1
        c["proj"]["tp"] += tf["shot"] and f["proj"] > 0
        c["proj"]["fp"] += (not tf["shot"]) and f["proj"] > 0
        c["proj"]["truth"] += tf["shot"]
        c["proj"]["drawn"] += tf["shot_drawn"]
        c["proj"]["drawn_tp"] += tf["shot_drawn"] and f["proj"] > 0
        for p in (1, 2):
            t, r = tf[p], f["p"][str(p)] if str(p) in f["p"] else f["p"][p]
            for grp in (chars[p], "all"):
                g = c["char:" + grp]
                g["n"] += 1
                g["found"] += r["found"]
                g["unknown"] += r["unknown"]
                far = tf["gap"] >= NEAR
                g["far"] += far
                xok = r["x"] is not None and abs(r["x"] - t["x"]) <= X_TOL
                g["x_ok_far"] += far and xok
                g["x_ok_lag1_far"] += far and r["x"] is not None and abs(r["x"] - t["x_lag1"]) <= X_TOL
                g["health_ok"] += r["health"] is not None and abs(r["health"] - t[HEALTH_TRUTH]) <= HEALTH_TOL
                g["health_life_ok"] += r["health"] is not None and abs(r["health"] - t["life"]) <= HEALTH_TOL
                g["health_hp0_ok"] += r["health"] is not None and abs(r["health"] - t["hp0"]) <= HEALTH_TOL
                g["air_ok"] += r["air"] is not None and r["air"] == t["air"]
                g["air0_ok"] += r["air"] is not None and r["air"] == t["air0"]
                if air_major is not None:
                    g["air_ceil_ok"] += (air_major.get(t["key"]) == "air") == t["air"]
                g["facing_ceil_ok"] += t["facing"] is not None and t["drawn"] == t["facing"]
                g["facing_drawn_n"] += t["drawn"] is not None
                g["facing_drawn_ok"] += t["drawn"] is not None and r["facing"] == t["drawn"]
                g["facing_n"] += t["facing"] is not None
                g["facing_ok"] += t["facing"] is not None and r["facing"] == t["facing"]
                g["facing0_ok"] += t["facing0"] is not None and r["facing"] == t["facing0"]
                g["key_n"] += t["key"] is not None
                g["key_in_cat"] += t["key"] in majority
                g["key_ok"] += t["key"] is not None and r["sprite"] == t["key"]
                if t["act"] is not None:
                    g["act_n"] += 1
                    g["act_ok"] += r["action"] == t["act"]
                    g["ceil_ok"] += majority.get(t["key"]) == t["act"]
                    if r["action"] != t["act"]:
                        why = "unknown" if r["unknown"] else ("label" if r["sprite"] == t["key"] else "sprite")
                        g["act_miss_" + why] += 1
                    if not far:
                        g["near_act_n"] += 1
                        g["near_act_ok"] += r["action"] == t["act"]
                if not far:
                    g["near"] += 1
                    g["near_x_ok"] += xok
    return {k: dict(v) for k, v in c.items()}


def merge(parts: List[Dict]) -> Dict[str, Counter]:
    out: Dict[str, Counter] = defaultdict(Counter)
    for p in parts:
        for k, v in p.items():
            out[k].update(v)
    return out


def _r(a, b) -> Optional[float]:
    return round(a / b, 4) if b else None


def summarize(c: Dict[str, Counter]) -> Dict:
    a = c["char:all"]
    rd, pj = c["round"], c["proj"]
    return dict(
        rounds=rd["n"], identity=_r(rd["identity"], rd["n"]), round_over=_r(rd["over_ok"], rd["n"]),
        round_over_early=rd["over_early"], rounds_without_next_start=rd["no_next_start"],
        frames=a["n"] // 1, x=_r(a["x_ok_far"], a["far"]), x_lag1=_r(a["x_ok_lag1_far"], a["far"]),
        health=_r(a["health_ok"], a["n"]), health_vs_ram_life=_r(a["health_life_ok"], a["n"]),
        health_vs_hp_lag0=_r(a["health_hp0_ok"], a["n"]),
        air=_r(a["air_ok"], a["n"]), air_lag0=_r(a["air0_ok"], a["n"]),
        facing=_r(a["facing_ok"], a["facing_n"]), facing_lag0=_r(a["facing0_ok"], a["facing_n"]),
        facing_ceiling_drawn_vs_ram=_r(a["facing_ceil_ok"], a["facing_n"]),
        facing_vs_drawn=_r(a["facing_drawn_ok"], a["facing_drawn_n"]), air_ceiling=_r(a["air_ceil_ok"], a["n"]),
        action_miss=dict(unknown=_r(a["act_miss_unknown"], a["act_n"]), same_sprite_label=_r(a["act_miss_label"],
                                                                                              a["act_n"]),
                         wrong_sprite=_r(a["act_miss_sprite"], a["act_n"])),
        action=_r(a["act_ok"], a["act_n"]), action_ceiling=_r(a["ceil_ok"], a["act_n"]),
        unknown_rate=_r(a["unknown"], a["n"]), not_found=_r(a["n"] - a["found"], a["n"]),
        true_sprite_not_in_catalog=_r(a["key_n"] - a["key_in_cat"], a["key_n"]),
        sprite_exact=_r(a["key_ok"], a["key_n"]),
        near_frames=a["near"], near_x=_r(a["near_x_ok"], a["near"]), near_action=_r(a["near_act_ok"], a["near_act_n"]),
        projectile_recall=_r(pj["tp"], pj["truth"]), projectile_recall_drawn=_r(pj["drawn_tp"], pj["drawn"]),
        projectile_false_rate=_r(pj["fp"], pj["n"] - pj["truth"]))


def per_char(c: Dict[str, Counter]) -> Dict[str, Dict]:
    out = {}
    for k, g in sorted(c.items()):
        if k.startswith("char:") and k != "char:all":
            out[k[5:]] = dict(frames=g["n"], x=_r(g["x_ok_far"], g["far"]), air=_r(g["air_ok"], g["n"]),
                              facing=_r(g["facing_ok"], g["facing_n"]), action=_r(g["act_ok"], g["act_n"]),
                              ceiling=_r(g["ceil_ok"], g["act_n"]), unknown=_r(g["unknown"], g["n"]),
                              not_in_catalog=_r(g["key_n"] - g["key_in_cat"], g["key_n"]),
                              health=_r(g["health_ok"], g["n"]))
    return out


def bars(s: Dict, speed_ms: Optional[float]) -> Dict[str, Dict]:
    """Each pre-registered bar: value, threshold, pass."""
    ceil = s["action_ceiling"] or 0.0
    out = {
        "identity": (s["identity"], BARS["identity"]),
        "x": (s["x"], BARS["x"]),
        "health": (s["health"], BARS["health"]),
        "round_over": (s["round_over"], BARS["round_over"]),
        "air": (s["air"], ceiling_bar(s["air_ceiling"])),
        "facing": (s["facing"], ceiling_bar(s["facing_ceiling_drawn_vs_ram"])),
        "action": (s["action"], round(ceil - BARS["action_margin"], 4)),
    }
    res = {k: dict(value=v, threshold=t, ok=v is not None and v >= t) for k, (v, t) in out.items()}
    for k, ce in (("air", s["air_ceiling"]), ("facing", s["facing_ceiling_drawn_vs_ram"]), ("action", ceil)):
        res[k]["ceiling"] = ce
    if speed_ms is not None:
        res["speed_ms"] = dict(value=speed_ms, threshold=BARS["speed_ms"], ok=speed_ms < BARS["speed_ms"])
    return res


# ---------------------------------------------------------------------------------------------- driver
def one_game(game_dir: str) -> Dict:
    truth = load_truth(game_dir)
    reader = run_reader(game_dir)
    return dict(name=truth["meta"]["name"], set=truth["meta"]["set"], stage=truth["meta"]["stage"],
                counts=compare(truth, reader, _catalog_majority(), _catalog_majority("air")),
                over_first=next((f["k"] for f in reader["frames"] if f["over"]), None), result_k=truth["result_k"],
                next_k=truth["next_k"],
                result=truth["meta"]["result"], locks=reader["locks"])


def speed(data: str, n: int = 400) -> float:
    """Median ms per frame of read_screen, one process, sequential, on frames from both sets."""
    from sf2.screen.reader import lock_round, read_screen
    from sf2.screen.assets import load_banks
    banks = load_banks()
    ts = []
    dirs = sorted(glob.glob(os.path.join(data, "*", "*")))
    for d in dirs[::max(1, len(dirs) // 8)]:
        fr = np.load(os.path.join(d, "frames.npz"))["frames"]
        lock = lock_round(fr[1], banks)
        for k in np.linspace(1, len(fr) - 1, n // 8).astype(int):
            t0 = time.perf_counter()
            read_screen(fr[k], lock, fr[k - 1], banks)
            ts.append(time.perf_counter() - t0)
    return round(1000 * float(np.median(ts)), 2)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=DATA)
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--workers", type=int, default=24)
    args = ap.parse_args()
    dirs = sorted(glob.glob(os.path.join(args.data, "*", "*")))
    dirs = [d for d in dirs if os.path.exists(os.path.join(d, "meta.json"))]
    if not dirs:
        print("no gate data under %s (scripts/collect_reader_gate.py)" % args.data)
        return 2
    with ProcessPoolExecutor(args.workers) as ex:
        games = list(ex.map(one_game, dirs))
    ms = speed(args.data)
    from sf2.screen.reader import DEFAULT_ACTION
    report = dict(lag=dict(x="row k (lag 0)", labels="row k - %d" % LAG), speed_median_ms=ms,
                  unknown_default_action=DEFAULT_ACTION, health_truth=HEALTH_TRUTH, sets={})
    ok_all = True
    for st in sorted({g["set"] for g in games}):
        gs = [g for g in games if g["set"] == st]
        c = merge([g["counts"] for g in gs])
        s = summarize(c)
        b = bars(s, ms)
        ok_all &= all(v["ok"] for v in b.values())
        by_stage = {stg: summarize(merge([g["counts"] for g in gs if g["stage"] == stg]))
                    for stg in sorted({g["stage"] for g in gs})}
        report["sets"][st] = dict(bars=b, summary=s, per_char=per_char(c), per_stage=by_stage,
                                  rounds=[dict(name=g["name"], lock=g["locks"][0]["chars"] if g["locks"] else None,
                                               over_delay=(g["over_first"] - LAG - g["result_k"])
                                               if g["over_first"] is not None else None,
                                               after_next_start=(g["over_first"] - LAG - g["next_k"])
                                               if g["over_first"] is not None and g["next_k"] is not None else None,
                                               result=g["result"])
                                          for g in gs])
    report["pass"] = bool(ok_all)
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "gate.json"), "w") as f:
        json.dump(report, f, indent=1)
    for st, r in report["sets"].items():
        print("set %s: %s" % (st, "  ".join("%s %s/%s %s" % (k, v["value"], v["threshold"], "ok" if v["ok"] else "FAIL")
                                            for k, v in r["bars"].items())))
    print("unknown sprite -> action %r (owner 2026-10-02); health truth: %s" % (DEFAULT_ACTION, HEALTH_TRUTH))
    print("gate %s -> %s" % ("PASS" if ok_all else "FAIL", os.path.join(args.out, "gate.json")))
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())
