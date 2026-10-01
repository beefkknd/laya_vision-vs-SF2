"""The movement dataset's gates before any training (docs/prereg_movement_data.md; scripts/gate_movement_data.py).

1 counts     every (opponent, answer) has >= min_train (1,000) train pairs and >= min_test (200) test pairs;
             shortfalls listed by name.
2 stages     for each (opponent, answer) with pairs from episodes of SHORT (6)+ frames: of those pairs, the start,
             middle and end bins (3 * pos // length, recomputed here) each hold >= min_share (20%).
3 labels     every row's answer re-derived from the collection's stored RAM by ``independent_answer`` (written from
             the prereg's rules, not importing sf2.data.movement), at the displayed row t, must match 100%; and its
             images must be the captures t - 4 + LAG and t + LAG of its game.
  alignment  RAM-to-image (lag 1), on a sample: the HUD clock digits (the clock's blue outline, the region and test
             scripts/calibrate_perception.display_lag measured the lag with) change between a pair's two images iff
             the RAM timer changed between the rows they show. Agreement at lag 1 >= min_agree on a random sample,
             and on the discriminating pairs (lags 0 / 1 / 2 predict differently) lag 1 must agree best and
             >= min_agree_disc, with at least min_disc of them.
  disk       the images (every file under the frames dirs) < max_gb (10 GB).
The report says pass per gate; the run passes only when all do.
"""
import collections
import json
import os
import random
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from . import movement as M
from . import movement_collect_io as IO
from .movement_collect import LAG, SHORT, STAGES
from .movement_data2 import FILES

GATES = ("counts", "stages", "labels", "alignment", "disk")
MIN_TRAIN, MIN_TEST, MIN_SHARE = 1000, 200, 0.2
MIN_AGREE, MIN_AGREE_DISC, MIN_DISC = 0.95, 0.9, 20
MAX_GB = 10.0
SAMPLE = 400
CLOCK = (slice(44, 62), slice(112, 144))      # the HUD clock (scripts/calibrate_perception.display_lag)
LAGS = (0, 1, 2)
THUMB = 256


# ---- the independent label (from docs/prereg_movement.md's rules; deliberately not sf2.data.movement) -------------

def independent_answer(rows: List[Dict[str, int]], t: int) -> str:
    if not 0 <= t < len(rows):
        return "unknown"
    now = rows[t]
    state, react, ground = now["p2_state"], now["p2_react"], 192
    if state == 0x14:
        return "being hit"
    if state == 0x0E:
        return "blocking" if react in (0x06, 0x08) else "being hit"
    if state == 0x08:
        return "blocking"
    if state in (0x0A, 0x0C):
        return "attacking"
    if state == 0x04 or now["p2_y"] != ground:
        return "jumping"
    if state == 0x02:
        return "crouching"
    if state != 0x00 or t < 4:
        return "unknown"
    before = rows[t - 4]
    xs = (before["p2_x"], now["p1_x"], now["p2_x"])
    if any(x < 0 or x > 512 for x in xs):
        return "unknown"
    moved = now["p2_x"] - before["p2_x"]
    if -2 < moved < 2:
        return "standing"
    her_side = now["p1_x"] - now["p2_x"]
    if her_side == 0:
        return "unknown"
    return "walking toward me" if (moved > 0) == (her_side > 0) else "walking away"
# ---- end independent ------------------------------------------------------------------------------------------------


def load(data: str) -> Tuple[Dict, Dict[str, Dict[str, List[Dict]]]]:
    meta = json.load(open(os.path.join(data, "build.json")))
    files = {o: {n: IO.read_jsonl(os.path.join(data, o, n + ".jsonl")) for n in FILES.values()} for o in meta["opps"]}
    return meta, files


def count_shortfalls(files: Dict[str, Dict[str, List[Dict]]], answers: Sequence[str], min_train: int,
                     min_test: int) -> List[str]:
    out = []
    for o, fs in sorted(files.items()):
        for split, need in (("train", min_train), ("test", min_test)):
            c = collections.Counter(r["answer"] for r in fs[FILES[split]])
            out += ["%s %s %s %d < %d" % (o, a, split, c[a], need) for a in answers if c[a] < need]
    return out


def stage_problems(rows: List[Dict], min_share: float = MIN_SHARE) -> List[str]:
    cells: Dict[Tuple[str, str], collections.Counter] = collections.defaultdict(collections.Counter)
    for r in rows:
        if r["length"] >= SHORT:
            cells[(r["opp"], r["answer"])][STAGES[3 * r["pos"] // r["length"]]] += 1
    out = []
    for (o, a), c in sorted(cells.items()):
        n = sum(c.values())
        thin = {s: round(c[s] / n, 3) for s in STAGES if c[s] / n < min_share}
        if thin:
            out.append("%s %s: %s below %.0f%% of %d pairs from 6+ frame episodes" % (o, a, thin, 100 * min_share, n))
    return out


def _ram(root: str, opp: str, game: int, cache: Dict) -> Optional[List[Dict[str, int]]]:
    key = (opp, game)
    if key not in cache:
        path = os.path.join(root, opp, "ram", "g%04d.json.gz" % game)
        cache[key] = IO.read_ram(path) if os.path.exists(path) else None
    return cache[key]


def label_check(meta: Dict, files: Dict[str, Dict[str, List[Dict]]]) -> Dict:
    cache: Dict = {}
    checked, bad, missing, examples = 0, 0, set(), []
    q = list(M.movement_question()["criteria"])
    for o, fs in sorted(files.items()):
        for rows in fs.values():
            for r in rows:
                ram = _ram(meta["root"], o, r["game"], cache)
                if ram is None:
                    missing.add("%s g%d" % (o, r["game"]))
                    continue
                t = r["t"]
                want_imgs = ["frames/g%04d_k%05d.png" % (r["game"], k) for k in (t - 4 + LAG, t + LAG)]
                got = independent_answer(ram, t)
                checked += 1
                ok = got == r["answer"] and r["images"] == want_imgs and q[r["label"]] == r["answer"]
                if not ok:
                    bad += 1
                    if len(examples) < 10:
                        examples.append({"id": r["id"], "row": r["answer"], "ram": got, "images": r["images"]})
            cache.clear()
    return {"pass": checked > 0 and bad == 0 and not missing, "checked": checked, "mismatches": bad,
            "missing_ram": sorted(missing), "examples": examples}


def _clock_changed(frames: str, a: str, b: str) -> bool:
    from PIL import Image

    def mask(p):
        im = np.asarray(Image.open(os.path.join(frames, os.path.basename(p))).convert("RGB")).astype(int)[CLOCK]
        return (im[..., 2] > 150) & (im[..., 0] < 60)
    return bool((mask(a) != mask(b)).any())


def alignment_check(data: str, meta: Dict, files: Dict[str, Dict[str, List[Dict]]], sample: int, min_disc: int,
                    min_agree: float, min_agree_disc: float, seed: int) -> Dict:
    cache: Dict = {}
    preds = []                      # (row, opp, {lag: timer changed})
    for o, fs in sorted(files.items()):
        for rows in fs.values():
            for r in rows:
                ram = _ram(meta["root"], o, r["game"], cache)
                if ram is None:
                    continue
                p = {L: ram[r["k_now"] - L]["timer"] != ram[r["k_prev"] - L]["timer"]
                     for L in LAGS if r["k_prev"] - L >= 0 and r["k_now"] - L < len(ram)}
                if len(p) == len(LAGS):
                    preds.append((r, o, p))
        cache.clear()
    rng = random.Random(seed)
    rand = rng.sample(preds, min(sample, len(preds)))
    disc_all = [x for x in preds if len(set(x[2].values())) > 1]
    disc = rng.sample(disc_all, min(sample, len(disc_all)))

    missing = []

    def agree(items):
        hits = {L: 0 for L in LAGS}
        for r, o, p in items:
            fr = os.path.join(data, o, "frames")
            gone = [x for x in r["images"] if not os.path.exists(os.path.join(fr, os.path.basename(x)))]
            if gone:
                missing.append("%s %s" % (o, gone[0]))
                continue
            seen = _clock_changed(fr, r["images"][0], r["images"][1])
            for L in LAGS:
                hits[L] += p[L] == seen
        return {str(L): round(hits[L] / len(items), 4) if items else None for L in LAGS}
    ag, ad = agree(rand), agree(disc)
    best = max(LAGS, key=lambda L: ad[str(L)] or 0.0) if disc else None
    ok = alignment_verdict(ag, ad, len(rand), len(disc), len(missing), min_disc, min_agree, min_agree_disc)
    return {"pass": bool(ok), "sample": len(rand), "agreement": ag, "discriminating": len(disc),
            "discriminating_total": len(disc_all), "agreement_discriminating": ad, "best_lag": best,
            "missing_images": missing[:10]}


def alignment_verdict(ag: Dict, ad: Dict, n_rand: int, n_disc: int, n_missing: int, min_disc: int,
                      min_agree: float, min_agree_disc: float) -> bool:
    """Lag 1 agrees >= min_agree on the random sample, and on >= min_disc discriminating pairs it agrees
    >= min_agree_disc and strictly better than every other lag; no image missing."""
    if n_missing or not n_rand or n_disc < min_disc or n_disc == 0:
        return False
    strict = all(ad[str(LAG)] > ad[str(L)] for L in LAGS if L != LAG)
    return ag[str(LAG)] >= min_agree and strict and ad[str(LAG)] >= min_agree_disc


def disk_check(data: str, opps: Sequence[str], max_gb: float) -> Dict:
    total = 0
    for o in opps:
        d = os.path.realpath(os.path.join(data, o, "frames"))
        with os.scandir(d) as it:
            total += sum(e.stat().st_size for e in it if e.is_file())
    gb = total / 1e9
    return {"pass": gb < max_gb, "gb": round(gb, 4), "max_gb": max_gb}


def run_gates(data: str, answers: Sequence[str] = M.ANSWERS, min_train: int = MIN_TRAIN, min_test: int = MIN_TEST,
              min_share: float = MIN_SHARE, min_disc: int = MIN_DISC, min_agree: float = MIN_AGREE,
              min_agree_disc: float = MIN_AGREE_DISC, max_gb: float = MAX_GB, sample: int = SAMPLE,
              seed: int = 0) -> Dict:
    if min(min_train, min_test, min_disc, sample) < 0 or not 0 <= min_share <= 1 or max_gb <= 0 or \
            not 0 <= min_agree <= 1 or not 0 <= min_agree_disc <= 1:
        raise ValueError("bad gate thresholds")
    meta, files = load(data)
    short = count_shortfalls(files, answers, min_train, min_test)
    all_rows = [r for fs in files.values() for rows in fs.values() for r in rows]
    stages = stage_problems(all_rows, min_share)
    gates = {
        "counts": {"pass": not short, "shortfalls": short, "min_train": min_train, "min_test": min_test},
        "stages": {"pass": not stages, "problems": stages, "min_share": min_share},
        "labels": label_check(meta, files),
        "alignment": alignment_check(data, meta, files, sample, min_disc, min_agree, min_agree_disc, seed),
        "disk": disk_check(data, meta["opps"], max_gb),
    }
    return {"pass": all(g["pass"] for g in gates.values()), "gates": gates, "rows": len(all_rows),
            "opps": meta["opps"]}


# ---- contact sheet ------------------------------------------------------------------------------------------------

def contact_sheets(data: str, out: str, per_answer: int = 3, answers: Sequence[str] = M.ANSWERS,
                   seed: int = 0) -> List[str]:
    """<out>/contact_<opp>.png: one band per answer, ``per_answer`` pairs (prev | now, full size), picked across
    the stage bins when it can, captioned with game, t, stage and split."""
    from PIL import Image, ImageDraw

    meta, files = load(data)
    os.makedirs(out, exist_ok=True)
    label_w, cap_h, gap = 170, 16, 8
    paths = []
    for o, fs in sorted(files.items()):
        rows = [r for rs in fs.values() for r in rs]
        rng = random.Random("%d:%s" % (seed, o))
        W = label_w + per_answer * (2 * THUMB + gap)
        sheet = Image.new("RGB", (W, len(answers) * (THUMB + cap_h + gap)), (40, 40, 40))
        draw = ImageDraw.Draw(sheet)
        for i, a in enumerate(answers):
            y = i * (THUMB + cap_h + gap)
            mine = [r for r in rows if r["answer"] == a]
            draw.text((6, y + THUMB // 2), "%s\n(%d pairs)" % (a, len(mine)), fill=(255, 255, 255))
            for j, r in enumerate(pick(mine, per_answer, rng)):
                x = label_w + j * (2 * THUMB + gap)
                for side, p in enumerate(r["images"]):
                    im = Image.open(os.path.join(data, o, p)).convert("RGB").resize((THUMB, THUMB))
                    sheet.paste(im, (x + side * THUMB, y))
                draw.text((x + 2, y + THUMB + 2), "g%d t%d %s %.2f %s%s" % (
                    r["game"], r["t"], r["stage_bin"], r["stage"], r["split"], " long" if r["long"] else ""),
                    fill=(255, 255, 0))
        path = os.path.join(out, "contact_%s.png" % o)
        sheet.save(path)
        paths.append(path)
    return paths


def pick(rows: List[Dict], n: int, rng: random.Random) -> List[Dict]:
    by = {s: [r for r in rows if r["stage_bin"] == s] for s in STAGES}
    out: List[Dict] = []
    while len(out) < n and any(by.values()):
        for s in STAGES:
            if by[s] and len(out) < n:
                out.append(by[s].pop(rng.randrange(len(by[s]))))
    return out
