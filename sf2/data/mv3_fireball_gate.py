"""The fireball dataset's gates before training (docs/prereg_movement_finetunes.md, round 3; scripts/gate_mv3_fireball.py).

labels     mv3_fireball.problems_fireball: every row re-derived from the stored RAM and move log (one drawn projectile
           of the named slot at t, the thrower's side by x, its projectile word at the spawn row, the flight stage;
           none rows with both slots off over t - 8 .. t; splits, caps, images at lag 1): 0 problems = 100%.
alignment  RAM-to-image lag 1 on the dataset's own pairs: the HUD clock digits change between the two images iff the
           RAM timer changed between the rows they show (sf2.data.movement_gate), lag 1 best. Pairs whose RAM clock
           is at or below CLOCK_BLINK (BCD 0x20 = 20 s) at either shown row are left out: the HUD clock blinks there
           (found on this build: every one of the 114 disagreements at lag 1 had the clock at 0x00-0x14 with the RAM
           timer unchanged), and projectiles come late in a round more often than the movement samples did.
drawn      the projectile is on the screen in the frame its label says: every hadoken row (Ryu / Ken; blue, the
           only colour test that is clean on this stage) has >= BLUE_MIN blue pixels below the HUD in its "now" image,
           and the none rows of Ryu-vs-Ken matches (no blue fighter) have fewer: >= DRAWN_MIN of each. Hadoken blinks 2
           frames on / 2 off, so an image one frame off the label misses about half of them.
ownership  every flight of the projectile rounds (games.jsonl "flights"): the slot's own player pressing a projectile
           word at the spawn row (reported; the flights where only the other player pressed one are counted).
disk       every image file of the collection < 10 GB.
"""
import collections
import json
import os
import random
from typing import Dict, List, Sequence

import numpy as np

from . import movement_collect_io as MIO
from . import mv3_fireball as F
from . import pairs_shots as S
from . import pairs_train as T
from .movement_gate import LAGS, MIN_AGREE, MIN_AGREE_DISC, MIN_DISC, SAMPLE, alignment_verdict, digits_changed
from .pairs_gate import disk_check

GATES = ("labels", "alignment", "drawn", "disk")
BLUE_MIN = 200
DRAWN_MIN = 0.95
HUD_ROWS = 50
HADOKEN = ("ryu", "ken")
CLOCK_BLINK = 0x20


def blue_pixels(path: str) -> int:
    from PIL import Image
    a = np.asarray(Image.open(path).convert("RGB")).astype(int)[HUD_ROWS:]
    return int(((a[..., 2] > a[..., 0] + 40) & (a[..., 2] > 160)).sum())


def rows_of(data: str) -> List[Dict]:
    return [dict(r, _dir=d) for d, fs in T.read_dataset(data).items() for f in T.FILES for r in fs[f]]


def drawn_check(data: str, rows: Sequence[Dict], blue_min: int = BLUE_MIN, need: float = DRAWN_MIN) -> Dict:
    had = [r for r in rows if r["answer"] != "none" and r.get("char") in HADOKEN]
    none = [r for r in rows if r["answer"] == "none" and set(r["pair_name"].split("_vs_")) <= set(HADOKEN)]
    seen = lambda r: blue_pixels(os.path.join(data, r["_dir"], r["images"][1])) >= blue_min
    hit = sum(seen(r) for r in had)
    clear = sum(not seen(r) for r in none)
    res = {"hadoken_rows": len(had), "hadoken_drawn": hit, "none_rows": len(none), "none_clear": clear,
           "blue_min": blue_min, "need": need}
    res["pass"] = bool(had) and bool(none) and hit >= need * len(had) and clear >= need * len(none)
    return res


def alignment_check(data: str, root: str, rows: Sequence[Dict], sample: int = SAMPLE, min_disc: int = MIN_DISC,
                    seed: int = 0) -> Dict:
    preds, cache, blink = [], {}, 0
    for r in rows:
        key = (r["pair_name"], r["game"])
        if key not in cache:
            if len(cache) > 64:
                cache.clear()
            cache[key] = MIO.read_ram(os.path.join(root, r["pair_name"], "ram", "g%04d.json.gz" % r["game"]))
        ram = cache[key]
        if min(ram[r["k_prev"] - 1]["timer"], ram[r["k_now"] - 1]["timer"]) <= CLOCK_BLINK:
            blink += 1
            continue
        p = {L: ram[r["k_now"] - L]["timer"] != ram[r["k_prev"] - L]["timer"]
             for L in LAGS if r["k_prev"] - L >= 0 and r["k_now"] - L < len(ram)}
        if len(p) == len(LAGS):
            preds.append((r, p))
    rng = random.Random(seed)
    rand = rng.sample(preds, min(sample, len(preds)))
    disc_all = [x for x in preds if len(set(x[1].values())) > 1]
    disc = rng.sample(disc_all, min(sample, len(disc_all)))
    missing: List[str] = []

    def agree(items):
        hits = {L: 0 for L in LAGS}
        for r, p in items:
            paths = [os.path.join(data, r["_dir"], x) for x in r["images"]]
            if not all(os.path.exists(x) for x in paths):
                missing.append(paths[0])
                continue
            seen = digits_changed(os.path.dirname(paths[0]), paths[0], paths[1])
            for L in LAGS:
                hits[L] += p[L] == seen
        return {str(L): round(hits[L] / len(items), 4) if items else None for L in LAGS}
    ag, ad = agree(rand), agree(disc)
    ok = alignment_verdict(ag, ad, len(rand), len(disc), len(missing), min_disc, MIN_AGREE, MIN_AGREE_DISC)
    return {"pass": bool(ok), "sample": len(rand), "agreement": ag, "discriminating": len(disc),
            "agreement_discriminating": ad, "missing_images": missing[:10], "left_out_clock_blink": blink}


def ownership(root: str) -> Dict:
    """Over every committed flight: whose player pressed a projectile word at the spawn row."""
    n = collections.Counter()
    for pair in sorted(x for x in os.listdir(root) if "_vs_" in x):
        chars = pair.split("_vs_")
        for g in MIO.read_jsonl(os.path.join(root, pair, "games.jsonl")):
            for fl in g.get("flights", []):
                s = fl["slot"]
                own = S.is_projectile(chars[s - 1], fl["spawn_words"][str(s)])
                other = S.is_projectile(chars[2 - s], fl["spawn_words"][str(3 - s)])
                n["own" if own else "other_only" if other else "neither"] += 1
                n["flights"] += 1
    return dict(n)


def run_gates(data: str, sample: int = SAMPLE, min_disc: int = MIN_DISC, seed: int = 0) -> Dict:
    meta = json.load(open(os.path.join(data, "build.json")))
    rows = rows_of(data)
    bad = F.problems_fireball(data, meta["source"], meta["root"], meta["caps"])
    pairs = sorted(x for x in os.listdir(meta["root"]) if "_vs_" in x)
    gates = {"labels": {"pass": not bad, "rows": len(rows), "problems": bad[:20], "n_problems": len(bad)},
             "alignment": alignment_check(data, meta["root"], rows, sample, min_disc, seed),
             "drawn": drawn_check(data, rows), "disk": disk_check(meta["root"], pairs)}
    return {"pass": all(g["pass"] for g in gates.values()), "gates": gates, "ownership": ownership(meta["root"]),
            "counts": meta["counts"], "short": meta["short"]}


THUMB = 192


def contact_sheets(data: str, out: str, per_cell: int = 3, seed: int = 0) -> List[str]:
    """<out>/fireball_<thrower>.png: one band per (side, flight stage), ``per_cell`` pairs (prev | now) with the RAM
    facts; <out>/fireball_none.png: none rows."""
    from PIL import Image, ImageDraw

    rows = rows_of(data)
    os.makedirs(out, exist_ok=True)
    paths = []
    groups = [(c, [("%s %s" % (side, st), [r for r in rows if r.get("char") == c and r["answer"] == side and
                                          r["flight_stage"] == st]) for side in T.SIDES for st in F.STAGES])
              for c in S.THROWERS]
    nones = [r for r in rows if r["answer"] == "none"]
    groups.append(("none", [("none %d" % i, nones[i::4]) for i in range(4)]))
    label_w, cap_h, gap = 120, 14, 6
    for name, bands in groups:
        rng = random.Random("%d:%s" % (seed, name))
        sheet = Image.new("RGB", (label_w + per_cell * (2 * THUMB + gap), len(bands) * (THUMB + cap_h + gap)),
                          (40, 40, 40))
        draw = ImageDraw.Draw(sheet)
        for i, (lab, these) in enumerate(bands):
            y = i * (THUMB + cap_h + gap)
            draw.text((4, y + THUMB // 2), "%s\n(%d rows)" % (lab, len(these)), fill=(255, 255, 255))
            for j, r in enumerate(rng.sample(these, min(per_cell, len(these)))):
                x = label_w + j * (2 * THUMB + gap)
                for k, p in enumerate(r["images"]):
                    im = Image.open(os.path.join(data, r["_dir"], p)).convert("RGB").resize((THUMB, THUMB))
                    sheet.paste(im, (x + k * THUMB, y))
                draw.text((x + 2, y + THUMB + 1), "%s g%d t%d P%s %s shot_x %s" % (
                    r["pair_name"], r["game"], r["t"], r.get("thrower_slot"), r["answer"], r.get("shot_x", "-")),
                    fill=(255, 255, 0))
        path = os.path.join(out, "fireball_%s.png" % name)
        sheet.save(path)
        paths.append(path)
    return paths
