"""The movement-pairs gates before any training (docs/prereg_movement_pairs.md; scripts/gate_pairs_data.py).

labels     every row of every question file: its five labels re-derived from the collection's stored RAM by
           ``independent_labels`` (written from the prereg / pairs_labels docstring, deliberately not importing
           sf2.data.pairs_labels) at the displayed row t must match 100%; the question's answer and label index must
           follow from them; the images must be the captures t - 4 + LAG and t + LAG of its game.
caps       per (split, char, controller, movement, direction, facing) at most the build's cap; per (pair, game, slot,
           movement, direction, facing) at most PER_GAME (re-counted here).
alignment  RAM-to-image lag 1 on a sample: the HUD clock digits change between a pair's two images iff the RAM timer
           changed between the rows they show (sf2.data.movement_gate.digits_changed / alignment_verdict).
disk       every image file the collection wrote < max_gb (10 GB).
The counts table (per cell and per question) and the shortfalls are reported, not gated (never padded).
"""
import collections
import json
import os
import random
from typing import Dict, List, Optional, Sequence

from . import movement_collect_io as MIO
from .movement_gate import LAGS, MIN_AGREE, MIN_AGREE_DISC, MIN_DISC, SAMPLE, alignment_verdict, digits_changed
from .pairs_collect import PER_GAME
from .pairs_data import FILES, QUESTION_ANSWERS, answer, cell, split_of_game

GATES = ("labels", "caps", "alignment", "disk")
MAX_GB = 10.0
LAG = 1


# ---- the independent labels (from the prereg's rules; deliberately not sf2.data.pairs_labels) ----------------------

def independent_labels(rows: List[Dict[str, int]], t: int, p: int, bands: Dict[str, int]) -> Dict[str, str]:
    me, him = "p%d_" % p, "p%d_" % (3 - p)
    if not 0 <= t < len(rows):
        return dict.fromkeys(("movement", "direction", "facing", "air", "distance"), "unknown")
    r = rows[t]
    state, sub, react, mcl = r[me + "state"], r[me + "sub"], r[me + "react"], r[me + "mclass"]
    blocked = react in (6, 8)
    xs_ok = 0 <= r["p1_x"] <= 512 and 0 <= r["p2_x"] <= 512
    old = rows[t - 4] if t >= 4 else None
    moved = r[me + "x"] - old[me + "x"] if old is not None and xs_ok and 0 <= old[me + "x"] <= 512 else None
    if state == 20 or (state == 14 and sub == 4 and not blocked):
        mv = "down"
    elif state == 8 or (state == 14 and blocked):
        mv = "block"
    elif state == 14:
        mv = "hit"
    elif state == 12 or (state == 10 and mcl == 8):
        mv = "special"
    elif state == 10:
        mv = "attack"
    elif state == 4 or r[me + "y"] != 192:
        mv = "jump"
    elif state == 2:
        mv = "crouch"
    elif state == 0 and moved is not None:
        mv = "stand" if -2 < moved < 2 else "walk"
    else:
        mv = "unknown"
    if mv == "unknown":
        d = "unknown"
    elif mv in ("walk", "jump"):
        side = r[him + "x"] - r[me + "x"]
        if moved is None:
            d = "unknown"
        elif -2 < moved < 2:
            d = "none"
        elif side == 0:
            d = "unknown"
        else:
            d = "toward" if (moved > 0) == (side > 0) else "away"
    else:
        d = "none"
    face = {64: "right", 0: "left"}.get(r[me + "facing"], "unknown")
    names = {0: "ryu", 1: "honda", 2: "blanka", 3: "guile", 4: "ken", 5: "chunli", 6: "zangief", 7: "dhalsim"}
    band = bands.get(names.get(r[me + "char"]), bands["all"])
    dist = ("close" if abs(r["p1_x"] - r["p2_x"]) <= band else "far") if xs_ok else "unknown"
    return {"movement": mv, "direction": d, "facing": face, "air": "ground" if r[me + "y"] == 192 else "air",
            "distance": dist}
# ---- end independent ------------------------------------------------------------------------------------------------


def load(data: str) -> Dict:
    meta = json.load(open(os.path.join(data, "build.json")))
    files = {q: {f: MIO.read_jsonl(os.path.join(data, q, f + ".jsonl")) for f in FILES} for q in QUESTION_ANSWERS}
    return {"meta": meta, "files": files}


def _ram(root: str, pair: str, game: int, cache: Dict) -> Optional[List[Dict[str, int]]]:
    key = (pair, game)
    if key not in cache:
        if len(cache) > 64:
            cache.clear()
        path = os.path.join(root, pair, "ram", "g%04d.json.gz" % game)
        cache[key] = MIO.read_ram(path) if os.path.exists(path) else None
    return cache[key]


def label_check(d: Dict, bands: Dict[str, int]) -> Dict:
    root, cache = d["meta"]["root"], {}
    checked, bad, missing, examples = 0, 0, set(), []
    for q, fs in sorted(d["files"].items()):
        for f, rows in fs.items():
            for r in sorted(rows, key=lambda x: (x["pair_name"], x["game"])):
                ram = _ram(root, r["pair_name"], r["game"], cache)
                if ram is None:
                    missing.add("%s g%d" % (r["pair_name"], r["game"]))
                    continue
                t = r["t"]
                got = independent_labels(ram, t, r["slot"], bands)
                want_imgs = ["frames/%s/g%04d_k%05d.png" % (r["pair_name"], r["game"], k) for k in (t - 4 + LAG,
                                                                                                   t + LAG)]
                checked += 1
                ok = (all(got[k] == r[k] for k in got) and r["answer"] == answer(q, got)
                      and QUESTION_ANSWERS[q][r["label"]] == r["answer"] and r["images"] == want_imgs
                      and r["split"] == f == split_of_game(r["game"])
                      and r["controller"] == {1: "directed", 2: "cpu"}[r["slot"]])
                if not ok:
                    bad += 1
                    if len(examples) < 10:
                        examples.append({"id": r["id"], "row": {k: r[k] for k in got}, "ram": got})
    return {"pass": checked > 0 and bad == 0 and not missing, "checked": checked, "mismatches": bad,
            "missing_ram": sorted(missing), "examples": examples}


def cap_check(d: Dict) -> Dict:
    caps = d["meta"]["caps"]
    rows = d["files"]["movement"]["train"] + d["files"]["movement"]["test"]
    per_cell = collections.Counter((r["split"],) + cell(r) for r in rows)
    per_game = collections.Counter((r["pair_name"], r["game"], r["slot"], r["movement"], r["direction"], r["facing"])
                                   for r in rows)
    over = ["%s: %d > %d" % ("|".join(k), n, caps[k[0]]) for k, n in per_cell.items() if n > caps[k[0]]]
    over += ["%s: %d > %d per game" % ("|".join(map(str, k)), n, PER_GAME) for k, n in per_game.items()
             if n > PER_GAME]
    return {"pass": bool(rows) and not over, "over": over[:20], "cells": len(per_cell)}


def alignment_check(data: str, d: Dict, sample: int = SAMPLE, min_disc: int = MIN_DISC, min_agree: float = MIN_AGREE,
                    min_agree_disc: float = MIN_AGREE_DISC, seed: int = 0) -> Dict:
    root, cache, preds = d["meta"]["root"], {}, []
    rows = d["files"]["movement"]["train"] + d["files"]["movement"]["test"]
    seen_imgs = set()
    for r in sorted(rows, key=lambda x: (x["pair_name"], x["game"])):
        if tuple(r["images"]) in seen_imgs:         # both slots share a pair of images: one test per pair
            continue
        seen_imgs.add(tuple(r["images"]))
        ram = _ram(root, r["pair_name"], r["game"], cache)
        if ram is None:
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
            paths = [os.path.join(data, "movement", x) for x in r["images"]]
            gone = [x for x in paths if not os.path.exists(x)]
            if gone:
                missing.append(gone[0])
                continue
            seen = _changed(paths)
            for L in LAGS:
                hits[L] += p[L] == seen
        return {str(L): round(hits[L] / len(items), 4) if items else None for L in LAGS}
    ag, ad = agree(rand), agree(disc)
    ok = alignment_verdict(ag, ad, len(rand), len(disc), len(missing), min_disc, min_agree, min_agree_disc)
    return {"pass": bool(ok), "sample": len(rand), "agreement": ag, "discriminating": len(disc),
            "discriminating_total": len(disc_all), "agreement_discriminating": ad, "missing_images": missing[:10]}


def _changed(paths: Sequence[str]) -> bool:
    """digits_changed reads ``os.path.join(frames, basename)``: give it the directory of the images."""
    d = os.path.dirname(paths[0])
    return digits_changed(d, paths[0], paths[1])


def disk_check(root: str, pairs: Sequence[str], max_gb: float = MAX_GB) -> Dict:
    total = 0
    for n in pairs:
        dd = os.path.join(root, n, "images")
        if os.path.isdir(dd):
            with os.scandir(dd) as it:
                total += sum(e.stat().st_size for e in it if e.is_file())
    return {"pass": total / 1e9 < max_gb, "gb": round(total / 1e9, 4), "max_gb": max_gb}


def counts_table(d: Dict) -> Dict:
    return {"cells": d["meta"]["cells"], "short": d["meta"]["short"], "questions": d["meta"]["questions"]}


def run_gates(data: str, bands: Dict[str, int], max_gb: float = MAX_GB, sample: int = SAMPLE,
              min_disc: int = MIN_DISC, seed: int = 0) -> Dict:
    d = load(data)
    gates = {"labels": label_check(d, bands), "caps": cap_check(d),
             "alignment": alignment_check(data, d, sample, min_disc, seed=seed),
             "disk": disk_check(d["meta"]["root"], d["meta"]["pairs"], max_gb)}
    return {"pass": all(g["pass"] for g in gates.values()), "gates": gates, "counts": counts_table(d)}


# ---- contact sheets -------------------------------------------------------------------------------------------------
THUMB = 256


def contact_sheets(data: str, out: str, per_answer: int = 3, seed: int = 0) -> List[str]:
    """<out>/contact_<char>.png: one band per movement answer, ``per_answer`` pairs (prev | now) of that character
    (either slot), captioned with pair, game, t, slot / controller and the other labels."""
    from PIL import Image, ImageDraw

    d = load(data)
    rows = d["files"]["movement"]["train"] + d["files"]["movement"]["test"]
    answers = QUESTION_ANSWERS["movement"]
    os.makedirs(out, exist_ok=True)
    label_w, cap_h, gap = 150, 16, 8
    paths = []
    for ch in sorted({r["char"] for r in rows}):
        mine = [r for r in rows if r["char"] == ch]
        rng = random.Random("%d:%s" % (seed, ch))
        sheet = Image.new("RGB", (label_w + per_answer * (2 * THUMB + gap), len(answers) * (THUMB + cap_h + gap)),
                          (40, 40, 40))
        draw = ImageDraw.Draw(sheet)
        for i, a in enumerate(answers):
            y = i * (THUMB + cap_h + gap)
            these = [r for r in mine if r["answer"] == a]
            draw.text((6, y + THUMB // 2), "%s\n(%d rows)" % (a, len(these)), fill=(255, 255, 255))
            for j, r in enumerate(rng.sample(these, min(per_answer, len(these)))):
                x = label_w + j * (2 * THUMB + gap)
                for side, p in enumerate(r["images"]):
                    im = Image.open(os.path.join(data, "movement", p)).convert("RGB").resize((THUMB, THUMB))
                    sheet.paste(im, (x + side * THUMB, y))
                draw.text((x + 2, y + THUMB + 2), "%s g%d t%d P%d %s | %s %s %s" % (
                    r["pair_name"], r["game"], r["t"], r["slot"], r["controller"], r["facing"], r["air"],
                    r["distance"]), fill=(255, 255, 0))
        path = os.path.join(out, "contact_%s.png" % ch)
        sheet.save(path)
        paths.append(path)
    return paths
