"""The movement-pairs gates before any training (docs/prereg_movement_pairs.md; scripts/gate_pairs_data.py).

labels     every row of every question file: its five labels re-derived from the collection's stored RAM by
           ``independent_labels`` (written from the prereg / pairs_labels docstring, deliberately not importing
           sf2.data.pairs_labels) at the displayed row t must match 100%; the question's answer and label index must
           follow from them; the images must be the captures t - 4 + LAG and t + LAG of its game.
caps       per (split, char, movement10, facing) - the owner's grid cell - at most the build's cap (its movement_caps
           for that movement when set, round 4: attack / special); per (pair, game, slot, movement10, facing) at most
           PER_GAME (re-counted here; movement10 re-derived here too).
kept       round 4: every movement row of the build named in keep_from is in this build, same id and split.
alignment  RAM-to-image lag 1 on a sample: the HUD clock digits change between a pair's two images iff the RAM timer
           changed between the rows they show (sf2.data.movement_gate.digits_changed / alignment_verdict).
disk       every image file the collection wrote < max_gb (10 GB).
second_fact (docs/prereg_movement_finetunes.md) each "down" pair on the ground (y == 192) at t - 4 .. t: 100%; "hit"
           pairs: health lost within the hit episode (reported, not gated).
The counts table (per cell and per question) and the shortfalls are reported, not gated (never padded).
"""
import collections
import json
import os
import random
from typing import Dict, List, Optional, Sequence

from . import movement_collect_io as MIO
from . import pairs_moves as PM
from .movement_gate import LAGS, MIN_AGREE, MIN_AGREE_DISC, MIN_DISC, SAMPLE, alignment_verdict, digits_changed
from .pairs_collect import PER_GAME
from .pairs_data import FILES, QUESTION_ANSWERS, answer, cell, split_of_game
from ..vocab import FULL_LIFE

CONTROLLER_OF = ({1: "directed", 2: "cpu"}, {1: "p1", 2: "p2"})   # P1 vs CPU, or Plan B (both ours)

GATES = ("labels", "episode", "caps", "kept", "alignment", "disk", "second_fact")
MAX_GB = 10.0
LAG = 1


# ---- the independent labels (from the prereg's rules; deliberately not sf2.data.pairs_labels) ----------------------

def independent_pressed(moves: List[list], t: int, p: int) -> Optional[str]:
    """The word slot ``p`` pressed for row ``t``: the logged move with k0 < t <= k1 (games.jsonl rows are
    [word, k0, k1, status, slot]; an old 3/4-field log is player 1's)."""
    for m in moves:
        slot = m[4] if len(m) >= 5 else 1
        if slot == p and m[1] < t <= m[2]:
            return m[0]
    return None


def independent_class(char: str, word: Optional[str]) -> Optional[str]:
    """Owner after round 1: normal / crouching normal / jump attack / throw -> attack, the character's special ->
    special (the kinds of the hardcoded list), anything else (walks, jumps, blocks) -> None."""
    if word is None:
        return None
    k = PM.kind(char, word)
    return {"normal": "attack", "crouch_normal": "attack", "jump_attack": "attack", "throw": "attack",
            "special": "special"}.get(k)


def independent_labels(rows: List[Dict[str, int]], t: int, p: int, bands: Dict[str, int],
                       pressed: Optional[str] = None) -> Dict[str, str]:
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
        mv = "down" if r[me + "y"] == 192 else "hit"           # down only once back on the ground (owner)
    elif state == 8 or (state == 14 and blocked):
        mv = "block"
    elif state == 14:
        mv = "hit"
    elif state == 12 or (state == 10 and mcl == 8):
        mv = "special"
    elif state == 10:
        mv = "attack"
    elif state == 4 or r[me + "y"] != 192:
        mv = "attack" if r[me + "aid"] != 0 else "jump"           # Plan B: a jump attack (box out) is an attack
    elif state == 2:
        mv = "crouch"
    elif state == 0 and moved is not None:
        mv = "stand" if -2 < moved < 2 else "walk"
    else:
        mv = "unknown"
    if mv in ("attack", "special") and pressed in ("attack", "special"):
        mv = pressed                                    # the move we pressed decides, RAM confirmed the attack
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


def _moves(root: str, pair: str, game: int, cache: Dict) -> List[list]:
    if (pair, "log") not in cache:
        cache[(pair, "log")] = {g["game"]: g.get("moves", []) for g in
                                MIO.read_jsonl(os.path.join(root, pair, "games.jsonl"))}
    return cache[(pair, "log")].get(game, [])


def label_check(d: Dict, bands: Dict[str, int]) -> Dict:
    root, cache, logs = d["meta"]["root"], {}, {}
    checked, bad, missing, examples = 0, 0, set(), []
    for q, fs in sorted(d["files"].items()):
        for f, rows in fs.items():
            for r in sorted(rows, key=lambda x: (x["pair_name"], x["game"])):
                ram = _ram(root, r["pair_name"], r["game"], cache)
                if ram is None:
                    missing.add("%s g%d" % (r["pair_name"], r["game"]))
                    continue
                t = r["t"]
                word = independent_pressed(_moves(root, r["pair_name"], r["game"], logs), t, r["slot"])
                got = independent_labels(ram, t, r["slot"], bands, independent_class(r["char"], word))
                want_imgs = ["frames/%s/g%04d_k%05d.png" % (r["pair_name"], r["game"], k) for k in (t - 4 + LAG,
                                                                                                   t + LAG)]
                checked += 1
                ok = (all(got[k] == r[k] for k in got) and r.get("pressed") == word and r["answer"] == answer(q, got)
                      and QUESTION_ANSWERS[q][r["label"]] == r["answer"] and r["images"] == want_imgs
                      and r["split"] == f == split_of_game(r["pair_name"], r["game"])
                      and any(r["controller"] == m[r["slot"]] for m in CONTROLLER_OF))
                if not ok:
                    bad += 1
                    if len(examples) < 10:
                        examples.append({"id": r["id"], "row": {k: r[k] for k in got}, "ram": got})
    return {"pass": checked > 0 and bad == 0 and not missing, "checked": checked, "mismatches": bad,
            "missing_ram": sorted(missing), "examples": examples}


def _grid(lab: Dict[str, str]) -> str:
    if lab["movement"] == "walk":
        return "walk " + lab["direction"] if lab["direction"] in ("toward", "away") else "unknown"
    return lab["movement"]


def episode_check(d: Dict, bands: Dict[str, int], gap: int = 4) -> Dict:
    """Every selected pair: rows t - gap .. t of its fighter all have the final grid movement of row t (re-derived
    here from RAM + the move log), so both frames lie inside one movement episode (owner fix)."""
    root, cache, logs = d["meta"]["root"], {}, {}
    rows = d["files"]["movement"]["train"] + d["files"]["movement"]["test"]
    checked, bad, examples = 0, 0, []
    for r in sorted(rows, key=lambda x: (x["pair_name"], x["game"])):
        ram = _ram(root, r["pair_name"], r["game"], cache)
        moves = _moves(root, r["pair_name"], r["game"], logs)
        t, p = r["t"], r["slot"]
        seq = [_grid(independent_labels(ram, u, p, bands, independent_class(r["char"],
                                                                            independent_pressed(moves, u, p))))
               if ram is not None and 0 <= u < len(ram) else "unknown" for u in range(t - gap, t + 1)]
        checked += 1
        if "unknown" in seq or len(set(seq)) != 1:
            bad += 1
            if len(examples) < 10:
                examples.append({"id": r["id"], "rows_t4_to_t": seq})
    return {"pass": checked > 0 and bad == 0, "checked": checked, "outside": bad, "examples": examples}


# ---- second facts (docs/prereg_movement_finetunes.md): each label against a different RAM field -------------------

def down_on_ground(rows: List[Dict[str, int]], t: int, p: int, gap: int = 4) -> bool:
    """A "down" pair: fighter ``p`` on the ground (y == 192) at every row t - gap .. t (both displayed frames)."""
    return t - gap >= 0 and t < len(rows) and all(rows[u]["p%d_y" % p] == 192 for u in range(t - gap, t + 1))


def hit_episode(rows: List[Dict[str, int]], t: int, p: int, bands: Dict[str, int]):
    """(first, last) row of the run of independent "hit" rows of fighter ``p`` around ``t`` (None if t is no hit)."""
    def is_hit(u):
        return 0 <= u < len(rows) and _grid(independent_labels(rows, u, p, bands)) == "hit"
    if not is_hit(t):
        return None
    s, e = t, t
    while is_hit(s - 1):
        s -= 1
    while is_hit(e + 1):
        e += 1
    return s, e


def _health(v: int) -> int:
    return v if 0 <= v <= FULL_LIFE else 0          # a KO under-runs the 2-byte health: count it as 0


def hit_health_lost(rows: List[Dict[str, int]], t: int, p: int, bands: Dict[str, int]) -> bool:
    """A "hit" pair: fighter ``p``'s health (hp) somewhere in its hit episode is below the row before the episode."""
    ep = hit_episode(rows, t, p, bands)
    if ep is None or ep[0] == 0:
        return False
    s, e = ep
    hp = "p%d_hp" % p
    return min(_health(rows[u][hp]) for u in range(s, e + 1)) < _health(rows[s - 1][hp])


def second_fact_check(d: Dict, bands: Dict[str, int], ram_of=None) -> Dict:
    """Every "down" pair on the ground at t - 4 .. t (gated: 100%); "hit" pairs with health lost in their hit episode
    (reported, not gated: a hit can be a throw's or a combo's later frames). ``ram_of(pair, game)``: the RAM rows
    (default: the collection's)."""
    if ram_of is None:
        cache: Dict = {}
        root = d["meta"]["root"]

        def ram_of(pair, game):
            return _ram(root, pair, game, cache)
    rows = d["files"]["movement"]["train"] + d["files"]["movement"]["test"]
    down = {"total": 0, "on_ground": 0, "examples": []}
    hit = {"total": 0, "health_lost": 0, "examples": []}
    for r in sorted(rows, key=lambda x: (x["pair_name"], x["game"])):
        if r["answer"] not in ("down", "hit"):
            continue
        ram = ram_of(r["pair_name"], r["game"])
        if r["answer"] == "down":
            ok = ram is not None and down_on_ground(ram, r["t"], r["slot"])
            down["total"] += 1
            down["on_ground"] += ok
            if not ok and len(down["examples"]) < 10:
                down["examples"].append(r.get("id", "%s g%s t%s" % (r["pair_name"], r["game"], r["t"])))
        else:
            ok = ram is not None and hit_health_lost(ram, r["t"], r["slot"], bands)
            hit["total"] += 1
            hit["health_lost"] += ok
            if not ok and len(hit["examples"]) < 10:
                hit["examples"].append(r.get("id", "%s g%s t%s" % (r["pair_name"], r["game"], r["t"])))
    for x, k in ((down, "on_ground"), (hit, "health_lost")):
        x["pct"] = round(100.0 * x[k] / x["total"], 1) if x["total"] else None
    return {"pass": down["on_ground"] == down["total"], "down": down, "hit": hit}


def cap_check(d: Dict) -> Dict:
    """Caps per cell: the build's movement_caps for that grid movement (round 4: attack / special), else its split's
    cap; movement10 re-derived here."""
    caps, mcaps = d["meta"]["caps"], d["meta"].get("movement_caps") or {}
    rows = d["files"]["movement"]["train"] + d["files"]["movement"]["test"]
    def grid(r):
        return ("walk " + r["direction"]) if r["movement"] == "walk" else r["movement"]
    per_cell = collections.Counter((r["split"], r["char"], grid(r), r["facing"]) for r in rows)
    per_game = collections.Counter((r["pair_name"], r["game"], r["slot"], grid(r), r["facing"]) for r in rows)
    def cap(k):
        return mcaps[k[2]][k[0]] if k[2] in mcaps else caps[k[0]]
    over = ["%s: %d > %d" % ("|".join(k), n, cap(k)) for k, n in per_cell.items() if n > cap(k)]
    over += ["%s: %d > %d per game" % ("|".join(map(str, k)), n, PER_GAME) for k, n in per_game.items()
             if n > PER_GAME]
    return {"pass": bool(rows) and not over, "over": over[:20], "cells": len(per_cell)}


def kept_check(d: Dict) -> Dict:
    """Round 4: every movement row of the kept build (build.json keep_from) is in this build, same id and split, so
    the earlier test rows stay comparable. Passes when nothing was kept."""
    src = d["meta"].get("keep_from")
    if not src:
        return {"pass": True, "keep_from": None}
    mine = {r["id"]: f for f in FILES for r in d["files"]["movement"][f]}
    theirs = {r["id"]: f for f in FILES for r in MIO.read_jsonl(os.path.join(src, "movement", f + ".jsonl"))}
    missing = sorted(i for i in theirs if i not in mine)
    moved = sorted(i for i in theirs if i in mine and mine[i] != theirs[i])
    return {"pass": bool(theirs) and not missing and not moved, "keep_from": src, "kept_rows": len(theirs),
            "per_split": dict(collections.Counter(theirs.values())), "missing": missing[:10],
            "n_missing": len(missing), "moved": moved[:10]}


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
    gates = {"labels": label_check(d, bands), "episode": episode_check(d, bands), "caps": cap_check(d), "kept": kept_check(d),
             "alignment": alignment_check(data, d, sample, min_disc, seed=seed),
             "disk": disk_check(d["meta"]["root"], d["meta"]["pairs"], max_gb),
             "second_fact": second_fact_check(d, bands)}
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
