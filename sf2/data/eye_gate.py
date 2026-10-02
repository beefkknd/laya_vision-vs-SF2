"""The eye datasets' gates before any training (docs/eye_questions_v1.md rules 5-6; scripts/gate_eye_data.py).

labels     every row's answer re-derived here from the stored RAM and move log at the displayed row t (written apart
           from sf2.data.eye_pool / eye_data; the movement rule is sf2.data.pairs_gate's independent one): q1 a
           projectile slot on AND drawn (blink bit 0 clear) whose own player pressed the character's projectile word
           at the flight's first row, inside the screen estimated from the fighters' x (8 px in from both edges),
           in either shown frame (v1.1); "no" = no such slot on in either frame (yoga flame is no fireball; an
           unknown projectile, or a fireball shown in neither frame, may not be a row); q3 the act of
           the grid movement; q4 y == 192; q5 |x1 - x2| <= poke_max["all"]; the side by x at t (q3 / q4); images =
           captures t - 3 and t + 1 (lag 1); 0 mismatches = PASS.
episode    q3: rows t - 4 .. t of the fighter asked share one grid movement (both frames inside one episode).
splits     ONE split table for every dataset: each match's split is pairs_train.split3's (test crc32 % 3 == 2, val 1
           in 6 training matches), the same in every dataset given, no match in two splits.
drawn      q1, matches of Ryu / Ken / Blanka / Zangief with a Ryu or Ken (no blue fighter, every projectile a blue
           hadoken): "yes" rows show >= BLUE_MIN blue pixels below the HUD in EVERY image whose RAM row has the
           fireball drawn on the screen (fire_frames); "no" rows in neither image - >= 95% each (the hard negatives
           among them reported apart).
alignment  RAM-to-image lag 1 (sf2.data.mv3_fireball_gate.alignment_check: HUD clock digits vs the RAM timer).
disk       every image of the collection < 10 GB.
shortcut   sf2.data.eye_shortcut.check: no metadata-only predictor beats chance by more than its margin on test.
"""
import collections
import json
import os
import random
import zlib
from typing import Dict, List, Optional, Sequence

from ..config import REPO
from . import movement_collect_io as MIO
from . import mv3_fireball_gate as FG
from . import pairs_gate as PG
from . import eye_shortcut as SC

GATES = ("labels", "episode", "splits", "drawn", "alignment", "disk", "shortcut")
NOT_BLUE = {"ryu", "ken", "blanka", "zangief"}       # drawn check: no blue fighter, and only hadokens are thrown
FRAME = {"n-4": 0, "n": 1}
THROWN = {"ryu": "hadoken", "ken": "hadoken", "guile": "sonic_boom", "dhalsim": "yoga_fire"}
THRESHOLDS = os.path.join(REPO, "lessons", "perception_thresholds_v2.json")
ANSWERS = {"q1": ("yes", "no"), "q3": ("moving", "attack", "special"), "q4": ("ground", "air"),
           "q5": ("close", "far")}


def band_all(path: str = THRESHOLDS) -> int:
    with open(path) as fh:
        return int(json.load(fh)["poke_max"]["all"])


def rows_of(data: str) -> List[Dict]:
    out = []
    for d in sorted(os.listdir(data)):
        if os.path.isdir(os.path.join(data, d)) and d != "frames":
            for f in ("train", "val", "test"):
                for r in MIO.read_jsonl(os.path.join(data, d, f + ".jsonl")):
                    out.append(dict(r, _dir=d, _file=f))
    return out


class Store:
    def __init__(self, root: str):
        self.root, self.ram, self.logs = root, {}, {}

    def rows(self, pair: str, game: int):
        if (pair, game) not in self.ram:
            if len(self.ram) > 64:
                self.ram.clear()
            self.ram[(pair, game)] = MIO.read_ram(os.path.join(self.root, pair, "ram", "g%04d.json.gz" % game))
        return self.ram[(pair, game)]

    def moves(self, pair: str, game: int) -> Optional[List[list]]:
        if pair not in self.logs:
            self.logs[pair] = {g["game"]: g.get("moves", []) for g in
                               MIO.read_jsonl(os.path.join(self.root, pair, "games.jsonl"))}
        return self.logs[pair].get(game)


def fire_at(ram, moves, t: int, chars: Sequence[str]) -> Optional[str]:
    """yes / no / None for the pair shown at rows (t - 4, t), written from the doc's rule (v1.1): yes = a fireball
    drawn inside the screen in either frame; no = no fireball / unknown projectile on in either; None otherwise."""
    seen = []
    for u in (t - 4, t):
        for s in (1, 2):
            if "shot%d_hide" % s not in ram[u]:
                return None
            if ram[u]["shot%d" % s] == 0:
                continue
            a = u
            while a > 0 and ram[a - 1]["shot%d" % s] != 0:
                a -= 1
            word = PG.independent_pressed(moves, a, s)
            char = chars[s - 1]
            if (char, word) == ("dhalsim", "yoga_flame"):
                continue
            if word != THROWN.get(char):
                seen.append("other")
                continue
            left = min(max((ram[u]["p1_x"] + ram[u]["p2_x"]) / 2.0 - 128, 32), 224)     # the screen's left edge
            inside = 8 <= ram[u]["shot%d_x" % s] - left <= 248
            seen.append("shown" if ram[u]["shot%d_hide" % s] % 2 == 0 and inside else "unseen")
    if "shown" in seen:
        return "yes"
    return None if seen else "no"


def _act(ram, moves, t: int, s: int, char: str, bands: Dict[str, int]) -> str:
    lab = PG.independent_labels(ram, t, s, bands, PG.independent_class(char, PG.independent_pressed(moves, t, s)))
    grid = PG._grid(lab)
    return "unknown" if grid == "unknown" else grid if grid in ("attack", "special") else "moving"


def _side(ram, t: int, s: int) -> Optional[str]:
    me, him = ram[t]["p%d_x" % s], ram[t]["p%d_x" % (3 - s)]
    return None if me == him else "left" if me < him else "right"


def expected(q: str, r: Dict, ram, moves, band: int) -> Dict:
    t, chars = r["t"], r["pair_name"].split("_vs_")
    if q == "q1":
        return {"answer": fire_at(ram, moves, t, chars)}
    if q == "q5":
        x1, x2 = ram[t]["p1_x"], ram[t]["p2_x"]
        ok = 0 <= x1 <= 512 and 0 <= x2 <= 512 and x1 != x2
        return {"answer": ("close" if abs(x1 - x2) <= band else "far") if ok else None}
    s = r["slot"]
    side = _side(ram, t, s)
    if q == "q4":
        return {"answer": "ground" if ram[t]["p%d_y" % s] == 192 else "air", "side": side}
    return {"answer": _act(ram, moves, t, s, chars[s - 1], {"all": band}), "side": side}


def label_check(q: str, data: str, rows: Sequence[Dict], store: Store, band: int) -> Dict:
    bad, examples, missing = 0, [], 0
    for r in sorted(rows, key=lambda x: (x["pair_name"], x["game"], x["t"])):
        moves = store.moves(r["pair_name"], r["game"])
        if moves is None:
            missing += 1
            continue
        ram = store.rows(r["pair_name"], r["game"])
        want = expected(q, r, ram, moves, band)
        imgs = ["frames/%s/g%04d_k%05d.png" % (r["pair_name"], r["game"], k) for k in (r["t"] - 3, r["t"] + 1)]
        ok = (want["answer"] == r["answer"] == r["_dir"] and ANSWERS[q][r["label"]] == r["answer"]
              and r["images"] == imgs and all(os.path.exists(os.path.join(data, r["_dir"], p)) for p in imgs)
              and ("side" not in want or want["side"] == r.get("side")))
        if not ok:
            bad += 1
            if len(examples) < 10:
                examples.append({"id": r["id"], "row": r["answer"], "ram": want})
    return {"pass": bool(rows) and bad == 0 and missing == 0, "checked": len(rows), "mismatches": bad,
            "uncommitted": missing, "examples": examples}


def episode_check(rows: Sequence[Dict], store: Store, band: int, gap: int = 4) -> Dict:
    bad, examples = 0, []
    for r in rows:
        ram, moves = store.rows(r["pair_name"], r["game"]), store.moves(r["pair_name"], r["game"]) or []
        s, char = r["slot"], r["pair_name"].split("_vs_")[r["slot"] - 1]
        seq = [PG._grid(PG.independent_labels(ram, u, s, {"all": band}, PG.independent_class(
            char, PG.independent_pressed(moves, u, s)))) for u in range(r["t"] - gap, r["t"] + 1)]
        if "unknown" in seq or len(set(seq)) != 1:
            bad += 1
            if len(examples) < 10:
                examples.append({"id": r["id"], "rows_t4_to_t": seq})
    return {"pass": bool(rows) and bad == 0, "checked": len(rows), "outside": bad, "examples": examples}


def split_of(pair: str, game: int) -> str:
    if zlib.crc32(("%s:%d" % (pair, game)).encode()) % 3 == 2:
        return "test"
    return "val" if zlib.crc32(("val:%s:%d" % (pair, game)).encode()) % 6 == 0 else "train"


def split_check(all_rows: Dict[str, Sequence[Dict]]) -> Dict:
    """Over every dataset given: each match in one split, that split = the rule, the same in every dataset."""
    seen: Dict[tuple, set] = collections.defaultdict(set)
    bad = []
    for name, rows in all_rows.items():
        for r in rows:
            m = (r["pair_name"], r["game"])
            seen[m].add(r["_file"])
            if r["_file"] != split_of(*m) or r["split"] != r["_file"]:
                bad.append("%s %s: in %s, rule %s" % (name, r["id"], r["_file"], split_of(*m)))
    bad += ["match %s:%d in %s" % (m[0], m[1], sorted(s)) for m, s in seen.items() if len(s) > 1]
    per = {f: sum(1 for s in seen.values() if s == {f}) for f in ("train", "val", "test")}
    return {"pass": bool(seen) and not bad, "matches": per, "problems": bad[:20], "n_problems": len(bad)}


def drawn_check(data: str, rows: Sequence[Dict], need: float = FG.DRAWN_MIN, blue_min: int = FG.BLUE_MIN) -> Dict:
    def hadoken(r):
        chars = set(r["pair_name"].split("_vs_"))
        return chars <= NOT_BLUE and bool(chars & set(FG.HADOKEN))    # no blue fighter: any blue is a hadoken
    yes = [r for r in rows if r["answer"] == "yes" and hadoken(r)]
    no = [r for r in rows if r["answer"] == "no" and hadoken(r)]
    blue = lambda r, i: FG.blue_pixels(os.path.join(data, r["_dir"], r["images"][i])) >= blue_min
    hit = sum(bool(r["fire_frames"]) and all(blue(r, FRAME[f]) for f in r["fire_frames"]) for r in yes)
    clear = [not blue(r, 0) and not blue(r, 1) for r in no]
    hard = [c for c, r in zip(clear, no) if r["hard"]]
    res = {"hadoken_yes": len(yes), "blue": hit, "ryu_ken_no": len(no), "clear": sum(clear),
           "ryu_ken_no_hard": len(hard), "hard_clear": sum(hard), "blue_min": blue_min, "need": need}
    res["pass"] = bool(yes) and bool(no) and hit >= need * len(yes) and sum(clear) >= need * len(no)
    return res


def run_gates(datas: Dict[str, str], root: str, sample: int = 400, seed: int = 0, margin: float = SC.MARGIN,
              max_gb: float = PG.MAX_GB) -> Dict:
    """datas: {question: data dir}. One report per dataset + the shared split table."""
    band, store = band_all(), Store(root)
    rows = {q: rows_of(d) for q, d in datas.items()}
    pairs = sorted(x for x in os.listdir(root) if "_vs_" in x)
    disk = PG.disk_check(root, pairs, max_gb)
    splits = split_check(rows)
    out = {"splits": splits, "datasets": {}}
    for q, d in datas.items():
        g = {"labels": label_check(q, d, rows[q], store, band),
             "episode": episode_check(rows[q], store, band) if q == "q3" else {"pass": True, "n/a": q},
             "splits": {"pass": splits["pass"]},
             "drawn": drawn_check(d, rows[q]) if q == "q1" else {"pass": True, "n/a": q},
             "alignment": FG.alignment_check(d, root, rows[q], sample, seed=seed),
             "disk": disk, "shortcut": SC.check(rows[q], ANSWERS[q], margin)}
        out["datasets"][q] = {"pass": all(x["pass"] for x in g.values()), "gates": g}
    out["pass"] = all(v["pass"] for v in out["datasets"].values()) and splits["pass"]
    return out


# ---- contact sheets ------------------------------------------------------------------------------------------------
THUMB = 192


def bands_of(q: str, rows: Sequence[Dict]) -> List:
    if q != "q1":
        return [(a, [r for r in rows if r["answer"] == a]) for a in ANSWERS[q]]
    no = [r for r in rows if r["answer"] == "no"]
    out = [("yes", [r for r in rows if r["answer"] == "yes"]), ("yes, throwing pose",
                                                                [r for r in rows if r["answer"] == "yes" and
                                                                 "projectile" in r["poses"]])]
    yes = [r for r in rows if r["answer"] == "yes"]
    out += [("yes: drawn in %s only" % f, [r for r in yes if r.get("fire_frames") == [f]]) for f in ("n-4", "n")]
    out += [("no: %s" % tag, [r for r in no if tag in r["hard"]]) for tag in
            ("before_spawn", "after_impact", "pose", "yoga_flame")]
    out.append(("no: not hard", [r for r in no if not r["hard"]]))
    return out


def contact_sheet(q: str, data: str, path: str, per_band: int = 4, seed: int = 0) -> str:
    from PIL import Image, ImageDraw

    rows = rows_of(data)
    bands = bands_of(q, rows)
    label_w, cap_h, gap = 150, 14, 6
    sheet = Image.new("RGB", (label_w + per_band * (2 * THUMB + gap), len(bands) * (THUMB + cap_h + gap)),
                      (40, 40, 40))
    draw = ImageDraw.Draw(sheet)
    rng = random.Random("%d:%s" % (seed, q))
    for i, (lab, these) in enumerate(bands):
        y = i * (THUMB + cap_h + gap)
        draw.text((4, y + THUMB // 2), "%s\n(%d rows)" % (lab, len(these)), fill=(255, 255, 255))
        for j, r in enumerate(rng.sample(these, min(per_band, len(these)))):
            x = label_w + j * (2 * THUMB + gap)
            for k, p in enumerate(r["images"]):
                im = Image.open(os.path.join(data, r["_dir"], p)).convert("RGB").resize((THUMB, THUMB))
                sheet.paste(im, (x + k * THUMB, y))
            what = r.get("side") or ",".join(r.get("hard", [])) or "-"
            draw.text((x + 2, y + THUMB + 1), "%s g%d t%d %s %s" % (r["pair_name"], r["game"], r["t"], r["answer"],
                                                                     what), fill=(255, 255, 0))
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    sheet.save(path)
    return path
