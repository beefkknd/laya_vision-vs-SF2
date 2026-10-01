"""The gates on the three movement datasets (act / where / dist; sf2.data.action_data; scripts/gate_action_data.py),
before any training. The exit code decides: a dataset passes only when every gate below passes.

  labels     every row re-derived from the collection's stored RAM by an INDEPENDENT implementation (below: the
             action probe's episodes and no-box classes, sf2.data.action_probe, plus this file's own row rules; not
             sf2.data.action_codes / perception), at the displayed row t: 100% match. The act set is also checked
             complete: a fighter labelled at t has both its rows (act + stage), an unlabelled one none. Images must be
             the captures t - 4 + LAG and t + LAG; the label index must name the answer.
  alignment  RAM-to-image lag 1 by the HUD clock (sf2.data.movement_gate.alignment_check, unchanged), one row per pair.
  disk       the images under the frames dirs < max_gb.
  trainable  scripts/train.py's own data checks on the dataset's dirs (sf2.data.train_data): rows load as laya
             examples, the note tags agree, and with --balance sampling every dir has >= 1,000 train and >= 100
             validation rows and equal shares; the validation size is reported against train.py's --val-limit.
  coverage   (report, never a fail; the owner decides) every (actor, code) the collection saw: its train / val / test
             rows in the act set; shortfalls (none in train or none in test) listed by name.
"""
import collections
import json
import os
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from . import action_probe as AP
from . import movement_collect_io as MIO
from .movement_gate import SAMPLE, MIN_AGREE, MIN_AGREE_DISC, MIN_DISC, MAX_GB, alignment_check, disk_check

FILES = ("train", "val", "test_real")
SPLIT_OF_FILE = {"train": "train", "val": "val", "test_real": "test"}
GATES = ("labels", "alignment", "disk", "trainable")
VAL_LIMIT = 4000                  # scripts/train.py --val-limit default
LAG, T0, GROUND = 1, 4, 192


# ---- the independent labels -----------------------------------------------------------------------------------------

def _row_code(rows: List[Dict[str, int]], t: int, p: int) -> Optional[int]:
    me, him = "p%d_" % p, "p%d_" % (3 - p)
    now = rows[t]
    s = now[me + "state"]
    if s == 0x14:
        return 77
    if s == 0x0E:
        return 75 if now[me + "react"] in (0x06, 0x08) else 76
    if s == 0x08:
        return 75
    if now[me + "y"] != GROUND:
        return 74
    if s == 0x02:
        return 73
    if s != 0x00:
        return None
    before = rows[t - 4]
    xs = (before[me + "x"], now[me + "x"], now[him + "x"])
    if min(xs) < 0 or max(xs) > 512:
        return None
    moved = xs[1] - xs[0]
    if -2 < moved < 2:
        return 70
    toward = xs[2] - xs[1]
    if toward == 0:
        return None
    return 71 if (moved > 0) == (toward > 0) else 72


def _run_code(struct: np.ndarray, ep: "AP.Episode", shot: Sequence[int]) -> Optional[int]:
    if ep.attack_id:
        return ep.attack_id if ep.attack_id < 60 else None
    if ep.state == 0x04:
        return 74
    kind = AP.no_box_kind(struct, ep, shot)
    if kind == "throw":
        return 60
    if kind == "projectile":
        return 61
    if kind.startswith("special_"):
        c = 80 + int(kind[len("special_"):], 16)
        return c if c <= 99 else None
    return None


def independent_actions(rows: List[Dict[str, int]], p: int) -> Dict[int, Tuple[int, int]]:
    """Displayed row t -> (code, stage 1-3) of player ``p`` for every labelled row."""
    n = len(rows) - T0
    if n <= 0:
        return {}
    pre = "p%d_" % p
    struct = np.zeros((n, AP.MOVE_CLASS + 1), np.uint8)
    for i, r in enumerate(rows[T0:]):
        struct[i, AP.STATE] = r[pre + "state"]
        struct[i, AP.ATTACK_ID] = r[pre + "aid"]
        struct[i, AP.MOVE_CLASS] = r[pre + "mclass"]
        struct[i, AP.SPECIAL_CLASS] = r[pre + "sclass"]
    shot = [r["shot%d" % p] for r in rows[T0:]]
    seg: List[Tuple] = [None] * n
    code: List[Optional[int]] = [None] * n
    for ep in AP.episodes(struct):
        run = ep.state in (0x0A, 0x0C, 0x04)
        c_run = _run_code(struct, ep, shot) if run else None
        for i in range(ep.start, ep.end):
            code[i] = c_run if run else _row_code(rows, i + T0, p)
            seg[i] = ("run", ep.start) if run else ("row", code[i])
    out: Dict[int, Tuple[int, int]] = {}
    i = 0
    while i < n:
        j = i
        while j + 1 < n and seg[j + 1] == seg[i]:
            j += 1
        if code[i] is not None:
            length = j - i + 1
            for k in range(i, j + 1):
                out[k + T0] = (code[i], 1 + (3 * (k - i)) // length)
        i = j + 1
    return out


def independent_band(r: Dict[str, int], th: Dict) -> Optional[str]:
    if r["p1_char"] != 5:                         # 5 = Chun-Li (sf2.vocab); the bands below are hers
        return None
    if not all(0 <= r[k] <= 512 for k in ("p1_x", "p2_x")):
        return None
    gap = abs(r["p1_x"] - r["p2_x"])
    if gap <= th["throw_max"].get("chunli", th["throw_max"]["all"]):
        return "throw"
    if gap <= th["poke_max"].get("chunli", th["poke_max"]["all"]):
        return "poke"
    return "mid" if gap < th["mid_max"] else "far"


# ---- loading ------------------------------------------------------------------------------------------------------

def load(data: str) -> Tuple[Dict, Dict[str, Dict[str, List[Dict]]]]:
    meta = json.load(open(os.path.join(data, "build.json")))
    files = {o: {n: MIO.read_jsonl(os.path.join(data, o, n + ".jsonl")) for n in FILES} for o in meta["opps"]}
    return meta, files


def _ram(root: str, opp: str, game: int) -> Optional[List[Dict[str, int]]]:
    path = os.path.join(root, opp, "ram", "g%04d.json.gz" % game)
    return MIO.read_ram(path) if os.path.exists(path) else None


def _row_ok(r: Dict) -> bool:
    t = r["t"]
    imgs = ["frames/g%04d_k%05d.png" % (r["game"], k) for k in (t - 4 + LAG, t + LAG)]
    return r["images"] == imgs and list(r["question"]["criteria"])[r["label"]] == r["answer"]


def label_check(meta: Dict, files: Dict[str, Dict[str, List[Dict]]], th: Dict) -> Dict:
    ds = meta["dataset"]
    checked, bad, missing, examples = 0, 0, set(), []

    def fail(r, want):
        nonlocal bad
        bad += 1
        if len(examples) < 10:
            examples.append({"id": r.get("id"), "row": r.get("answer"), "ram": want})

    for o, fs in sorted(files.items()):
        by_game: Dict[int, List[Dict]] = collections.defaultdict(list)
        for rows in fs.values():
            for r in rows:
                by_game[r["game"]].append(r)
        for game, rs in sorted(by_game.items()):
            ram = _ram(meta["root"], o, game)
            if ram is None:
                missing.add("%s g%d" % (o, game))
                continue
            if ds == "act":
                labs = {p: independent_actions(ram, p) for p in (1, 2)}
                chars = {1: meta["me"], 2: o}
                by_dec: Dict[str, List[Dict]] = collections.defaultdict(list)
                for r in rs:
                    by_dec[r["decision"]].append(r)
                for dec, group in by_dec.items():
                    t = group[0]["t"]
                    for p in (1, 2):
                        mine = [r for r in group if r["player"] == p]
                        want = labs[p].get(t)
                        checked += 1
                        if want is None:
                            if mine:
                                fail(mine[0], None)
                            continue
                        keys = sorted(r["key"] for r in mine)
                        answers = {r["key"]: r["answer"] for r in mine}
                        ok = keys == ["act", "stage"] and answers == {"act": "act%02d" % want[0],
                                                                      "stage": "stg%d" % want[1]}
                        ok = ok and all(_row_ok(r) and r["actor"] == chars[p] and r["code"] == want[0]
                                        and r["stg"] == want[1] for r in mine)
                        if not ok:
                            fail(mine[0] if mine else {"id": dec}, list(want))
            else:
                for r in rs:
                    checked += 1
                    now = ram[r["t"]]
                    want = (("on the ground" if now["p2_y"] == GROUND else "in the air") if ds == "where"
                            else independent_band(now, th))
                    if want != r["answer"] or not _row_ok(r):
                        fail(r, want)
    return {"pass": checked > 0 and bad == 0 and not missing, "checked": checked, "mismatches": bad,
            "missing_ram": sorted(missing), "examples": examples}


def one_per_pair(files: Dict[str, Dict[str, List[Dict]]]) -> Dict[str, Dict[str, List[Dict]]]:
    out = {}
    for o, fs in files.items():
        out[o] = {}
        for n, rows in fs.items():
            seen, keep = set(), []
            for r in rows:
                if r["decision"] not in seen:
                    seen.add(r["decision"])
                    keep.append(r)
            out[o][n] = keep
    return out


# ---- coverage (report) --------------------------------------------------------------------------------------------

def observed_codes(root: str, opps: Sequence[str]) -> collections.Counter:
    """(actor, code) -> episodes the collection saw (games.jsonl "observed"), over every committed game."""
    c: collections.Counter = collections.Counter()
    for o in opps:
        for g in MIO.read_jsonl(os.path.join(root, o, "games.jsonl")):
            for k, n in g.get("observed", {}).items():
                actor, code = k.split("|")
                c[(actor, int(code))] += n
    return c


def coverage(meta: Dict, files: Dict[str, Dict[str, List[Dict]]]) -> Dict:
    rows_by: Dict[Tuple[str, int], collections.Counter] = collections.defaultdict(collections.Counter)
    for fs in files.values():
        for name, rows in fs.items():
            for r in rows:
                if r["key"] == "act":
                    rows_by[(r["actor"], r["code"])][SPLIT_OF_FILE[name]] += 1
    obs = observed_codes(meta["root"], meta["opps"])
    keys = sorted(set(obs) | set(rows_by))
    table = {"%s act%02d" % k: {"episodes_seen": obs.get(k, 0), "train": rows_by[k]["train"],
                                "val": rows_by[k]["val"], "test": rows_by[k]["test"]} for k in keys}
    short = ["%s act%02d: %s (episodes seen %d)" % (k[0], k[1], ", ".join(
        "no %s rows" % s for s in ("train", "test") if not rows_by[k][s]), obs.get(k, 0))
        for k in keys if not rows_by[k]["train"] or not rows_by[k]["test"]]
    return {"pass": True, "report_only": True, "codes": len(keys), "shortfalls": short, "table": table}


# ---- train.py's data checks ---------------------------------------------------------------------------------------

def trainable(data: str, opps: Sequence[str], val_limit: int = VAL_LIMIT) -> Dict:
    from . import train_data as TD

    dirs = [os.path.join(data, o) for o in opps]
    train, val = TD.load_data(dirs)
    n_rows = sum(len(MIO.read_jsonl(os.path.join(d, f + ".jsonl"))) for d in dirs for f in ("train", "val"))
    out = {"train": len(train), "val": len(val), "loaded_of_rows": [len(train) + len(val), n_rows]}
    try:
        out["tags"] = TD.checkpoint_tags(train + val)
    except ValueError as e:
        out["tags_error"] = str(e)
    out["sampling_problems"] = TD.sampling_problems(train, val, dirs)
    out["rows_problems"] = TD.coverage_problems(train, val, dirs)
    out["val_over_limit"] = len(val) > val_limit
    out["val_limit"] = val_limit
    out["pass"] = (len(train) + len(val) == n_rows and "tags" in out and not out["sampling_problems"]
                   and not TD.coverage_problems(train, val, dirs, share_check=False) and not out["val_over_limit"])
    out["train_py_flags"] = ["--balance sampling"] + ([] if len(val) <= VAL_LIMIT else ["--val-limit %d" % len(val)])
    return out


def run_gates(data: str, thresholds: str, max_gb: float = MAX_GB, sample: int = SAMPLE, seed: int = 0,
              val_limit: int = VAL_LIMIT, check_train: bool = True) -> Dict:
    meta, files = load(data)
    th = json.load(open(thresholds))
    gates = {
        "labels": label_check(meta, files, th),
        "alignment": alignment_check(data, meta, one_per_pair(files), sample, MIN_DISC, MIN_AGREE, MIN_AGREE_DISC,
                                     seed),
        "disk": disk_check(data, meta["opps"], max_gb),
        "trainable": trainable(data, meta["opps"], val_limit) if check_train else {"pass": True, "skipped": True},
    }
    rep = {"pass": all(g["pass"] for g in gates.values()), "dataset": meta["dataset"], "gates": gates,
           "rows": {o: {n: len(r) for n, r in fs.items()} for o, fs in files.items()}, "opps": meta["opps"]}
    if meta["dataset"] == "act":
        rep["coverage"] = coverage(meta, files)
    return rep


# ---- contact sheets (act) -----------------------------------------------------------------------------------------

def move_names(path: str) -> Dict[int, str]:
    if not os.path.exists(path):
        return {}
    d = json.load(open(path))
    out = {int(k): "/".join(v["moves"]) for k, v in d.get("id_to_moves", {}).items()}
    for k, v in d.get("no_box_fallback", {}).items():      # "state 0x0C, 0x0C49 = 0x09": "spinning_bird_kick (46)"
        if "0x0C49 = 0x" in k:
            out[-(80 + int(k.rsplit("0x", 1)[1], 16))] = v      # keyed by -code: a no-box special's name
    return out


def contact_sheets(data: str, out: str, chunli_ids: str, reserved: Dict[int, str], thumb: int = 192,
                   per_row: int = 3) -> List[str]:
    """<out>/contact_<actor>.png: one pair per action code of that actor, at stage 2 (middle) when it has one,
    captioned with the code (and, for Chun-Li, her move name from ``chunli_ids``; reserved codes by name)."""
    from PIL import Image, ImageDraw

    meta, files = load(data)
    names = move_names(chunli_ids)
    picks: Dict[str, Dict[int, Tuple[str, Dict]]] = collections.defaultdict(dict)
    for o, fs in sorted(files.items()):
        for name in FILES:
            for r in fs[name]:
                if r["key"] != "act":
                    continue
                have = picks[r["actor"]].get(r["code"])
                if have is None or (have[1]["stg"] != 2 and r["stg"] == 2):
                    picks[r["actor"]][r["code"]] = (o, r)
    os.makedirs(out, exist_ok=True)
    cap_h, gap = 30, 8
    paths = []
    for actor, codes in sorted(picks.items()):
        items = sorted(codes.items())
        nrows = (len(items) + per_row - 1) // per_row
        W, Hc = per_row * (2 * thumb + gap), thumb + cap_h + gap
        sheet = Image.new("RGB", (W, max(1, nrows) * Hc), (40, 40, 40))
        draw = ImageDraw.Draw(sheet)
        for i, (code, (o, r)) in enumerate(items):
            x, y = (i % per_row) * (2 * thumb + gap), (i // per_row) * Hc
            for side, p in enumerate(r["images"]):
                im = Image.open(os.path.join(data, o, p)).convert("RGB").resize((thumb, thumb))
                sheet.paste(im, (x + side * thumb, y))
            what = reserved.get(code) or (names.get(code, "") if actor == "chunli" else "")
            if 80 <= code <= 99:
                what = "special without box, class 0x%02X %s" % (code - 80, names.get(-code, "") if actor == "chunli"
                                                                  else "")
            draw.text((x + 2, y + thumb + 1), "%s act%02d stg%d  %s" % (actor, code, r["stg"], what[:40]),
                      fill=(255, 255, 0))
            draw.text((x + 2, y + thumb + 14), "vs %s g%d t%d %s%s" % (o, r["game"], r["t"], r["split"],
                                                                     "" if r["stg"] == 2 else " (no stg2 pair)"),
                      fill=(200, 200, 200))
        path = os.path.join(out, "contact_%s.png" % actor)
        sheet.save(path)
        paths.append(path)
    return paths
