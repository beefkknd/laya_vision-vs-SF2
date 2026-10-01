"""The U arm's dataset (docs/prereg_u_perception.md incl. its amendments; scripts/build_u_data.py -> test_data_u/<char>/).

One decision of the U collection (rollouts/u_perception/<sub>/<char>/: actions.jsonl + ram.jsonl joined by game and
frame, images/) becomes up to ~24 laya rows that share its two pictures and its note:

- the pictures: frames n-4 and n with the HUD VISIBLE (``hud_frame``: padded to 256x256 with black rows at the bottom,
  nothing blanked; frames v3). A player reads the health bars there. No mirroring: the HUD is not mirror-symmetric
  (her bar is always on the left), and live play covers both sides.
- the note: v3, ``eye_note`` = "me=<char>" only (a player knows her own character; nothing else, no RAM).
- the questions (``questions``): the perception questions 1-7 (PERCEPTION: Q1 band and trend, Q2 his phase, Q3 his
  air, Q4 projectile, Q5 me / him, Q6 corner, Q7 my bar / his bar), labelled from the stored RAM rows by
  sf2.data.perception.labels with a thresholds file given as an argument (v1 now, v2 swappable); and question 8, one
  question per move of choices(me) except forward, with the SOFT target of the decision's RAM cell
  (sf2.data.value_oracle._cell_of of the action entry: exactly the cell the T arm reads) from the q8 targets file.
  A question whose label is "unknown" (or a move without a target in that cell) gets no row: masked, never guessed.
- the split, by whole games: game % 10 in TEST_INDEX (2, 5, 8) -> test_real.jsonl; game % 10 in VAL_INDEX (0) ->
  val.jsonl (whole games, so validation never shares neighbouring frames with training); the rest -> train.jsonl.
  Chun-Li vs Guile -> test_heldout_guile.jsonl only.

Every question is a choice question whose criteria keys are the answers themselves (perception.QUESTIONS order;
question 8: perception.Q8_ANSWERS, text laya's own rating words), each with a short general description. The wording
is general (no opponent or move names of his) and byte-identical at train and play time: ask only through
``questions``.
"""
import collections
import hashlib
import json
import os
import shutil
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from ..config import IMAGE_SIZE
from ..vocab import FIGHTERS
from .perception import FORWARD, Q8_ANSWERS, QUESTIONS, UNKNOWN, _life, decode, labels
from .vs_sweep import TEST_INDEX

OUT = "test_data_u"
U_ROOT = os.path.join("rollouts", "u_perception")
FRAMES_VERSION = 3          # 1-2: model_frame (HUD blanked, mirrored training rows); 3: hud_frame (HUD visible)
NOTE_VERSION = 3            # 1: runs/all8's RAM note; 2: + opp_attacking; 3: "me=<char>" only
VAL_INDEX = (0,)
HELDOUT = ("chunli", "guile")
FILES = {"train": "train", "val": "val", "test": "test_real", "heldout": "test_heldout_guile"}
Q8_KEY = "q8:"
VAL_REST = "val_rest"         # the val games' decisions past --val-decisions (never trained on; eval reference)
VAL_SALT = "u_val_cap:"
CONFIDENT = 0.9             # a q8 target is "confident" when one answer has >= 0.9 (calibrate_perception's count)

_SAME = {"full": "full or nearly full", "high": "more than half", "half": "about half", "low": "less than a third"}
_ACT = {"free": "free to move", "stunned": "reeling from a hit", "knocked down": "knocked down",
        "dizzy": "dizzy, stars over the head"}
PERCEPTION: Dict[str, Tuple[str, Dict[str, str]]] = {
    "range": ("How far away is he?", {"throw": "close enough to grab", "poke": "in reach of a ground attack",
                                      "mid": "a few steps away", "far": "far away"}),
    "trend": ("Is the gap between us changing?", {"closing": "getting smaller", "steady": "about the same",
                                                  "opening": "getting bigger"}),
    "phase": ("What is he doing?", {"neutral": "moving or waiting, not attacking", "attacking": "attacking",
                                    "recovering after a miss": "recovering after an attack that missed",
                                    "blocking": "blocking", "being hit": "being hit"}),
    "air": ("Is he in the air?", {"grounded": "on the ground", "jumping at me": "jumping toward me",
                                  "jumping away or straight up": "jumping away or straight up",
                                  "landing": "about to land"}),
    "projectile": ("Is a projectile coming at me?", {"none": "no", "far": "yes, still far", "near": "yes, close"}),
    "me_can_act": ("Can I act now?", dict(_ACT)),
    "him_can_act": ("Can he act now?", dict(_ACT)),
    "corner": ("Is one of us in a corner?", {"me": "I am", "him": "he is", "neither": "neither"}),
    "my_bar": ("How full is my health bar?", dict(_SAME)),
    "his_bar": ("How full is his health bar?", dict(_SAME)),
}
Q8_CRITERIA = {"likely works": "likely better than walking in", "may work": "maybe better than walking in",
               "likely fails": "not better than walking in"}


# ---- pictures and note ----------------------------------------------------------------------------------------

def hud_frame(img: np.ndarray) -> np.ndarray:
    """Frames v3: the screen padded with black rows at the bottom to 256x256, the HUD left as it is (a 256x256
    frame passes unchanged, so a stored training frame can be fed again)."""
    h, w = img.shape[:2]
    if img.ndim != 3 or img.shape[2] != 3 or h > IMAGE_SIZE or w != IMAGE_SIZE:
        raise ValueError("expected a %d-wide RGB screen at most %d tall, got %s" % (IMAGE_SIZE, IMAGE_SIZE, img.shape))
    out = np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3), np.uint8)
    out[:h] = img
    return out


def eye_note(me: str) -> str:
    if me not in FIGHTERS:
        raise ValueError("unknown character %r" % me)
    return "me=%s" % me


# ---- questions ----------------------------------------------------------------------------------------------------

def perception_question(key: str) -> Dict:
    ins, crit = PERCEPTION[key]
    if tuple(crit) != QUESTIONS[key]:
        raise AssertionError("question %s answers %s != perception.QUESTIONS %s" % (key, tuple(crit), QUESTIONS[key]))
    return {"type": "choice", "instructions": ins, "criteria": dict(crit)}


def q8_question(move: str) -> Dict:
    if move == FORWARD:
        raise ValueError("walking in is the baseline: question 8 is not asked for it")
    return {"type": "choice", "instructions": "If I do %s now, is it better than walking in?" % move,
            "criteria": dict(Q8_CRITERIA)}


def q8_moves(me: str) -> List[str]:
    """The moves question 8 is asked for: every choice of System 1 except walking in (choices order)."""
    from ..system1.system1 import choices

    return [m for m in choices(me) if m != FORWARD]


def questions(me: str) -> Dict[str, Dict]:
    """Every question of one decision, keyed as the rows and the eye key them: the perception questions, then
    "q8:<move>" per move."""
    out = {k: perception_question(k) for k in PERCEPTION}
    out.update({Q8_KEY + m: q8_question(m) for m in q8_moves(me)})
    return out


# ---- split and targets ------------------------------------------------------------------------------------------

def split_of(me: str, opp: str, game: int) -> str:
    if (me, opp) == HELDOUT:
        return "heldout"
    if game % 10 in TEST_INDEX:
        return "test"
    return "val" if game % 10 in VAL_INDEX else "train"


def load_targets(path: str) -> Dict[Tuple, Dict[str, Dict[str, float]]]:
    """lessons/perception_q8_targets_v*.json -> {cell: {move: {answer: p}}}; answers must be Q8_ANSWERS."""
    with open(path) as f:
        doc = json.load(f)
    if tuple(doc.get("answers", ())) != Q8_ANSWERS:
        raise ValueError("%s: answers %r != %r" % (path, doc.get("answers"), Q8_ANSWERS))
    out: Dict[Tuple, Dict[str, Dict[str, float]]] = {}
    for t in doc["targets"]:
        p = t["p"]
        if sorted(p) != sorted(Q8_ANSWERS) or abs(sum(p.values()) - 1.0) > 1e-6 or min(p.values()) < 0:
            raise ValueError("%s: bad target %r for %r %r" % (path, p, t["cell"], t["move"]))
        out.setdefault(tuple(t["cell"]), {})[t["move"]] = dict(p)
    return out


def cell_of(entry: Dict) -> Tuple:
    """The decision's RAM cell, as the table and the T arm read it (sf2.data.value_oracle._cell_of)."""
    from .value_oracle import _cell_of

    return _cell_of(entry)


def argmax(p: Dict[str, float], order: Sequence[str]) -> str:
    """The most likely answer; a tie goes to the first in ``order``."""
    return max(order, key=lambda a: (p.get(a, 0.0), -list(order).index(a)))


# ---- one decision ------------------------------------------------------------------------------------------------

def decision_id(me: str, sub: str, game: int, frame: int) -> str:
    return "%s-u_%s_g%03d_f%05d" % (me, sub, game, frame)


def frame_names(sub: str, game: int, frame: int) -> List[str]:
    base = "%s_g%03d_f%05d" % (sub, game, frame)
    return ["frames/%s_prev.png" % base, "frames/%s_now.png" % base]


def decision_rows(me: str, sub: str, entry: Dict, ram: Dict, th: Dict, targets: Dict) -> Tuple[List[Dict], Dict]:
    """One decision's rows (masked questions left out) and its label record (every question's answer, "unknown"
    included, and the cell), for the stats. ``ram``: its ram.jsonl record; ``th``: the thresholds."""
    rows, n = decode(ram)
    lab = labels(rows, n, th)
    cell = cell_of(entry)
    did = decision_id(me, sub, entry["game"], entry["frame"])
    common = {"decision": did, "char": me, "opp": entry["opp"], "sub": sub, "game": entry["game"],
              "frame": entry["frame"], "split": split_of(me, entry["opp"], entry["game"]),
              "images": frame_names(sub, entry["game"], entry["frame"]), "state_text": eye_note(me),
              "perception": True, "note_version": NOTE_VERSION, "frames_version": FRAMES_VERSION,
              "cell": list(cell), "played": entry["action"], "explored": entry.get("explored")}
    out = []
    for key in PERCEPTION:
        if lab[key] == UNKNOWN:
            continue
        q = perception_question(key)
        out.append(dict(common, id="%s-%s" % (did, key), task="perception", key=key, question=q,
                        label=list(q["criteria"]).index(lab[key]), answer=lab[key]))
    cell_targets = targets.get(tuple(cell), {})
    q8 = {}
    for m in q8_moves(me):
        p = cell_targets.get(m)
        if p is None:
            continue
        q = q8_question(m)
        word = argmax(p, Q8_ANSWERS)
        q8[m] = word
        out.append(dict(common, id="%s-%s%s" % (did, Q8_KEY, m), task="q8", key=Q8_KEY + m, question=q,
                        label=Q8_ANSWERS.index(word), answer=word, target=[float(p[a]) for a in Q8_ANSWERS],
                        confident=max(p.values()) >= CONFIDENT))
    return out, {"labels": lab, "q8": q8, "cell": cell}


def join_problem(entry: Dict, ram: Dict) -> Optional[str]:
    """The action entry and the ram record must be the same decision: game, frame, and the decision row's gap and
    lives as the entry logged them (sf2.system1.game_log.action_entry reads the same row; a KO wraps the life byte
    to ~255, which the entry logs as 0)."""
    rows, n = decode(ram)
    r = rows[n]
    got = (ram["game"], ram["frame"], abs(r["p2_x"] - r["p1_x"]), _life(r["p1_life"]), _life(r["p2_life"]))
    want = (entry["game"], entry["frame"], entry["gap"], entry["my_life"], entry["opp_life"])
    if got != want:
        return "g%d f%d: ram (game, frame, gap, my life, his life) %s != action entry %s" % (
            entry["game"], entry["frame"], got, want)
    return None


# ---- reading the collection ---------------------------------------------------------------------------------------

def read_jsonl(path: str) -> Tuple[List[Dict], int]:
    """Complete lines of a file a collection may still be writing; (records, 1 if a last partial line was skipped)."""
    with open(path) as f:
        text = f.read()
    lines = text.split("\n")
    partial = 1 if lines[-1].strip() else 0
    return [json.loads(x) for x in lines[:-1] if x.strip()], partial


def find_logs(root: str) -> List[Tuple[str, str, str]]:
    """(sub, char, dir) of every <root>/<sub>/<char>/ holding actions.jsonl and ram.jsonl."""
    out = []
    for sub in sorted(os.listdir(root)) if os.path.isdir(root) else []:
        for char in sorted(os.listdir(os.path.join(root, sub))) if os.path.isdir(os.path.join(root, sub)) else []:
            d = os.path.join(root, sub, char)
            if char in FIGHTERS and all(os.path.isfile(os.path.join(d, f)) for f in ("actions.jsonl", "ram.jsonl")):
                out.append((sub, char, d))
    return out


Decision = Tuple[List[Dict], Dict, List[Tuple[str, str]], Dict]   # (rows, label record, [(dst, raw)], action entry)


def log_decisions(sub: str, char: str, d: str, th: Dict, targets: Dict) -> Tuple[List[Decision], List[str], Dict]:
    """One log's decisions: every action entry joined with its ram record by (game, frame); an entry or a record
    without its partner is a problem (a partial last line of a running collection is counted, not a problem)."""
    acts, pa = read_jsonl(os.path.join(d, "actions.jsonl"))
    rams, pr = read_jsonl(os.path.join(d, "ram.jsonl"))
    by_key = collections.defaultdict(list)
    for rec in rams:
        by_key[(rec["game"], rec["frame"])].append(rec)
    problems: List[str] = []
    out: List[Decision] = []
    seen = set()
    for e in acts:
        key = (e["game"], e["frame"])
        hits = by_key.get(key, [])
        where = "%s/%s" % (sub, char)
        if e.get("me") != char:
            problems.append("%s g%d f%d: logged me=%r" % (where, key[0], key[1], e.get("me")))
            continue
        if len(hits) != 1:
            problems.append("%s g%d f%d: %d ram records" % (where, key[0], key[1], len(hits)))
            continue
        bad = join_problem(e, hits[0])
        if bad:
            problems.append("%s %s" % (where, bad))
            continue
        seen.add(key)
        rows, rec = decision_rows(char, sub, e, hits[0], th, targets)
        raws = [os.path.join(d, "images", p) for p in (e.get("images") or [])]
        if len(raws) != 2:
            problems.append("%s g%d f%d: %d images logged" % (where, key[0], key[1], len(raws)))
            continue
        out.append((rows, rec, list(zip(frame_names(sub, e["game"], e["frame"]), raws)), e))
    orphans = sorted(set(by_key) - seen - {(e["game"], e["frame"]) for e in acts})
    problems += ["%s/%s g%d f%d: ram record without an action entry" % (sub, char, g, f) for g, f in orphans]
    return out, problems, {"partial_lines": pa + pr, "entries": len(acts), "ram_records": len(rams)}


# ---- frames (parallel) ------------------------------------------------------------------------------------------

def _write_one(job: Tuple[str, str]) -> Optional[str]:
    from .build import _load
    from .dataset import save_png

    dst, raw = job
    if not os.path.exists(raw):
        return "missing raw image %s (for %s)" % (raw, dst)
    try:
        save_png(hud_frame(_load(raw)), dst)
    except (OSError, ValueError) as e:
        return "%s: %s" % (raw, e)
    return None


def write_frames(base: str, jobs: Iterable[Tuple[str, str]], workers: int = 1) -> List[str]:
    """frames/<x>.png = hud_frame(raw), over ``workers`` processes (frames are independent)."""
    os.makedirs(os.path.join(base, "frames"), exist_ok=True)
    todo = sorted({(os.path.join(base, dst), raw) for dst, raw in jobs})
    if workers <= 1 or len(todo) < 64:
        res = [_write_one(j) for j in todo]
    else:
        from concurrent.futures import ProcessPoolExecutor

        with ProcessPoolExecutor(workers) as ex:
            res = list(ex.map(_write_one, todo, chunksize=64))
    return [r for r in res if r]


# ---- stats and gates ------------------------------------------------------------------------------------------------

def row_problems(char: str, files: Dict[str, List[Dict]]) -> List[str]:
    """Unique ids; every note exactly "me=<char>"; no Chun-Li vs Guile row outside its file; no game in two files;
    every label in range and every q8 target a distribution over the three answers."""
    problems = []
    ids = collections.Counter(r["id"] for rows in files.values() for r in rows)
    problems += ["%s: duplicate id %s (x%d)" % (char, i, n) for i, n in sorted(ids.items()) if n > 1]
    games = collections.defaultdict(set)
    for name, rows in sorted(files.items()):
        for r in rows:
            if r["state_text"] != eye_note(char):
                problems.append("%s %s: %s note is not v3: %r" % (char, name, r["id"], r["state_text"]))
            if (char, r["opp"]) == HELDOUT and name != FILES["heldout"]:
                problems.append("%s %s: %s is vs %s (held out)" % (char, name, r["id"], HELDOUT[1]))
            if not 0 <= r["label"] < len(r["question"]["criteria"]):
                problems.append("%s %s: %s label %r out of range" % (char, name, r["id"], r["label"]))
            if "target" in r and (len(r["target"]) != 3 or abs(sum(r["target"]) - 1) > 1e-6):
                problems.append("%s %s: %s bad target %r" % (char, name, r["id"], r["target"]))
            games[(r["sub"], r["game"])].add(FILES["val"] if name == VAL_REST else name)
    problems += ["%s: game %s g%d in %s" % (char, s, g, sorted(n)) for (s, g), n in sorted(games.items()) if len(n) > 1]
    return problems


def image_problems(base: str, files: Dict[str, List[Dict]]) -> List[str]:
    paths = sorted({p for rows in files.values() for r in rows for p in r["images"]})
    return ["%s: missing image %s" % (base, p) for p in paths if not os.path.exists(os.path.join(base, p))]


def question_stats(rows: List[Dict]) -> Dict[str, Dict]:
    """Per question key (q8 pooled as "q8"): rows and the label distribution."""
    out: Dict[str, Dict] = {}
    for r in rows:
        k = "q8" if r["task"] == "q8" else r["key"]
        s = out.setdefault(k, {"rows": 0, "labels": collections.Counter()})
        s["rows"] += 1
        s["labels"][r["answer"]] += 1
    return {k: {"rows": v["rows"], "labels": dict(sorted(v["labels"].items()))} for k, v in sorted(out.items())}


def unknown_stats(recs: List[Dict]) -> Dict[str, int]:
    """Per perception question: decisions whose label is "unknown" (masked)."""
    return {k: sum(r["labels"][k] == UNKNOWN for r in recs) for k in PERCEPTION}


def mapping_agreement(recs: List[Dict], entries: List[Dict]) -> Dict:
    """How often the eye's mapping of the RAM-true answers to text laya's words (sf2.system1.eye.situation_of) equals
    what text laya reads from RAM on the T and A arms (sf2.system1.system1.situation: range_of(gap), opp_doing at
    the decision row). Disagreement is the price of the 1-frame lag, the band edges and the vocabularies."""
    from ..system1.advice import opp_doing
    from ..system1.eye import doing_of, range_word
    from ..vocab import range_of

    rng = doing = n = 0
    for rec, e in zip(recs, entries):
        lab = rec["labels"]
        if UNKNOWN in (lab["range"], lab["phase"], lab["air"], lab["him_can_act"]):
            continue
        n += 1
        rng += range_word(lab["range"]) == range_of(e["gap"])
        doing += doing_of(lab["phase"], lab["air"], lab["him_can_act"]) == opp_doing(e)
    return {"decisions": n, "range": rng / n if n else None, "doing": doing / n if n else None}


def training_steps(train_rows: int, epochs: float = 2.0, batch: int = 8) -> int:
    """scripts/train.py's step count: epochs * rows / batch, at least 1."""
    return max(1, int(epochs * train_rows / batch))


def step_budget(hours: float, steps_per_min: float) -> int:
    """A fixed step budget for ``hours`` of training at a measured speed (evaluations included in the speed)."""
    return int(hours * 60 * steps_per_min)


# ---- build -----------------------------------------------------------------------------------------------------------

def sha256_file(path: str) -> str:
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def val_rank(decision: str) -> str:
    """The order the validation cap keeps decisions in: the sha256 of the decision id (no random state)."""
    return hashlib.sha256((VAL_SALT + decision).encode()).hexdigest()


def cap_val(rows: List[Dict], n: Optional[int]) -> Tuple[List[Dict], List[Dict], Optional[Dict]]:
    """(kept, rest, report): the ``n`` decisions of the val games with the smallest val_rank, every question of
    each (a decision is never split); ``n`` None keeps everything (the default)."""
    if n is None:
        return rows, [], None
    decisions = sorted({r["decision"] for r in rows}, key=val_rank)
    keep = set(decisions[:max(0, n)])
    return ([r for r in rows if r["decision"] in keep], [r for r in rows if r["decision"] not in keep],
            {"decisions": len(decisions), "kept": len(keep), "rest": len(decisions) - len(keep)})


def char_build(char: str, logs: List[Tuple[str, str, str]], th: Dict, targets: Dict, out: str,
               workers: int = 1, val_decisions: Optional[int] = None) -> Tuple[Dict, List[str]]:
    files: Dict[str, List[Dict]] = {v: [] for v in FILES.values() if v != FILES["heldout"] or char == HELDOUT[0]}
    problems: List[str] = []
    recs: List[Dict] = []
    jobs: List[Tuple[str, str]] = []
    counts = {"logs": [], "partial_lines": 0, "entries": 0, "ram_records": 0, "decisions": 0,
              "decisions_by_split": collections.Counter()}
    base = os.path.join(out, char)
    entries: List[Dict] = []
    for sub, _, d in logs:
        decs, p, c = log_decisions(sub, char, d, th, targets)
        problems += p
        counts["logs"].append(sub)
        for k in ("partial_lines", "entries", "ram_records"):
            counts[k] += c[k]
        for rows, rec, js, e in decs:
            if not rows:
                continue
            split = rows[0]["split"]
            files[FILES[split]] += rows
            recs.append(rec)
            entries.append(e)
            jobs += js
            counts["decisions"] += 1
            counts["decisions_by_split"][split] += 1
    files["val"], rest, val_cap = cap_val(files["val"], val_decisions)
    if val_cap is not None:
        files[VAL_REST] = rest
    files = {k: sorted(v, key=lambda r: r["id"]) for k, v in files.items()}
    problems += write_frames(base, jobs, workers)
    problems += row_problems(char, files) + image_problems(base, files)
    for name, rows in files.items():
        with open(os.path.join(base, name + ".jsonl"), "w") as f:
            f.write("".join(json.dumps(r) + "\n" for r in rows))
    stats = dict(counts, decisions_by_split=dict(counts["decisions_by_split"]),
                 files={k: {"rows": len(v), "questions": question_stats(v)} for k, v in sorted(files.items())},
                 unknown=unknown_stats(recs), **({"val_cap": val_cap} if val_cap is not None else {}), mapping=mapping_agreement(recs, entries),
                 rows_per_decision=(sum(len(v) for v in files.values()) / counts["decisions"]
                                    if counts["decisions"] else None))
    with open(os.path.join(base, "stats.json"), "w") as f:
        json.dump(stats, f, indent=1, sort_keys=True)
    return stats, problems


def build(root: str = U_ROOT, out: str = OUT, thresholds: str = "", q8_targets: str = "", workers: int = 1,
          chars: Optional[Sequence[str]] = None, val_decisions: Optional[int] = None) -> Dict:
    """Every character's dir in ``out`` (which must not exist or be empty). Returns {"problems", "counts", "meta"}."""
    if os.path.exists(out) and os.listdir(out):
        return {"problems": ["%s exists: remove it first (or --overwrite)" % out], "counts": {}, "meta": {}}
    with open(thresholds) as f:
        th = json.load(f)
    targets = load_targets(q8_targets)
    found = find_logs(root)
    want = sorted({c for _, c, _ in found} if chars is None else set(chars))
    meta = {"root": root, "thresholds": thresholds, "thresholds_sha256": sha256_file(thresholds),
            "q8_targets": q8_targets, "q8_targets_sha256": sha256_file(q8_targets), "frames_version": FRAMES_VERSION,
            "note_version": NOTE_VERSION, "val_decisions": val_decisions, "logs": [[s, c] for s, c, _ in found if c in want]}
    problems, counts = [], {}
    for char in want:
        os.makedirs(os.path.join(out, char), exist_ok=True)
        counts[char], p = char_build(char, [x for x in found if x[1] == char], th, targets, out, workers,
                                     val_decisions)
        problems += p
    with open(os.path.join(out, "build.json"), "w") as f:
        json.dump(dict(meta, problems=len(problems)), f, indent=1, sort_keys=True)
    return {"problems": problems, "counts": counts, "meta": meta}


def overwrite(out: str) -> None:
    """Remove a previous build of ``out`` (only one that looks like ours: a build.json and character dirs)."""
    if not os.path.exists(out):
        return
    for d in os.listdir(out):
        if d == "build.json":
            continue
        if d not in FIGHTERS or not os.path.exists(os.path.join(out, d, "stats.json")):
            raise SystemExit("%s holds %s, not a U-data build: not removing it" % (out, d))
    shutil.rmtree(out)
