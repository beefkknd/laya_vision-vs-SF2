"""The movement dataset (docs/prereg_movement.md; scripts/build_movement_data.py -> test_data_mv/<char>/).

One row per decision of test_data_u (scripts/build_u_data.py), asking sf2.data.movement's one question:
- the SAME pictures: <out>/<char>/frames is a symlink to test_data_u/<char>/frames (no copies), and every row's
  images are the u row's own (u_data.frame_names);
- the SAME note: v3, "me=<char>" (u_data.eye_note); rows carry "perception": true, as the U arm's do (train.py's
  coverage gate and checkpoint tags read them so);
- the SAME split per decision, read from test_data_u's files (train, test_real, test_heldout_guile; the val games
  are u's val.jsonl + val_rest.jsonl) and checked against u_data.split_of. The val games' decisions are capped to
  ``val_decisions`` per character by u_data.val_rank (the order u's own cap used, so u's kept decisions come first);
  the rest -> val_rest.jsonl (never trained on);
- the label from the decision's stored RAM rows (rollouts/u_perception, joined by sub, game and frame exactly as
  u_data joins them): movement.movement_at; "unknown" -> no row (masked, counted).

Class balance (the prereg: "each answer equally often" in training). train.jsonl holds, per character, every answer
the same number of times: each answer's decisions repeated in a fixed order (sha256 of the decision id) up to the
character's most common answer's count; the repeats are rows of their own ("copy" k >= 1, id suffix "~k"). With
train.py --balance sampling every character dir is drawn equally, so every answer is drawn equally overall. val,
val_rest and the test files stay natural (one row per decision).
"""
import collections
import hashlib
import json
import math
import os
from typing import Dict, List, Optional, Sequence, Tuple

from . import movement as M
from . import u_data as U
from .perception import UNKNOWN, decode

OUT = "test_data_mv"
VAL_DECISIONS = 200          # per character; >= train_data.MIN_SAMPLED_VAL (100) after masking
U_FILES = ("train", "val", U.VAL_REST, "test_real", "test_heldout_guile")
SPLIT_OF_FILE = {"train": "train", "val": "val", U.VAL_REST: "val", "test_real": "test",
                 "test_heldout_guile": "heldout"}
BALANCE_SALT = "mv_balance:"
NOTE_VERSION, FRAMES_VERSION = U.NOTE_VERSION, U.FRAMES_VERSION


# ---- test_data_u's decisions ----------------------------------------------------------------------------------------

def _decision_of(line: str) -> Optional[str]:
    """The decision id at the start of a u row (u_data writes "decision" first), without parsing the whole line."""
    head = '{"decision": "'
    if line.startswith(head):
        end = line.find('"', len(head))
        if end > 0:
            return line[len(head):end]
    return None


def u_decisions(u_dir: str) -> Tuple[Dict[str, Dict], List[str]]:
    """{decision: {decision, char, opp, sub, game, frame, images, u_file}} from a test_data_u/<char>/ dir (one entry
    per decision, from its first row), and the problems (a decision in two files)."""
    out: Dict[str, Dict] = {}
    problems: List[str] = []
    for name in U_FILES:
        path = os.path.join(u_dir, name + ".jsonl")
        if not os.path.exists(path):
            continue
        last = None
        with open(path) as f:
            for line in f:
                did = _decision_of(line)
                if did is not None and did == last:
                    continue
                r = json.loads(line)
                did = last = r["decision"]
                if did in out:
                    if out[did]["u_file"] != name:
                        problems.append("%s: decision %s in %s and %s" % (u_dir, did, out[did]["u_file"], name))
                    continue
                out[did] = {k: r[k] for k in ("decision", "char", "opp", "sub", "game", "frame", "images")}
                out[did]["u_file"] = name
    return out, problems


# ---- the collection --------------------------------------------------------------------------------------------------

def collection_index(root: str, char: str) -> Tuple[Dict[Tuple, Tuple[List[Dict], List[Dict]]], int]:
    """{(sub, game, frame): ([action entries], [ram records])} over the character's logs; and the action entries."""
    idx: Dict[Tuple, Tuple[List[Dict], List[Dict]]] = collections.defaultdict(lambda: ([], []))
    entries = 0
    for sub, c, d in U.find_logs(root):
        if c != char:
            continue
        acts, _ = U.read_jsonl(os.path.join(d, "actions.jsonl"))
        rams, _ = U.read_jsonl(os.path.join(d, "ram.jsonl"))
        entries += len(acts)
        for e in acts:
            idx[(sub, e["game"], e["frame"])][0].append(e)
        for r in rams:
            idx[(sub, r["game"], r["frame"])][1].append(r)
    return dict(idx), entries


# ---- rows --------------------------------------------------------------------------------------------------------------

def row_of(dec: Dict, answer: str, split: str) -> Dict:
    q = M.movement_question()
    char = dec["char"]
    return {"decision": dec["decision"], "char": char, "opp": dec["opp"], "sub": dec["sub"], "game": dec["game"],
            "frame": dec["frame"], "split": split, "images": list(dec["images"]), "state_text": U.eye_note(char),
            "perception": True, "note_version": NOTE_VERSION, "frames_version": FRAMES_VERSION,
            "id": "%s-%s" % (dec["decision"], M.KEY), "task": M.KEY, "key": M.KEY, "question": q,
            "label": list(q["criteria"]).index(answer), "answer": answer, "copy": 0}


def label_decision(dec: Dict, idx: Dict) -> Tuple[Optional[str], Optional[str]]:
    """(answer, problem) for one u decision: its one action entry and one ram record, joined as u_data joins them."""
    key = (dec["sub"], dec["game"], dec["frame"])
    acts, rams = idx.get(key, ([], []))
    where = "%s %s" % (dec["char"], dec["decision"])
    if len(acts) != 1 or len(rams) != 1:
        return None, "%s: %d action entries, %d ram records" % (where, len(acts), len(rams))
    bad = U.join_problem(acts[0], rams[0])
    if bad:
        return None, "%s: %s" % (where, bad)
    if dec["images"] != U.frame_names(dec["sub"], dec["game"], dec["frame"]):
        return None, "%s: images %s are not u_data's frame names" % (where, dec["images"])
    rows, n = decode(rams[0])
    return M.movement_at(rows, n), None


def balance_rank(decision: str) -> str:
    return hashlib.sha256((BALANCE_SALT + decision).encode()).hexdigest()


def balance(rows: List[Dict], char: str) -> Tuple[List[Dict], List[str]]:
    """Every answer repeated (in balance_rank order) up to the most common answer's count; (rows, problems: an
    answer with no training row cannot be balanced)."""
    by: Dict[str, List[Dict]] = {a: [] for a in M.ANSWERS}
    for r in rows:
        by[r["answer"]].append(r)
    missing = [a for a in M.ANSWERS if not by[a]]
    if missing:
        return [], ["%s train: no %s rows, the answers cannot be balanced" % (char, a) for a in missing]
    k = max(len(v) for v in by.values())
    out = []
    for a in M.ANSWERS:
        src = sorted(by[a], key=lambda r: balance_rank(r["decision"]))
        for i in range(k):
            r, copy = src[i % len(src)], i // len(src)
            out.append(r if copy == 0 else dict(r, id="%s~%d" % (r["id"], copy), copy=copy))
    return sorted(out, key=lambda r: r["id"]), []


def extra_problems(char: str, files: Dict[str, List[Dict]]) -> List[str]:
    """Beyond u_data.row_problems: every answer the criteria's own at its label, the one question, natural files
    without copies, a train balanced exactly."""
    out = []
    q = M.movement_question()
    for name, rows in sorted(files.items()):
        for r in rows:
            if r["question"] != q or list(q["criteria"])[r["label"]] != r["answer"]:
                out.append("%s %s: %s question / label mismatch" % (char, name, r["id"]))
            if name != "train" and r["copy"]:
                out.append("%s %s: %s is a repeat outside train" % (char, name, r["id"]))
    counts = collections.Counter(r["answer"] for r in files.get("train", []))
    if files.get("train") and len(set(counts[a] for a in M.ANSWERS)) != 1:
        out.append("%s train: answers not balanced %s" % (char, dict(counts)))
    return out


# ---- one character ---------------------------------------------------------------------------------------------------

def _counts(rows: List[Dict]) -> Dict[str, int]:
    c = collections.Counter(r["answer"] for r in rows if r["copy"] == 0)
    return {a: c[a] for a in M.ANSWERS if c[a]}


def _by_opp(rows: List[Dict]) -> Dict[str, Dict[str, int]]:
    opps = sorted({r["opp"] for r in rows})
    return {o: _counts([r for r in rows if r["opp"] == o]) for o in opps}


def char_build(job: Tuple[str, str, str, str, int]) -> Tuple[str, Dict, List[str]]:
    char, root, u_data, out, val_decisions = job
    u_dir, base = os.path.join(u_data, char), os.path.join(out, char)
    decs, problems = u_decisions(u_dir)
    idx, entries = collection_index(root, char)
    natural: Dict[str, List[Dict]] = collections.defaultdict(list)
    unknown = 0
    for did in sorted(decs):
        dec = decs[did]
        split = SPLIT_OF_FILE[dec["u_file"]]
        want = U.split_of(char, dec["opp"], dec["game"])
        if split != want:
            problems.append("%s %s: in %s.jsonl (split %s) but its game's split is %s" % (
                char, did, dec["u_file"], split, want))
            continue
        answer, bad = label_decision(dec, idx)
        if bad:
            problems.append(bad)
            continue
        if answer == UNKNOWN:
            unknown += 1
            continue
        natural["val" if split == "val" else U.FILES[split]].append(row_of(dec, answer, split))
    val_pool = sorted(natural.pop("val", []), key=lambda r: U.val_rank(r["decision"]))
    files: Dict[str, List[Dict]] = {"train": natural.get("train", []), "val": val_pool[:val_decisions],
                                    U.VAL_REST: val_pool[val_decisions:], "test_real": natural.get("test_real", [])}
    if char == U.HELDOUT[0]:
        files["test_heldout_guile"] = natural.get("test_heldout_guile", [])
    nat_train = files["train"]
    files["train"], p = balance(nat_train, char)
    problems += p
    files = {k: sorted(v, key=lambda r: r["id"]) for k, v in files.items()}
    os.makedirs(base)
    os.symlink(os.path.abspath(os.path.join(u_dir, "frames")), os.path.join(base, "frames"))
    problems += U.row_problems(char, files) + extra_problems(char, files) + U.image_problems(base, files)
    for name, rows in files.items():
        with open(os.path.join(base, name + ".jsonl"), "w") as f:
            f.write("".join(json.dumps(r) + "\n" for r in rows))
    in_u = {(d["sub"], d["game"], d["frame"]) for d in decs.values()}
    labelled = len(nat_train) + len(val_pool) + sum(len(v) for k, v in files.items() if k in U_FILES[3:])
    stats = {"decisions": len(decs), "labelled": labelled,
             "unknown": unknown, "collection_entries": entries,
             "collection_not_in_u": sum(1 for k, (a, _) in idx.items() if a and k not in in_u),
             "natural": {k: _counts(nat_train if k == "train" else v) for k, v in files.items()},
             "balanced_train": dict(collections.Counter(r["answer"] for r in files["train"])),
             "by_opponent": {k: _by_opp(nat_train if k == "train" else v) for k, v in files.items()},
             "val_cap": {"val_games_decisions": len(val_pool), "kept": len(files["val"])},
             "train_rows": len(files["train"]), "rows": {k: len(v) for k, v in files.items()}}
    with open(os.path.join(base, "stats.json"), "w") as f:
        json.dump(stats, f, indent=1, sort_keys=True)
    return char, stats, problems


# ---- build -------------------------------------------------------------------------------------------------------------

def build(root: str, u_data: str, out: str = OUT, val_decisions: int = VAL_DECISIONS,
          chars: Optional[Sequence[str]] = None, workers: int = 1) -> Dict:
    """Every character of ``u_data`` into ``out`` (which must not exist or be empty)."""
    if os.path.exists(out) and os.listdir(out):
        raise SystemExit("%s exists and is not empty: remove it first" % out)
    want = sorted(chars or (c for c in os.listdir(u_data) if os.path.isdir(os.path.join(u_data, c))))
    os.makedirs(out, exist_ok=True)
    jobs = [(c, root, u_data, out, val_decisions) for c in want]
    if workers > 1 and len(jobs) > 1:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(min(workers, len(jobs))) as ex:
            done = list(ex.map(char_build, jobs))
    else:
        done = [char_build(j) for j in jobs]
    problems = [p for _, _, ps in done for p in ps]
    counts = {c: s for c, s, _ in done}
    ub = os.path.join(u_data, "build.json")
    meta = {"root": root, "u_data": u_data, "u_build_sha256": U.sha256_file(ub) if os.path.exists(ub) else None,
            "val_decisions": val_decisions, "lag": M.LAG, "walk_px": M.WALK_PX, "question": M.movement_question(),
            "answers": list(M.ANSWERS), "balance": "train.jsonl per character: every answer repeated to the most "
            "common answer's count (balance_rank order); train with --balance sampling", "chars": want,
            "frames_version": FRAMES_VERSION, "note_version": NOTE_VERSION}
    with open(os.path.join(out, "build.json"), "w") as f:
        json.dump(dict(meta, problems=len(problems)), f, indent=1, sort_keys=True)
    return {"problems": problems, "counts": counts, "meta": meta}


def epochs_for_steps(steps: int, rows: int, batch: int = 8) -> str:
    """The --epochs (as a string to pass) that makes scripts/train.py run exactly ``steps``:
    max(1, int(epochs * rows / batch)) == steps."""
    e0 = steps * batch / rows
    for d in range(4, 16):
        e = math.ceil(e0 * 10 ** d) / 10 ** d
        s = ("%%.%df" % d) % e
        if max(1, int(float(s) * rows / batch)) == steps:
            return s
    raise ValueError("no --epochs gives %d steps on %d rows at batch %d" % (steps, rows, batch))
