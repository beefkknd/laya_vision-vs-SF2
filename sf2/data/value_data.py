"""The value fine-tune's dataset (docs/prereg_lv_value.md; scripts/build_value_data.py -> test_data_v2/<char>/):

- the all8 rows (test_data/<char>/train.jsonl, test_real_{left,right}.jsonl) with note v2: the still dummy is never
  attacking (opp_attacking=0); a block row's decision frame is 3 frames into a ground probe (his state 0x0A: 1) or the
  jump-in's descent before the kick (his state is the jump 0x04: 0); a live row takes the opponent's state logged at
  that decision (rollouts/<log>/<char>/actions.jsonl, joined by game + frame from the row id: a row that does not join
  is a problem, never guessed). Their images stay in test_data, by a path relative to test_data_v2/<char>/.
- the new live rows from the collection logs (rollouts/lv_value/<sub>/<char>/): as scripts/vs_dataset.py import-live
  and build do it (model frames, mirrored training rows, dx == 0 dropped, game % 10 in TEST_INDEX held out), with
  no cap. Chun-Li vs Guile goes to test_heldout_guile.jsonl only (real rows), never to train or test_real.
- next to every live row (old and new, mirrored too) its value row: the same picture and note, the value question,
  the label value_bucket(dealt - taken). A still dummy never punishes: no value rows for the other rows.
"""
import collections
import json
import os
import re
import shutil
from typing import Callable, Dict, List, Optional, Tuple

from .dataset import read, save_png, write_jsonl
from .frames import mirror_frame, model_frame
from .value import VALUE_BUCKETS, value_bucket, value_question
from .vs_sweep import OUTCOMES, TEST_INDEX, mirror_record, outcome_question
from ..vocab import FIGHTERS

OUT = "test_data_v2"
LV_ROOT = os.path.join("rollouts", "lv_value")
HELDOUT_CHAR, HELDOUT_OPP = "chunli", "guile"
HELDOUT_FILE = "test_heldout_guile"
MIN_VALUE_TRAIN = 20            # new real value training rows per move of System 1's choices
OLD_FILES = ("train", "test_real_left", "test_real_right")
# his state at a block row's decision frame (sf2.data.vs_defense.decision_frame): ground probes are 3 frames into the
# attack (0x0A); the jump-in is decided as he comes down, before the kick is pressed (still the jump, 0x04)
DEFENSE_ATTACKING = {"s.hk": 1, "c.mk": 1, "sweep": 1, "jump_in": 0}
LIVE_ID = re.compile(r"^(?P<char>[a-z]+)-live_(?P<log>.+)_g(?P<game>\d+)_f(?P<frame>\d+)-(?P<action>[a-z0-9_.]+)"
                     r"(?P<m>-m)?$")
V1_NOTE = re.compile(r"^me=\S+ dist=\S+ side=(left|right) dx=[+-]\d+ my_bar=\S+ opp_bar=\S+ opp_airborne=[01] "
                     r"opp_crouch=[01]$")
V2_TAIL = re.compile(r" opp_attacking=[01]$")


def with_attacking(text: str, flag: int) -> str:
    """A v1 note as note v2 (sf2.data.vs_sweep.note version 2): one trailing opp_attacking field."""
    if "opp_attacking=" in text or "\n" in text:
        raise ValueError("not a plain v1 note: %r" % text)
    return "%s opp_attacking=%d" % (text, int(flag))


def attacking(state: str) -> int:
    """opp_attacking from a game log's named opponent state (sf2.system1.game_log.STATE: 0x0A attack, 0x0C special)."""
    return int(state in ("attack", "special"))


def parse_live_id(rid: str) -> Optional[Dict]:
    """A live row's id (sf2.data.collect.import_live; a mirror adds -m) -> its log, game, frame and action."""
    m = LIVE_ID.match(rid)
    if not m:
        return None
    return {"char": m["char"], "log": m["log"], "game": int(m["game"]), "frame": int(m["frame"]),
            "action": m["action"], "mirrored": bool(m["m"])}


def split_of_game(game: int) -> str:
    return "test" if game % 10 in TEST_INDEX else "train"


def value_row(r: Dict) -> Dict:
    """The value row of a live outcome row: same picture and note, the value question, the net bucket."""
    net = r["damage"] - r["damage_taken"]
    return dict(r, id=r["id"] + "-value", task="value", question=value_question(r["action"]),
                label=list(VALUE_BUCKETS).index(value_bucket(net)))


def read_log(path: str) -> Tuple[List[Dict], int]:
    """A game log's entries; a last line with no newline is still being written (collection runs): skipped, counted."""
    with open(path) as f:
        text = f.read()
    lines = text.split("\n")
    partial = 1 if lines[-1].strip() else 0
    return [json.loads(line) for line in lines[:-1] if line.strip()], partial


# --- the all8 rows ---

def _log_index(path: str) -> Dict[Tuple[int, int], List[Dict]]:
    idx: Dict[Tuple[int, int], List[Dict]] = collections.defaultdict(list)
    for a in (read_log(path)[0] if os.path.exists(path) else []):
        idx[(a["game"], a["frame"])].append(a)
    return idx


def _join_live(r: Dict, repo: str, logs: Callable[[str], Dict]) -> Tuple[Optional[int], Optional[str]]:
    """opp_attacking of an old live row from its game log, and every check that the join is the right decision."""
    p = parse_live_id(r["id"])
    src = r.get("source", "")
    if not p or not src.startswith("live:") or p["char"] != r["char"]:
        return None, "%s: not a live id / source %r" % (r["id"], src)
    log_dir = src[len("live:"):]
    if os.path.basename(os.path.normpath(log_dir)) != p["log"]:
        return None, "%s: id log %s != source %s" % (r["id"], p["log"], src)
    path = os.path.join(repo, log_dir, r["char"], "actions.jsonl")
    hits = logs(path).get((p["game"], p["frame"]), [])
    if len(hits) != 1:
        return None, "%s: %d log entries at game %d frame %d in %s" % (r["id"], len(hits), p["game"], p["frame"], path)
    a = hits[0]
    got = (a["action"], a["dealt"], a["taken"], a["game"])
    want = (p["action"], r["damage"], r["damage_taken"], r.get("game"))
    if got != want or r["action"] != p["action"]:
        return None, "%s: log (action, dealt, taken, game) %s != row %s" % (r["id"], got, want)
    return attacking(a["opp_state"]), None


def _old_flag(r: Dict, repo: str, logs: Callable[[str], Dict]) -> Tuple[Optional[int], Optional[str]]:
    kind = r.get("kind")
    if kind in ("attack", "movement"):
        return 0, None
    if kind == "defense":
        if r.get("probe") not in DEFENSE_ATTACKING:
            return None, "%s: unknown probe %r" % (r["id"], r.get("probe"))
        return DEFENSE_ATTACKING[r["probe"]], None
    if kind == "live":
        return _join_live(r, repo, logs)
    return None, "%s: unknown kind %r" % (r["id"], kind)


def convert_old(rows: List[Dict], rel: str, repo: str, logs: Callable[[str], Dict]) -> Tuple[List[Dict], List[str]]:
    """The all8 rows in note v2, images by ``rel`` (test_data/<char> seen from test_data_v2/<char>), plus a value
    row per live row. Rows that cannot be converted are dropped and named."""
    out, problems = [], []
    for r in rows:
        flag, problem = _old_flag(r, repo, logs)
        if not problem and "opp_attacking=" in r["state_text"]:
            problem = "%s: already a v2 note: %r" % (r["id"], r["state_text"])
        if problem:
            problems.append(problem)
            continue
        new = dict(r, state_text=with_attacking(r["state_text"], flag),
                   images=[rel + "/" + p for p in r["images"]])
        out += [new, value_row(new)] if r["kind"] == "live" else [new]
    return out, problems


# --- the new live rows ---

def new_record(char: str, sub: str, a: Dict) -> Dict:
    """One logged decision (sf2.system1.game_log.action_entry) as a real laya row, like import_live's."""
    key = "live_lv_%s_g%02d_f%05d" % (sub, a["game"], a["frame"])
    held = char == HELDOUT_CHAR and a["opp"] == HELDOUT_OPP
    return {
        "id": "%s-%s-%s" % (char, key, a["action"]), "char": char, "opp": a["opp"], "side": a["side"],
        "facing": "right" if a["side"] == "left" else "left", "range": a["range"], "gap": a["gap"],
        "gap_index": a["game"] % 10, "game": a["game"], "frame": a["frame"],
        "dx": a["gap"] if a["side"] == "left" else -a["gap"], "posture": "live", "action": a["action"],
        "kind": "live", "move_kind": a["kind"], "buttons": [],
        "images": ["frames/%s_prev.png" % key, "frames/%s_now.png" % key],
        "state_text": with_attacking(a["prompt"], attacking(a["opp_state"])),
        "split": "heldout_" + HELDOUT_OPP if held else split_of_game(a["game"]), "mirrored": False,
        "source": "live:" + os.path.join(LV_ROOT, sub), "boot": None, "outcome": a["actual"],
        "damage": a["dealt"], "damage_taken": a["taken"], "dealt": a["dealt"], "taken": a["taken"],
        "opp_state": a["opp_state"], "explored": a.get("explored"), "thrown": False, "executed": True,
        "attacked": a["kind"] == "attack", "busy_frames": a["frames"], "travel": 0, "my_life": a["my_life"],
        "opp_life": a["opp_life"], "opp_air": a["opp_air"]}


def _entry_problem(char: str, a: Dict) -> Optional[str]:
    where = "%s g%d f%d" % (char, a.get("game", -1), a.get("frame", -1))
    prompt = a.get("prompt", "")
    if "\n" in prompt:
        return "%s: prompt has more than the note (memory?): %r" % (where, prompt[:120])
    if not V1_NOTE.match(prompt):
        return "%s: prompt is not a v1 note: %r" % (where, prompt)
    if a.get("me") != char:
        return "%s: logged me=%r" % (where, a.get("me"))
    dx = a["gap"] if a["side"] == "left" else -a["gap"]
    if "side=%s dx=%+d " % (a["side"], dx) not in prompt:
        return "%s: prompt side/dx disagree with side %s gap %d" % (where, a["side"], a["gap"])
    if a["actual"] not in OUTCOMES:
        return "%s: unknown outcome %r" % (where, a["actual"])
    return None


def laya_row(r: Dict) -> Dict:
    return dict(r, question=outcome_question(r["action"]), label=OUTCOMES.index(r["outcome"]))


def mirror_row(r: Dict) -> Dict:
    m = mirror_record(r)
    return dict(m, images=[p.replace("frames/", "frames/mirror_") for p in r["images"]])


def new_rows(char: str, sub: str, log_dir: str, entries: List[Dict]) -> Tuple[Dict[str, List[Dict]], List, List, int]:
    """One log's decisions -> rows per output file, frame jobs (dst, raw source, mirrored), problems, dx==0 drops."""
    files: Dict[str, List[Dict]] = collections.defaultdict(list)
    jobs, problems, dropped = [], [], 0
    for a in entries:
        if a["gap"] == 0:
            dropped += 1        # the fighters at the same x: no left or right, a mirror would contradict it
            continue
        problem = _entry_problem(char, a)
        if problem:
            problems.append(problem)
            continue
        r = laya_row(new_record(char, sub, a))
        raws = [os.path.join(log_dir, "images", p) for p in a["images"]]
        jobs += [(dst, raw, False) for dst, raw in zip(r["images"], raws)]
        if r["split"] == "train":
            m = mirror_row(r)
            jobs += [(dst, raw, True) for dst, raw in zip(m["images"], raws)]
            files["train"] += [r, value_row(r), m, value_row(m)]
        elif r["split"] == "test":
            files["test_real_" + r["side"]] += [r, value_row(r)]
        else:
            files[HELDOUT_FILE] += [r, value_row(r)]
    return dict(files), jobs, problems, dropped


def find_logs(lv_root: str) -> List[Tuple[str, str, str]]:
    """(sub, char, dir) of every rollouts/lv_value/<sub>/<char>/actions.jsonl there is (collection may be partial)."""
    if not os.path.isdir(lv_root):
        return []
    out = []
    for sub in sorted(os.listdir(lv_root)):
        for char in (sorted(os.listdir(os.path.join(lv_root, sub))) if os.path.isdir(os.path.join(lv_root, sub))
                     else []):
            d = os.path.join(lv_root, sub, char)
            if char in FIGHTERS and os.path.isfile(os.path.join(d, "actions.jsonl")):
                out.append((sub, char, d))
    return out


def write_frames(base: str, jobs: List[Tuple[str, str, bool]]) -> List[str]:
    """frames/<x>.png = model_frame(raw), frames/mirror_<x>.png = mirror_frame(raw) (as sf2.data.build does)."""
    from .build import _load

    os.makedirs(os.path.join(base, "frames"), exist_ok=True)
    problems = []
    for dst, raw, mirrored in sorted(set(jobs)):
        if not os.path.exists(raw):
            problems.append("missing raw image %s (for %s)" % (raw, dst))
            continue
        img = _load(raw)
        save_png(mirror_frame(img) if mirrored else model_frame(img), os.path.join(base, dst))
    return problems


# --- gates ---

def row_problems(char: str, files: Dict[str, List[Dict]]) -> List[str]:
    """Row-level gates on one character's output files: unique ids, one trailing opp_attacking field, and no Chun-Li
    vs Guile row outside the hold-out file."""
    problems = []
    ids = collections.Counter(r["id"] for rows in files.values() for r in rows)
    problems += ["%s: duplicate id %s (x%d)" % (char, i, n) for i, n in sorted(ids.items()) if n > 1]
    for name, rows in sorted(files.items()):
        for r in rows:
            t = r["state_text"]
            if t.count("opp_attacking=") != 1 or not V2_TAIL.search(t):
                problems.append("%s %s: %s note is not v2: %r" % (char, name, r["id"], t))
            if char == HELDOUT_CHAR and name != HELDOUT_FILE and r.get("opp") == HELDOUT_OPP:
                problems.append("%s %s: %s is vs %s (held out)" % (char, name, r["id"], HELDOUT_OPP))
    return problems


def image_problems(base: str, files: Dict[str, List[Dict]]) -> List[str]:
    paths = sorted({p for rows in files.values() for r in rows for p in r["images"]})
    return ["%s: missing image %s" % (base, p) for p in paths if not os.path.exists(os.path.join(base, p))]


def move_problems(char: str, train: List[Dict], moves: List[str], need: int) -> List[str]:
    """Every move System 1 can choose needs ``need`` new real value training rows (mirrors not counted)."""
    n = collections.Counter(r["action"] for r in train
                            if r.get("task") == "value" and not r["mirrored"] and r["id"].startswith(char + "-live_lv_"))
    return ["%s: %s has %d new value training rows < %d" % (char, a, n[a], need) for a in moves if n[a] < need]


# --- build ---

def _stats(files: Dict[str, List[Dict]], counts: Dict) -> Dict:
    def tally(rows):
        return {"rows": len(rows), "value": sum(r.get("task") == "value" for r in rows),
                "new": sum("-live_lv_" in r["id"] for r in rows),
                "mirrored": sum(bool(r.get("mirrored")) for r in rows)}
    value_labels = collections.Counter(list(VALUE_BUCKETS)[r["label"]] for r in files.get("train", [])
                                       if r.get("task") == "value")
    return dict(counts, files={k: tally(v) for k, v in sorted(files.items())}, value_train=dict(value_labels))


def _char_build(char: str, test_data: str, out: str, repo: str, logs: Callable[[str], Dict],
                new: List[Tuple[str, str, str]], min_value_train: int) -> Tuple[Dict, List[str]]:
    from ..system1.system1 import choices

    src, base = os.path.join(test_data, char), os.path.join(out, char)
    rel = os.path.relpath(os.path.abspath(src), os.path.abspath(base))
    files: Dict[str, List[Dict]] = {n: [] for n in OLD_FILES}
    if char == HELDOUT_CHAR:
        files[HELDOUT_FILE] = []
    problems: List[str] = []
    for name in OLD_FILES:
        rows, p = convert_old(read(os.path.join(src, name + ".jsonl"), missing_ok=True), rel, repo, logs)
        files[name] += rows
        problems += p
    jobs, counts = [], {"dropped_dx0": 0, "partial_lines": 0, "logs": []}
    for sub, _, d in (x for x in new if x[1] == char):
        entries, partial = read_log(os.path.join(d, "actions.jsonl"))
        got, j, p, dropped = new_rows(char, sub, d, entries)
        for name, rows in got.items():
            files.setdefault(name, [])
            files[name] += rows
        jobs, problems = jobs + j, problems + p
        counts = dict(counts, dropped_dx0=counts["dropped_dx0"] + dropped,
                      partial_lines=counts["partial_lines"] + partial, logs=counts["logs"] + [sub])
    files = {k: sorted(v, key=lambda r: r["id"]) for k, v in files.items()}
    problems += write_frames(base, jobs) + row_problems(char, files) + image_problems(base, files)
    if counts["logs"]:
        problems += move_problems(char, files["train"], choices(char), min_value_train)
    for name, rows in files.items():
        write_jsonl(os.path.join(base, name + ".jsonl"), rows)
    stats = _stats(files, counts)
    with open(os.path.join(base, "stats.json"), "w") as f:
        json.dump(stats, f, indent=1, sort_keys=True)
    return stats, problems


def build(test_data: str = "test_data", lv_root: str = LV_ROOT, out: str = OUT, repo: str = ".",
          min_value_train: int = MIN_VALUE_TRAIN) -> Dict:
    """Build every character's dir in ``out`` (which must not exist yet). Returns {"problems", "counts"}."""
    if os.path.exists(out) and os.listdir(out):
        return {"problems": ["%s exists: remove it first (or --overwrite)" % out], "counts": {}}
    from .dataset import dataset_chars

    new = find_logs(lv_root)
    chars = sorted(set(dataset_chars(test_data)) | {c for _, c, _ in new})
    cache: Dict[str, Dict] = {}

    def logs(path: str) -> Dict:
        if path not in cache:
            cache[path] = _log_index(path)
        return cache[path]

    problems, counts = [], {}
    for char in chars:
        os.makedirs(os.path.join(out, char), exist_ok=True)
        counts[char], p = _char_build(char, test_data, out, repo, logs, new, min_value_train)
        problems += p
    return {"problems": problems, "counts": counts}


def overwrite(out: str) -> None:
    """Remove a previous build of ``out`` (only one that looks like ours: character dirs with a stats.json)."""
    if not os.path.exists(out):
        return
    for d in os.listdir(out):
        if d not in FIGHTERS or not os.path.exists(os.path.join(out, d, "stats.json")):
            raise SystemExit("%s holds %s, not a value-data build: not removing it" % (out, d))
    shutil.rmtree(out)
