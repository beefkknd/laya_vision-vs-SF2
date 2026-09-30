"""The value fine-tune's dataset (docs/prereg_lv_value.md; scripts/build_value_data.py -> test_data_v2/<char>/):

- the all8 rows (test_data/<char>/train.jsonl, test_real_{left,right}.jsonl) with note v2: the still dummy is never
  attacking (opp_attacking=0); a block row's decision frame is 3 frames into a ground probe (his state 0x0A: 1) or the
  jump-in's descent before the kick (his state is the jump 0x04: 0); a live row takes the opponent's state logged at
  that decision (rollouts/<log>/<char>/actions.jsonl, joined by game + frame from the row id: a row that does not join
  is a problem, never guessed). Their images stay in test_data, by a path relative to test_data_v2/<char>/.
- the new live rows from the collection logs (rollouts/lv_value/<sub>/<char>/): as scripts/vs_dataset.py import-live
  and build do it (model frames, mirrored training rows, dx == 0 dropped, game % 10 in TEST_INDEX held out). The
  live cap: every character trains on the same number of new decisions (the smallest character's count; its earliest
  games, evenly across its opponents); training games past the cap go, real only, to test_extra.jsonl (evaluation). Chun-Li vs Guile goes to test_heldout_guile.jsonl only (real rows), never to train or test_real.
  Opt-in for the second run (docs/reviews/2026-09-30_dr_fable_lv_value.md B.2-3), both off by default so the first
  run's build stays byte for byte: ``no_cap`` (no live cap: every character keeps all its training games, no
  test_extra; balance comes from equal-share sampling in training) and ``forward_value_cap="median_attack"`` (per
  character, the new live forward VALUE training rows are cut, by a hash of the row id, to the median count of its
  attacks' new value training rows; the mirror follows its source; outcome rows and other moves untouched).
- next to every live row (old and new, mirrored too) its value row: the same picture and note, the value question,
  the label value_bucket(dealt - taken). A still dummy never punishes: no value rows for the other rows.
"""
import collections
import hashlib
import json
import os
import re
import shutil
import statistics
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
FORWARD = "forward"
FORWARD_VALUE_CAPS = (None, "median_attack")
FORWARD_HASH_SALT = "lv_value_forward_cap:"


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


Decision = Tuple[str, Dict, List[str]]      # (log sub-dir, real laya row, its raw images)
EXTRA_FILE = "test_extra"


def new_decisions(char: str, sub: str, log_dir: str, entries: List[Dict]) -> Tuple[List[Decision], List[str], int]:
    """One log's decisions as real laya rows (with their raw images), problems, and the dx == 0 drops."""
    decs, problems, dropped = [], [], 0
    for a in entries:
        if a["gap"] == 0:
            dropped += 1        # the fighters at the same x: no left or right, a mirror would contradict it
            continue
        problem = _entry_problem(char, a)
        if problem:
            problems.append(problem)
            continue
        decs.append((sub, laya_row(new_record(char, sub, a)), [os.path.join(log_dir, "images", p) for p in a["images"]]))
    return decs, problems, dropped


def train_games(decs: List[Decision]) -> Dict[Tuple[str, int], int]:
    """New training decisions per (log, game): what the cap counts."""
    return dict(collections.Counter((sub, r["game"]) for sub, r, _ in decs if r["split"] == "train"))


def cap_of(totals: Dict[str, int]) -> Optional[int]:
    """The live cap: the smallest positive count of new training decisions over the characters (None: no new data)."""
    return min([n for n in totals.values() if n > 0], default=None)


def keep_games(counts: Dict[Tuple[str, int], int], cap: int) -> set:
    """A character's earliest training games, evenly across its opponents (logs): game 0 of each, then game 1 of each,
    ...; whole games only, stopping before the first one that would pass the cap."""
    per = {sub: sorted(g for s, g in counts if s == sub) for sub in sorted({s for s, _ in counts})}
    order = [(sub, gs[i]) for i in range(max(map(len, per.values()), default=0)) for sub, gs in per.items()
             if i < len(gs)]
    keep, total = set(), 0
    for key in order:
        if total + counts[key] > cap:
            break
        keep, total = keep | {key}, total + counts[key]
    return keep


def median_attack(counts: Dict[str, int], attacks: List[str]) -> int:
    """The median of the new value training rows per attack move (every attack of ``attacks``, none counted as 0),
    rounded down."""
    return int(statistics.median([counts.get(a, 0) for a in attacks])) if attacks else 0


def forward_value_keep(ids: List[str], n: int) -> set:
    """``n`` of ``ids`` chosen by the sha256 of each id (no random state): the same ids for the same input in any
    order, and a larger ``n`` keeps a superset."""
    ranked = sorted(ids, key=lambda i: hashlib.sha256((FORWARD_HASH_SALT + i).encode()).hexdigest())
    return set(ranked[:max(0, n)])


def _labels(rows: List[Dict]) -> Dict[str, int]:
    return dict(collections.Counter(list(VALUE_BUCKETS)[value_row(r)["label"]] for r in rows))


def forward_value_plan(kept: List[Dict], attacks: List[str]) -> Tuple[set, Dict]:
    """The real forward rows (of ``kept``, a character's kept new real training rows) whose value rows are dropped,
    and the report: forward value rows before/after, the median attack count, the label counts before/after."""
    counts = collections.Counter(r["action"] for r in kept)
    med = median_attack(counts, attacks)
    fwd = [r for r in kept if r["action"] == FORWARD]
    keep = forward_value_keep([r["id"] for r in fwd], med)
    return ({r["id"] for r in fwd} - keep,
            {"median_attack": med, "before": len(fwd), "after": len(keep), "labels_before": _labels(fwd),
             "labels_after": _labels([r for r in fwd if r["id"] in keep])})


def assign(decs: List[Decision], keep: set, drop_value: frozenset = frozenset()
           ) -> Tuple[Dict[str, List[Dict]], List[Tuple[str, str, bool]]]:
    """Rows per output file and frame jobs (dst, raw, mirrored): kept training games with mirrors, training games past
    the cap real only in test_extra, test games in test_real_<side>, Chun-Li vs Guile in the hold-out file. A kept
    training row whose id is in ``drop_value`` keeps its outcome rows but loses its value row and its mirror's."""
    files: Dict[str, List[Dict]] = collections.defaultdict(list)
    jobs = []
    for sub, r, raws in decs:
        jobs += [(dst, raw, False) for dst, raw in zip(r["images"], raws)]
        if r["split"] == "train" and (sub, r["game"]) in keep:
            m = mirror_row(r)
            jobs += [(dst, raw, True) for dst, raw in zip(m["images"], raws)]
            files["train"] += [r, m] if r["id"] in drop_value else [r, value_row(r), m, value_row(m)]
        elif r["split"] == "train":
            files[EXTRA_FILE] += [r, value_row(r)]
        elif r["split"] == "test":
            files["test_real_" + r["side"]] += [r, value_row(r)]
        else:
            files[HELDOUT_FILE] += [r, value_row(r)]
    return dict(files), jobs


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


def read_new(char: str, new: List[Tuple[str, str, str]]) -> Tuple[List[Decision], List[str], Dict]:
    """Every new log of one character: its decisions, problems and counts."""
    decs, problems, counts = [], [], {"dropped_dx0": 0, "partial_lines": 0, "logs": []}
    for sub, _, d in (x for x in new if x[1] == char):
        entries, partial = read_log(os.path.join(d, "actions.jsonl"))
        got, p, dropped = new_decisions(char, sub, d, entries)
        decs, problems = decs + got, problems + p
        counts = dict(counts, dropped_dx0=counts["dropped_dx0"] + dropped,
                      partial_lines=counts["partial_lines"] + partial, logs=counts["logs"] + [sub])
    return decs, problems, counts


def _forward_plan(char: str, decs: List[Decision], keep: set, forward_value_cap: Optional[str]
                  ) -> Tuple[frozenset, Optional[Dict]]:
    from ..system1.system1 import choices
    from .vs_defense import BLOCKS

    if forward_value_cap is None:
        return frozenset(), None
    kept = [r for sub, r, _ in decs if r["split"] == "train" and (sub, r["game"]) in keep]
    attacks = [a for a in choices(char) if a not in BLOCKS and a != FORWARD]
    drop, report = forward_value_plan(kept, attacks)
    return frozenset(drop), report


def _char_build(char: str, test_data: str, out: str, repo: str, logs: Callable[[str], Dict], new: Tuple,
                cap: Optional[int], min_value_train: int, no_cap: bool = False,
                forward_value_cap: Optional[str] = None) -> Tuple[Dict, List[str]]:
    from ..system1.system1 import choices

    decs, problems, counts = new
    src, base = os.path.join(test_data, char), os.path.join(out, char)
    rel = os.path.relpath(os.path.abspath(src), os.path.abspath(base))
    files: Dict[str, List[Dict]] = {n: [] for n in OLD_FILES + (() if no_cap else (EXTRA_FILE,))}
    if char == HELDOUT_CHAR:
        files[HELDOUT_FILE] = []
    for name in OLD_FILES:
        rows, p = convert_old(read(os.path.join(src, name + ".jsonl"), missing_ok=True), rel, repo, logs)
        files[name] += rows
        problems = problems + p
    games = train_games(decs)
    keep = set(games) if no_cap else keep_games(games, cap) if cap is not None else set()
    drop, fwd_report = _forward_plan(char, decs, keep, forward_value_cap)
    got, jobs = assign(decs, keep, drop)
    for name, rows in got.items():
        files[name] = files.get(name, []) + rows
    counts = dict(counts, new_train_decisions=sum(games[k] for k in keep),
                  extra_decisions=sum(n for k, n in games.items() if k not in keep))
    if fwd_report is not None:
        counts = dict(counts, forward_value_cap=fwd_report)
    files = {k: sorted(v, key=lambda r: r["id"]) for k, v in files.items()}
    problems = problems + write_frames(base, jobs) + row_problems(char, files) + image_problems(base, files)
    if counts["logs"]:
        problems = problems + move_problems(char, files["train"], choices(char), min_value_train)
    for name, rows in files.items():
        write_jsonl(os.path.join(base, name + ".jsonl"), rows)
    stats = _stats(files, dict(counts, cap=cap, **({"no_cap": True} if no_cap else {})))
    with open(os.path.join(base, "stats.json"), "w") as f:
        json.dump(stats, f, indent=1, sort_keys=True)
    return stats, problems


def build(test_data: str = "test_data", lv_root: str = LV_ROOT, out: str = OUT, repo: str = ".",
          min_value_train: int = MIN_VALUE_TRAIN, no_cap: bool = False, forward_value_cap: Optional[str] = None) -> Dict:
    """Build every character's dir in ``out`` (which must not exist yet). Returns {"problems", "counts", "cap"}
    (cap None with ``no_cap``). ``no_cap`` and ``forward_value_cap``: see the module doc; both off by default."""
    if forward_value_cap not in FORWARD_VALUE_CAPS:
        raise ValueError("forward_value_cap must be one of %s, got %r" % (FORWARD_VALUE_CAPS, forward_value_cap))
    if os.path.exists(out) and os.listdir(out):
        return {"problems": ["%s exists: remove it first (or --overwrite)" % out], "counts": {}, "cap": None}
    from .dataset import dataset_chars

    found = find_logs(lv_root)
    chars = sorted(set(dataset_chars(test_data)) | {c for _, c, _ in found})
    new = {c: read_new(c, found) for c in chars}
    cap = None if no_cap else cap_of({c: sum(train_games(n[0]).values()) for c, n in new.items()})
    cache: Dict[str, Dict] = {}

    def logs(path: str) -> Dict:
        if path not in cache:
            cache[path] = _log_index(path)
        return cache[path]

    problems, counts = [], {}
    for char in chars:
        os.makedirs(os.path.join(out, char), exist_ok=True)
        counts[char], p = _char_build(char, test_data, out, repo, logs, new[char], cap, min_value_train, no_cap,
                                      forward_value_cap)
        problems += p
    return {"problems": problems, "counts": counts, "cap": cap}


def overwrite(out: str) -> None:
    """Remove a previous build of ``out`` (only one that looks like ours: character dirs with a stats.json)."""
    if not os.path.exists(out):
        return
    for d in os.listdir(out):
        if d not in FIGHTERS or not os.path.exists(os.path.join(out, d, "stats.json")):
            raise SystemExit("%s holds %s, not a value-data build: not removing it" % (out, d))
    shutil.rmtree(out)
