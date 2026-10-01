"""The three movement datasets of the owner's label decision (docs/prereg_movement_data.md, "Owner decisions on the
label"; scripts/build_action_data.py), built from the SAME pairs of the action collection (sf2.data.action_collect):

  act    per pair, per fighter (her = player 1 = "me", him = player 2), two choice questions:
           key "act"    "What move is <Name> (me|him) doing?"   options: that actor's codes "act<NN>" (every code the
                        actor has anywhere in the dataset, numeric order, the same list in every dir)
           key "stage"  "Which part of <Name>'s (me|him) move is this?"   options: stg1 start / stg2 middle / stg3 end
         together the owner's "<actor> act<NN> stg<k>" (kept whole on both rows as ``label_full``). One question with
         every "act<NN> stg<k>" option does not fit laya's 256-token head (options are cut and collide; see
         ``head_tokens``), so the action and its stage are asked separately. A fighter whose row t is unknown
         (sf2.data.action_codes) gets no rows on that pair.
  where  "Where is he?"  on the ground (his y == GROUND_Y at t) / in the air.
  dist   "How far is he?"  throw / poke / mid / far: sf2.data.perception.range_band with
         lessons/perception_thresholds_v2.json; an impossible x: no row.

Every row: the pair's two images (frames/g<game>_k<k_prev>.png, frames/g<game>_k<k_now>.png; frames is a symlink to the
collection's <opp>/images, no copies), the note "me=chunli" (v3), "perception": true, the label read at the displayed
row t (lag 1). Split by whole games (game % 10 in 2, 5, 8 -> test_real.jsonl, 0 -> val.jsonl, else train.jsonl). One
dir per opponent in each dataset. Natural counts, no repetition copies.
"""
import collections
import json
import os
from typing import Dict, List, Optional, Sequence, Tuple

from . import action_codes as A
from . import action_collect_io as IO
from . import u_data as U
from .movement_collect import LAG, split_of_game
from .perception import UNKNOWN, range_band

ME = "chunli"
DATASETS = ("act", "where", "dist")
OUTS = {"act": "test_data_act", "where": "test_data_where", "dist": "test_data_dist"}
FILES = {"train": "train", "val": "val", "test": "test_real"}
THRESHOLDS = os.path.join("lessons", "perception_thresholds_v2.json")
NAMES = {"chunli": "Chun-Li", "ryu": "Ryu", "ken": "Ken", "guile": "Guile", "blanka": "Blanka", "dhalsim": "Dhalsim",
         "honda": "E. Honda", "zangief": "Zangief"}
WHO = {1: "me", 2: "him"}
STAGE_CRITERIA = {"stg1": "start", "stg2": "middle", "stg3": "end"}
WHERE_CRITERIA = {"on the ground": "", "in the air": ""}
DIST_CRITERIA = {"throw": "close enough to throw", "poke": "in poke range", "mid": "mid range", "far": "far"}
HEAD_MAX = 256                       # laya.vlm_train.make_item's head_max_len (scripts/train.py keeps the default)
TOKENIZER = os.path.join("runs", "all8", "best", "processor")
KEEP = ("t", "k_prev", "k_now", "pos", "length", "episode", "long", "cut_end")


def image_name(game: int, k: int) -> str:
    return "g%04d_k%05d.png" % (game, k)


def act_question(actor: str, player: int, codes: Sequence[int]) -> Dict:
    return {"type": "choice", "instructions": "What move is %s (%s) doing?" % (NAMES[actor], WHO[player]),
            "criteria": {A.code_name(c): "" for c in sorted(codes)}}


def stage_question(actor: str, player: int) -> Dict:
    return {"type": "choice", "instructions": "Which part of %s's (%s) move is this?" % (NAMES[actor], WHO[player]),
            "criteria": dict(STAGE_CRITERIA)}


def where_question() -> Dict:
    return {"type": "choice", "instructions": "Where is he?", "criteria": dict(WHERE_CRITERIA)}


def dist_question() -> Dict:
    return {"type": "choice", "instructions": "How far is he?", "criteria": dict(DIST_CRITERIA)}


def where_answer(rows: A.Rows, t: int) -> str:
    from ..emu.vs import GROUND_Y
    return "on the ground" if rows[t]["p2_y"] == GROUND_Y else "in the air"


def pair_problems(opp: str, p: Dict, img_dir: str) -> List[str]:
    where = "%s g%s t%s" % (opp, p.get("game"), p.get("t"))
    out = []
    if p["split"] != split_of_game(p["game"]):
        out.append("%s: split %s but game %d belongs to %s" % (where, p["split"], p["game"], split_of_game(p["game"])))
    want = [image_name(p["game"], p["t"] - 4 + LAG), image_name(p["game"], p["t"] + LAG)]
    if p["images"] != want or (p["k_prev"], p["k_now"]) != (p["t"] - 4 + LAG, p["t"] + LAG):
        out.append("%s: images %s are not frames (t - 4, t) shown at lag %d %s" % (where, p["images"], LAG, want))
    out += ["%s: missing image %s" % (where, n) for n in p["images"] if not os.path.exists(os.path.join(img_dir, n))]
    return out


def label_opp(job: Tuple[str, str, Dict]) -> Tuple[str, List[Dict], Dict, List[str]]:
    """One opponent's committed pairs -> (opp, labelled pairs, stats, problems). A labelled pair is the collector's
    pair plus "who": {player: (actor, code, stg, [s, e], cut_end)} for each fighter labelled at t, "where", "dist"."""
    opp, root, th = job
    src = os.path.join(root, opp)
    pairs, dropped = IO.committed_pairs(src)
    img_dir = os.path.join(src, "images")
    actors = {1: ME, 2: opp}
    problems: List[str] = []
    seen = collections.Counter((p["game"], p["t"]) for p in pairs)
    problems += ["%s g%d t%d: %d pairs at one displayed row" % (opp, g, t, n) for (g, t), n in seen.items() if n > 1]
    out: List[Dict] = []
    stats = collections.Counter()
    by_game: Dict[int, List[Dict]] = collections.defaultdict(list)
    for p in pairs:
        by_game[p["game"]].append(p)
    for game in sorted(by_game):
        rows = _read_ram(src, game)
        if rows is None:
            problems.append("%s g%d: no RAM file" % (opp, game))
            continue
        labs = {pl: A.labels_at(rows, pl) for pl in (1, 2)}
        for p in by_game[game]:
            bad = pair_problems(opp, p, img_dir)
            src_lab = labs[p["player"]].get(p["t"])
            if src_lab is None or (src_lab[0].code, src_lab[1]) != (p["code"], p["stg"]) or \
                    actors[p["player"]] != p["actor"]:
                bad.append("%s g%d t%d: collector label %s %s stg%s, RAM gives %s" % (
                    opp, game, p["t"], p["actor"], p["code"], p["stg"],
                    None if src_lab is None else (src_lab[0].code, src_lab[1])))
            if bad:
                problems += bad
                continue
            who = {}
            for pl in (1, 2):
                lab = labs[pl].get(p["t"])
                if lab is None:
                    stats["unlabelled_%s" % WHO[pl]] += 1
                    continue
                ep, stg = lab
                who[pl] = (actors[pl], ep.code, stg, [ep.start, ep.end], ep.cut_end)
            band = range_band(rows, p["t"], th)
            stats["dist_unknown"] += band == UNKNOWN
            out.append(dict(p, who=who, where=where_answer(rows, p["t"]), dist=None if band == UNKNOWN else band))
    stats["dropped_uncommitted"] = dropped
    return opp, out, dict(stats), problems


def _read_ram(src: str, game: int) -> Optional[A.Rows]:
    path = os.path.join(src, "ram", "g%04d.json.gz" % game)
    return IO.read_ram(path) if os.path.exists(path) else None


def _base(opp: str, p: Dict, key: str, tag: str, q: Dict, answer: str) -> Dict:
    did = "%s-mv3_%s_g%04d_t%05d" % (ME, opp, p["game"], p["t"])
    return dict({"decision": did, "char": ME, "opp": opp, "sub": opp, "game": p["game"], "frame": p["k_now"],
                 "split": p["split"], "images": ["frames/" + n for n in p["images"]], "state_text": U.eye_note(ME),
                 "perception": True, "note_version": U.NOTE_VERSION, "frames_version": U.FRAMES_VERSION,
                 "id": "%s-%s%s" % (did, key, tag), "task": key, "key": key, "question": q,
                 "label": list(q["criteria"]).index(answer), "answer": answer, "copy": 0,
                 "src_actor": p["actor"], "src_code": p["code"], "src_stg": p["stg"]}, **{k: p[k] for k in KEEP})


def act_rows(opp: str, p: Dict, options: Dict[str, Sequence[int]]) -> List[Dict]:
    out = []
    for pl, (actor, code, stg, ep, cut_end) in sorted(p["who"].items()):
        extra = {"actor": actor, "player": pl, "code": code, "stg": stg, "label_full": A.label(actor, code, stg),
                 "actor_episode": ep, "actor_cut_end": cut_end}
        tag = "-%s" % WHO[pl]
        out.append(dict(_base(opp, p, "act", tag, act_question(actor, pl, options[actor]), A.code_name(code)),
                        **extra))
        out.append(dict(_base(opp, p, "stage", tag, stage_question(actor, pl), "stg%d" % stg), **extra))
    return out


def rows_for(dataset: str, opp: str, p: Dict, options: Dict[str, Sequence[int]]) -> List[Dict]:
    if dataset == "act":
        return act_rows(opp, p, options)
    if dataset == "where":
        return [_base(opp, p, "where", "", where_question(), p["where"])]
    if dataset == "dist":
        return [_base(opp, p, "dist", "", dist_question(), p["dist"])] if p["dist"] else []
    raise ValueError("unknown dataset %r" % dataset)


def options_of(labelled: Dict[str, List[Dict]]) -> Dict[str, List[int]]:
    """actor -> every code it has on any labelled pair of any opponent (numeric order)."""
    out: Dict[str, set] = collections.defaultdict(set)
    for pairs in labelled.values():
        for p in pairs:
            for actor, code, _, _, _ in p["who"].values():
                out[actor].add(code)
    return {a: sorted(c) for a, c in out.items()}


def head_tokens(q: Dict, tokenizer) -> int:
    """laya's head length for a choice question: the question text plus every rendered option and its terminator
    (laya.vlm.build_vlm_inputs); above HEAD_MAX, options are cut and the instructions lose their middle."""
    from laya.common import render_options
    from laya.vlm import OPTION_BULLET, QUESTION_TEXT, VLMAgent

    enc = lambda s: tokenizer(s, add_special_tokens=False)["input_ids"]  # noqa: E731
    qi = VLMAgent._to_internal(q)
    return len(enc(QUESTION_TEXT % (qi["t"], qi["ins"]))) + sum(len(enc(OPTION_BULLET + o)) + 1
                                                               for o in render_options(qi))


def write_dataset(dataset: str, out: str, root: str, labelled: Dict[str, List[Dict]],
                  options: Dict[str, List[int]]) -> Tuple[Dict, List[str]]:
    os.makedirs(out)
    stats, problems = {}, []
    for opp, pairs in sorted(labelled.items()):
        base = os.path.join(out, opp)
        os.makedirs(base)
        os.symlink(os.path.abspath(os.path.join(root, opp, "images")), os.path.join(base, "frames"))
        files: Dict[str, List[Dict]] = {n: [] for n in FILES.values()}
        for p in pairs:
            files[FILES[p["split"]]] += rows_for(dataset, opp, p, options)
        ids = collections.Counter(r["id"] for rows in files.values() for r in rows)
        problems += ["%s %s: id %s on %d rows" % (dataset, opp, i, n) for i, n in ids.items() if n > 1]
        for name, rows in files.items():
            with open(os.path.join(base, name + ".jsonl"), "w") as f:
                f.write("".join(json.dumps(r) + "\n" for r in sorted(rows, key=lambda r: r["id"])))
        stats[opp] = {"rows": {k: len(v) for k, v in files.items()},
                      "answers": {k: dict(collections.Counter("%s %s" % (r["key"], r["answer"]) for r in v))
                                  for k, v in files.items()}}
    return stats, problems


def build(root: str, outs: Dict[str, str] = OUTS, opps: Optional[Sequence[str]] = None, workers: int = 1,
          thresholds: str = THRESHOLDS, tokenizer=None) -> Dict:
    """The three datasets of the collection ``root`` into ``outs`` (none may exist and be non-empty)."""
    for d in outs.values():
        if os.path.exists(d) and os.listdir(d):
            raise SystemExit("%s exists and is not empty: remove it first" % d)
    th = json.load(open(thresholds))
    want = sorted(opps or (o for o in os.listdir(root) if os.path.exists(os.path.join(root, o, "games.jsonl"))))
    jobs = [(o, root, th) for o in want]
    if workers > 1 and len(jobs) > 1:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(min(workers, len(jobs))) as ex:
            done = list(ex.map(label_opp, jobs))
    else:
        done = [label_opp(j) for j in jobs]
    labelled = {o: pairs for o, pairs, _, _ in done}
    problems = [x for _, _, _, ps in done for x in ps]
    options = options_of(labelled)
    heads = {}
    if tokenizer is not None:
        qs = {"act %s %s" % (a, WHO[pl]): act_question(a, pl, options[a]) for a in options for pl in (1, 2)
              if (a == ME) == (pl == 1)}
        qs.update({"where": where_question(), "dist": dist_question(), "stage": stage_question(ME, 1)})
        heads = {k: head_tokens(q, tokenizer) for k, q in qs.items()}
        problems += ["question %s: %d head tokens > %d (laya cuts it)" % (k, n, HEAD_MAX)
                     for k, n in heads.items() if n > HEAD_MAX]
    stats = {}
    for ds, out in outs.items():
        stats[ds], ps = write_dataset(ds, out, root, labelled, options)
        problems += ps
    for ds, out in outs.items():
        meta = {"root": os.path.abspath(root), "dataset": ds, "me": ME, "opps": want, "lag": LAG,
                "options": {a: [A.code_name(c) for c in cs] for a, cs in options.items()}, "head_tokens": heads,
                "split": {"test": [2, 5, 8], "val": [0]}, "copies": 0, "codes": A.table(),
                "thresholds": thresholds, "frames_version": U.FRAMES_VERSION, "note_version": U.NOTE_VERSION,
                "problems": len(problems)}
        with open(os.path.join(out, "build.json"), "w") as f:
            json.dump(meta, f, indent=1, sort_keys=True)
    return {"problems": problems, "stats": stats, "label_stats": {o: s for o, _, s, _ in done}, "options": options,
            "head_tokens": heads}
