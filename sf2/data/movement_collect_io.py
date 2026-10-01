"""The movement collector's files and per-opponent loop (docs/prereg_movement_data.md; scripts/collect_movement.py).

<out>/<opp>/
  images/g<game>_k<capture>.png   hud_frame of the captured screen (the model's picture, HUD visible), written once,
                                  only for kept pairs; PNG is lossless, so the pixels are the model's exactly
  ram/g<game>.json.gz             every RAM row of the game: {"game", "names", "rows": [[...], ...]} (stream order)
  pairs.jsonl                     one line per kept pair (sf2.data.movement_collect.EpisodeSampler)
  games.jsonl                     one line per finished game: its commit (written after its RAM and pairs)
  stop.json                       why the loop stopped, with the counts and the shortfalls

A game counts only once its games.jsonl line is written: a run resumes from the committed games (pairs of a game
without that line are ignored here and dropped by the builder) and never reuses a game number.
"""
import gzip
import json
import os
import random
import resource
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from . import movement as M
from .movement_collect import RING, SPLIT_QUOTA, EpisodeSampler, split_of_game

PNG_LEVEL = 6                    # lossless; level 1 (dataset.save_png) makes ~2x bigger files for the same pixels
MEM_CAP_GB = 8.0                 # per worker (its laya-vision on MPS + the ring + Mesen's frames in flight)


class MemoryCapExceeded(RuntimeError):
    pass


class ImageSaver:
    def __init__(self, img_dir: str, game: int):
        self.dir, self.game = img_dir, game
        os.makedirs(img_dir, exist_ok=True)

    def __call__(self, k: int, im: np.ndarray) -> str:
        from PIL import Image

        from .u_data import hud_frame

        name = "g%04d_k%05d.png" % (self.game, k)
        path = os.path.join(self.dir, name)
        if not os.path.exists(path):
            tmp = path + ".tmp.png"
            Image.fromarray(hud_frame(im)).save(tmp, compress_level=PNG_LEVEL)
            os.replace(tmp, path)
        return name


def write_ram(ram_dir: str, game: int, rows: List[Dict[str, int]]) -> str:
    os.makedirs(ram_dir, exist_ok=True)
    names = list(rows[0]) if rows else []
    path = os.path.join(ram_dir, "g%04d.json.gz" % game)
    tmp = path + ".tmp"
    with gzip.open(tmp, "wt", compresslevel=6) as f:
        json.dump({"game": game, "names": names, "rows": [[r[n] for n in names] for r in rows]}, f,
                  separators=(",", ":"))
    os.replace(tmp, path)
    return path


def read_ram(path: str) -> List[Dict[str, int]]:
    with gzip.open(path, "rt") as f:
        rec = json.load(f)
    return [dict(zip(rec["names"], r)) for r in rec["rows"]]


def read_jsonl(path: str, torn: Optional[List[int]] = None) -> List[Dict]:
    """The records of a JSONL file. A torn line (a crash mid-write; the next append closes it) is skipped and its
    line number added to ``torn``."""
    if not os.path.exists(path):
        return []
    out = []
    with open(path) as f:
        for i, line in enumerate(f):
            if not line.strip():
                continue
            try:
                out.append(json.loads(line))
            except ValueError:
                if torn is not None:
                    torn.append(i)
    return out


def _append(path: str, recs: Sequence[Dict]) -> None:
    """Append whole lines; a torn last line left by a crash is closed first, so it stays the only bad line."""
    torn = False
    if os.path.exists(path) and os.path.getsize(path):
        with open(path, "rb") as f:
            f.seek(-1, os.SEEK_END)
            torn = f.read(1) != b"\n"
    with open(path, "a") as f:
        if torn:
            f.write("\n")
        f.write("".join(json.dumps(r, separators=(",", ":")) + "\n" for r in recs))
        f.flush()
        os.fsync(f.fileno())


@dataclass
class Progress:
    counts: Dict[Tuple[str, str], int] = field(default_factory=dict)    # (answer, split) -> committed pairs
    played: int = 0
    next_game: int = 0


def load_progress(base: str, targets: Dict[str, int] = SPLIT_QUOTA,
                  answers: Sequence[str] = M.ANSWERS) -> Progress:
    games = read_jsonl(os.path.join(base, "games.jsonl"))
    pairs = read_jsonl(os.path.join(base, "pairs.jsonl"))
    done = {g["game"] for g in games}
    counts = {(a, s): 0 for a in answers for s in targets}
    for p in pairs:
        if p.get("game") in done and (p["answer"], p["split"]) in counts:
            counts[(p["answer"], p["split"])] += 1
    seen = done | {p["game"] for p in pairs if "game" in p} | _games_on_disk(base)
    return Progress(counts, len(done), max(seen) + 1 if seen else 0)


def _games_on_disk(base: str) -> set:
    """Game numbers a crashed game may have left files under (RAM, images): never reused, so no stale image is
    kept under a new game's name."""
    out = set()
    for sub, head in (("ram", "g"), ("images", "g")):
        d = os.path.join(base, sub)
        if os.path.isdir(d):
            out |= {int(n[len(head):len(head) + 4]) for n in os.listdir(d)
                    if n.startswith(head) and n[len(head):len(head) + 4].isdigit()}
    return out


def _left(prog: Progress, targets: Dict[str, int], split: str, answers: Sequence[str]) -> Dict[str, int]:
    return {a: max(0, targets[split] - prog.counts[(a, split)]) for a in answers}


def _open(prog: Progress, targets: Dict[str, int], split: str, answers: Sequence[str]) -> bool:
    return any(_left(prog, targets, split, answers).values())


def peak_rss_gb() -> float:
    """This process's peak resident memory (macOS reports ru_maxrss in bytes)."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9


def collect_opponent(base: str, play: Callable[[int, EpisodeSampler], Dict], opp: str, cap: int, seed: int,
                     targets: Dict[str, int] = SPLIT_QUOTA, answers: Sequence[str] = M.ANSWERS, ring: int = RING,
                     mem_cap_gb: float = MEM_CAP_GB, rss_gb: Callable[[], float] = peak_rss_gb,
                     log: Callable[..., None] = print) -> Dict:
    """Play games (``play(game, sampler)`` feeds every stream row to the sampler and returns the game summary) until
    every (answer, split) quota is full or ``cap`` games are committed; the stop record."""
    if cap < 0 or any(v < 0 for v in targets.values()):
        raise ValueError("cap %r and quotas %r must be >= 0" % (cap, targets))
    os.makedirs(base, exist_ok=True)
    prog = load_progress(base, targets, answers)
    reason = None
    while reason is None:
        if not any(_open(prog, targets, s, answers) for s in targets):
            reason = "quotas full"
            break
        if prog.played >= cap:
            reason = "game cap"
            break
        game = prog.next_game
        while not _open(prog, targets, split_of_game(game), answers):
            game += 1
        split = split_of_game(game)
        sampler = EpisodeSampler(game, split, _left(prog, targets, split, answers),
                                 random.Random("%d:%s:%d" % (seed, opp, game)),
                                 ImageSaver(os.path.join(base, "images"), game), ring)
        t0 = time.time()
        summary = play(game, sampler)
        pairs = sampler.finish()
        write_ram(os.path.join(base, "ram"), game, sampler.rows)
        _append(os.path.join(base, "pairs.jsonl"), pairs)
        by = {a: sum(p["answer"] == a for p in pairs) for a in answers}
        rss = rss_gb()
        _append(os.path.join(base, "games.jsonl"), [dict(summary or {}, game=game, split=split, opp=opp,
                                                         rows=len(sampler.rows), pairs=len(pairs), by_answer=by,
                                                         seconds=round(time.time() - t0, 2), rss_gb=round(rss, 2))])
        for p in pairs:
            prog.counts[(p["answer"], split)] += 1
        prog = Progress(prog.counts, prog.played + 1, game + 1)
        log("%s game %d (%s): %d rows, %d pairs %s, %.0f s, rss %.1f GB" % (
            opp, game, split, len(sampler.rows), len(pairs), {a: n for a, n in by.items() if n},
            time.time() - t0, rss))
        if rss > mem_cap_gb:
            _stop(base, "memory", prog, targets, answers, cap)
            raise MemoryCapExceeded("%s: %.1f GB resident > the %.1f GB cap" % (opp, rss, mem_cap_gb))
    return _stop(base, reason, prog, targets, answers, cap)


def _stop(base: str, reason: str, prog: Progress, targets: Dict[str, int], answers: Sequence[str],
          cap: int) -> Dict:
    short = {a: {s: targets[s] - prog.counts[(a, s)] for s in targets if prog.counts[(a, s)] < targets[s]}
             for a in answers}
    rec = {"reason": reason, "games": prog.played, "cap": cap, "targets": targets,
           "counts": {"%s|%s" % k: v for k, v in sorted(prog.counts.items())},
           "short": {a: v for a, v in short.items() if v}}
    with open(os.path.join(base, "stop.json"), "w") as f:
        json.dump(rec, f, indent=1, sort_keys=True)
    return rec


def read_games(base: str) -> List[Dict]:
    return read_jsonl(os.path.join(base, "games.jsonl"))


def committed_pairs(base: str) -> Tuple[List[Dict], int]:
    """(pairs of committed games, pairs dropped as uncommitted)."""
    done = {g["game"] for g in read_games(base)}
    pairs = read_jsonl(os.path.join(base, "pairs.jsonl"))
    kept = [p for p in pairs if p.get("game") in done]
    return kept, len(pairs) - len(kept)

