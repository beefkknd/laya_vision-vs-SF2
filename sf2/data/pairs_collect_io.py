"""The movement-pairs collector's files and per-pair loop (docs/prereg_movement_pairs.md; scripts/collect_pairs.py).

<out>/<A>_vs_<B>/                 A = player 1 (directed, or Plan B: ours), B = player 2 (the CPU, or Plan B: ours)
  images/g<game>_k<capture>.png   hud_frame of the captured screen, written once, only for kept pairs
  ram/g<game>.json.gz             every RAM row of the game (sf2.emu.vs.VARS + the attack-ID bytes, both fighters)
  pairs.jsonl                     one line per kept pair (sf2.data.pairs_collect.PairSampler), both slots
  games.jsonl                     one line per finished game, its commit: the summary and the move log
                                  [[word, k_start, k_end, status, slot], ...] (status: sf2.data.pairs_moves.executed
                                  for that slot's character; Plan B logs both slots)
  stop.json                       the games played and the budget

A fixed budget: ``games`` committed games per pair, then the loop stops. A game counts only once its games.jsonl line
is written; a re-run resumes, never reusing a game number (a crashed game's files keep theirs).
"""
import json
import os
import random
import time
from typing import Callable, Dict, List, Tuple

from . import movement_collect_io as MIO
from . import pairs_moves as PM
from .movement_collect import RING
from .movement_collect_io import ImageSaver, MemoryCapExceeded, peak_rss_gb, read_jsonl, write_ram
from .pairs_collect import PER_GAME, SLOTS, PairSampler

MEM_CAP_GB = 3.0                  # per worker: Python + the ring (no model) + Mesen's frames in flight


def pair_name(a: str, b: str) -> str:
    return "%s_vs_%s" % (a, b)


def committed(base: str) -> List[Dict]:
    return read_jsonl(os.path.join(base, "games.jsonl"))


def next_game(base: str) -> int:
    seen = {g["game"] for g in committed(base)} | {p["game"] for p in read_jsonl(os.path.join(base, "pairs.jsonl"))
                                                   if "game" in p} | MIO._games_on_disk(base)
    return max(seen) + 1 if seen else 0


def move_log(chars: Dict[int, str], moves: List[List], rows: List[Dict[str, int]]) -> List[List]:
    """The move log with each move's status (sf2.data.pairs_moves.executed over its rows, for its slot): a move is
    [word, k0, k1] (slot 1) or [word, k0, k1, slot]."""
    out = []
    for m in moves:
        w, k0, k1 = m[:3]
        p = m[3] if len(m) > 3 else 1
        out.append([w, k0, k1, PM.executed(chars[p], w, rows[k0:k1 + 1], p)["status"], p])
    return out


def collect_pair(base: str, play: Callable[[int, PairSampler], Dict], a: str, b: str, games: int, seed: int,
                 bands: Dict[str, int], ring: int = RING, per_game: int = PER_GAME, mem_cap_gb: float = MEM_CAP_GB,
                 rss_gb: Callable[[], float] = peak_rss_gb, log: Callable[..., None] = print,
                 controllers: Dict[int, str] = SLOTS) -> Dict:
    """Play games (``play(game, sampler)`` feeds every stream row to the sampler and returns the summary with
    "moves") until ``games`` are committed; the stop record."""
    if games < 0:
        raise ValueError("games %r must be >= 0" % games)
    os.makedirs(base, exist_ok=True)
    while len(committed(base)) < games:
        game = next_game(base)
        sampler = PairSampler(game, {1: a, 2: b}, random.Random("%d:%s:%s:%d" % (seed, a, b, game)),
                              ImageSaver(os.path.join(base, "images"), game), bands, ring, per_game, controllers)
        t0 = time.time()
        summary = play(game, sampler)
        pairs = sampler.finish()
        write_ram(os.path.join(base, "ram"), game, sampler.rows)
        MIO._append(os.path.join(base, "pairs.jsonl"), pairs)
        moves = move_log({1: a, 2: b}, summary.get("moves", []), sampler.rows)
        rss = rss_gb()
        rec = dict(summary, game=game, pair=[a, b], rows=len(sampler.rows), pairs=len(pairs), moves=moves,
                   seconds=round(time.time() - t0, 2), rss_gb=round(rss, 2))
        MIO._append(os.path.join(base, "games.jsonl"), [rec])
        done = sum(m[3] == "done" for m in moves)
        log("%s vs %s game %d: %s, %d rows, %d pairs, %d moves (%d done), %.0f s, rss %.1f GB" % (
            a, b, game, rec.get("result"), len(sampler.rows), len(pairs), len(moves), done, time.time() - t0, rss))
        if rss > mem_cap_gb:
            _stop(base, "memory", games)
            raise MemoryCapExceeded("%s vs %s: %.1f GB resident > the %.1f GB cap" % (a, b, rss, mem_cap_gb))
    return _stop(base, "budget spent", games)


def _stop(base: str, reason: str, games: int) -> Dict:
    played = committed(base)
    rec = {"reason": reason, "games": len(played), "budget": games,
           "pairs": sum(g["pairs"] for g in played), "seconds": round(sum(g["seconds"] for g in played), 1)}
    with open(os.path.join(base, "stop.json"), "w") as f:
        json.dump(rec, f, indent=1, sort_keys=True)
    return rec


def committed_pairs(base: str) -> Tuple[List[Dict], int]:
    """(pairs of committed games, pairs dropped as uncommitted)."""
    done = {g["game"] for g in committed(base)}
    pairs = read_jsonl(os.path.join(base, "pairs.jsonl"))
    kept = [p for p in pairs if p.get("game") in done]
    return kept, len(pairs) - len(kept)
