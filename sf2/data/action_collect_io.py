"""The action collector's per-opponent loop and stop rule (docs/prereg_movement_data.md, "Owner decisions on the
label"; scripts/collect_actions.py). Files as sf2.data.movement_collect_io (images/, ram/, pairs.jsonl, games.jsonl,
stop.json); the RAM rows carry sf2.data.action_codes.EXTRA_NAMES too.

Stop rule (owner: "see each action once, then move on"): a NEW key is an (actor, code, split) with split in
NOVEL_SPLITS (train, test: the dataset gate asks every (actor, code) in both) that no committed pair had before.
An opponent stops when ``min_games`` games are committed and the last ``patience`` games brought no new key
("no new action"), or at ``cap`` games ("game cap"). Every game number is played in order (no split is skipped).
"""
import json
import os
import random
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field, replace
from typing import Callable, Dict, Iterable, List, Optional, Set, Tuple

from .action_collect import CAPS, ActionSampler, split_of_game
from .movement_collect import RING
from .movement_collect_io import (  # noqa: F401  (read_ram re-exported)
    MEM_CAP_GB,
    ImageSaver,
    MemoryCapExceeded,
    _append,
    _games_on_disk,
    peak_rss_gb,
    read_jsonl,
    read_ram,
    write_ram,
)

MIN_GAMES, PATIENCE, GAME_CAP = 40, 15, 150
NOVEL_SPLITS = ("train", "test")
Actors = Optional[Iterable[str]]     # the actors whose (code, split) count as new; None = every actor


class WrongCharacters(RuntimeError):
    """A game's RAM does not hold the expected characters (the game is not committed)."""


def char_problems(rows: List[Dict[str, int]], ids: Dict[int, int]) -> List[str]:
    """Rows whose p<n>_char is not ``ids[n]`` (sf2.vocab.IDS), per player; empty = every row holds the pair."""
    if not rows:
        return ["no rows"]
    out = []
    for p, want in sorted(ids.items()):
        bad = [k for k, r in enumerate(rows) if r.get("p%d_char" % p) != want]
        if bad:
            out.append("player %d: %d of %d rows hold character %s, not %d (first at row %d)" % (
                p, len(bad), len(rows), rows[bad[0]].get("p%d_char" % p), want, bad[0]))
    return out


@dataclass
class Progress:
    counts: Dict[str, Counter] = field(default_factory=lambda: {s: Counter() for s in CAPS})  # (actor, code, stg)
    seen: Set[Tuple[str, int, str]] = field(default_factory=set)       # (actor, code, split), NOVEL_SPLITS only
    played: int = 0
    since_new: int = 0
    last_new_game: Optional[int] = None
    next_game: int = 0


def novel(pairs: List[Dict], seen: Set[Tuple[str, int, str]], actors: Actors = None) -> List[Tuple[str, int, str]]:
    keys = {(p["actor"], p["code"], p["split"]) for p in pairs if p["split"] in NOVEL_SPLITS
            and (actors is None or p["actor"] in actors)}
    return sorted(keys - seen)


def stop_reason(prog: Progress, min_games: int, patience: int, cap: int) -> Optional[str]:
    if prog.played >= cap:
        return "game cap"
    if prog.played >= min_games and prog.since_new >= patience:
        return "no new action"
    return None


def advance(prog: Progress, game: int, pairs: List[Dict], actors: Actors = None) -> Progress:
    """The progress after committing ``game`` with ``pairs`` (a new object); only ``actors``' keys count as new."""
    new = novel(pairs, prog.seen, actors)
    counts = {s: Counter(c) for s, c in prog.counts.items()}
    for p in pairs:
        counts[p["split"]][(p["actor"], p["code"], p["stg"])] += 1
    return Progress(counts, prog.seen | set(new), prog.played + 1, 0 if new else prog.since_new + 1,
                    game if new else prog.last_new_game, game + 1)


def load_progress(base: str, actors: Actors = None) -> Progress:
    games = sorted(read_jsonl(os.path.join(base, "games.jsonl")), key=lambda g: g["game"])
    pairs = read_jsonl(os.path.join(base, "pairs.jsonl"))
    by_game: Dict[int, List[Dict]] = defaultdict(list)
    for p in pairs:
        by_game[p.get("game")].append(p)
    prog = Progress()
    for g in games:
        prog = advance(prog, g["game"], by_game.get(g["game"], []), actors)
    seen_games = {g["game"] for g in games} | {p["game"] for p in pairs if "game" in p} | _games_on_disk(base)
    return replace(prog, next_game=max(seen_games) + 1 if seen_games else 0)


def _keyed(c: Counter) -> Dict[str, int]:
    return {"|".join(str(x) for x in k): v for k, v in sorted(c.items())}


def collect_opponent(base: str, play: Callable[[int, ActionSampler], Dict], opp: str, actors: Dict[int, str],
                     seed: int, cap: int = GAME_CAP, min_games: int = MIN_GAMES, patience: int = PATIENCE,
                     caps: Dict[str, int] = CAPS, ring: int = RING, mem_cap_gb: float = MEM_CAP_GB,
                     rss_gb: Callable[[], float] = peak_rss_gb, log: Callable[..., None] = print,
                     sample_actors: Actors = None, provenance: Optional[Dict] = None,
                     expect_ids: Optional[Dict[int, int]] = None) -> Dict:
    """Play games (``play(game, sampler)`` feeds every stream row and returns the game summary) until the stop rule
    holds; the stop record (also <base>/stop.json).

    ``sample_actors``: only these actors' episodes are sampled, and only their (code, split) count as new for the
    stop rule (None: both fighters). ``provenance``: fields copied onto every pair and games.jsonl line (its
    "p1_char", if any, must be ``actors[1]``). ``expect_ids``: player -> character id every RAM row of a game must
    hold; a game that does not is not committed (stop "wrong characters", WrongCharacters raised)."""
    if cap < 0 or min_games < 0 or patience < 1 or any(v < 0 for v in caps.values()) or set(caps) != set(CAPS):
        raise ValueError("cap %r, min_games %r, patience %r, caps %r" % (cap, min_games, patience, caps))
    if provenance is not None and provenance.get("p1_char", actors[1]) != actors[1]:
        raise ValueError("provenance p1_char %r but player 1 is %r" % (provenance.get("p1_char"), actors[1]))
    keep = None if sample_actors is None else sorted(sample_actors)
    extra = {"sample_actors": keep, "provenance": provenance} if keep is not None or provenance is not None else {}
    prov = dict(provenance or {})
    os.makedirs(base, exist_ok=True)
    prog = load_progress(base, keep)
    while True:
        reason = stop_reason(prog, min_games, patience, cap)
        if reason:
            break
        game = prog.next_game
        split = split_of_game(game)
        sampler = ActionSampler(game, split, actors, prog.counts[split], caps[split],
                                random.Random("%d:%s:%d" % (seed, opp, game)),
                                ImageSaver(os.path.join(base, "images"), game), ring, keep)
        t0 = time.time()
        summary = play(game, sampler)
        pairs = [dict(p, **prov) for p in sampler.finish()]
        wrong = char_problems(sampler.rows, expect_ids) if expect_ids else []
        if wrong:
            write_stop(base, "wrong characters", prog, cap, min_games, patience, caps, extra)
            raise WrongCharacters("%s game %d: %s" % (opp, game, "; ".join(wrong)))
        write_ram(os.path.join(base, "ram"), game, sampler.rows)
        _append(os.path.join(base, "pairs.jsonl"), pairs)
        new = novel(pairs, prog.seen, keep)
        rss = rss_gb()
        _append(os.path.join(base, "games.jsonl"), [dict(
            summary or {}, **prov, game=game, split=split, opp=opp, rows=len(sampler.rows), pairs=len(pairs),
            by_code=_keyed(Counter((p["actor"], p["code"]) for p in pairs)), observed=_keyed(sampler.observed),
            unknown=_keyed(sampler.unknown), new=["%s|%d|%s" % k for k in new],
            seconds=round(time.time() - t0, 2), rss_gb=round(rss, 2))])
        prog = advance(prog, game, pairs, keep)
        log("%s game %d (%s): %d rows, %d pairs, %d new, %d since new, %.0f s, rss %.1f GB" % (
            opp, game, split, len(sampler.rows), len(pairs), len(new), prog.since_new, time.time() - t0, rss))
        if rss > mem_cap_gb:
            write_stop(base, "memory", prog, cap, min_games, patience, caps, extra)
            raise MemoryCapExceeded("%s: %.1f GB resident > the %.1f GB cap" % (opp, rss, mem_cap_gb))
    return write_stop(base, reason, prog, cap, min_games, patience, caps, extra)


def write_stop(base: str, reason: str, prog: Progress, cap: int, min_games: int, patience: int,
               caps: Dict[str, int], extra: Optional[Dict] = None) -> Dict:
    per: Dict[str, Dict[str, int]] = defaultdict(dict)
    for s, c in prog.counts.items():
        for (actor, code, _), n in c.items():
            key = "%s|%d" % (actor, code)
            per[key][s] = per[key].get(s, 0) + n
    rec = {"reason": reason, "games": prog.played, "cap": cap, "min_games": min_games, "patience": patience,
           "caps": caps, "since_new": prog.since_new, "last_new_game": prog.last_new_game,
           "pairs_by_code": dict(sorted(per.items())),
           "missing_in": {k: [s for s in NOVEL_SPLITS if not v.get(s)] for k, v in sorted(per.items())
                          if any(not v.get(s) for s in NOVEL_SPLITS)}}
    rec.update(extra or {})
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
