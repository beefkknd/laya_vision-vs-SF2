"""The action collector's core (docs/prereg_movement_data.md, "Owner decisions on the label"): pure logic, no
emulator, no files. Same frames and ring as sf2.data.movement_collect (the image captured at stream index k shows row
k - LAG; a pair for displayed row u is the captures u - 3 and u + 1), but episodes are per FIGHTER and ACTION CODE
(sf2.data.action_codes): when either fighter's episode ends, pairs are sampled from it, one per stage third (start /
middle / end of the whole episode) whose rows are still in the ring; an episode shorter than SHORT rows gives one.

Small caps, not quotas (owner: "see each action once, then move on"): a pair is saved only while its (actor, code,
stage) has fewer than ``cap`` pairs in this game's split (CAPS). A displayed row is used by at most one pair per game
(both fighters are labelled on every pair by the builder, so a second pair at the same row would be a copy).
"""
import random
from collections import Counter, deque
from typing import Callable, Deque, Dict, Iterable, List, Optional, Set, Tuple

import numpy as np

from . import action_codes as A
from .movement_collect import LAG, RING, SHORT, split_of_game  # noqa: F401  (split_of_game re-exported)

CAPS = {"train": 30, "val": 5, "test": 10}      # pairs per (actor, code, stage) per split


def positions(s: int, e: int, lo: int, hi: int, used: Set[int], rng: random.Random) -> List[int]:
    """Displayed rows to sample from the episode [s, e]: one per stage third of the WHOLE episode among rows in
    [lo, hi] not in ``used`` (a third with none is skipped); an episode shorter than SHORT gives one row."""
    a, b = max(s, lo), min(e, hi)
    cand = [u for u in range(a, b + 1) if u not in used]
    if not cand:
        return []
    if e - s + 1 < SHORT:
        return [rng.choice(cand)]
    out = []
    for stg in A.STAGES:
        mine = [u for u in cand if A.stage_of(u, s, e) == stg]
        if mine:
            out.append(rng.choice(mine))
    return out


class ActionSampler:
    """One game: feed every stream row (and its image; None only for row 0) in order, then ``finish``.
    ``counts`` holds this split's committed pairs per (actor, code, stage); ``actors`` names player 1 and 2."""

    def __init__(self, game: int, split: str, actors: Dict[int, str], counts: Counter, cap: int,
                 rng: random.Random, save_image: Callable[[int, np.ndarray], str], ring: int = RING,
                 sample_actors: Optional[Iterable[str]] = None):
        if ring < 5:
            raise ValueError("a ring of %d frames cannot hold a (t - 4, t) pair" % ring)
        if split not in CAPS:
            raise ValueError("unknown split %r" % split)
        if sorted(actors) != [1, 2]:
            raise ValueError("actors must name players 1 and 2, got %r" % actors)
        if cap < 0:
            raise ValueError("cap %r < 0" % cap)
        if sample_actors is not None and not set(sample_actors) <= set(actors.values()):
            raise ValueError("sample_actors %r are not fighters of %r" % (sorted(sample_actors), actors))
        # None: both fighters' episodes are sampled; else only these actors' (both are still tracked and observed)
        self.sample_actors = None if sample_actors is None else frozenset(sample_actors)
        self.game, self.split, self.actors, self.cap = game, split, dict(actors), cap
        self.rng, self.save_image = rng, save_image
        self.counts: Counter = Counter(counts)
        self.rows: List[Dict[str, int]] = []
        self.ring: Deque[Tuple[int, np.ndarray]] = deque(maxlen=ring)
        self.tracks = {p: A.ActorTrack(p) for p in (1, 2)}
        self.pairs: List[Dict] = []
        self.used: Set[int] = set()
        self.names: Dict[int, str] = {}
        self.observed: Counter = Counter()       # (actor, code) -> episodes seen (sampled or not)
        self.unknown: Counter = Counter()        # (actor, reason) -> unknown episodes

    def feed(self, row: Dict[str, int], image: Optional[np.ndarray]) -> None:
        k = len(self.rows)
        if k > 0 and image is None:
            raise ValueError("stream row %d has no image" % k)
        self.rows.append(row)
        if image is not None:
            self.ring.append((k, image))
        for p in (1, 2):
            ep = self.tracks[p].step(self.rows, k)
            if ep is not None:
                self._close(ep)

    def finish(self) -> List[Dict]:
        for p in (1, 2):
            ep = self.tracks[p].finish(self.rows)
            if ep is not None:
                self._close(ep)
        return list(self.pairs)

    def _close(self, ep: A.Episode) -> None:
        actor = self.actors[ep.player]
        if ep.code is None:
            self.unknown[(actor, ep.reason)] += 1
            return
        self.observed[(actor, ep.code)] += 1
        if self.sample_actors is not None and actor not in self.sample_actors:
            return
        if not self.ring:
            return
        oldest, newest = self.ring[0][0], self.ring[-1][0]
        lo = max(A.FIRST_T, oldest + 4 - LAG)            # the prev image u - 3 must still be in the ring
        hi = newest - LAG                                 # the now image u + 1 must exist
        for u in positions(ep.start, ep.end, lo, hi, self.used, self.rng):
            stg = A.stage_of(u, ep.start, ep.end)
            key = (actor, ep.code, stg)
            if self.counts[key] >= self.cap:
                continue
            self.counts[key] += 1
            self.used.add(u)
            k_prev, k_now = u - 4 + LAG, u + LAG
            self.pairs.append({
                "game": self.game, "split": self.split, "t": u, "k_prev": k_prev, "k_now": k_now, "actor": actor,
                "player": ep.player, "code": ep.code, "stg": stg, "reason": ep.reason, "pos": u - ep.start,
                "length": ep.length, "episode": [ep.start, ep.end], "long": ep.start < lo, "cut_end": ep.cut_end,
                "images": [self._image(k_prev), self._image(k_now)]})

    def _image(self, k: int) -> str:
        if k not in self.names:
            im = dict(self.ring).get(k)
            if im is None:
                raise AssertionError("capture %d is not in the ring" % k)
            self.names[k] = self.save_image(k, im)
        return self.names[k]
