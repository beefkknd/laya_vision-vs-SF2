"""The movement-dataset collector's core (docs/prereg_movement_data.md): pure logic, no emulator, no files.

Frames. Every frame of a round is captured (``CapturingBridge``): stream index k is the RAM row k of the round
(play_round's own stream: the first RUN's rows all, later RUNs without their row 0) and the image captured at that
same boundary. The image at k shows row k - LAG (perception.LAG = 1). A ring buffer keeps the last ``RING`` images.

The answer. His movement answer at displayed row t is sf2.data.movement.movement(rows, t), unchanged (it reads rows
t and t - 4 only, so it is known as soon as row t arrives). Rows t < 4 are not used (no t - 4 row, and the pair's
first image would be capture 0, taken right after the savestate load).

Episodes. A run of displayed rows with one answer (unknown breaks runs and is never sampled). When the answer
changes at row k, the episode [s, k - 1] ends and is sampled: a pair for displayed row u is the images captured at
u - 3 (showing row u - 4) and u + 1 (showing row u), exactly the (n - 4, n) pair of play_round with n = u + 1.
Positions: with length L >= SHORT, one position in each third of the episode (stage bin = 3 * pos // L: start,
middle, end); a shorter episode gives one position. Only positions whose two images are still in the ring are used
(the buffered part, u >= oldest capture + 3); an episode starting before that is flagged ``long`` and its thirds are
taken over the buffered part (its stage stays pos / L of the WHOLE episode). The round's last episode is closed at
the round end (flagged ``cut_end``; its last row has no "now" image yet and is skipped).

Quota. ``left`` is what each answer may still save in this game's split; a pair of a full answer is not saved (its
images are never written).
"""
import random
from collections import deque
from typing import Callable, Deque, Dict, List, Optional, Tuple

import numpy as np

from . import movement as M
from .perception import LAG, UNKNOWN

RING = 120                        # displayed frames kept (the prereg)
SHORT = 6                         # episodes shorter than this give one pair
QUOTA = 1500                      # pairs per (opponent, answer), all splits
# The prereg's gate 1 asks >= 1,000 train and >= 200 test pairs per (opponent, answer); a single 1,500 target over a
# 6/3/1 game split would give ~900 train. So the 1,500 is split per split, filled from that split's own games.
SPLIT_QUOTA = {"train": 1100, "val": 100, "test": 300}
TEST_INDEX, VAL_INDEX = (2, 5, 8), (0,)
STAGES = ("start", "middle", "end")
FIRST_T = 4                       # first displayed row with a t - 4 row (and a pair whose first capture is >= 1)


def split_of_game(game: int) -> str:
    if game % 10 in TEST_INDEX:
        return "test"
    return "val" if game % 10 in VAL_INDEX else "train"


def stage_bin(pos: int, length: int) -> str:
    if not 0 <= pos < length:
        raise ValueError("position %d outside an episode of %d frames" % (pos, length))
    return STAGES[3 * pos // length]


def sample_positions(s: int, e: int, lo: int, hi: int, rng: random.Random) -> List[int]:
    """Displayed rows to sample from the episode [s, e], restricted to [lo, hi] (the buffered part with a now-image):
    for L >= SHORT one per stage bin of the whole episode when its start is buffered, else one per third of the
    buffered part (a long episode); a shorter episode gives one."""
    a, b = max(s, lo), min(e, hi)
    if a > b:
        return []
    L = e - s + 1
    if L < SHORT:
        return [rng.randint(a, b)]
    base, m = (s, L) if a == s else (a, b - a + 1)
    out = []
    for i in range(len(STAGES)):
        cand = [u for u in range(a, b + 1) if 3 * (u - base) // m == i]
        if cand:
            out.append(rng.choice(cand))
    return out


class EpisodeSampler:
    """One game: feed every stream row (and its image; None only for row 0) in order, then ``finish``."""

    def __init__(self, game: int, split: str, left: Dict[str, int], rng: random.Random,
                 save_image: Callable[[int, np.ndarray], str], ring: int = RING):
        if ring < 5:
            raise ValueError("a ring of %d frames cannot hold a (t - 4, t) pair" % ring)
        if split not in SPLIT_QUOTA:
            raise ValueError("unknown split %r" % split)
        self.game, self.split, self.rng, self.save_image = game, split, rng, save_image
        self.left = dict(left)
        self.rows: List[Dict[str, int]] = []
        self.ring: Deque[Tuple[int, np.ndarray]] = deque(maxlen=ring)
        self.pairs: List[Dict] = []
        self.names: Dict[int, str] = {}
        self.cur: Optional[str] = None       # the answer of the open episode
        self.start = 0

    def feed(self, row: Dict[str, int], image: Optional[np.ndarray]) -> None:
        k = len(self.rows)
        if k > 0 and image is None:
            raise ValueError("stream row %d has no image" % k)
        self.rows.append(row)
        if image is not None:
            self.ring.append((k, image))
        ans = M.movement(self.rows, k) if k >= FIRST_T else UNKNOWN
        if ans != self.cur:
            self._close(k - 1, cut_end=False)
            self.cur, self.start = ans, k

    def finish(self) -> List[Dict]:
        self._close(len(self.rows) - 1, cut_end=True)
        self.cur = None
        return list(self.pairs)

    def _close(self, e: int, cut_end: bool) -> None:
        ans, s = self.cur, self.start
        if ans is None or ans == UNKNOWN or e < s or self.left.get(ans, 0) <= 0:
            return
        oldest, newest = self.ring[0][0], self.ring[-1][0]
        lo = max(FIRST_T, oldest + 4 - LAG)            # the prev image u - 3 must still be in the ring
        hi = newest - LAG                               # the now image u + 1 must exist
        L = e - s + 1
        for u in sample_positions(s, e, lo, hi, self.rng):
            if self.left[ans] <= 0:
                break
            self.left[ans] -= 1
            k_prev, k_now = u - 4 + LAG, u + LAG
            self.pairs.append({
                "game": self.game, "split": self.split, "t": u, "k_prev": k_prev, "k_now": k_now, "answer": ans,
                "pos": u - s, "length": L, "stage": round((u - s) / L, 4), "stage_bin": stage_bin(u - s, L),
                "episode": [s, e], "long": s < lo, "cut_end": cut_end,
                "images": [self._image(k_prev), self._image(k_now)]})

    def _image(self, k: int) -> str:
        if k not in self.names:
            im = dict(self.ring).get(k)
            if im is None:
                raise AssertionError("capture %d is not in the ring" % k)
            self.names[k] = self.save_image(k, im)
        return self.names[k]


class CapturingBridge:
    """A MesenBridge that captures every frame of every RUN and hands (row, image) to ``sink`` in stream order:
    after ``load_state`` the first RUN's rows from 0 (row 0 without an image), later RUNs from row 1 (row 0 is the
    previous RUN's last row). The caller's own caps are kept, so play_round reads its frames as before."""

    def __init__(self, bridge, names: Optional[List[str]] = None):
        self.inner, self.names = bridge, names
        self.sink: Optional[Callable] = None
        self.fresh = True

    def load_state(self, state):
        self.fresh = True
        return self.inner.load_state(state)

    def run(self, frames, caps=(), p2=None):
        if self.sink is None:
            raise RuntimeError("CapturingBridge.run without a sink")
        n = len(frames)
        first = 0 if self.fresh else 1
        obs = self.inner.run(frames, caps=sorted(set(caps) | set(range(n + 1))), p2=p2)
        for k in range(first, n + 1):
            r = obs.rams[k]
            r = dict(zip(self.names, r)) if self.names else r
            self.sink(r, None if (self.fresh and k == 0) else obs.images[k])
        self.fresh = False
        return obs

    def __getattr__(self, name):
        return getattr(self.inner, name)
