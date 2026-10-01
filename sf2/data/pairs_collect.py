"""The movement-pairs collector's core (docs/prereg_movement_pairs.md, "Fallback"): the directed player 1 and the
two-fighter episode sampler. No files here (sf2.data.pairs_collect_io).

Frames, as sf2.data.movement_collect: every frame of a game is captured (CapturingBridge); stream index k is RAM row k
and the image captured at that boundary, which shows row k - LAG. A ring keeps the last RING images.

Sampler. For EACH fighter (slot 1 = the directed player 1, slot 2 = the CPU), an episode is a run of displayed rows
with one key (movement, direction, facing) (sf2.data.pairs_labels.episode_key; an unknown breaks runs and is never
sampled). When an episode ends its pairs are drawn as in movement_collect (one position per third: start / middle /
end; one for an episode shorter than SHORT): a pair for displayed row u is the images captured at u - 3 (showing
u - 4) and u + 1 (showing u). At most PER_GAME pairs per (slot, key) per game. The global cap per (character,
controller, movement, direction, facing) is the builder's (sf2.data.pairs_data), over all games.

Directed play (``play_directed``): from the pair's savestate (a random 4-43 idle frames first, so games differ), every
time player 1 can act (standing or crouching, on the ground) the next word of the cycle is pressed, then the game runs
until player 1 can act again; the round ends on the ROM's result. Each move is logged as [word, k_start, k_end].
"""
import random
from collections import deque
from typing import Callable, Deque, Dict, List, Optional, Tuple

import numpy as np

from ..config import PAD
from ..emu.vs import GROUND_Y, physical
from . import pairs_labels as L
from . import pairs_moves as PM
from .movement_collect import FIRST_T, LAG, RING, SHORT, sample_positions, stage_bin

PER_GAME = 3                      # pairs per (slot, key) per game
SLOTS = {1: "directed", 2: "cpu"}
WAIT = 4                          # idle frames per step while player 1 cannot act
MAX_RECOVER = 120                 # frames to wait after a move for player 1 to act again
MAX_FRAMES = 12000                # a round cannot last longer (99 s clock plus the KO)


class PairSampler:
    """One game: feed every stream row (and its image; None only for row 0) in order, then ``finish``."""

    def __init__(self, game: int, chars: Dict[int, str], rng: random.Random,
                 save_image: Callable[[int, np.ndarray], str], bands: Dict[str, int], ring: int = RING,
                 per_game: int = PER_GAME):
        if ring < 5:
            raise ValueError("a ring of %d frames cannot hold a (t - 4, t) pair" % ring)
        if per_game < 0:
            raise ValueError("per_game %d < 0" % per_game)
        self.game, self.chars, self.rng, self.save_image = game, dict(chars), rng, save_image
        self.bands, self.per_game = bands, per_game
        self.rows: List[Dict[str, int]] = []
        self.ring: Deque[Tuple[int, np.ndarray]] = deque(maxlen=ring)
        self.pairs: List[Dict] = []
        self.names: Dict[int, str] = {}
        self.taken: Dict[Tuple[int, Tuple[str, str, str]], int] = {}
        self.cur: Dict[int, Optional[Tuple[str, str, str]]] = {1: None, 2: None}
        self.start: Dict[int, int] = {1: 0, 2: 0}

    def feed(self, row: Dict[str, int], image: Optional[np.ndarray]) -> None:
        k = len(self.rows)
        if k > 0 and image is None:
            raise ValueError("stream row %d has no image" % k)
        self.rows.append(row)
        if image is not None:
            self.ring.append((k, image))
        for p in SLOTS:
            key = L.episode_key(self.rows, k, p) if k >= FIRST_T else None
            if key != self.cur[p]:
                self._close(p, k - 1, cut_end=False)
                self.cur[p], self.start[p] = key, k

    def finish(self) -> List[Dict]:
        for p in SLOTS:
            self._close(p, len(self.rows) - 1, cut_end=True)
            self.cur[p] = None
        return list(self.pairs)

    def _close(self, p: int, e: int, cut_end: bool) -> None:
        key, s = self.cur[p], self.start[p]
        if key is None or e < s or self.taken.get((p, key), 0) >= self.per_game:
            return
        oldest, newest = self.ring[0][0], self.ring[-1][0]
        lo = max(FIRST_T, oldest + 4 - LAG)            # the prev image u - 3 must still be in the ring
        hi = newest - LAG                               # the now image u + 1 must exist
        n = len(self.rows)
        for u in sample_positions(s, e, lo, hi, self.rng):
            if self.taken.get((p, key), 0) >= self.per_game or u + LAG >= n:
                break
            self.taken[(p, key)] = self.taken.get((p, key), 0) + 1
            lab = L.labels(self.rows, u, p, self.bands)
            k_prev, k_now = u - 4 + LAG, u + LAG
            self.pairs.append(dict(
                lab, game=self.game, slot=p, controller=SLOTS[p], char=self.chars[p], opp=self.chars[3 - p],
                t=u, k_prev=k_prev, k_now=k_now, pos=u - s, length=e - s + 1,
                stage=round((u - s) / (e - s + 1), 4), stage_bin=stage_bin(u - s, e - s + 1), episode=[s, e],
                long=s < lo, cut_end=cut_end, images=[self._image(k_prev), self._image(k_now)]))

    def _image(self, k: int) -> str:
        if k not in self.names:
            im = dict(self.ring).get(k)
            if im is None:
                raise AssertionError("capture %d is not in the ring" % k)
            self.names[k] = self.save_image(k, im)
        return self.names[k]


# ---- directed play ------------------------------------------------------------------------------------------------

def can_act(r: Dict[str, int]) -> bool:
    return r["p1_state"] in (0, 2) and r["p1_y"] == GROUND_Y


def press_frames(char: str, word: str, r: Dict[str, int]) -> List[List[str]]:
    """The physical buttons per frame for ``word`` from row ``r``: F / B from x for the walks and jumps, from the
    facing byte otherwise (as System 1's _act); at least WAIT frames."""
    right = (r["p1_x"] < r["p2_x"]) if word in PM.BY_X else r["p1_facing"] == 0x40
    steps = [t for toks, n in PM.moves(char)[word] for t in [toks] * n]
    frames = [physical(t, right, PAD) for t in steps]
    return frames + [[]] * max(0, WAIT - len(frames))


RESULT = {1: "win", 2: "loss", 0xFF: "draw"}


def play_directed(bridge, names: List[str], char: str, cycle: PM.Cycle, state: bytes, rng: random.Random,
                  stream_len: Callable[[], int], max_frames: int = MAX_FRAMES) -> Dict:
    """One game (a round) from ``state``. ``bridge`` (a CapturingBridge) feeds the sampler, whose row count
    ``stream_len`` gives the stream index of the last row; ``names`` the RAM row's names. The summary with the move
    log."""
    def run(frames):
        return [dict(zip(names, x)) for x in bridge.run(frames).rams]

    bridge.load_state(state)
    rows = run([[]] * (4 + rng.randrange(40)))
    frames, log = 0, []
    while True:
        r = rows[-1]
        if r["result"] or frames >= max_frames:
            break
        if not can_act(r):
            rows = run([[]] * WAIT)
            frames += WAIT
            continue
        word = cycle.next()
        k0 = stream_len() - 1
        pressed = press_frames(char, word, r)
        rows = run(pressed)
        frames += len(pressed)
        for _ in range(MAX_RECOVER // WAIT):
            if can_act(rows[-1]) or rows[-1]["result"]:
                break
            rows = run([[]] * WAIT)
            frames += WAIT
        log.append([word, k0, stream_len() - 1])
    res = r["result"]
    return {"result": RESULT.get(res, "result_%d" % res) if res else "unfinished",
            "frames": frames, "moves": log, "cycle_rounds": cycle.rounds}
