"""Round 3 of docs/prereg_movement_finetunes.md: the projectile sampling trigger of the 2P collector
(scripts/collect_pairs.py --mode vs --shots).

Slots (confirmed before trusting them; scripts/probe_shots.py and the round-2 collection): shot1 (0x1000) is player
1's projectile, shot2 (0x1050) player 2's - in rollouts/pairs2p games 0-3, 255 of 258 projectile flights started while
their own slot's player was pressing a projectile word, and the probe's frames show each player's hadoken / sonic boom
/ yoga fire leaving that player. The game BLINKS some projectiles: while the slot is on, the slot's byte +0x3A
(0x103A / 0x108A) bit 0 set = not drawn on that frame (hadoken 2 on / 2 off, yoga fire 1 in 4 hidden, sonic boom never;
probe: a blue-pixel detector agreed with the bit on 102 / 102 hadoken frames of player 2 and 196 / 200 of player 1;
spawn: the first "on" row is drawn in the capture of the next row, lag 1). So a projectile is VISIBLE at displayed row
t when its slot is on and that bit is clear (``visible``); only visible rows are sampled.

The trigger (``ShotSampler``, a PairSampler; the movement sampling runs too, normally with per_game 0): a flight is a
run of rows with the slot on; when it ends its rows are split in thirds (start / middle / end of the whole flight) and
one visible row u is drawn at random per third (inside the image ring, with a capture u + 1); at most SHOT_PER_GAME
per (slot, stage) per game. Each sample records the thrower (the slot's character), side (the thrower's screen side by
x at row t, None when equal), the projectile word the thrower was pressing at the spawn row (spawn_word: Dhalsim's
yoga flame also uses the slot and is no fireball; the builder keeps hadoken / sonic boom / yoga fire only), the other
slot at t, and the images (captures t - 3 and t + 1: frames t - 4 and t). Every flight is listed in the game's
games.jsonl record ("flights") for the ownership report.
"""
import os
from typing import Dict, List, Optional, Tuple

from ..emu.ram import Var
from . import movement_collect_io as MIO
from . import pairs_collect_io as IO
from .movement_collect import FIRST_T, LAG, STAGES
from .pairs_collect import PairSampler, SLOTS

SHOT_VARS: List[Var] = [Var("shot1_hide", 0x103A, 1, False), Var("shot2_hide", 0x108A, 1, False)]
SHOT_NAMES = [v.name for v in SHOT_VARS]
PROJECTILES: Dict[str, Tuple[str, ...]] = {"ryu": ("hadoken",), "ken": ("hadoken",), "guile": ("sonic_boom",),
                                           "dhalsim": ("yoga_fire",)}
THROWERS = tuple(sorted(PROJECTILES))
SHOT_PER_GAME = 3


def is_projectile(char: str, word: Optional[str]) -> bool:
    return word is not None and word in PROJECTILES.get(char, ())


def visible(row: Dict[str, int], slot: int) -> bool:
    return bool(row["shot%d" % slot]) and not row["shot%d_hide" % slot] & 1


def side_of(row: Dict[str, int], slot: int) -> Optional[str]:
    me, other = row["p%d_x" % slot], row["p%d_x" % (3 - slot)]
    return None if me == other else "left" if me < other else "right"


class ShotSampler(PairSampler):
    def __init__(self, *args, shot_per_game: int = SHOT_PER_GAME, **kw):
        super().__init__(*args, **kw)
        if shot_per_game < 0:
            raise ValueError("shot_per_game %d < 0" % shot_per_game)
        self.shot_per_game = shot_per_game
        self.run_start: Dict[int, Optional[int]] = {1: None, 2: None}
        self.shot_taken: Dict[Tuple[int, str], int] = {}
        self.shots: List[Dict] = []
        self.flights: List[Dict] = []

    def feed(self, row: Dict[str, int], image) -> None:
        super().feed(row, image)
        k = len(self.rows) - 1
        for s in SLOTS:
            on = bool(row["shot%d" % s])
            if on and self.run_start[s] is None:
                self.run_start[s] = k
            elif not on and self.run_start[s] is not None:
                self._close_flight(s, self.run_start[s], k - 1, cut_end=False)
                self.run_start[s] = None

    def finish(self) -> List[Dict]:
        for s in SLOTS:
            if self.run_start[s] is not None:
                self._close_flight(s, self.run_start[s], len(self.rows) - 1, cut_end=True)
                self.run_start[s] = None
        return super().finish()

    def _close_flight(self, s: int, a: int, e: int, cut_end: bool) -> None:
        words = {str(p): self.pressed[p][a] for p in SLOTS}
        self.flights.append({"slot": s, "start": a, "end": e, "spawn_words": words, "thrower": self.chars[s],
                             "cut_end": cut_end})
        oldest, newest, n = self.ring[0][0], self.ring[-1][0], len(self.rows)
        lo = max(FIRST_T, oldest + 4 - LAG, a)
        hi = min(newest - LAG, e, n - 1 - LAG)
        length = e - a + 1
        for i, stage in enumerate(STAGES):
            if self.shot_taken.get((s, stage), 0) >= self.shot_per_game:
                continue
            third = [u for u in range(lo, hi + 1)
                     if 3 * (u - a) // length == i and visible(self.rows[u], s)]
            if not third:
                continue
            u = self.rng.choice(third)
            self.shot_taken[(s, stage)] = self.shot_taken.get((s, stage), 0) + 1
            r = self.rows[u]
            k_prev, k_now = u - 4 + LAG, u + LAG
            self.shots.append(dict(
                game=self.game, slot=s, thrower=self.chars[s], other=self.chars[3 - s], t=u, k_prev=k_prev,
                k_now=k_now, flight=[a, e], flight_len=length, pos=u - a, flight_stage=stage, cut_end=cut_end,
                spawn_word=words[str(s)], spawn_words=words, side=side_of(r, s), shot_x=r["shot%d_x" % s],
                thrower_x=r["p%d_x" % s], other_x=r["p%d_x" % (3 - s)], other_shot=r["shot%d" % (3 - s)],
                images=[self._image(k_prev), self._image(k_now)]))


def committed_shots(base: str) -> Tuple[List[Dict], int]:
    """(shot samples of committed games, samples dropped as uncommitted)."""
    done = {g["game"] for g in IO.committed(base)}
    rows = MIO.read_jsonl(os.path.join(base, "shots.jsonl"))
    kept = [r for r in rows if r.get("game") in done]
    return kept, len(rows) - len(kept)
