"""stable-retro wrapper: one call = one Laya decision (a held input or a whole special-move macro).

Tracks rounds and matches from RAM so every script counts wins and damage the same way. An episode is one match
from the savestate: it ends when either side has two round wins, and ``reset()`` loads the savestate again, so the
opponent never changes under the gate.
"""
import gzip
import os
from collections import deque
from dataclasses import dataclass, field
from typing import Callable, Deque, List, Optional

import numpy as np

from . import actions as A
from . import ram
from .config import DEFAULT_STATE, FULL_HP, GAME, PREV_GAP

AIR_DY = 6          # |y - standing y| above this = airborne
INTRO_SKIP = 90     # frames after the life bars refill before a round really starts ("ROUND 2 ... FIGHT!")
MAX_WAIT = 1500     # safety cap while waiting through KO / time-over screens


@dataclass
class ActResult:
    frames: int
    dmg_for: int
    dmg_against: int
    round_over: bool = False
    winner: Optional[str] = None      # "me" | "opp" | "draw"


@dataclass
class Context:
    """What the scripted teacher is allowed to know besides the RAM snapshot."""
    my_air: bool
    opp_air: bool
    dx_trend: int                      # dx now minus dx 8 frames ago (negative = closing in)
    frames_since_hit: int              # since my life last dropped
    frames_since_fireball: int         # since I last threw a hadouken
    history: List[ram.Fighters] = field(default_factory=list)


class FightEnv:
    def __init__(self, state: str = DEFAULT_STATE, me: str = "ryu", opp: str = "guile", render: bool = False):
        import stable_retro as retro

        self.retro = retro
        self.me, self.opp = me, opp
        if os.path.isfile(state):  # a savestate you made yourself
            self.env = retro.make(GAME, use_restricted_actions=retro.Actions.ALL,
                                  render_mode="human" if render else "rgb_array")
            with gzip.open(state, "rb") as fh:
                self.env.initial_state = fh.read()
        else:
            self.env = retro.make(GAME, state=state, use_restricted_actions=retro.Actions.ALL,
                                  render_mode="human" if render else "rgb_array")
        ram.register(self.env)
        self.buttons: List[str] = list(self.env.buttons)
        self.episode = -1

    # ------------------------------------------------------------------ frames
    def reset(self) -> np.ndarray:
        obs, _ = self.env.reset()
        self.episode += 1
        self.frame_no = 0
        self.round = 0
        self.wins = {"me": 0, "opp": 0}
        self.last = "idle"
        self.done = False
        self.frames: Deque[np.ndarray] = deque([obs] * (PREV_GAP + 1), maxlen=PREV_GAP + 1)
        self.hist: Deque[ram.Fighters] = deque(maxlen=120)
        self.f = ram.read(self.env)
        self.hist.append(self.f)
        self.ground = (self.f.my_y, self.f.opp_y)
        self.last_hit = -10_000
        self.last_fireball = -10_000
        self.in_round = True
        return obs

    @property
    def frame(self) -> np.ndarray:
        return self.frames[-1]

    @property
    def prev_frame(self) -> np.ndarray:
        return self.frames[0]

    def step_frame(self, names: List[str]) -> None:
        obs, _, done, trunc, _ = self.env.step(A.to_array(names, self.buttons))
        self.frame_no += 1
        self.frames.append(obs)
        prev, self.f = self.f, ram.read(self.env)
        self.hist.append(self.f)
        if 0 <= self.f.my_hp < prev.my_hp:
            self.last_hit = self.frame_no
        self.done = self.done or done or trunc

    # ------------------------------------------------------------------ state for model / teacher
    def airborne(self):
        return abs(self.f.my_y - self.ground[0]) > AIR_DY, abs(self.f.opp_y - self.ground[1]) > AIR_DY

    def text(self) -> str:
        my_air, opp_air = self.airborne()
        return ram.text_state(self.f, self.me, self.opp, self.last, my_air, opp_air)

    def context(self) -> Context:
        my_air, opp_air = self.airborne()
        old = self.hist[-9] if len(self.hist) >= 9 else self.hist[0]
        return Context(my_air, opp_air, self.f.dx - old.dx, self.frame_no - self.last_hit,
                       self.frame_no - self.last_fireball, list(self.hist))

    # ------------------------------------------------------------------ one decision
    def act(self, action: str, on_frame: Optional[Callable[["FightEnv"], None]] = None) -> ActResult:
        facing = self.f.facing_right
        start = self.f
        if action == "hadouken":
            self.last_fireball = self.frame_no
        res = ActResult(0, 0, 0)
        for tokens in A.expand(action):
            before = self.f
            self.step_frame(A.to_physical(tokens, facing))
            res.frames += 1
            if on_frame:
                on_frame(self)
            over, winner = self._round_check(before)
            if over:
                res.round_over, res.winner = True, winner
                break
            if self.done:
                break
        # KO life is -1 (clamped to 0); a time-over refill makes the difference negative (clamped to 0)
        res.dmg_for = max(0, start.opp_hp - max(0, self.f.opp_hp))
        res.dmg_against = max(0, start.my_hp - max(0, self.f.my_hp))
        self.last = action
        return res

    def _round_check(self, before: ram.Fighters):
        f = self.f
        if f.my_hp < 0 or f.opp_hp < 0:  # KO
            winner = "draw" if f.my_hp < 0 and f.opp_hp < 0 else "me" if f.opp_hp < 0 else "opp"
        elif f.my_hp == FULL_HP and f.opp_hp == FULL_HP and (before.my_hp < FULL_HP or before.opp_hp < FULL_HP):
            # time over: the bars refilled without a KO; higher life won
            winner = "me" if before.my_hp > before.opp_hp else "opp" if before.opp_hp > before.my_hp else "draw"
        else:
            return False, None
        self.in_round = False
        if winner != "draw":
            self.wins[winner] += 1
        return True, winner

    def next_round(self) -> bool:
        """Advance through KO / time-over screens. Returns False when the match (episode) is over."""
        if self.wins["me"] >= 2 or self.wins["opp"] >= 2 or self.done:
            return False
        for _ in range(MAX_WAIT):
            self.step_frame([])
            if self.done:
                return False
            if self.f.my_hp == FULL_HP and self.f.opp_hp == FULL_HP:
                break
        for _ in range(INTRO_SKIP):
            self.step_frame([])
        self.round += 1
        self.last = "idle"
        self.ground = (self.f.my_y, self.f.opp_y)
        self.in_round = True
        return not self.done

    def close(self):
        self.env.close()
