"""The fight environment on top of the Mesen bridge: one call = one Laya decision.

A decision's whole input sequence (a held direction, a button tap, or a special-move macro) goes to Mesen as one
RUN; Mesen sends back the RAM variables for every frame plus two screenshots (4 frames before the end, and the
end). Rounds and matches are tracked from the life values. An episode is one match from the savestate: it ends
when either side has two round wins, and ``reset()`` reloads the savestate, so the opponent never changes under
the gate.
"""
from collections import deque
from dataclasses import dataclass, field, replace
from typing import Callable, Deque, List, Optional

import numpy as np

from . import actions as A
from . import ram
from .config import PREV_GAP

AIR_DY = 6          # |y - standing y| above this = airborne
INTRO_SKIP = 182    # frames from the life bars refilling to the first decision: input first moves her at +183..185
MAX_WAIT = 1200     # round end to refill: 600-940 frames on the ROM; the next opponent refills at ~1354
MAX_ROUNDS = 4      # the 4th round is the "FINAL ROUND" on the ROM: no 5th, even after draws
WAIT_CHUNK = 30


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
    """``backend``: a MesenBridge (or a test fake) with set_vars / run / load_state."""

    def __init__(self, backend, ram_map: List[ram.Var], savestate: bytes, me: str = "ryu", opp: str = "ken",
                 jitter: int = 0, jitter_base: int = 0):
        self.backend = backend
        self.names = [v.name for v in ram_map]
        backend.set_vars(ram_map)
        self.savestate = savestate
        self.me, self.opp = me, opp
        self.episode = -1
        self.done = False
        self.jitter, self.jitter_base = jitter, jitter_base

    # ------------------------------------------------------------------ helpers
    def _f(self, values) -> ram.Fighters:
        return ram.Fighters.from_values(self.names, values)

    def _ingest(self, rows: List[List[int]]) -> List[ram.Fighters]:
        """RAM rows *after* each executed frame -> Fighters, updating history / hit timer."""
        out = []
        for values in rows:
            prev, self.f = self.f, self._f(values)
            self.frame_no += 1
            self.hist.append(self.f)
            if 0 <= self.f.my_hp < prev.my_hp:
                self.last_hit = self.frame_no
            out.append(self.f)
        return out

    def run_frames(self, frames: List[List[str]], capture: bool = True) -> List[ram.Fighters]:
        n = len(frames)
        caps = {max(0, n - PREV_GAP), n} if capture else set()
        obs = self.backend.run(frames, caps)
        fs = self._ingest(obs.rams[1:])
        if capture:
            cur = obs.images[n]
            prev = obs.images.get(n - PREV_GAP, self.frame)
            self.frames.extend([prev, cur])
        return fs

    # ------------------------------------------------------------------ episode
    def reset(self) -> np.ndarray:
        obs = self.backend.load_state(self.savestate)
        self.episode += 1
        self.frame_no = 0
        self.round = 0
        self.wins = {"me": 0, "opp": 0}
        self.last = "idle"
        img = obs.images[0]
        self.frames: Deque[np.ndarray] = deque([img, img], maxlen=2)
        self.hist: Deque[ram.Fighters] = deque(maxlen=120)
        self.f = self._f(obs.rams[0])
        self.hist.append(self.f)
        self.full_hp = max(self.f.my_hp, self.f.opp_hp)  # the savestate starts with full bars
        self.ground = (self.f.my_y, self.f.opp_y)
        self.last_hit = -10_000
        self.last_fireball = -10_000
        self.in_round = True
        if self.jitter:  # a different idle count per worker (jitter_base) and match: a different CPU fight
            self.run_frames([[]] * (self.jitter_base + self.episode % self.jitter + 1))
            self.frame_no = 0
        return self.frame

    @property
    def frame(self) -> np.ndarray:
        return self.frames[-1]

    @property
    def prev_frame(self) -> np.ndarray:
        return self.frames[0]

    # ------------------------------------------------------------------ state for model / teacher
    def airborne(self):
        """Off the ground by choice (a jump or jump attack): being knocked into the air (0E), lifted and thrown
        (00 at y 136, then 14) or falling after a KO (00) does not count."""
        f = self.f
        air = (None, ram.JUMP_STATE, ram.ATTACK_STATE)
        return (abs(f.my_y - self.ground[0]) > AIR_DY and f.my_state in air,
                abs(f.opp_y - self.ground[1]) > AIR_DY and f.opp_state in air)

    def controllable(self) -> bool:
        """Does the stick do anything now? Not while she is hit, thrown or knocked down (state 0E), or between
        rounds. Such decisions stay in rollouts (gate, damage) but not in training data."""
        return self.in_round and self.f.my_state != ram.HIT_STATE

    def text(self) -> str:
        my_air, opp_air = self.airborne()
        return ram.text_state(self.f, self.me, self.opp, self.last, my_air, opp_air, self.full_hp)

    def context(self) -> Context:
        my_air, opp_air = self.airborne()
        old = self.hist[-9] if len(self.hist) >= 9 else self.hist[0]
        return Context(my_air, opp_air, self.f.dx - old.dx, self.frame_no - self.last_hit,
                       self.frame_no - self.last_fireball, list(self.hist))

    # ------------------------------------------------------------------ one decision
    def act(self, action: str, on_frame: Optional[Callable[[ram.Fighters], None]] = None) -> ActResult:
        facing = self.f.facing_right
        start = self.f
        if action == "hadouken":
            self.last_fireball = self.frame_no
        frames = [A.to_physical(t, facing) for t in A.expand(action)]
        res = ActResult(len(frames), 0, 0)
        before = start
        for f in self.run_frames(frames):
            if on_frame:
                on_frame(f)
            over, winner, judged = self._round_check(before, f)
            before = f
            if over:
                res.round_over, res.winner = True, winner
                self.f = judged  # judge damage at the deciding frame
                break
        # KO life is clamped to zero on the SNES ROM; a time-over refill makes the difference negative.
        res.dmg_for = max(0, start.opp_hp - max(0, self.f.opp_hp))
        res.dmg_against = max(0, start.my_hp - max(0, self.f.my_hp))
        self.last = action
        return res

    def _round_check(self, before: ram.Fighters, f: ram.Fighters):
        full = self.full_hp
        judged = f
        if f.result is not None:
            # the ROM's own round result (0x1ACF): 1 Chun-Li, 2 Dhalsim, FF draw, set on the KO frame or 30 frames
            # after the clock shows 00. The rules below infer it from the life bars when the map lacks it.
            if not f.result or before.result:
                return False, None, f
            winner = {1: "me", 2: "opp"}.get(f.result, "draw")
            if f.timer and winner != "draw":  # KO: the loser's bar is still draining to zero
                judged = replace(f, **{"opp_hp" if winner == "me" else "my_hp": 0})
        elif f.my_hp <= 0 and f.opp_hp <= 0 < min(before.my_hp, before.opp_hp):
            # both bars emptied on the same frame: the timer ran out and the ROM zeroed them; higher life won
            winner = "me" if before.my_hp > before.opp_hp else "opp" if before.opp_hp > before.my_hp else "draw"
            judged = before
        elif f.my_hp <= 0 or f.opp_hp <= 0:  # KO
            winner = "draw" if f.my_hp <= 0 and f.opp_hp <= 0 else "me" if f.opp_hp <= 0 else "opp"
        elif f.timer == 0 and before.timer:
            # time over: the ROM runs its end-of-round screens for ~480 frames before zeroing both bars
            winner = "me" if f.my_hp > f.opp_hp else "opp" if f.opp_hp > f.my_hp else "draw"
        elif f.my_hp == full and f.opp_hp == full and (before.my_hp < full or before.opp_hp < full):
            # the bars refilled without a KO (time over, or a cart that stops at 0): higher life won
            winner = "me" if before.my_hp > before.opp_hp else "opp" if before.opp_hp > before.my_hp else "draw"
        else:
            return False, None, f
        self.in_round = False
        if winner != "draw":
            self.wins[winner] += 1
        return True, winner, judged

    def next_round(self) -> bool:
        """Advance through KO / time-over screens. Returns False when the match (episode) is over."""
        if self.wins["me"] >= 2 or self.wins["opp"] >= 2 or self.done or self.round + 1 >= MAX_ROUNDS:
            return False
        waited = 0
        while True:
            fs = self.run_frames([[]] * WAIT_CHUNK, capture=False)
            waited += WAIT_CHUNK
            full = [i for i, f in enumerate(fs) if f.my_hp == self.full_hp and f.opp_hp == self.full_hp]
            if full:
                break
            if waited >= MAX_WAIT:  # bars never refilled (continue screen, match over): end the episode
                print("next_round: life bars did not refill within %d frames; ending the match" % MAX_WAIT,
                      flush=True)
                return False
        self.run_frames([[]] * (INTRO_SKIP - (len(fs) - 1 - full[0])))   # counted from the refill frame
        self.round += 1
        self.last = "idle"
        self.ground = (self.f.my_y, self.f.opp_y)
        self.in_round = True
        return not self.done

    def close(self):
        self.backend.close()
