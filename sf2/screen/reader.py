"""The screen reader: RGB frame(s) -> ScreenFacts. Input is ONLY the image (256x224 RGB, HUD visible; optionally the
frame before) plus fixed files (sf2.screen.assets, sf2.screen.hud). No RAM, no OAM, no emulator.

    lock = lock_round(first_frame)            # identify both characters among all 8 catalogs, lock them
    facts = read_screen(frame, lock)          # pure: same inputs, same facts
    facts, track = step_round(facts, track)   # pure: round over across frames (track: a RoundTrack, None at first)
    rr = RoundReader(); facts = rr.feed(frame) # convenience: locks at a round start, tracks round over, re-locks

Round start: the frame a round begins on (both fighters standing at their start places). identify() matches every
character's own colours on the left and the right half; the best character per half (two different characters) wins.
lock_round() also drops from each character's LOCATING colours the other fighter's colours and every colour common in
the background of that frame (>= BG_SHARE of the pixels outside both fighters): the stage's own palette.

Round over (gate amendment, owner 2026-10-02): a bar that looks empty is NOT a cue (2 hp per bar pixel: 1 hp looks
empty). read_screen (single frame): "over" when the clock reads 00. step_round (pure, the state is a RoundTrack passed in
and returned) / RoundReader: the round is over when
  - the clock has read 00 for TIME_OVER_FRAMES frames (time over), or
  - the NEXT round visibly starts: both bars full and both fighters within START_TOL px of their start places
    (START_X), after the round progressed: a bar below full or a non-fight screen (no bars: the black between two
    rounds) since the last start. Not the clock: in a new round both fighters stand at their places while it runs.
That frame carries new_round=True and round_state "over" (the played round ended); RoundReader re-locks on it, and
the frames after it are "fighting" again. A KO is therefore reported when the next round starts (inputs after a KO do
nothing, so late is free).
"""
from collections import Counter
from dataclasses import dataclass, replace
from typing import Dict, List, Optional, Tuple

import numpy as np

from . import assets as A
from .facts import FighterFacts, ProjectileFacts, ScreenFacts
from .hud import load_digits, read_hud
from .unknown_log import UnknownLog
from .match import Match, centroids, locate, match, padded_codes, padded_mask

COMMON_SHARE = 4           # a colour in >= this many other characters' palettes is shared (hit flash): never locates
BG_SHARE = 0.002
BG_ROW0 = 16               # rows above are blank (the HUD below it counts as background: never a fighter colour)
DEFAULT_ACTION = "block"   # owner 2026-10-02: no confident match -> answer "block", flag unknown, log it
UNKNOWN_SCORE = 0.3          # below: the true sprite is mostly not in the catalog (gate set b, 2026-10-02)
PROJ_SCORE = 0.7
PROJ_MIN_PEAK = 6
AIR_Y = -1                 # (off) a sprite whose feet are above this screen row is in the air whatever its label says
TIME_OVER_FRAMES = 30
START_X = (80, 176)        # screen x of player 1 / player 2 at a round start (RAM x 208 / 304, camera 128; probe 2026-10-02)
START_TOL = 8              # px: "back near the start place"
START_FREE_X, START_HEAD_Y = 40, 105
MIN_ON_SCREEN = 0.8
HALF = A.W // 2


@dataclass(frozen=True)
class RoundLock:
    chars: Tuple[str, str]                 # (player 1: left at the round start, player 2)
    locating: Tuple[np.ndarray, np.ndarray]   # bool[32768] per character
    proj_idx: np.ndarray                   # projectile templates the two fighters can throw
    proj_owner: Tuple[str, ...]            # owner character per proj_idx entry
    proj_locating: np.ndarray              # bool[32768]
    cent: Tuple[np.ndarray, np.ndarray] = (None, None)   # per character: template centroids of locating colours
    id_scores: Tuple[Tuple[str, str, float], ...] = ()


def _lut(colours) -> np.ndarray:
    lut = np.zeros(32768, bool)
    if colours:
        lut[np.fromiter(colours, np.int64)] = True
    return lut


def own_colours(banks: Dict[str, A.Bank]) -> Dict[str, frozenset]:
    """Each character's palette minus the colours COMMON_SHARE or more other characters also use."""
    use = Counter(c for ch in A.CHARS for c in banks[ch].palette)
    return {ch: frozenset(c for c in banks[ch].palette if use[c] - 1 < COMMON_SHARE) for ch in A.CHARS}


def _find(bank: A.Bank, codes: np.ndarray, code_pad: np.ndarray, lut: np.ndarray,
          cols: Optional[Tuple[int, int]] = None, cent: Optional[np.ndarray] = None
          ) -> Tuple[Optional[Match], Optional[object]]:
    mask = lut[codes]
    blob = locate(mask, cols)
    if blob is None:
        return None, None
    return match(bank, code_pad, padded_mask(mask), blob, cent=cent), blob


def start_background(codes: np.ndarray) -> set:
    """Colours common in the parts of a round-start frame no fighter covers (fighters start at screen x ~60-200,
    feet at ~195, heads below ~105): the side strips and the band above their heads."""
    bg = np.zeros(codes.shape, bool)
    bg[BG_ROW0:, :START_FREE_X] = bg[BG_ROW0:, A.W - START_FREE_X:] = True
    bg[BG_ROW0:START_HEAD_Y, :] = True
    cnt = Counter(codes[bg].tolist())
    return {c for c, n in cnt.items() if n >= BG_SHARE * bg.sum()}


def identify(frame: np.ndarray, banks: Optional[Dict[str, A.Bank]] = None):
    """((left character, right character), {(char, side): (score, Match)}) at a round start: each character's own
    colours (minus the start background's) located in each half, matched; a match counts only with most of the sprite
    on the screen (MIN_ON_SCREEN) and its centre in that half."""
    banks = banks or A.load_banks()
    codes = A.codes15(frame)
    code_pad = padded_codes(codes)
    own = own_colours(banks)
    common = start_background(codes)
    found = {}
    for ch in A.CHARS:
        lut = _lut(own[ch] - common)
        for side, cols in (("left", (0, HALF)), ("right", (HALF, A.W))):
            m, _ = _find(banks[ch], codes, code_pad, lut, cols, centroids(banks[ch], lut))
            ok = m is not None and m.visible >= MIN_ON_SCREEN * banks[ch].templates[m.t].n and \
                cols[0] <= m.ox + banks[ch].templates[m.t].w // 2 < cols[1]        # the sprite's centre is there
            found[(ch, side)] = (m.score if ok else 0.0, m)
    left = max(A.CHARS, key=lambda c: found[(c, "left")][0])
    right = max((c for c in A.CHARS if c != left), key=lambda c: found[(c, "right")][0])
    return (left, right), found


def lock_round(frame: np.ndarray, banks: Optional[Dict[str, A.Bank]] = None) -> RoundLock:
    banks = banks or A.load_banks()
    (c1, c2), found = identify(frame, banks)
    codes = A.codes15(frame)
    bg = np.ones(codes.shape, bool)
    bg[:BG_ROW0] = False
    for ch, side in ((c1, "left"), (c2, "right")):
        m = found[(ch, side)][1]
        if m is not None:
            t = banks[ch].templates[m.t]
            bg[max(0, m.oy):max(0, m.oy + t.h), max(0, m.ox):max(0, m.ox + t.w)] = False
    cnt = Counter(codes[bg].tolist())
    common = {c for c, n in cnt.items() if n >= BG_SHARE * bg.sum()}
    own = own_colours(banks)
    loc1 = _lut(own[c1] - banks[c2].palette - common)
    loc2 = _lut(own[c2] - banks[c1].palette - common)
    owners = A.projectile_owners()
    pb = banks.get(A.PROJ)
    idx, own_by = [], []
    if pb is not None:
        for i, t in enumerate(pb.templates):
            owner, shot = owners.get(t.ck, ("?", False))
            if shot and owner in (c1, c2):
                idx.append(i)
                own_by.append(owner)
    pcol = set()
    for i in idx:
        pcol.update(pb.full[i][1].tolist())
    ploc = _lut(pcol - common)          # may share fighter colours: the fighters' boxes are masked out per frame
    scores = tuple((ch, side, round(s, 4)) for (ch, side), (s, _) in sorted(found.items()))
    return RoundLock((c1, c2), (loc1, loc2), np.array(idx, np.int64), tuple(own_by), ploc,
                     cent=(centroids(banks[c1], loc1), centroids(banks[c2], loc2)), id_scores=scores)


def _fighter(ch: str, player: int, m: Optional[Match], banks, health) -> FighterFacts:
    if m is None:
        return FighterFacts("?", ch, False, None, None, None, None, None, DEFAULT_ACTION, 0.0, True, None, health,
                            player)
    t = banks[ch].templates[m.t]
    x = int(round(m.ox - t.dx))
    feet = m.oy + t.h
    unknown = m.score < UNKNOWN_SCORE or t.act2 is None
    in_air = (t.air == "air") or feet < AIR_Y
    return FighterFacts("?", ch, True, x, int(feet), bool(in_air), "right" if t.face == A.FACE_RIGHT else "left",
                        t.ck, DEFAULT_ACTION if unknown else t.act2, round(m.score, 4), unknown,
                        (m.ox, m.oy, m.ox + t.w, m.oy + t.h), health, player)


def _projectiles(lock: RoundLock, banks, codes, code_pad, sides: Dict[str, str],
                 boxes: List[Tuple[int, int, int, int]]) -> Tuple[ProjectileFacts, ...]:
    """Up to two thrown shots of the two fighters: their colours (not the stage's) outside both fighters' boxes."""
    if len(lock.proj_idx) == 0:
        return ()
    mask = lock.proj_locating[codes]
    for x0, y0, x1, y1 in boxes:
        mask[max(0, y0):max(0, y1), max(0, x0):max(0, x1)] = False
    if mask.sum() < 8:
        return ()
    pb = banks[A.PROJ]
    out: List[ProjectileFacts] = []
    for _ in range(2):
        blob = locate(mask, min_peak=PROJ_MIN_PEAK)
        if blob is None:
            break
        m = match(pb, code_pad, padded_mask(mask), blob, lock.proj_idx)
        mask = mask.copy()
        mask[blob.y0:blob.y1, blob.x0:blob.x1] = False
        if m is None or m.score < PROJ_SCORE:
            continue
        t = pb.templates[m.t]
        owners = {lock.proj_owner[j] for j, i in enumerate(lock.proj_idx) if pb.templates[i].ck == t.ck}
        side = sides.get(next(iter(owners))) if len(owners) == 1 else None
        out.append(ProjectileFacts(m.ox + t.w // 2, m.oy + t.h // 2, side, t.ck, round(m.score, 4)))
    return tuple(out)


def by_side(f1: FighterFacts, f2: FighterFacts) -> Tuple[FighterFacts, FighterFacts]:
    """(left, right): the smaller x is on the left; one fighter not found: the found one's half decides."""
    if f1.found and f2.found:
        p1_left = f1.x <= f2.x
    elif f1.found or f2.found:
        p1_left = (f1.x < HALF) if f1.found else (f2.x >= HALF)
    else:
        p1_left = True
    left, right = (f1, f2) if p1_left else (f2, f1)
    return replace(left, side="left"), replace(right, side="right")


def read_screen(frame: np.ndarray, lock: RoundLock, prev: Optional[np.ndarray] = None,
                banks: Optional[Dict[str, A.Bank]] = None) -> ScreenFacts:
    """The facts of ``frame`` for the locked round. ``prev`` (the frame before) is accepted for the interface and not
    used by this reader."""
    if frame.shape != (A.H, A.W, 3) or frame.dtype != np.uint8:
        raise ValueError("expected a (224, 256, 3) uint8 frame, got %s %s" % (frame.shape, frame.dtype))
    banks = banks or A.load_banks()
    codes = A.codes15(frame)
    code_pad = padded_codes(codes)
    hud = read_hud(frame, load_digits())
    fs = []
    for i, ch in enumerate(lock.chars):
        m, _ = _find(banks[ch], codes, code_pad, lock.locating[i], cent=lock.cent[i])
        fs.append(_fighter(ch, i + 1, m, banks, hud.facts.health[i]))
    left, right = by_side(*fs)
    sides = {left.character: "left", right.character: "right"}
    proj = _projectiles(lock, banks, codes, code_pad, sides, [f.box for f in (left, right) if f.box is not None])
    over = hud.facts.timer == 0             # single frame: the clock at 00 (a bar that looks empty is not a cue)
    gap = right.x - left.x if left.found and right.found else None
    return ScreenFacts(left, right, proj, hud.facts, "over" if over else "fighting", gap)


@dataclass(frozen=True)
class RoundTrack:
    """What step_round carries from frame to frame (immutable: step_round returns a new one)."""
    zeros: int = 0                  # consecutive frames the clock has read 00
    time_over: bool = False         # the time-over rule fired: "over" until the next round starts
    progressed: bool = False        # a bar below full / no bars since the round's start (bars never refill in a round)


def is_round_start(facts: ScreenFacts) -> bool:
    """Both bars full and both fighters found within START_TOL px of their start places (player 1 on the left)."""
    if facts.hud.health != (1.0, 1.0):
        return False
    for f in (facts.left, facts.right):
        if not f.found or f.player not in (1, 2) or abs(f.x - START_X[f.player - 1]) > START_TOL:
            return False
    return True


def step_round(facts: ScreenFacts, track: Optional[RoundTrack] = None) -> Tuple[ScreenFacts, RoundTrack]:
    """Pure: (this frame's facts from read_screen, the track so far) -> (the facts with round_state / new_round over
    frames, the next track). ``track`` None: the first frame seen (a round start there is the CURRENT round)."""
    track = track or RoundTrack()
    hud = facts.hud
    if track.progressed and is_round_start(facts):
        return replace(facts, round_state="over", new_round=True), RoundTrack()
    bars_moved = any(h is None or h < 1.0 for h in hud.health)
    zeros = track.zeros + 1 if hud.timer == 0 else 0
    nxt = RoundTrack(zeros=zeros, time_over=track.time_over or zeros >= TIME_OVER_FRAMES,
                     progressed=track.progressed or bars_moved)
    return replace(facts, round_state="over" if nxt.time_over else "fighting", new_round=False), nxt


class RoundReader:
    """Locks the characters on the first frame of a round and reads every frame after; tracks round over across
    frames with step_round (time-over rule, the next round visibly starting) and re-locks on a new round. Unknown
    fighters (answered DEFAULT_ACTION, "block") are saved to ``log`` for a later catalog top-up."""

    def __init__(self, banks: Optional[Dict[str, A.Bank]] = None, log: Optional[UnknownLog] = None):
        """``log``: where unknown fighters are saved (sf2.screen.unknown_log; None: not logged)."""
        self.banks = banks or A.load_banks()
        self.log = log
        self.frames = 0
        self.lock: Optional[RoundLock] = None
        self.track = RoundTrack()
        self.prev: Optional[np.ndarray] = None

    @property
    def over(self) -> bool:
        return self.track.time_over

    @property
    def zeros(self) -> int:
        return self.track.zeros

    def feed(self, frame: np.ndarray) -> ScreenFacts:
        if self.lock is None:
            self.lock = lock_round(frame, self.banks)
        facts = read_screen(frame, self.lock, self.prev, self.banks)
        if self.log is not None:
            self.log.add(frame, facts, self.frames)
        self.frames += 1
        self.prev = frame
        facts, self.track = step_round(facts, self.track)
        if facts.new_round:
            self.lock = lock_round(frame, self.banks)
        return facts
