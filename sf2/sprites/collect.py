"""One pair's sprite collection (scripts/collect_sprites.py): 2P versus, both controllers driven by their own
character's move list (sf2.data.pairs_collect.play_both, the Plan B collector), stepped one frame at a time with the
sprite snapshot (sf2.sprites.emu.SnapBridge). No labels here: the RAM rows are stored and the labels are joined
later (sf2.sprites.build), so the lag can be checked.

<out>/_pairs/<A>_vs_<B>/
  sprites/<key>.png        each canonical sprite the first time this pair sees it (fighters and projectiles)
  snaps/<key>.npz          the snapshot of that first sighting (oam, cg, vram, mode/base/off) - for re-render checks
  g<game>.npz              rows (int32, one per stream row, columns = names), pressed class per row and slot
  g<game>_frames.jsonl.gz  one line per (row k >= 1, group): k, group (p1 / p2 / pal5 / pal7), key, index_key,
                           bbox, size, oam_facing (from the entries' hflip), pals (palette -> entry count, all groups)
  done.json                the game summary (result, frames, moves, seconds)
"""
import gzip
import json
import os
import random
import time
from collections import Counter
from typing import Dict, List, Optional

import numpy as np
from PIL import Image

from ..data import pairs_collect as PC
from ..data import pairs_moves as PM
from ..data.action_codes import EXTRA_NAMES, EXTRA_VARS
from ..emu.ram import Var
from ..emu.vs import NAMES, VARS
from ..vocab import IDS
from .catalog import FACE_LEFT, FACE_RIGHT, canonical, key_of
from .emu import SnapBridge, open_mesen
from .oam import Snapshot, entries, render_groups

RAW = [Var("p%d_raw%02x" % (p, o), 0x0C00 + 0x200 * (p - 1) + o, 1, False) for p in (1, 2) for o in range(0x80)]
ALL_VARS = VARS + EXTRA_VARS + RAW
ALL_NAMES = NAMES + EXTRA_NAMES + [v.name for v in RAW]
GROUPS = {4: "p1", 6: "p2", 5: "pal5", 7: "pal7"}     # palette -> group name (fighters and projectile / spark groups)
FIGHTER_SLOT = {"p1": 1, "p2": 2}


def oam_facing(snap: Snapshot, pal: int) -> Optional[int]:
    """FACE_RIGHT when most of the group's entries are h-flipped (this game draws a fighter facing right flipped),
    FACE_LEFT when most are not; None on a tie."""
    flips = [e.hflip for e in entries(snap) if e.pal == pal]
    n = sum(flips)
    if 2 * n == len(flips):
        return None
    return FACE_RIGHT if 2 * n > len(flips) else FACE_LEFT


class PairSink:
    """Feeds on (row, snapshot) in stream order; writes sprites / snaps once per new key."""

    def __init__(self, base: str, chars: Dict[int, str]):
        self.base, self.chars = base, chars
        for d in ("sprites", "snaps"):
            os.makedirs(os.path.join(base, d), exist_ok=True)
        self.seen = {k[:-4] for k in os.listdir(os.path.join(base, "sprites"))}
        self.new_game()

    def new_game(self):
        self.rows: List[List[int]] = []
        self.frames: List[Dict] = []
        self.word: Dict[int, Optional[tuple]] = {1: None, 2: None}
        self.pressed: Dict[int, List[Optional[str]]] = {1: [], 2: []}

    def press(self, p: int, word: str, k0: int) -> None:
        if k0 != len(self.rows) - 1:
            raise ValueError("a word starts on the last row fed (%d), not %d" % (len(self.rows) - 1, k0))
        self.word[p] = (word, k0)

    def feed(self, row: Dict[str, int], snap: Optional[Snapshot]) -> None:
        k = len(self.rows)
        if (snap is None) != (k == 0):
            raise ValueError("row %d: a snapshot on every row but the first" % k)
        self.rows.append([row[n] for n in ALL_NAMES])
        for p in (1, 2):
            w = self.word[p]
            self.pressed[p].append(PM.pressed_class(self.chars[p], w[0]) if w and k > w[1] else None)
        if snap is not None:
            self._cut(k, snap)

    def _cut(self, k: int, snap: Snapshot) -> None:
        pals = dict(Counter(e.pal for e in entries(snap)))
        groups = render_groups(snap, pals=set(GROUPS))
        for pal, g in sorted(groups.items()):
            face = oam_facing(snap, pal)
            rgba = canonical(g.rgba, face)
            key = key_of(rgba)
            if key not in self.seen:
                self.seen.add(key)
                Image.fromarray(rgba, "RGBA").save(os.path.join(self.base, "sprites", key + ".png"))
                np.savez_compressed(os.path.join(self.base, "snaps", key + ".npz"),
                                    oam=np.frombuffer(snap.oam, np.uint8), cg=np.frombuffer(snap.cg, np.uint8),
                                    vram=np.frombuffer(snap.vram, np.uint8),
                                    regs=np.array([snap.mode, snap.base, snap.off]), k=k, pal=pal)
            self.frames.append(dict(k=k, group=GROUPS[pal], key=key, index_key=key_of(canonical(g.index, face)),
                                    bbox=list(g.bbox), size=[int(rgba.shape[1]), int(rgba.shape[0])],
                                    oam_facing=face, pals=pals))

    def write(self, game: int, summary: Dict) -> None:
        np.savez_compressed(os.path.join(self.base, "g%d.npz" % game), rows=np.array(self.rows, np.int32),
                            names=np.array(ALL_NAMES),
                            pressed=np.array([[c or "" for c in self.pressed[p]] for p in (1, 2)]))
        with gzip.open(os.path.join(self.base, "g%d_frames.jsonl.gz" % game), "wt") as f:
            for fr in self.frames:
                f.write(json.dumps(fr) + "\n")
        with open(os.path.join(self.base, "done.json"), "w") as f:
            json.dump(dict(summary, game=game, rows=len(self.rows), frames=len(self.frames)), f)


def collect_pair(out: str, a: str, b: str, port: int, state_path: str, seed: int = 0, game: int = 0,
                 rom: Optional[str] = None) -> Dict:
    """Play one game of A (player 1) vs B (player 2), both ours, every frame cut; the game summary."""
    base = os.path.join(out, "_pairs", "%s_vs_%s" % (a, b))
    chars = {1: a, 2: b}
    with open(state_path, "rb") as f:
        state = f.read()
    t0 = time.time()
    with open_mesen(port, rom) as br:
        br.set_vars(ALL_VARS)
        r = dict(zip(ALL_NAMES, br.load_state(state).rams[-1]))
        if (r["p1_char"], r["p2_char"]) != (IDS[a], IDS[b]):
            raise SystemExit("%s holds %s, expected %s" % (state_path, (r["p1_char"], r["p2_char"]), (a, b)))
        sink = PairSink(base, chars)
        proxy = SnapBridge(br, ALL_NAMES)
        proxy.sink = sink.feed
        words = {p: list(PM.moves(c)) for p, c in chars.items()}
        cycles = {p: PM.Cycle(words[p], random.Random("cycle%d:%d:%s:%s:%d" % (p, seed, a, b, game))) for p in chars}
        start = random.Random("start:%d:%s:%s:%d" % (seed, a, b, game))
        summary = PC.play_both(proxy, ALL_NAMES, chars, cycles, state, start, lambda: len(sink.rows),
                               on_word=sink.press)
        summary = dict(summary, seconds=round(time.time() - t0, 1), pair=[a, b], seed=seed)
        sink.write(game, summary)
    return summary
