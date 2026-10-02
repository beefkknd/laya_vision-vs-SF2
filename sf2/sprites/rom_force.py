"""Force a fighter into each pose of its table and cut the sprite (scripts/collect_full_sprites.py). Collection time
only; nothing here runs in play.

How (sf2/sprites/rom_table.py for the layout): the animation scripts live in player 1's WRAM buffer (a copy of the
ROM), so they can be patched after a savestate load:
  * "chain" (poses some script names): the stance loop's end record jumps to that script's start instead of back to
    the stance, and the target record's duration is set to 0xFF. The game plays the script's records before the
    target in order, as in play - VRAM tiles are uploaded as deltas along a script (a pose can reuse its
    predecessor's tiles), so a pose shown straight from the stance can come out with stale tiles.
  * "stance" (poses no script names): every stance record's pose byte is set to the target.
Writing the pose pointer 0x0C1E alone is not used: the tile upload belongs to the animation step.
A pose is accepted once the anim pointer holds the target record (chain) / 0x0C1E holds the pose record (stance) and
the cut sprite is the same on STABLE consecutive snapshots.
"""
from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np

from ..emu.ram import Var
from .catalog import canonical, key_of
from .collect import oam_facing
from .emu import video
from .oam import Snapshot, render_groups
from .rom_table import ANIM_PTR, BUFFER, END_FLAG, IMAGE_OFF, POSE_PTR, RECORD, walk_anim

P1 = 0x0C00
PAL_P1 = 4
PALS = (PAL_P1, 5)
FORCE_VARS = [Var("anim", P1 + ANIM_PTR, 2, False), Var("pose", P1 + POSE_PTR, 2, False)]
STABLE = 3
MAX_FRAMES = 160


@dataclass(frozen=True)
class Forced:
    pose: int
    record: int                       # WRAM address of the pose record
    ok: bool
    why: str
    key: Optional[str] = None
    rgba: Optional[np.ndarray] = None
    snap: Optional[Snapshot] = None
    frames: int = 0
    method: str = ""
    index_key: Optional[str] = None
    pal: Optional[int] = None


def cut(snap: Snapshot, pals: Sequence[int] = PALS):
    """(canonical rgba, key, index key, palette) of the first of ``pals`` drawn (the catalog cuts player 1 on
    palette 4 only; a few poses are drawn wholly on palette 5, player 1's projectile palette), or Nones."""
    groups = render_groups(snap, pals=set(pals))
    for pal in pals:
        g = groups.get(pal)
        if g is not None:
            face = oam_facing(snap, pal)
            rgba = canonical(g.rgba, face)
            return rgba, key_of(rgba), key_of(canonical(g.index, face)), pal
    return None, None, None, None


def stance_records(bridge, state: bytes, frames: int = 90) -> List[int]:
    """The animation records player 1 steps through while idling from ``state`` (the stance loop)."""
    bridge.set_vars(FORCE_VARS)
    bridge.load_state(state)
    obs = bridge.run([["-"]] * frames)
    recs = sorted({r[0] for r in obs.rams})
    if not recs:
        raise RuntimeError("no stance records seen")
    return recs


def chain_pokes(wram: bytes, stance: Sequence[int], script: int, target: int) -> dict:
    """WRAM writes for the chain method: the stance's end record jumps to ``script``; ``target`` holds (0xFF)."""
    ends = [r for r in stance if wram[r + 1] & END_FLAG]
    if not ends:            # the end record can have been skipped over while sampling: walk the script from its start
        recs = walk_anim(wram, min(stance), BUFFER[1] + IMAGE_OFF)
        ends = [recs[-1][0]] if recs and recs[-1][2] & END_FLAG else []
    if len(ends) != 1:
        raise RuntimeError("stance loop has %d end records" % len(ends))
    end = ends[0]
    rel = (script - (end + RECORD)) & 0xFFFF
    return {end + RECORD: rel & 0xFF, end + RECORD + 1: rel >> 8, target: 0xFF}


def force_chain(bridge, state: bytes, wram: bytes, stance: Sequence[int], pose: int, record: int, script: int,
                target: int) -> Forced:
    """Chain method: play ``script`` from its start up to its record ``target`` (which names ``pose``) and hold."""
    if wram[target + 2] != pose:
        raise ValueError("record %04x names pose %d, not %d" % (target, wram[target + 2], pose))
    return _run(bridge, state, chain_pokes(wram, stance, script, target), lambda ram: ram[0] == target, pose, record,
                "chain")


def _run(bridge, state, pokes, reached_when, pose, record, method) -> Forced:
    bridge.set_vars(FORCE_VARS)
    bridge.load_state(state)
    bridge.poke(pokes)
    keys: List[Optional[str]] = []
    reached = None
    for f in range(MAX_FRAMES):
        ram = bridge.run([["-"]]).rams[-1]
        snap = video(bridge)
        if reached is None and reached_when(ram):
            reached = f
            if ram[1] != record:
                return Forced(pose, record, False, "pose pointer %04x, not the record" % ram[1], frames=f + 1,
                              method=method)
        if reached is None or f <= reached:
            continue
        rgba, key, ikey, pal = cut(snap)
        keys.append(key)
        if len(keys) >= STABLE and key is not None and len(set(keys[-STABLE:])) == 1:
            return Forced(pose, record, True, "ok", key, rgba, snap, f + 1, method, ikey, pal)
    why = "target never reached" if reached is None else "sprite not stable (%d distinct, last %s)" % (
        len(set(keys)), keys[-1] if keys else None)
    return Forced(pose, record, False, why, frames=MAX_FRAMES, method=method)


def force(bridge, state: bytes, stance: Sequence[int], pose: int, record: int) -> Forced:
    """Stance method: load ``state``, point every stance record at pose index ``pose``, step until the pose pointer
    holds ``record`` (the table's record of that pose) and the sprite is stable; the cut."""
    return _run(bridge, state, {r + 2: pose & 0xFF for r in stance}, lambda ram: ram[1] == record, pose, record,
                "stance")
