"""Rebuild the SNES sprites from one frame's snapshot (OAM 544 B, CGRAM 512 B, VRAM 0xC000-0xFFFF) - pure numpy.

Each OAM palette group is rendered on its own canvas (no occlusion by the other fighter, sparks or shadows), lower
OAM index on top (SNES order). The canvas is wider and taller than the screen (x -256..319, y -64..319), so a sprite
partly off-screen is still cut whole. Palettes in this game (probe, scripts/probe_sprite_cut.py): 4 = player 1,
6 = player 2, 5 = player 1's projectile, 7 = hit sparks (and player 2's projectile), 0 = shadows, 3 = a stage prop.
"""
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

VRAM_FROM = 0xC000                     # the snapshot's VRAM slice starts here (bytes)
VRAM_LEN = 0x10000 - VRAM_FROM
SIZES = {0: ((8, 8), (16, 16)), 1: ((8, 8), (32, 32)), 2: ((8, 8), (64, 64)), 3: ((16, 16), (32, 32)),
         4: ((16, 16), (64, 64)), 5: ((32, 32), (64, 64)), 6: ((16, 32), (32, 64)), 7: ((16, 32), (32, 32))}
X0, Y0 = -256, -64                     # canvas origin in OAM coordinates
CW, CH = 576, 384


@dataclass(frozen=True)
class Snapshot:
    oam: bytes
    cg: bytes
    vram: bytes                         # VRAM_FROM .. 0xFFFF
    mode: int
    base: int                           # OBJ name base, words
    off: int                            # table-2 offset, words

    def __post_init__(self):
        if (len(self.oam), len(self.cg), len(self.vram)) != (544, 512, VRAM_LEN):
            raise ValueError("snapshot sizes %s" % ((len(self.oam), len(self.cg), len(self.vram)),))
        if self.mode not in SIZES:
            raise ValueError("OBJ size mode %r" % self.mode)


@dataclass(frozen=True)
class Entry:
    i: int
    x: int
    y: int
    w: int
    h: int
    tile: int                           # 9-bit (table << 8 | tile)
    pal: int
    hflip: bool
    vflip: bool


def cg_rgb(cg: bytes) -> np.ndarray:
    w = np.frombuffer(cg, "<u2").astype(np.int32)
    rgb = np.stack([(w & 31), (w >> 5) & 31, (w >> 10) & 31], -1)
    return ((rgb << 3) | (rgb >> 2)).astype(np.uint8)


def decode_tiles(vram: bytes) -> np.ndarray:
    """(n, 8, 8) 4bpp colour indices of every 32-byte tile of the slice."""
    t = np.frombuffer(vram, np.uint8).reshape(-1, 32)
    planes = [t[:, 0:16:2], t[:, 1:16:2], t[:, 16:32:2], t[:, 17:32:2]]     # (n, 8 rows)
    bits = [np.unpackbits(p[..., None], axis=-1) for p in planes]           # (n, 8, 8), MSB = column 0
    return (bits[0] | (bits[1] << 1) | (bits[2] << 2) | (bits[3] << 3)).astype(np.uint8)


def entries(snap: Snapshot) -> List[Entry]:
    """The visible OAM entries (parked ones, wholly in rows 224-255, left out). y >= 224 wraps to the top (y - 256)."""
    oam, out = snap.oam, []
    for i in range(128):
        x, y, tile, attr = oam[4 * i:4 * i + 4]
        hi = (oam[512 + i // 4] >> (2 * (i % 4))) & 3
        x = x | ((hi & 1) << 8)
        x = x - 512 if x >= 256 else x
        w, h = SIZES[snap.mode][hi >> 1]
        if y >= 224 and y + h <= 256:
            continue
        y = y - 256 if y >= 224 else y
        out.append(Entry(i, x, y, w, h, tile | ((attr & 1) << 8), (attr >> 1) & 7, bool(attr & 0x40),
                         bool(attr & 0x80)))
    return out


def _tile_index(snap: Snapshot, table: int, t: int) -> Optional[int]:
    byte = ((snap.base + (snap.off if table else 0) + t * 16) * 2) & 0xFFFF
    return (byte - VRAM_FROM) // 32 if byte >= VRAM_FROM else None


def entry_pixels(snap: Snapshot, e: Entry, tiles: np.ndarray) -> np.ndarray:
    """(h, w) colour indices of one entry, flips applied."""
    tile, table = e.tile & 0xFF, e.tile >> 8
    spr = np.zeros((e.h, e.w), np.uint8)
    for tr in range(e.h // 8):
        for tc in range(e.w // 8):
            idx = _tile_index(snap, table, ((tile & 0xF0) + tr * 16 + ((tile + tc) & 0x0F)) & 0xFF)
            if idx is not None:
                spr[tr * 8:tr * 8 + 8, tc * 8:tc * 8 + 8] = tiles[idx]
    if e.hflip:
        spr = spr[:, ::-1]
    if e.vflip:
        spr = spr[::-1]
    return spr


@dataclass(frozen=True)
class Group:
    """One palette group's pixels: ``rgba`` cropped to ``bbox`` (x0, y0, x1, y1 exclusive, OAM coordinates),
    ``index`` its 4bpp colour indices (0 = transparent), ``entries`` its OAM indices."""
    pal: int
    rgba: np.ndarray
    index: np.ndarray
    bbox: Tuple[int, int, int, int]
    entries: Tuple[int, ...]


def render_groups(snap: Snapshot, pals=None) -> Dict[int, Group]:
    """Every palette group (or only ``pals``) rendered on its own canvas and cropped to its pixels."""
    tiles = decode_tiles(snap.vram)
    colours = cg_rgb(snap.cg)
    by_pal: Dict[int, List[Entry]] = {}
    for e in entries(snap):
        if pals is None or e.pal in pals:
            by_pal.setdefault(e.pal, []).append(e)
    out = {}
    for pal, es in by_pal.items():
        canvas = np.zeros((CH, CW), np.uint8)
        for e in sorted(es, key=lambda e: -e.i):            # high index first: the lower index ends on top
            spr = entry_pixels(snap, e, tiles)
            view = canvas[e.y - Y0:e.y - Y0 + e.h, e.x - X0:e.x - X0 + e.w]
            np.copyto(view, spr, where=spr > 0)
        ys, xs = np.nonzero(canvas)
        if len(ys) == 0:
            continue
        y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
        index = canvas[y0:y1, x0:x1]
        rgba = np.zeros(index.shape + (4,), np.uint8)
        rgba[..., :3] = colours[128 + pal * 16 + index.astype(np.int32)]
        rgba[..., 3] = np.where(index > 0, 255, 0)
        rgba[index == 0, :3] = 0
        out[pal] = Group(pal, rgba, index, (int(x0 + X0), int(y0 + Y0), int(x1 + X0), int(y1 + Y0)),
                         tuple(sorted(e.i for e in es)))
    return out


def screen_layer(snap: Snapshot, pals=None) -> Tuple[np.ndarray, np.ndarray]:
    """(rgb 224x256x3, mask 224x256) of all sprites (or ``pals``) as drawn on screen (one canvas, lower index on
    top): for checking the rebuild against Mesen's sprites-only screen."""
    tiles = decode_tiles(snap.vram)
    colours = cg_rgb(snap.cg)
    canvas = np.zeros((CH, CW), np.int32) - 1
    for e in sorted(entries(snap), key=lambda e: -e.i):
        if pals is not None and e.pal not in pals:
            continue
        spr = entry_pixels(snap, e, tiles).astype(np.int32)
        view = canvas[e.y - Y0:e.y - Y0 + e.h, e.x - X0:e.x - X0 + e.w]
        np.copyto(view, 128 + e.pal * 16 + spr, where=spr > 0)
    scr = canvas[-Y0:-Y0 + 224, -X0:-X0 + 256]
    mask = scr >= 0
    rgb = np.where(mask[..., None], colours[np.clip(scr, 0, 255)], 0).astype(np.uint8)
    return rgb, mask
