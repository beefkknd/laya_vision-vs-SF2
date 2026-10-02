"""Locate and match sprites in one frame - exact colours, masked (only a sprite's opaque pixels are compared, so the
background never counts), both facings, vectorised numpy.

1. locate: the pixels in the character's LOCATING colours (its own colours, minus the other fighter's and the stage's
   common ones - sf2.screen.reader.lock_round) are summed in a BOX_H x BOX_W box (integral image); the peak is the
   fighter. The blob around it (columns / rows with such pixels, gaps up to GAP px) gives its extent.
2. coarse: every template of the character, both facings, at 6 alignments to the blob (left / right edge x bottom /
   top, centred on the bottom, and its locating-colour centroid on the blob's), scored on N_SAMPLE pixels: the share
   whose screen colour equals the sprite's.
3. fine: the TOP_K best (template, offset) are re-scored at +-JITTER px with all pixels:
   score = matched / (visible + unexplained + missing), visible = the sprite's pixels on the screen, unexplained =
   locating-colour pixels inside the sprite's box where the sprite is transparent, missing = the blob's locating
   pixels the sprite does not cover (so a smaller sprite that fits inside the real one does not win). Overlap and
   the HUD lower the score of the right sprite: a confidence question.
"""
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from .assets import HP, H, PAD, WP, W, Bank

BOX_H, BOX_W = 32, 24
MIN_PEAK = 24              # locating pixels in the best box for "found"
GAP = 8
MIN_LINE = 2               # a blob column / row holds at least this many locating pixels
TOP_K = 16
JITTER = 2
WINDOW = 96                # columns either side of the peak searched for the blob


@dataclass(frozen=True)
class Blob:
    x0: int
    y0: int
    x1: int                # exclusive
    y1: int
    peak: int              # locating pixels in the best box
    cy: float = 0.0        # centroid of the locating pixels in the blob
    cx: float = 0.0
    count: int = 0


@dataclass(frozen=True)
class Match:
    t: int                 # template index in the bank
    ox: int                # screen x / y of the template's top-left
    oy: int
    score: float
    matched: int
    visible: int


def padded_codes(codes: np.ndarray) -> np.ndarray:
    """(HP * WP,) int32 flat code image with a -1 border of PAD px."""
    out = np.full((HP, WP), -1, np.int32)
    out[PAD:PAD + H, PAD:PAD + W] = codes
    return out.ravel()


def padded_mask(mask: np.ndarray) -> np.ndarray:
    out = np.zeros((HP, WP), bool)
    out[PAD:PAD + H, PAD:PAD + W] = mask
    return out.ravel()


def _runs(profile: np.ndarray, centre: int) -> Optional[Tuple[int, int]]:
    """The extent [a, b) of nonzero entries around ``centre``, bridging gaps up to GAP."""
    nz = np.nonzero(profile >= MIN_LINE)[0]
    if len(nz) == 0:
        return None
    i = int(np.searchsorted(nz, centre))
    i = min(max(i, 0), len(nz) - 1)
    if i > 0 and abs(nz[i - 1] - centre) < abs(nz[i] - centre):
        i -= 1
    a = b = i
    while a > 0 and nz[a] - nz[a - 1] <= GAP:
        a -= 1
    while b < len(nz) - 1 and nz[b + 1] - nz[b] <= GAP:
        b += 1
    return int(nz[a]), int(nz[b]) + 1


def locate(mask: np.ndarray, cols: Optional[Tuple[int, int]] = None, min_peak: int = MIN_PEAK) -> Optional[Blob]:
    """The densest BOX of ``mask`` (H, W bool), optionally only with its centre in columns ``cols``, and its blob
    (None when the box holds fewer than ``min_peak`` pixels)."""
    integ = np.zeros((H + 1, W + 1), np.int32)
    np.cumsum(np.cumsum(mask, 0, dtype=np.int32), 1, out=integ[1:, 1:])
    s = integ[BOX_H:, BOX_W:] - integ[:-BOX_H, BOX_W:] - integ[BOX_H:, :-BOX_W] + integ[:-BOX_H, :-BOX_W]
    if cols is not None:
        lo, hi = max(0, cols[0] - BOX_W // 2), max(0, min(s.shape[1], cols[1] - BOX_W // 2))
        if hi <= lo:
            return None
        s = s[:, lo:hi]
    else:
        lo = 0
    py, px = np.unravel_index(int(s.argmax()), s.shape)
    peak = int(s[py, px])
    if peak < min_peak:
        return None
    cx, cy = px + lo + BOX_W // 2, py + BOX_H // 2
    wx0, wx1 = max(0, cx - WINDOW), min(W, cx + WINDOW)
    sub = mask[:, wx0:wx1]
    xr = _runs(sub.sum(0), cx - wx0)
    if xr is None:
        return None
    x0, x1 = xr[0] + wx0, xr[1] + wx0
    yr = _runs(mask[:, x0:x1].sum(1), cy)
    if yr is None:
        return None
    ys, xs = np.nonzero(mask[yr[0]:yr[1], x0:x1])
    return Blob(x0, yr[0], x1, yr[1], peak, float(ys.mean() + yr[0]), float(xs.mean() + x0), int(len(ys)))


def _alignments(bank: Bank, blob: Blob, idx: np.ndarray,
                cent: Optional[np.ndarray]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(template, ox, oy) candidate arrays: each template in ``idx`` at the 5 edge alignments, plus (with ``cent``,
    (T, 2) centroids of each template's locating pixels) its locating centroid on the blob's."""
    h, w = bank.sizes[idx, 0], bank.sizes[idx, 1]
    xs = [np.full_like(w, blob.x0), blob.x1 - w, np.full_like(w, blob.x0), blob.x1 - w, (blob.x0 + blob.x1 - w) // 2]
    ys = [blob.y1 - h, blob.y1 - h, np.full_like(h, blob.y0), np.full_like(h, blob.y0), blob.y1 - h]
    n = 5
    if cent is not None:
        c = cent[idx]
        ok = np.isfinite(c[:, 0])
        xs.append(np.where(ok, np.round(blob.cx - np.nan_to_num(c[:, 1])), blob.x0).astype(w.dtype))
        ys.append(np.where(ok, np.round(blob.cy - np.nan_to_num(c[:, 0])), blob.y1 - h).astype(h.dtype))
        n += 1
    return np.tile(idx, n), np.concatenate(xs), np.concatenate(ys)


def _coarse(bank: Bank, code: np.ndarray, t: np.ndarray, ox: np.ndarray, oy: np.ndarray) -> np.ndarray:
    base = (oy + PAD).astype(np.int64) * WP + (ox + PAD)
    got = code[base[:, None] + bank.samp_off[t]]
    vis = got >= 0
    hit = (got == bank.samp_code[t]).sum(1)
    n = vis.sum(1)
    return np.where(n >= 12, hit / np.maximum(n, 1), 0.0)


def _fine(bank: Bank, code: np.ndarray, loc: np.ndarray, t: int, ox: int, oy: int, blob_count: int = 0) -> Match:
    off, col, _, _ = bank.full[t]
    hoff = bank.hole[t][0]
    d = np.arange(-JITTER, JITTER + 1)
    dy, dx = np.meshgrid(d, d, indexing="ij")
    base = ((oy + PAD + dy.ravel()).astype(np.int64) * WP + (ox + PAD + dx.ravel()))
    got = code[base[:, None] + off[None]]
    matched = (got == col[None]).sum(1)
    visible = (got >= 0).sum(1)
    unexplained = loc[base[:, None] + hoff[None]].sum(1) if len(hoff) else np.zeros(len(base), np.int64)
    covered = loc[base[:, None] + off[None]].sum(1)
    missing = np.maximum(0, blob_count - covered)          # the blob's locating pixels the sprite does not cover
    score = np.where(visible >= max(12, len(off) // 4),
                     matched / np.maximum(visible + unexplained + missing, 1), 0.0)
    j = int(score.argmax())
    return Match(t, int(ox + dx.ravel()[j]), int(oy + dy.ravel()[j]), float(score[j]), int(matched[j]),
                 int(visible[j]))


def centroids(bank: Bank, lut: np.ndarray) -> np.ndarray:
    """(T, 2) centroid (row, col) of each template's pixels in the locating colours ``lut`` (NaN when none)."""
    out = np.full((len(bank.templates), 2), np.nan)
    for i, (_, col, dy, dx) in enumerate(bank.full):
        sel = lut[col]
        if sel.any():
            out[i] = dy[sel].mean(), dx[sel].mean()
    return out


def match(bank: Bank, code: np.ndarray, loc_flat: np.ndarray, blob: Blob,
          idx: Optional[np.ndarray] = None, cent: Optional[np.ndarray] = None) -> Optional[Match]:
    """Best (template, offset) of ``bank`` for ``blob``. ``code``: padded_codes; ``loc_flat``: padded_mask of the
    locating colours; ``idx``: the templates to try (default all); ``cent``: centroids() for the same colours."""
    if idx is None:
        idx = np.arange(len(bank.templates))
    t, ox, oy = _alignments(bank, blob, idx, cent)
    keep = (ox > -PAD + JITTER) & (ox + bank.sizes[t, 1] < W + PAD - JITTER) & \
           (oy > -PAD + JITTER) & (oy + bank.sizes[t, 0] < H + PAD - JITTER)
    t, ox, oy = t[keep], ox[keep], oy[keep]
    if len(t) == 0:
        return None
    s = _coarse(bank, code, t, ox, oy)
    order = np.argsort(-s, kind="stable")
    seen, best = set(), None
    for i in order:
        key = (int(t[i]), int(ox[i]), int(oy[i]))
        if key in seen:
            continue
        seen.add(key)
        m = _fine(bank, code, loc_flat, *key, blob_count=blob.count)
        if best is None or m.score > best.score:
            best = m
        if len(seen) >= TOP_K:
            break
    return best
