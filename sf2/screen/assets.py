"""The files the screen reader loads (no emulator, no RAM): the sprite catalog (out/sprite_catalog: catalog.json and
<char>/<key>.png, canonical = drawn facing right), the anchor table (out/screen_reader/anchors.json: where a fighter's
x sits relative to each sprite's box) and the HUD digit glyphs (sf2.screen.hud).

Colours are compared exactly, as SNES 15-bit codes: code = (r >> 3) << 10 | (g >> 3) << 5 | (b >> 3) (the emulator
expands 5-bit channels as (c << 3) | (c >> 2), so >> 3 recovers them).
"""
import json
import os
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

from ..config import REPO

CATALOG = os.path.join(REPO, "out", "sprite_catalog")
READER_DIR = os.path.join(REPO, "out", "screen_reader")
ANCHORS = os.path.join(READER_DIR, "anchors.json")
CHARS = ("blanka", "chunli", "dhalsim", "guile", "honda", "ken", "ryu", "zangief")
PROJ = "_projectiles"
FACE_RIGHT, FACE_LEFT = 0x40, 0x00
FACES = (FACE_RIGHT, FACE_LEFT)    # every sprite is tried drawn both ways (a mirrored sprite = the same label)
N_SAMPLE = 48                      # opaque pixels per template for the coarse score
H, W = 224, 256
PAD = 160                          # the padded code image: sprites may hang off the screen by up to this
HP, WP = H + 2 * PAD, W + 2 * PAD


def codes15(rgb: np.ndarray) -> np.ndarray:
    """(..., 3) uint8 RGB -> (...) int32 SNES colour codes."""
    c = rgb.astype(np.int32) >> 3
    return (c[..., 0] << 10) | (c[..., 1] << 5) | c[..., 2]


@dataclass(frozen=True)
class Template:
    ck: str                        # "<char>/<key>"
    face: int                      # FACE_RIGHT: drawn as the canonical PNG; FACE_LEFT: mirrored
    h: int
    w: int
    dx: float                      # screen x of the fighter = box x0 - dx
    act2: Optional[str]            # majority 7-answer label (None: only "unknown" seen)
    act2_share: float              # share of the majority among labelled frames
    air: Optional[str]             # majority "ground" / "air"
    n: int                         # opaque pixels


@dataclass(frozen=True)
class Bank:
    """All templates of one character (both facings), with flat pixel arrays for vectorised scoring.
    Flat offsets are into the padded code image (row stride WP), relative to the template's top-left."""
    char: str
    templates: Tuple[Template, ...]
    palette: frozenset             # every colour code any of its sprites uses
    sizes: np.ndarray              # (T, 2) h, w
    samp_off: np.ndarray           # (T, N_SAMPLE) flat offsets of sampled opaque pixels
    samp_code: np.ndarray          # (T, N_SAMPLE)
    samp_dy: np.ndarray            # (T, N_SAMPLE) rows / cols, for the on-screen test
    samp_dx: np.ndarray
    full: Tuple[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray], ...]   # per template: off, code, dy, dx
    hole: Tuple[Tuple[np.ndarray, np.ndarray, np.ndarray], ...]               # transparent pixels in the box


def _majority(c: Dict[str, int], skip=("unknown",)) -> Tuple[Optional[str], float]:
    c = {k: v for k, v in c.items() if k not in skip}
    if not c:
        return None, 0.0
    k = max(sorted(c), key=c.get)
    return k, c[k] / sum(c.values())


def _anchor(anchors: Dict[str, list], ck: str, face: int, w: int, mirror_c: int) -> Optional[float]:
    a = anchors.get("%s|%d" % (ck, face))
    if a is not None:
        return float(a[0])
    other = anchors.get("%s|%d" % (ck, FACE_LEFT if face == FACE_RIGHT else FACE_RIGHT))
    if other is not None:                       # anchor column from the left edge mirrors: a' = w - a + c
        return float(-(w - (-other[0]) + mirror_c))
    return None


def mirror_constant(anchors: Dict[str, list], sizes: Dict[str, int]) -> int:
    """The c in a_left = w - a_right + c, from sprites anchored in both facings (a = -dx)."""
    c = Counter()
    for u, v in anchors.items():
        ck, face = u.rsplit("|", 1)
        if int(face) == FACE_RIGHT and "%s|%d" % (ck, FACE_LEFT) in anchors and ck in sizes:
            c[-anchors["%s|%d" % (ck, FACE_LEFT)][0] - sizes[ck] + (-v[0])] += 1
    return c.most_common(1)[0][0] if c else 0


def _load_png(path: str) -> np.ndarray:
    with Image.open(path) as im:
        return np.asarray(im.convert("RGBA"))


def _flat(dy: np.ndarray, dx: np.ndarray) -> np.ndarray:
    return (dy * WP + dx).astype(np.int64)


def make_bank(char: str, items: List[Tuple[str, np.ndarray, Dict]], anchors: Dict[str, list]) -> Bank:
    """``items``: (ck, canonical RGBA, catalog entry). Builds both facings of every sprite."""
    sizes = {ck: rgba.shape[1] for ck, rgba, _ in items}
    mc = mirror_constant(anchors, sizes)
    temps, full, hole, palette = [], [], [], set()
    so, sc, sdy, sdx = [], [], [], []
    for ck, rgba, e in items:
        act2, share = _majority(e.get("labels", {}).get("act2", {}))
        air, _ = _majority(e.get("labels", {}).get("air", {}))
        for face in FACES:
            img = rgba if face == FACE_RIGHT else rgba[:, ::-1]
            h, w = img.shape[:2]
            op = img[..., 3] > 0
            dy, dx = np.nonzero(op)
            code = codes15(img[..., :3])[op]
            palette.update(code.tolist())
            hy, hx = np.nonzero(~op)
            anc = _anchor(anchors, ck, face, w, mc)
            if anc is None:
                anc = -w / 2.0
            temps.append(Template(ck, face, h, w, anc, act2, share, air, int(op.sum())))
            full.append((_flat(dy, dx), code.astype(np.int32), dy.astype(np.int32), dx.astype(np.int32)))
            hole.append((_flat(hy, hx), hy.astype(np.int32), hx.astype(np.int32)))
            pick = np.linspace(0, len(dy) - 1, N_SAMPLE).round().astype(int)
            so.append(_flat(dy[pick], dx[pick]))
            sc.append(code[pick])
            sdy.append(dy[pick])
            sdx.append(dx[pick])
    return Bank(char, tuple(temps), frozenset(palette), np.array([[t.h, t.w] for t in temps], np.int32),
                np.stack(so), np.stack(sc).astype(np.int32), np.stack(sdy).astype(np.int32),
                np.stack(sdx).astype(np.int32), tuple(full), tuple(hole))


@lru_cache(maxsize=4)
def load_banks(catalog: str = CATALOG, anchors_path: str = ANCHORS) -> Dict[str, Bank]:
    """Every character's bank and the projectile bank (key PROJ), from the catalog files."""
    with open(os.path.join(catalog, "catalog.json")) as f:
        sprites = json.load(f)["sprites"]
    anchors = {}
    if os.path.exists(anchors_path):
        with open(anchors_path) as f:
            anchors = json.load(f)["anchors"]
    by_char: Dict[str, List] = {}
    for ck, e in sorted(sprites.items()):
        by_char.setdefault(e["char"], []).append((ck, _load_png(os.path.join(catalog, ck + ".png")), e))
    return {c: make_bank(c, items, anchors) for c, items in by_char.items()}


@lru_cache(maxsize=4)
def projectile_owners(catalog: str = CATALOG) -> Dict[str, Tuple[str, bool]]:
    """Projectile sprite -> (owner character, is a thrown shot): a shot when most of its frames had a shot slot
    active (the rest of the palette 5 / 7 sprites are hit sparks and the like)."""
    with open(os.path.join(catalog, "catalog.json")) as f:
        sprites = json.load(f)["sprites"]
    out = {}
    for ck, e in sprites.items():
        if e["char"] != PROJ:
            continue
        shots = e.get("shots", {})
        active = sum(n for k, n in shots.items() if k != "shot1=0 shot2=0")
        owner = max(sorted(e.get("owners", {"?": 1})), key=e.get("owners", {"?": 1}).get)
        out[ck] = (owner, 2 * active > sum(shots.values()))
    return out
