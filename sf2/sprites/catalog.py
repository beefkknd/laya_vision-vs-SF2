"""Canonical sprites and their keys - pure.

A sprite (an RGBA crop, transparent outside the sprite) is normalised to face right: a fighter facing left is
mirrored. key = sha1 of the canonical RGBA (shape + bytes), 16 hex digits; ``index_key`` the same over the 4bpp
colour indices only (a colour-only variant - e.g. a hit flash - has another key but the same index key).
"""
import hashlib
from typing import Optional

import numpy as np

FACE_RIGHT, FACE_LEFT = 0x40, 0x00


def canonical(img: np.ndarray, facing: Optional[int]) -> np.ndarray:
    """``img`` (h, w[, c]) as if facing right: mirrored when ``facing`` is FACE_LEFT; otherwise as is."""
    return np.ascontiguousarray(img[:, ::-1]) if facing == FACE_LEFT else np.ascontiguousarray(img)


def key_of(img: np.ndarray) -> str:
    a = np.ascontiguousarray(img, np.uint8)
    return hashlib.sha1(repr(a.shape).encode() + a.tobytes()).hexdigest()[:16]


def mirror_key(img: np.ndarray) -> str:
    """The key the sprite would have if it were mirrored (for the 'mirror twins' check)."""
    return key_of(img[:, ::-1])
