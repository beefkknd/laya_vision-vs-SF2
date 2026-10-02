"""RAM-free frame helpers: what laya-vision is shown of a screen frame, plus the "me=<char>" context note, with NO
RAM (nothing here imports sf2.emu or any RAM-row builder). The screen-only play runner feeds the eye through these.

``hud_frame`` and ``eye_note`` are the perception-time copies of the two pure helpers in sf2.data.u_data (a data
BUILDER that is RAM-tangled at import through sf2.data.perception / sf2.data.vs_sweep, so play code cannot import it).
tests/test_m5_moves_free.py pins these against u_data's so the two cannot drift. ``model_frame`` / ``mirror_frame``
are re-exported from the already-RAM-free sf2.data.frames.
"""
import numpy as np

from ..config import IMAGE_SIZE
from ..vocab import FIGHTERS
from .frames import mirror_frame, model_frame  # noqa: F401  (RAM-free; re-exported for the runner)


def hud_frame(img: np.ndarray) -> np.ndarray:
    """Frames v3: the screen padded with black rows at the bottom to 256x256, the HUD left as it is (a 256x256
    frame passes unchanged, so a stored training frame can be fed again)."""
    h, w = img.shape[:2]
    if img.ndim != 3 or img.shape[2] != 3 or h > IMAGE_SIZE or w != IMAGE_SIZE:
        raise ValueError("expected a %d-wide RGB screen at most %d tall, got %s" % (IMAGE_SIZE, IMAGE_SIZE, img.shape))
    out = np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3), np.uint8)
    out[:h] = img
    return out


def eye_note(me: str) -> str:
    """The note v3 context string: the character only, nothing of the game's memory."""
    if me not in FIGHTERS:
        raise ValueError("unknown character %r" % me)
    return "me=%s" % me
