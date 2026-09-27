"""What the model sees of a screen frame: the HUD blanked out. Every path that hands a frame to laya-vision
(sf2.policy.make_state at play time, sf2.dataset.Writer and scripts/vs_dataset.py at train time) goes through
``model_frame``, so train and play always see the same pixels.

Why blank: the HUD (score, life bars, names, timer: rows 0-61, measured on the fight screen 2026-09-27) is the one
part of the screen that does not mirror. Left in, a mirrored training frame shows the character on the right under
its own name on the LEFT bar, a picture the real game never shows. Life is in the text note; the timer and names
add nothing the note lacks. Blanked, a mirrored frame is an exact left-right flip of the whole screen.
"""
import numpy as np

HUD_ROWS = 62


def model_frame(img: np.ndarray) -> np.ndarray:
    out = np.array(img, copy=True)
    out[:HUD_ROWS] = 0
    return out


def mirror_frame(img: np.ndarray) -> np.ndarray:
    """The model frame of the same scene with the fighters' sides swapped."""
    return model_frame(img)[:, ::-1].copy()
