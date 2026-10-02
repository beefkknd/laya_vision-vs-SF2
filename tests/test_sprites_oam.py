"""The OAM rebuild (sf2.sprites.oam) on a frozen real frame: tests/fixtures/sprites/frame.npz is one frame of
states/vs_ken_vs_ryu.state (player 1 Ken throws a fireball: palettes 4, 6, 5, 0, 3 up), with the snapshot from
mesen/sf2_bridge_sprites.lua AND the same frame's screen from a Mesen with all BG layers hidden (sprites only) - an
independent truth. Golden crops expected_<group>.png are this code's output at the time, regenerated only on purpose:

    .venv/bin/python tests/test_sprites_oam.py --regen        # needs Mesen + the ROM
"""
import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sf2.sprites import oam  # noqa: E402

FIX = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures", "sprites")
GROUPS = {4: "p1", 6: "p2", 5: "pal5"}


def _frame():
    z = np.load(os.path.join(FIX, "frame.npz"))
    mode, base, off = map(int, z["regs"])
    return oam.Snapshot(z["oam"].tobytes(), z["cg"].tobytes(), z["vram"].tobytes(), mode, base, off), z["screen"]


def _sprite_mask(screen):
    """Sprites-only screen: the backdrop is one colour per row; anything else is a sprite pixel."""
    out = np.zeros(screen.shape[:2], bool)
    for y in range(screen.shape[0]):
        vals, counts = np.unique(screen[y], axis=0, return_counts=True)
        out[y] = np.any(screen[y] != vals[counts.argmax()], -1)
    return out


def test_rebuild_matches_mesens_sprites_only_screen():
    snap, screen = _frame()
    rgb, mask = oam.screen_layer(snap)
    truth = _sprite_mask(screen)
    iou = (mask & truth).sum() / (mask | truth).sum()
    assert iou > 0.99, iou
    both = mask & truth
    close = np.abs(rgb.astype(int) - screen.astype(int)).max(-1) <= 8
    assert (close & both).sum() / both.sum() > 0.99


def test_rebuild_equals_the_golden_crops():
    snap, _ = _frame()
    groups = oam.render_groups(snap, pals=set(GROUPS))
    assert set(groups) == set(GROUPS)
    for pal, name in GROUPS.items():
        want = np.asarray(Image.open(os.path.join(FIX, "expected_%s.png" % name)).convert("RGBA"))
        assert groups[pal].rgba.shape == want.shape, name
        assert np.array_equal(groups[pal].rgba, want), name


def test_groups_are_cut_without_each_other():
    """Each group's crop holds only its own palette's pixels: alpha exactly where its colour index is non-zero."""
    snap, _ = _frame()
    for g in oam.render_groups(snap).values():
        assert np.array_equal(g.rgba[..., 3] > 0, g.index > 0)
        i = g.index > 0                                                       # tight bbox: every edge has a pixel
        assert i[0].any() and i[-1].any() and i[:, 0].any() and i[:, -1].any()


def test_snapshot_rejects_wrong_sizes():
    import pytest
    with pytest.raises(ValueError):
        oam.Snapshot(b"\0" * 543, b"\0" * 512, b"\0" * oam.VRAM_LEN, 0, 24576, 4096)


def regen():
    from sf2.emu.vs import VARS
    from sf2.sprites.emu import open_mesen, video
    state = open(os.path.join(os.path.dirname(FIX), "..", "..", "states", "vs_ken_vs_ryu.state"), "rb").read()
    qcf = [["down"]] * 3 + [["down", "right"]] * 3 + [["right", "y"]] * 2
    p1 = qcf + [[]] * 40
    best = None
    with open_mesen(52190, hide_bg=True) as b:
        b.set_capture("raw")
        b.set_vars(VARS)
        b.load_state(state)
        for f in range(len(p1)):
            o = b.run([p1[f]], caps=[1], p2=[[]])
            snap = video(b)
            pals = {e.pal for e in oam.entries(snap)}
            if {4, 5, 6} <= pals:
                best = (snap, o.images[1])
                break
    if best is None:
        raise SystemExit("no frame with palettes 4, 5 and 6")
    snap, screen = best
    os.makedirs(FIX, exist_ok=True)
    np.savez_compressed(os.path.join(FIX, "frame.npz"), oam=np.frombuffer(snap.oam, np.uint8),
                        cg=np.frombuffer(snap.cg, np.uint8), vram=np.frombuffer(snap.vram, np.uint8),
                        regs=np.array([snap.mode, snap.base, snap.off]), screen=screen)
    for pal, g in oam.render_groups(snap, pals=set(GROUPS)).items():
        Image.fromarray(g.rgba, "RGBA").save(os.path.join(FIX, "expected_%s.png" % GROUPS[pal]))


if __name__ == "__main__" and "--regen" in sys.argv:
    regen()
