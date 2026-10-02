"""Headless Mesen with the sprite bridge (mesen/sf2_bridge_sprites.lua): launch, VIDEO, and a bridge proxy that
steps one frame at a time and hands every (RAM row, snapshot) to a sink in stream order.

Stream: after ``load_state`` the first row (index 0) has no snapshot; every later row k comes with the snapshot of
the frame emulated just before it. That snapshot is what the screen captured at row k shows, i.e. displayed row
k - LAG (sf2.data.perception.LAG = 1; checked on real frames in tests/test_sprites_live.py).
"""
import os
import re
from contextlib import contextmanager
from typing import Callable, List, Optional

from ..config import DEFAULT_ROM, REPO
from ..emu.headless import find_mesen
from ..emu.mesen import MesenBridge, Obs
from .oam import Snapshot

LUA = os.path.join(REPO, "mesen", "sf2_bridge_sprites.lua")
BRIDGE_DIR = os.path.join(REPO, "out", "sprite_bridge")


def bridge_copy(port: int) -> str:
    with open(LUA) as f:
        src = f.read()
    src, n = re.subn(r'local HOST, PORT = "127\.0\.0\.1", \d+', 'local HOST, PORT = "127.0.0.1", %d' % port, src)
    src, m = re.subn(r"local EXIT_ON_DISCONNECT = false", "local EXIT_ON_DISCONNECT = true", src)
    if (n, m) != (1, 1):
        raise RuntimeError("could not patch the port / exit lines of %s" % LUA)
    os.makedirs(BRIDGE_DIR, exist_ok=True)
    path = os.path.join(BRIDGE_DIR, "sf2_bridge_sprites_%d.lua" % port)
    with open(path, "w") as f:
        f.write(src)
    return path


def argv(port: int, rom: Optional[str] = None, hide_bg: bool = False) -> List[str]:
    rom = rom or os.environ.get("SF2_ROM") or os.path.join(REPO, DEFAULT_ROM)
    if not os.path.exists(rom):
        raise FileNotFoundError("ROM not found: %s" % rom)
    extra = ["--snes.hideBgLayer%d=true" % i for i in (1, 2, 3, 4)] if hide_bg else []
    return [find_mesen(), "--testrunner", "--timeout=604800", "--snes.disableFrameSkipping=true",
            "--snes.port2.type=SnesController", *extra, rom, bridge_copy(port)]


@contextmanager
def open_mesen(port: int, rom: Optional[str] = None, hide_bg: bool = False):
    b = MesenBridge(port, launch=argv(port, rom, hide_bg))
    try:
        yield b
    finally:
        b.close()


def video(b: MesenBridge) -> Snapshot:
    """The snapshot of the last emulated frame (VIDEO)."""
    b._send("VIDEO")
    head = b._line().split()
    if head[0] != "VID" or len(head) != 7:
        raise RuntimeError("expected VID, got %r" % head)
    mode, base, off, lo, lc, lv = map(int, head[1:])
    oam, cg, vram = b._read(lo), b._read(lc), b._read(lv)
    return Snapshot(oam, cg, vram, mode, base, off)


class SnapBridge:
    """Wraps a MesenBridge: ``run`` steps frame by frame (one RUN + one VIDEO each) and feeds ``sink(row, snap)``;
    returns the same Obs rows a plain run would (no images)."""

    def __init__(self, bridge: MesenBridge, names: List[str]):
        self.inner, self.names = bridge, names
        self.sink: Optional[Callable] = None
        self.fresh = True

    def load_state(self, state: bytes):
        self.fresh = True
        return self.inner.load_state(state)

    def run(self, frames, caps=(), p2=None) -> Obs:
        if self.sink is None:
            raise RuntimeError("SnapBridge.run without a sink")
        if caps:
            raise ValueError("SnapBridge takes no screenshots")
        rams = []
        for j, f in enumerate(frames):
            obs = self.inner.run([f], p2=None if p2 is None else [p2[j]])
            if self.fresh:
                self.sink(dict(zip(self.names, obs.rams[0])), None)
                self.fresh = False
            if not rams:
                rams.append(obs.rams[0])
            rams.append(obs.rams[1])
            self.sink(dict(zip(self.names, obs.rams[1])), video(self.inner))
        return Obs(rams)

    def __getattr__(self, name):
        return getattr(self.inner, name)
