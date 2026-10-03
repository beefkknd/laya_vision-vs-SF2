"""The emulator handle of screen-only play (docs/laya_text_only_plan.md, "The rule: no RAM in play"): screen frames out,
buttons in, the start savestate loaded - nothing else. Any other attribute (a RAM read, VARS, DUMP, POKE, a savestate
save, the bridge itself) raises RamForbidden; a report that carries RAM rows is refused too. Mesen runs
mesen/sf2_bridge_screen.lua, which has no RAM path at all.

The handle is also the run's input record: every frame's buttons (index = frame since the start state was loaded)
and a sha256 of every captured frame, so scripts/replay_score.py can replay the game offline and prove it identical.
"""
import contextlib
import hashlib
import os
import re
from typing import Dict, Iterator, List, Optional, Sequence

import numpy as np

from ..config import REPO
from ..emu.headless import KeepMesenSettings, launch_argv, window_argv
from ..emu.mesen import MesenBridge

SCREEN_BRIDGE = os.path.join(REPO, "mesen", "sf2_bridge_screen.lua")


class RamForbidden(RuntimeError):
    """Something in screen-only play tried to read RAM (or reach past the frames + buttons interface)."""


def frame_hash(img: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(img, np.uint8).tobytes()).hexdigest()


class ScreenEmu:
    """``load`` the start state once, then ``run`` frames of buttons; captures come back by absolute frame index."""

    __slots__ = ("__bridge", "frame", "inputs", "captures", "loaded")

    def __init__(self, bridge):
        self.__bridge = bridge
        self.frame = 0                      # frames run since the start state
        self.inputs: List[List[str]] = []   # buttons per frame
        self.captures: Dict[int, str] = {}  # frame index -> sha256 of the RGB bytes
        self.loaded = False

    def __getattr__(self, name):            # only reached for names that are not the interface
        raise RamForbidden("screen-only play: the emulator gives frames and takes buttons; %r is not available" % name)

    def _images(self, obs, offset: int) -> Dict[int, np.ndarray]:
        if obs.rams:
            raise RamForbidden("the bridge sent %d RAM rows: screen-only play needs mesen/sf2_bridge_screen.lua"
                               % len(obs.rams))
        out = {}
        for k, img in obs.images.items():
            out[offset + k] = img
            self.captures[offset + k] = frame_hash(img)
        return out

    def load(self, state: bytes) -> np.ndarray:
        """Load the start state; the frame it lands on (frame 0)."""
        if self.loaded:
            raise ValueError("this handle already holds a game from a start state: one record per round")
        self.loaded = True
        return self._images(self.__bridge.load_state(state), 0)[0]

    def run(self, frames: Sequence[Sequence[str]], caps: Sequence[int] = ()) -> Dict[int, np.ndarray]:
        """Press ``frames`` (one button list per frame); the frames captured at ``caps`` (relative: 0 = before the
        first, len(frames) = after the last), keyed by absolute frame index."""
        if not self.loaded:
            raise ValueError("load the start state first")
        frames = [list(f) for f in frames]
        imgs = self._images(self.__bridge.run(frames, caps=caps), self.frame)
        self.inputs.extend(frames)
        self.frame += len(frames)
        return imgs

    def new_round(self) -> "ScreenEmu":
        """A fresh record on the same emulator (the next round loads its start state again)."""
        return ScreenEmu(self.__bridge)


def screen_bridge_for_port(port: int, out_dir: str = os.path.join(REPO, "out", "bridge")) -> str:
    with open(SCREEN_BRIDGE) as f:
        src = f.read()
    src, n = re.subn(r'local HOST, PORT = "127\.0\.0\.1", \d+', 'local HOST, PORT = "127.0.0.1", %d' % port, src)
    src, m = re.subn(r"local EXIT_ON_DISCONNECT = false", "local EXIT_ON_DISCONNECT = true", src)
    if n != 1 or m != 1:
        raise RuntimeError("could not find the PORT / EXIT_ON_DISCONNECT lines in %s" % SCREEN_BRIDGE)
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "sf2_bridge_screen_%d.lua" % port)
    with open(path, "w") as f:
        f.write(src)
    return path


@contextlib.contextmanager
def open_screen(port: int, rom: Optional[str] = None, show_window: bool = False, speed: int = 100) -> Iterator[ScreenEmu]:
    """Mesen running the screen-only bridge, raw capture; always closed. The handle yielded is the first
    round's record (``new_round`` for the next). ``watch=True`` opens a VISIBLE Mesen window you can watch the match
    in (``speed`` percent), instead of the headless test runner; the bridge is identical, so play is unchanged."""
    argv = window_argv(port, rom, speed=speed) if show_window else launch_argv(port, rom)
    if not argv[-1].endswith("sf2_bridge_%d.lua" % port):
        raise RuntimeError("unexpected Mesen command line: %r" % argv)
    argv = argv[:-1] + [screen_bridge_for_port(port)]
    keep = KeepMesenSettings() if show_window else contextlib.nullcontext()
    with keep:
        b = MesenBridge(port, launch=argv)
        try:
            b.set_capture("raw")
            yield ScreenEmu(b)
        finally:
            b.close()
