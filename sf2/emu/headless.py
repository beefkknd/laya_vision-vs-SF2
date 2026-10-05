"""Headless Mesen: one windowless Mesen process per worker, each with its own port.

Mesen's test runner (``Mesen --testrunner <rom> <script.lua>``) emulates without a window and as fast as the
host allows. Each worker gets a copy of mesen/sf2_bridge.lua with its port written in, so no Lua I/O
permission is needed; "Allow network access" must still be ticked once in Mesen's Script Window settings
(the test runner reads the same settings).
"""
import copy
import json
import os
import re
from typing import List

from ..config import DEFAULT_ROM, MESEN_CANDIDATES, MESEN_SETTINGS, REPO

ROOT = REPO
BRIDGE = os.path.join(ROOT, "mesen", "sf2_bridge.lua")


def bridge_for_port(port: int, out_dir: str = os.path.join(ROOT, "out", "bridge")) -> str:
    with open(BRIDGE) as f:
        src = f.read()
    src, n = re.subn(r'local HOST, PORT = "127\.0\.0\.1", \d+', 'local HOST, PORT = "127.0.0.1", %d' % port, src)
    src, m = re.subn(r"local EXIT_ON_DISCONNECT = false", "local EXIT_ON_DISCONNECT = true", src)
    if n != 1 or m != 1:
        raise RuntimeError("could not find the PORT / EXIT_ON_DISCONNECT lines in %s" % BRIDGE)
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "sf2_bridge_%d.lua" % port)
    with open(path, "w") as f:
        f.write(src)
    return path


def find_mesen(explicit: str = None) -> str:
    for cand in (explicit, *MESEN_CANDIDATES):
        if cand and os.path.exists(os.path.expanduser(cand)):
            return os.path.expanduser(cand)
    raise FileNotFoundError("Mesen binary not found: pass --mesen or set SF2_MESEN (on macOS usually %s)"
                            % MESEN_CANDIDATES[0])


def launch_argv(port: int, rom: str, mesen: str = None) -> List[str]:
    rom = os.path.expanduser(rom or os.environ.get("SF2_ROM") or os.path.join(ROOT, DEFAULT_ROM))
    if not os.path.exists(rom):
        raise FileNotFoundError("ROM not found at %s: pass --rom, set SF2_ROM, or copy it to %s" % (rom, DEFAULT_ROM))
    # --timeout is the test runner's total wall-clock limit (default 100 s); past it Mesen exits mid-run. A week.
    # Running unthrottled, Mesen skips rendering frames on a wall-clock timer: screenshots would lag the RAM by
    # 0-3 frames, differently every run. Rendering every frame makes them exact and deterministic.
    # Controller 2 is plugged in for this run only (Mesen's saved settings leave SNES port 2 empty), so Python can
    # play both sides of a VS BATTLE.
    return [find_mesen(mesen), "--testrunner", "--timeout=604800", "--snes.disableFrameSkipping=true",
            "--snes.port2.type=SnesController", rom, bridge_for_port(port)]


def window_argv(port: int, rom: str = None, mesen: str = None, speed: int = 100, volume: int = 100) -> List[str]:
    """A Mesen window you can watch, running the bridge for ``port`` (same bridge copy as headless). ``speed``: emulation
    speed in percent (150 measured at 90 fps). ``volume``: master volume in percent (0-100) for the watch session;
    Mesen's saved setting defaults it to ~3/100 (silent) and also ducks volume when the window is not the focused app,
    so the watch command forces an audible level and disables background ducking (dotted overrides, same mechanism as
    --emulation.emulationSpeed). The window saves these overrides into Mesen's settings when it closes: run it inside
    KeepMesenSettings, which restores the real settings afterward."""
    argv = launch_argv(port, rom, mesen)
    volume = max(0, min(100, int(volume)))      # Mesen rejects an out-of-range MasterVolume override
    overrides = ["--emulation.emulationSpeed=%d" % speed,
                 "--audio.masterVolume=%d" % volume,
                 "--audio.reduceSoundInBackground=false"]
    return [argv[0]] + overrides + [
        a for a in argv[1:] if not a.startswith(("--testrunner", "--timeout"))]


SETTINGS = MESEN_SETTINGS
# Parked-window coordinates: far off any normal desktop, so Mesen opens the Script Window out of the
# screen-recording. Mesen restores each window's SAVED geometry on open, so writing this into the settings
# before launch places the window there; KeepMesenSettings restores the real settings right after.
OFFSCREEN = 32000


def park_script_window(settings: dict) -> dict:
    """A COPY of ``settings`` with Mesen's Script Window ("console") parked off-screen and its log pane
    collapsed, for a clean --watch recording. The script still auto-runs (AutoStartScriptOnLoad is left on),
    so the game still plays; only the window is moved out of the way. Pure: the input is not mutated."""
    out = copy.deepcopy(settings)
    sw = out.setdefault("Debug", {}).setdefault("ScriptWindow", {})
    sw["WindowLocation"] = {"X": OFFSCREEN, "Y": OFFSCREEN}
    sw["WindowIsMaximized"] = False
    sw["LogWindowHeight"] = 0
    return out


class KeepMesenSettings:
    """A windowed Mesen writes its settings back when it closes, including this run's command-line overrides (speed,
    controller 2). Use around a window's lifetime: the settings file is put back exactly as it was before.

    ``hide_console=True`` ALSO parks the Script Window off-screen for the duration of the run (``park_script_window``),
    so a --watch recording shows only the game; the original settings are restored byte-for-byte on exit either way.
    ``path`` is the settings file (defaults to Mesen's; overridable for tests)."""

    def __init__(self, path: str = SETTINGS, hide_console: bool = False):
        self.path = path
        self.hide_console = hide_console

    def __enter__(self):
        self.saved = open(self.path, "rb").read() if os.path.exists(self.path) else None
        if self.hide_console and self.saved is not None:
            try:
                parked = park_script_window(json.loads(self.saved.decode("utf-8-sig")))
                # keep Mesen's UTF-8 BOM so it reads the file the same way it wrote it
                with open(self.path, "wb") as f:
                    f.write(("﻿" + json.dumps(parked, indent=2)).encode("utf-8"))
            except (ValueError, OSError):
                pass            # a settings file we cannot parse: leave it; the console just stays visible
        return self

    def __exit__(self, *exc):
        if self.saved is not None:
            with open(self.path, "wb") as f:
                f.write(self.saved)
        return False
