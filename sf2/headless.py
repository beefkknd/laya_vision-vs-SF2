"""Headless Mesen: one windowless Mesen process per worker, each with its own port.

Mesen's test runner (``Mesen --testrunner <rom> <script.lua>``) emulates without a window and as fast as the
host allows. Each worker gets a copy of mesen/sf2_bridge.lua with its port written in, so no Lua I/O
permission is needed; "Allow network access" must still be ticked once in Mesen's Script Window settings
(the test runner reads the same settings).
"""
import os
import re
from typing import List

from .config import DEFAULT_ROM

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BRIDGE = os.path.join(ROOT, "mesen", "sf2_bridge.lua")
MAC_MESEN = "/Applications/Mesen.app/Contents/MacOS/Mesen"


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
    # ~/Applications is where a per-user drag install lands; the app may keep its "Mesen 2" download name
    user_apps = ["~/Applications/%s.app/Contents/MacOS/Mesen" % app for app in ("Mesen", "Mesen 2")]
    for cand in (explicit, os.environ.get("SF2_MESEN"), MAC_MESEN, *user_apps):
        if cand and os.path.exists(os.path.expanduser(cand)):
            return os.path.expanduser(cand)
    raise FileNotFoundError("Mesen binary not found: pass --mesen or set SF2_MESEN (on macOS usually %s)"
                            % MAC_MESEN)


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


def window_argv(port: int, rom: str = None, mesen: str = None, speed: int = 100) -> List[str]:
    """A Mesen window you can watch, running the bridge for ``port`` (same bridge copy as headless). ``speed``: emulation
    speed in percent (150 measured at 90 fps). The window saves these overrides into Mesen's settings when it closes:
    run it inside KeepMesenSettings."""
    argv = launch_argv(port, rom, mesen)
    return [argv[0], "--emulation.emulationSpeed=%d" % speed] + [
        a for a in argv[1:] if not a.startswith(("--testrunner", "--timeout"))]


SETTINGS = os.path.expanduser("~/Library/Application Support/MesenCE/settings.json")


class KeepMesenSettings:
    """A windowed Mesen writes its settings back when it closes, including this run's command-line overrides (speed,
    controller 2). Use around a window's lifetime: the settings file is put back exactly as it was before."""

    def __enter__(self):
        self.saved = open(SETTINGS, "rb").read() if os.path.exists(SETTINGS) else None
        return self

    def __exit__(self, *exc):
        if self.saved is not None:
            with open(SETTINGS, "wb") as f:
                f.write(self.saved)
        return False
