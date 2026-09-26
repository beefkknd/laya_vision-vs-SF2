"""Headless Mesen: one windowless Mesen process per worker, each with its own port.

Mesen's test runner (``Mesen --testrunner <rom> <script.lua>``) emulates without a window and as fast as the
host allows. Each worker gets a copy of mesen/sf2_bridge.lua with its port written in, so no Lua I/O
permission is needed; "Allow network access" must still be ticked once in Mesen's Script Window settings
(the test runner reads the same settings).
"""
import os
import re
from typing import List

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BRIDGE = os.path.join(ROOT, "mesen", "sf2_bridge.lua")
MAC_MESEN = "/Applications/Mesen.app/Contents/MacOS/Mesen"


def bridge_for_port(port: int, out_dir: str = os.path.join(ROOT, "out", "bridge")) -> str:
    with open(BRIDGE) as f:
        src = f.read()
    src, n = re.subn(r'local HOST, PORT = "127\.0\.0\.1", \d+', 'local HOST, PORT = "127.0.0.1", %d' % port, src)
    if n != 1:
        raise RuntimeError("could not find the PORT line in %s" % BRIDGE)
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
    rom = os.path.expanduser(rom or os.environ.get("SF2_ROM", ""))
    if not rom or not os.path.exists(rom):
        raise FileNotFoundError("ROM not found: pass --rom or set SF2_ROM")
    # The test runner's default wall-clock limit is 100 seconds, shorter than a full student match.
    # Running unthrottled, Mesen skips rendering frames on a wall-clock timer: screenshots would lag the RAM by
    # 0-3 frames, differently every run. Rendering every frame makes them exact and deterministic.
    return [find_mesen(mesen), "--testrunner", "--timeout=3600", "--snes.disableFrameSkipping=true", rom,
            bridge_for_port(port)]
