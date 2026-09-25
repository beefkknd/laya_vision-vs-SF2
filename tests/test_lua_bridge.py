"""The real mesen/sf2_bridge.lua, run by a stock Lua 5.4 + LuaSocket against a mock of Mesen's emu API,
talking to the real MesenBridge / FightEnv / RAM finder. Skipped when lua5.4 or LuaSocket is missing."""
import os
import shutil
import subprocess

import numpy as np
import pytest

from sf2 import ramsearch
from sf2.env import FightEnv
from sf2.mesen import MesenBridge
from sf2.ram import parse_map

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LUA = shutil.which("lua5.4")

pytestmark = pytest.mark.skipif(
    not LUA or subprocess.run([LUA or "false", "-e", 'require("socket.core")'], capture_output=True).returncode,
    reason="needs lua5.4 with LuaSocket")

MAP = parse_map("""
my_hp 0x530 2 1
opp_hp 0x730 2 1
my_x 0x522 2 0
opp_x 0x722 2 0
my_y 0x526 2 0
opp_y 0x726 2 0
""")


@pytest.fixture
def bridge(tmp_path, monkeypatch):
    monkeypatch.setenv("SF2_BRIDGE_PORT", "47923")
    from PIL import Image

    png = tmp_path / "screen.png"
    Image.fromarray(np.full((224, 256, 3), 77, np.uint8)).save(png)
    cmd = "%s %s %s %s" % (LUA, os.path.join(ROOT, "tests/lua/mock_mesen.lua"),
                           os.path.join(ROOT, "mesen/sf2_bridge.lua"), png)
    b = MesenBridge(47923, launch=cmd, timeout=10)
    yield b
    b.close()


def test_env_over_the_real_lua_bridge(bridge):
    assert bridge.rom_sha1 == "mocksha1"
    bridge.set_vars([])
    state = bridge.save_state()
    assert state.startswith(b"0,144,144,80,176")
    env = FightEnv(bridge, MAP, state)
    img = env.reset()
    assert img.shape == (224, 256, 3) and img[0, 0, 0] == 77
    assert env.full_hp == 144 and env.f.my_x == 80 and env.f.facing_right
    x0 = env.f.my_x
    res = env.act("forward")
    assert res.frames == 4 and env.f.my_x == x0 + 8  # right held 4 frames, 2 px each
    env.act("jump")
    assert env.airborne()[0]
    for _ in range(12):
        env.act("forward")
    dealt = sum(env.act("hp").dmg_for for _ in range(6))
    assert dealt > 0
    env.reset()                                        # savestate reload puts everything back
    assert env.f.my_x == 80 and env.f.opp_hp == 144
    watch = bridge.watch(8, 4)
    assert len(watch.inputs) == 8 and set(watch.images) == {0, 4} and len(watch.rams) == 9


def test_find_ram_scripted_phases_over_the_real_lua_bridge(bridge, monkeypatch):
    import importlib.util

    monkeypatch.syspath_prepend(os.path.join(ROOT, "scripts"))
    spec = importlib.util.spec_from_file_location("find_ram", os.path.join(ROOT, "scripts/find_ram.py"))
    find_ram = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(find_ram)
    bridge.set_vars([])
    state = bridge.save_state()
    ph = find_ram.scripted(bridge, state)
    chosen, report = ramsearch.analyze(ph)
    assert {v.name: v.addr for v in chosen} == {v.name: v.addr for v in MAP}
    assert report["stride"] == 0x200


def test_bridge_reconnects_to_the_next_python_run(bridge):
    import threading

    bridge.set_vars([])
    bridge.run([["right"]] * 3)
    nxt = {}
    proc, bridge.proc = bridge.proc, None     # a QUIT (keep Mesen running), not an EXIT
    port = bridge.sock.getsockname()[1]
    t = threading.Thread(target=lambda: nxt.update(b=MesenBridge(port, timeout=10)), daemon=True)
    t.start()
    bridge.close()
    t.join(10)
    b2 = nxt["b"]
    assert b2.rom_sha1 == "mocksha1"
    b2.set_vars(MAP)
    obs = b2.run([[]], caps=[1])
    assert obs.rams[0][2] == 86 and 1 in obs.images     # the game kept its state: x = 80 + 3 * 2
    b2.proc = proc                                        # let the fixture's close() end the Lua process
    bridge.sock, bridge.f, bridge.proc = b2.sock, b2.f, proc
