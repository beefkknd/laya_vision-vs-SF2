"""Headless runs, with tests/lua/fake_mesen_binary.sh standing in for `Mesen --testrunner`."""
import os
import shutil
import subprocess

import numpy as np
import pytest

from sf2.headless import bridge_for_port
from sf2.mesen import MesenBridge

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LUA = shutil.which("lua5.4")
FAKE = os.path.join(ROOT, "tests/lua/fake_mesen_binary.sh")
pytestmark = pytest.mark.skipif(
    not LUA or subprocess.run([LUA or "false", "-e", 'require("socket.core")'], capture_output=True).returncode,
    reason="needs lua5.4 with LuaSocket")

MAP = "my_hp 0x530 2 1\nopp_hp 0x730 2 1\nmy_x 0x522 2 0\nopp_x 0x722 2 0\nmy_y 0x526 2 0\nopp_y 0x726 2 0\n"


@pytest.fixture
def work(tmp_path, monkeypatch):
    from PIL import Image

    monkeypatch.delenv("SF2_BRIDGE_PORT", raising=False)
    monkeypatch.setenv("SF2_UNVERIFIED", "1")  # the fake Mesen is not the ROM the harness was verified on
    Image.fromarray(np.full((224, 256, 3), 77, np.uint8)).save(tmp_path / "rom.png")
    (tmp_path / "map.txt").write_text(MAP)
    (tmp_path / "start.state").write_bytes(b"0,144,144,80,176,200,200")  # the mock's savestate format
    monkeypatch.chdir(tmp_path)
    return tmp_path


def test_port_is_baked_into_a_copy_of_the_script(work):
    path = bridge_for_port(47955, out_dir=str(work / "bridge"))
    assert 'local HOST, PORT = "127.0.0.1", 47955' in open(path).read()
    b = MesenBridge(47955, launch=[FAKE, "--testrunner", str(work / "rom.png"), path], timeout=10)
    b.set_vars([])
    b.set_capture("raw")
    obs = b.run([["right"]] * 3, caps=[3])
    img = obs.images[3]
    assert img.shape == (224, 256, 3) and (img[:, 86] == 255).all() and img[0, 0].tolist() == [0x10, 0x20, 0x30]
    b.close()

