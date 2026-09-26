"""Headless + parallel runs, with tests/lua/fake_mesen_binary.sh standing in for `Mesen --testrunner`."""
import json
import os
import shutil
import subprocess
import sys

import numpy as np
import pytest

from sf2 import dataset as D
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


def test_parallel_collect_and_play_over_headless_workers(work):
    common = ["--rom", str(work / "rom.png"), "--mesen", FAKE, "--ram-map", "map.txt",
              "--savestate", "start.state", "--jitter", "20"]
    run = lambda *a: subprocess.run([sys.executable, os.path.join(ROOT, "scripts/parallel.py"), *a],  # noqa: E731
                                    capture_output=True, text=True, timeout=300)
    out = run("--workers", "2", "--base-port", "47960", "collect_teacher", "--name", "seed", "--decisions",
              "300", "--eps", "0.3", *common)
    assert out.returncode == 0, out.stdout + out.stderr + open("out/parallel/seed_w0.log").read()
    assert "switching to the raw screen buffer" in open("out/parallel/seed_w0.log").read()
    merged = D.read("data/seed/train.jsonl") + D.read("data/seed/val.jsonl")
    assert len(merged) == 300
    assert {r["id"].split("-")[0] for r in merged} == {"seed_w0", "seed_w1"}
    for im in merged[-1]["images"]:
        assert os.path.exists(os.path.normpath(os.path.join("data/seed", im)))
    # the two workers did not replay the identical fight
    w0 = [r["meta"]["dx"] for r in D.read("data/seed_w0/train.jsonl")][:50]
    w1 = [r["meta"]["dx"] for r in D.read("data/seed_w1/train.jsonl")][:50]
    assert w0 != w1

    vt = pytest.importorskip("laya.vlm_train")
    ex = vt.load_jsonl_examples("data", "seed", "train")
    assert ex and all(os.path.exists(p) for p in ex[0]["state"]["images"])

    out = run("--workers", "2", "--base-port", "47970", "play_teacher", "--name", "teacher", "--matches", "2",
              *common)
    assert out.returncode == 0, out.stdout + out.stderr
    g = json.load(open("rollouts/teacher/gate.json"))
    assert g["workers"] == 2 and g["rounds"] >= 2 and g["decisions"] > 0
