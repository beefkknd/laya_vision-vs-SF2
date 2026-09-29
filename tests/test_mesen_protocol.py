"""MesenBridge against a scripted fake of mesen/sf2_bridge.lua's side of the socket."""
import io
import socket
import threading
import time

import numpy as np

from sf2.emu.mesen import MesenBridge
from sf2.emu.ram import Var


def png(color):
    from PIL import Image

    buf = io.BytesIO()
    Image.fromarray(np.full((224, 256, 3), color, np.uint8)).save(buf, format="PNG")
    return buf.getvalue()


def fake_lua(port, seen):
    for _ in range(100):
        try:
            s = socket.create_connection(("127.0.0.1", port))
            break
        except OSError:
            time.sleep(0.02)
    f = s.makefile("rwb", buffering=0)
    s.sendall(b"HELLO abc123 Street_Fighter_II\n")
    line = lambda: f.readline().decode().rstrip("\n")  # noqa: E731
    head = line()
    seen.append(head)
    n = int(head.split()[1])
    seen.extend(line() for _ in range(n))
    s.sendall(b"OK\n")
    run = line().split()
    seen.append(run)
    frames = int(run[1])
    rows = "".join("%d,%d\n" % (176 - i, 80 + i) for i in range(frames + 1))
    img = png(200)
    s.sendall(("OBS %d 0 1 0\n" % (frames + 1)).encode() + rows.encode()
              + ("IMG %d %d\n" % (frames, len(img))).encode() + img)
    seen.append(line())  # QUIT
    s.close()


def test_vars_run_and_images():
    port = 47911
    seen = []
    t = threading.Thread(target=fake_lua, args=(port, seen), daemon=True)
    t.start()
    b = MesenBridge(port, timeout=5)
    assert b.rom_sha1 == "abc123"
    b.set_vars([Var("my_hp", 0x530, 2, True), Var("my_x", 0x522, 2, False)])
    obs = b.run([["down"], ["down", "right"], ["right", "l"], []], caps=[4])
    b.close()
    t.join(2)
    assert seen[0] == "VARS 2" and seen[1] == "my_hp 1328 2 1"
    assert seen[3] == ["RUN", "4", "4", "down", "down+right", "right+l", "-"]
    assert seen[4] == "QUIT"
    assert obs.rams[0] == [176, 80] and obs.rams[-1] == [172, 84] and len(obs.rams) == 5
    assert set(obs.images) == {4} and obs.images[4].shape == (224, 256, 3) and obs.images[4][0, 0, 0] == 200
