"""Python end of mesen/sf2_bridge.lua: a local TCP server that Mesen's Lua script connects to.

Start the Python script first (it listens), then load the Lua script in Mesen. With ``launch`` set (e.g.
``/Applications/Mesen.app/Contents/MacOS/Mesen --testrunner ~/roms/sf2.sfc mesen/sf2_bridge.lua``), Python
starts Mesen itself once it is listening.
"""
import io
import shlex
import socket
import subprocess
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np

from ..config import MESEN_PORT


@dataclass
class Obs:
    rams: List[List[int]]                 # RAM vars before frame 0..n (n+1 rows for a RUN of n frames)
    inputs: List[List[str]] = field(default_factory=list)   # WATCH: buttons the player held, per frame
    images: Dict[int, np.ndarray] = field(default_factory=dict)  # frame index -> RGB HxWx3
    state: Optional[bytes] = None         # savestate bytes (SAVESTATE, or F9 during WATCH)


def decode_png(data: bytes) -> np.ndarray:
    from PIL import Image

    with Image.open(io.BytesIO(data)) as im:
        return np.asarray(im.convert("RGB")).copy()


def _pad(buttons: Sequence[str]) -> str:
    return "+".join(buttons) if buttons else "-"


class MesenBridge:
    def __init__(self, port: int = MESEN_PORT, launch=None, timeout: float = 300.0):
        srv = socket.create_server(("127.0.0.1", port))
        srv.settimeout(timeout)
        self.proc = None
        print("waiting for Mesen on 127.0.0.1:%d - load mesen/sf2_bridge.lua in Mesen's Script Window" % port,
              flush=True)
        if launch:  # a command line string, or an argv list (paths with spaces)
            if isinstance(launch, str):
                self.proc = subprocess.Popen(shlex.split(launch))
            else:
                # Mesen's test runner emits emulator diagnostics for every worker.
                self.proc = subprocess.Popen(list(launch), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        try:
            conn, _ = srv.accept()
            conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            self.sock = conn
            self.f = conn.makefile("rwb", buffering=0)
            self._send_lock = threading.Lock()
            hello = self._line().split(" ", 2)
            if hello[0] != "HELLO":
                raise RuntimeError("unexpected greeting from Mesen: %r" % hello)
        except BaseException:
            if self.proc:  # a Mesen we launched must not outlive us (headless: --timeout is a week)
                self.proc.kill()
                self.proc.wait()
            raise
        finally:
            srv.close()
        self.rom_sha1 = hello[1]
        self.rom_name = hello[2] if len(hello) > 2 else "?"
        print("Mesen connected: %s (sha1 %s)" % (self.rom_name, self.rom_sha1), flush=True)
        self.nvars = 0

    # ------------------------------------------------------------------ wire
    def _line(self) -> str:
        buf = bytearray()
        while True:
            c = self.f.read(1)
            if not c:
                raise ConnectionError("Mesen closed the connection")
            if c == b"\n":
                return buf.decode()
            buf += c

    def _read(self, n: int) -> bytes:
        out = bytearray()
        while len(out) < n:
            chunk = self.f.read(n - len(out))
            if not chunk:
                raise ConnectionError("Mesen closed the connection")
            out += chunk
        return bytes(out)

    def _send(self, s: str, payload: bytes = b"") -> None:
        with self._send_lock:
            self.sock.sendall(s.encode() + b"\n" + payload)

    @contextmanager
    def keep_alive(self, interval: float = 0.5):
        """Prevent a headless Mesen test runner timing out while a model predicts."""
        done = threading.Event()

        def tick():
            while not done.wait(interval):
                try:
                    self._send("KEEP")
                except OSError:
                    return

        thread = threading.Thread(target=tick, daemon=True)
        thread.start()
        try:
            yield
        finally:
            done.set()
            thread.join()

    def _obs(self) -> Obs:
        head = self._line().split()
        if head[0] != "OBS":
            raise RuntimeError("expected OBS, got %r" % head)
        nr, ni, nimg, slen = map(int, head[1:5])
        rams = [[int(v) for v in line.split(",")] if (line := self._line()) else [] for _ in range(nr)]
        inputs = [[] if (l := self._line()) == "-" else l.split("+") for _ in range(ni)]
        images = {}
        for _ in range(nimg):
            h = self._line().split()
            if h[0] == "RAW":
                idx, w, hgt, ln = map(int, h[1:5])
                img = np.frombuffer(self._read(ln), np.uint8).reshape(hgt, w, 3)
                if hgt == 239:  # the raw buffer keeps the overscan rows Mesen's PNGs crop (top 7, bottom 8)
                    img = img[7:231]
                images[idx] = img.copy()
            else:
                images[int(h[1])] = decode_png(self._read(int(h[2])))
        state = self._read(slen) if slen else None
        return Obs(rams, inputs, images, state)

    # ------------------------------------------------------------------ commands
    def set_vars(self, specs: Sequence) -> None:
        """specs: objects with addr / size / signed (sf2.emu.ram.Var)."""
        lines = ["%s %d %d %d" % (v.name, v.addr, v.size, int(v.signed)) for v in specs]
        self._send("\n".join(["VARS %d" % len(lines)] + lines))
        if self._line() != "OK":
            raise RuntimeError("VARS not accepted")
        self.nvars = len(lines)

    def run(self, frames: Sequence[Sequence[str]], caps: Iterable[int] = (),
            p2: Optional[Sequence[Sequence[str]]] = None) -> Obs:
        """Apply one input set per frame (SNES button names); screenshots at the given frame indices
        (0 = before the first frame, len(frames) = after the last). ``p2``: controller 2's inputs, one per frame
        (VS BATTLE); without it controller 2 is left alone."""
        caps = sorted({c for c in caps if 0 <= c <= len(frames)})
        if p2 is not None and len(p2) != len(frames):
            raise ValueError("p2 has %d frames, p1 has %d" % (len(p2), len(frames)))
        spec = " ".join(_pad(f) if p2 is None else _pad(f) + "/" + _pad(g)
                        for f, g in zip(frames, p2 if p2 is not None else frames))
        self._send("RUN %d %s %s" % (len(frames), ",".join(map(str, caps)) or "-", spec))
        return self._obs()

    def watch(self, n: int, every: int) -> Obs:
        self._send("WATCH %d %d" % (n, every))
        return self._obs()

    def load_state(self, data: bytes) -> Obs:
        self._send("LOADSTATE %d" % len(data), data)
        return self._obs()

    def save_state(self) -> bytes:
        self._send("SAVESTATE")
        return self._obs().state

    def set_capture(self, mode: str) -> None:
        """'png' (emu.takeScreenshot) or 'raw' (screen buffer; for headless runs where PNGs come back blank)."""
        self._send("CAPTURE %s" % mode)
        if self._line() != "OK":
            raise RuntimeError("CAPTURE not accepted")

    def reset(self) -> Obs:
        self._send("RESET")
        return self._obs()

    def poke(self, writes: Dict[int, int]) -> None:
        """Write WRAM bytes now ({offset: value}, offsets into WRAM, values 0-255)."""
        for a, v in writes.items():
            if not (0 <= a < 0x20000 and 0 <= v < 256):
                raise ValueError("bad POKE %r=%r" % (a, v))
        if not writes:
            return
        self._send("POKE " + " ".join("%d %d" % kv for kv in writes.items()))
        if self._line() != "OK":
            raise RuntimeError("POKE not accepted")

    def dump_wram(self) -> bytes:
        self._send("DUMP")
        head = self._line().split()
        return self._read(int(head[1]))

    def close(self) -> None:
        """Disconnect. Mesen keeps running (the script waits for the next Python run) unless we launched it."""
        try:
            self._send("EXIT" if self.proc else "QUIT")
        except OSError:
            pass
        self.sock.close()
        if self.proc:        # a Mesen we launched: a window ignores EXIT, so terminate, then kill
            for stop in (None, self.proc.terminate, self.proc.kill):
                if stop:
                    stop()
                try:
                    self.proc.wait(3)
                    break
                except subprocess.TimeoutExpired:
                    continue
        time.sleep(0.1)
