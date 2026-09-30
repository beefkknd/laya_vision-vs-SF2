"""One text laya process per checkpoint, shared by every run (2026-09-30: each run's own ~6.3 GB helper, 12-24 per
batch, held 75-150 GB of identical weights). Plain stdlib: the server side runs in the laya-mlx venv
(scripts/text_laya_server.py --shared), the client side in the torch venv (sf2.system1.advisor.Advisor).

    socket  config.TEXT_LAYA_CACHE/text_laya_<hash of the checkpoint>.sock, with .lock (flock) and .log next to it
    on connect the server writes {"ready": <checkpoint>, "pid": <server pid>}
    then, like the per-run helper: request {"text", "question"} per line -> {"probabilities"} or {"error"} per line

Requests from all clients go through one queue to the thread that owns the model, one at a time (the MLX model is not
shared across threads), and each reply goes back on its own client's connection. ``connect`` starts the server when
none answers, under an flock, so runs starting together end with exactly one; the server leaves (under the same lock)
after ``idle_s`` with no client connected, so nothing stays running.
"""
import errno
import fcntl
import hashlib
import json
import os
import queue
import socket
import subprocess
import threading
import time
from typing import Callable, Dict, List, Optional

from ..config import TEXT_LAYA_BASE, TEXT_LAYA_CACHE, TEXT_LAYA_IDLE_S, TEXT_LAYA_START_S, TEXT_LAYA_TIMEOUT_S

POLL = 0.2          # seconds between checks: the server's idle check and accept loop, a starter's wait for ready
BACKLOG = 64
ATTEMPTS = 3        # connect tries (a server may be leaving just as a client arrives)
PROBE_S = 10.0      # a server that does not greet in this long is busy, not dead

Predict = Callable[[str, Dict], Dict[str, float]]


def socket_path(checkpoint: str, cache: str = TEXT_LAYA_CACHE) -> str:
    """The socket of ``checkpoint``'s server: one per checkpoint folder and adapter file version (a retrained
    checkpoint at the same path gets a new server, never the old weights)."""
    key = "none" if checkpoint == "none" else os.path.realpath(checkpoint)
    adapter = os.path.join(key, "adapter.safetensors")
    stamp = str(os.stat(adapter).st_mtime_ns) if key != "none" and os.path.exists(adapter) else ""
    digest = hashlib.sha1("\0".join((TEXT_LAYA_BASE, key, stamp)).encode()).hexdigest()[:16]
    return os.path.join(cache, "text_laya_%s.sock" % digest)


# ---- server ----

def answer(predict: Predict, line: str) -> str:
    """One request line -> its reply line, exactly as the per-run helper writes it."""
    try:
        req = json.loads(line)
        reply = {"probabilities": predict(req["text"], req["question"])}
    except Exception as e:                      # a bad request must not kill the server: report it
        reply = {"error": "%s: %s" % (type(e).__name__, e)}
    return json.dumps(reply) + "\n"


class _State:
    def __init__(self):
        self.lock = threading.Lock()
        self.clients = 0
        self.idle_since = time.monotonic()
        self.closing = False


def _client_thread(conn: socket.socket, jobs: "queue.Queue", st: _State, hello: str) -> None:
    try:
        conn.sendall(hello.encode())
        with conn.makefile("r", encoding="utf-8", newline="\n") as lines:   # (an open makefile keeps conn open)
            for line in lines:
                slot: "queue.Queue" = queue.Queue(1)
                jobs.put((line, slot))
                conn.sendall(slot.get().encode())
    except OSError:
        pass                                    # the client left; nothing to answer
    finally:
        conn.close()
        with st.lock:
            st.clients -= 1
            if not st.clients:
                st.idle_since = time.monotonic()


def _accept_loop(listener: socket.socket, jobs: "queue.Queue", st: _State, hello: str) -> None:
    while True:
        try:
            conn, _ = listener.accept()
        except socket.timeout:
            if st.closing:
                return
            continue
        except OSError:
            return
        with st.lock:
            if st.closing:                      # arrived as the server leaves: the client sees EOF and restarts one
                conn.close()
                continue
            st.clients += 1
        conn.setblocking(True)
        threading.Thread(target=_client_thread, args=(conn, jobs, st, hello), daemon=True).start()


def _idle(st: _State, idle_s: float) -> bool:
    """No client connected, for ``idle_s`` seconds (the caller holds ``st.lock``)."""
    return not st.clients and time.monotonic() - st.idle_since >= idle_s


def _try_leave(sock: str, inode: int, st: _State, idle_s: float) -> bool:
    """Leave if idle, under the starters' lock (never while one is starting or checking a server)."""
    with st.lock:
        if not _idle(st, idle_s):
            return False
    with open(sock + ".lock", "a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return False
        with st.lock:
            if not _idle(st, idle_s):           # a client may have come while we took the lock
                return False
            st.closing = True
            try:
                if os.stat(sock).st_ino == inode:   # only our own socket file
                    os.unlink(sock)
            except FileNotFoundError:
                pass
        return True


def serve(predict: Predict, sock: str, label: str, idle_s: float = TEXT_LAYA_IDLE_S) -> None:
    """Answer every client on ``sock`` until no client has been connected for ``idle_s`` seconds. ``predict`` runs
    only in the calling thread."""
    listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    listener.bind(sock)
    inode = os.stat(sock).st_ino
    listener.listen(BACKLOG)
    listener.settimeout(POLL)
    jobs: "queue.Queue" = queue.Queue()
    st = _State()
    hello = json.dumps({"ready": label, "pid": os.getpid()}) + "\n"
    acceptor = threading.Thread(target=_accept_loop, args=(listener, jobs, st, hello), daemon=True)
    acceptor.start()
    try:
        while True:
            try:
                line, slot = jobs.get(timeout=POLL)
            except queue.Empty:
                if _try_leave(sock, inode, st, idle_s):
                    return
                continue
            slot.put(answer(predict, line))
    finally:
        st.closing = True
        acceptor.join(POLL * 5)
        listener.close()


# ---- client ----

class Client:
    """One connection to a running server; raises RuntimeError (never hangs past ``timeout``) if it dies."""

    def __init__(self, sock: str, timeout: float = TEXT_LAYA_TIMEOUT_S):
        self.sock, self.timeout = sock, timeout
        self.conn = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.conn.settimeout(timeout)
        self.rfile = self.conn.makefile("r", encoding="utf-8", newline="\n")
        try:
            self.conn.connect(sock)
            hello = json.loads(self._line())
        except (OSError, ValueError, RuntimeError):
            self.close()
            raise
        if "ready" not in hello:
            self.close()
            raise RuntimeError("text laya server at %s did not greet: %s" % (sock, hello))
        self.label, self.server_pid = hello["ready"], hello["pid"]

    def _line(self) -> str:
        try:
            line = self.rfile.readline()
        except socket.timeout:
            raise RuntimeError("text laya server at %s: no reply in %.0f s" % (self.sock, self.timeout)) from None
        except OSError as e:
            raise RuntimeError("text laya server at %s closed the connection (%s)" % (self.sock, e)) from None
        if not line:
            raise RuntimeError("text laya server at %s closed the connection" % self.sock)
        return line

    def request_line(self, line: str) -> str:
        """Send one request line, return the reply line as the server wrote it."""
        try:
            self.conn.sendall(line.encode())
        except OSError as e:
            raise RuntimeError("text laya server at %s closed the connection (%s)" % (self.sock, e)) from None
        return self._line()

    def ask(self, text: str, q: Dict) -> Dict:
        """The reply: {"probabilities": ...} or {"error": ...}."""
        return json.loads(self.request_line(json.dumps({"text": text, "question": q}) + "\n"))

    def close(self) -> None:
        self.rfile.close()                      # an open makefile would keep the connection (and the server) up
        self.conn.close()


def _probe(sock: str) -> str:
    """"up" (a server greeted), "dead" (no socket file, or nobody listening on it) or "busy" (listening but no
    greeting in PROBE_S: a live server under load, never to be replaced)."""
    try:
        Client(sock, timeout=PROBE_S).close()
        return "up"
    except (FileNotFoundError, ConnectionRefusedError):
        return "dead"
    except OSError as e:
        if e.errno == errno.ENOTSOCK:           # a plain file where the socket should be
            return "dead"
        return "busy"
    except (ValueError, RuntimeError):
        return "busy"


def _start(sock: str, cmd: List[str], start_timeout: float, env: Optional[Dict], cwd: Optional[str]) -> None:
    """Start a server unless one is there, holding the lock until it is ready (so exactly one starts). A start that
    failed while this run waited for the lock is not tried again (every waiting run would repeat its wait)."""
    os.makedirs(os.path.dirname(sock), exist_ok=True)
    failed = sock + ".failed"
    asked = time.time()
    with open(sock + ".lock", "a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if _probe(sock) != "dead":
            return                              # up, or busy (the caller's connect tries report it)
        if os.path.exists(failed) and os.stat(failed).st_mtime >= asked:
            raise RuntimeError("text laya server start failed just now: %s" % open(failed).read().strip())
        if os.path.lexists(sock):
            os.unlink(sock)                     # left by a server that was killed
        with open(sock + ".log", "a") as log:
            proc = subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, env=env,
                                    cwd=cwd, start_new_session=True)   # outlives the run that started it
        try:
            _wait_ready(sock, proc, start_timeout)
        except RuntimeError as e:
            with open(failed, "w") as f:
                f.write(str(e))
            raise
        if os.path.exists(failed):
            os.unlink(failed)
        threading.Thread(target=proc.wait, daemon=True).start()   # reap it when it leaves (no zombie in our run)


def _wait_ready(sock: str, proc: subprocess.Popen, start_timeout: float) -> None:
    end = time.monotonic() + start_timeout
    while not (os.path.exists(sock) and _probe(sock) == "up"):
        if proc.poll() is not None:
            raise RuntimeError("text laya server exited (code %s) before it was ready; see %s.log"
                               % (proc.returncode, sock))
        if time.monotonic() > end:
            proc.kill()
            proc.wait()
            raise RuntimeError("text laya server not ready after %.0f s; see %s.log" % (start_timeout, sock))
        time.sleep(POLL)


def connect(sock: str, cmd: List[str], timeout: float = TEXT_LAYA_TIMEOUT_S,
            start_timeout: float = TEXT_LAYA_START_S, env: Optional[Dict] = None, cwd: Optional[str] = None) -> Client:
    """A client of the server on ``sock``, starting it with ``cmd`` if none answers."""
    last: Exception = RuntimeError("no try")
    for _ in range(ATTEMPTS):
        try:
            return Client(sock, timeout)
        except (OSError, ValueError, RuntimeError) as e:
            last = e
        _start(sock, cmd, start_timeout, env, cwd)
    raise RuntimeError("cannot reach the text laya server at %s: %s" % (sock, last))
