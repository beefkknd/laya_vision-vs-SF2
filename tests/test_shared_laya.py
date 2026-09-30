"""One shared text laya per checkpoint (sf2.system1.shared_laya, scripts/text_laya_server.py --shared): the same
JSON-lines answers as the per-run helper, to many clients at once, exactly one server however many runs start it,
clear errors (never a hang) when the server dies, and an idle exit. The model is tests/fixtures/fake_laya.py."""
import io
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

import pytest

from sf2.system1 import shared_laya
from sf2.system1.advice import question

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "fixtures"))
import fake_laya  # noqa: E402

ROOT = os.path.dirname(HERE)
FAKE_SERVER = os.path.join(HERE, "fixtures", "fake_text_laya_server.py")
REAL_SERVER = os.path.join(ROOT, "scripts", "text_laya_server.py")
Q = question({"c.mk": "good", "s.hp": "fair", "forward": None})


@pytest.fixture
def short():
    """A short folder: a unix socket path must fit 104 bytes on macOS (pytest's tmp_path does not)."""
    d = tempfile.mkdtemp(prefix="sl", dir="/tmp")
    yield d
    shutil.rmtree(d, ignore_errors=True)


def _env(short, **extra):
    env = dict(os.environ, FAKE_LAYA_STARTS=os.path.join(short, "starts"))
    env.update({k: str(v) for k, v in extra.items()})
    return env


def _cmd(sock, idle=30.0):
    return [sys.executable, FAKE_SERVER, "none", "--shared", sock, "--idle", str(idle)]


def _starts(short):
    p = os.path.join(short, "starts")
    return open(p).read().split() if os.path.exists(p) else []


def _wait(cond, timeout=10.0):
    end = time.time() + timeout
    while time.time() < end:
        if cond():
            return True
        time.sleep(0.05)
    return False


def _alive(pid):
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False


@pytest.fixture
def servers():
    """Server pids a test started; killed at the end so no test leaves one running."""
    pids = []
    yield pids
    for pid in pids:
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


def _connect(short, servers, idle=30.0, timeout=10.0, **env):
    sock = os.path.join(short, "t.sock")
    c = shared_laya.connect(sock, _cmd(sock, idle), env=_env(short, **env), timeout=timeout, start_timeout=30)
    servers.append(c.server_pid)
    return c


# ---- the socket name ----

def test_socket_path_is_one_per_checkpoint(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    for d in (a, b):
        d.mkdir()
        (d / "adapter.safetensors").write_bytes(b"x")
    cache = "/tmp/sf2cache"
    pa, pb = shared_laya.socket_path(str(a), cache), shared_laya.socket_path(str(b), cache)
    assert pa != pb and pa == shared_laya.socket_path(str(a) + "/", cache)
    assert os.path.dirname(pa) == cache and pa.endswith(".sock") and len(pa) < 104
    assert shared_laya.socket_path("none", cache) not in (pa, pb)


def test_a_retrained_checkpoint_gets_a_new_server(tmp_path):
    (tmp_path / "adapter.safetensors").write_bytes(b"x")
    before = shared_laya.socket_path(str(tmp_path), "/tmp/c")
    os.utime(tmp_path / "adapter.safetensors", ns=(1, 1))
    assert shared_laya.socket_path(str(tmp_path), "/tmp/c") != before


# ---- the protocol ----

def test_round_trip_answers_like_the_per_run_helper(short, servers):
    """The shared server's reply line is byte for byte the per-run helper's for the same request."""
    req = json.dumps({"text": "close range, he is idle. use more c.mk", "question": Q}) + "\n"
    bad = json.dumps({"text": "no question"}) + "\n"
    old = subprocess.run([sys.executable, FAKE_SERVER, "none"], input=req + bad, capture_output=True, text=True,
                         timeout=30, env=_env(short)).stdout.splitlines(keepends=True)
    assert json.loads(old[0]) == {"ready": "aac6fef/laya-mlx"}
    # the reply exactly as the helper wrote it before the shared server (json.dumps of the model's dict, in its order)
    text = "close range, he is idle. use more c.mk"
    assert old[1] == json.dumps({"probabilities": fake_laya.probabilities(text, Q)}) + "\n"
    assert list(json.loads(old[1])["probabilities"]) == ["c.mk", "s.hp", "forward"]
    c = _connect(short, servers)
    try:
        assert c.request_line(req) == old[1]
        assert c.request_line(bad) == old[2] and "error" in json.loads(old[2])
        assert c.ask("close range, he is idle. use more c.mk", Q) == \
            {"probabilities": fake_laya.probabilities("close range, he is idle. use more c.mk", Q)}
    finally:
        c.close()


def _many(sock, who, n):
    """One client's ``n`` requests; every reply must be the answer to its own request."""
    c = shared_laya.Client(sock, timeout=30)
    try:
        wrong = 0
        for i in range(n):
            text = "client %s request %d" % (who, i)
            wrong += c.ask(text, Q) != {"probabilities": fake_laya.probabilities(text, Q)}
        return wrong
    finally:
        c.close()


def test_many_concurrent_clients_get_their_own_replies(short, servers):
    first = _connect(short, servers, FAKE_LAYA_DELAY=0.002)
    try:
        with ThreadPoolExecutor(16) as pool:
            wrong = list(pool.map(lambda k: _many(first.sock, "t%d" % k, 25), range(16)))
        with ProcessPoolExecutor(6) as pool:
            wrong += list(pool.map(_many, [first.sock] * 6, ["p%d" % k for k in range(6)], [25] * 6))
    finally:
        first.close()
    assert wrong == [0] * 22


def test_two_simultaneous_starters_end_with_one_server(short, servers):
    sock = os.path.join(short, "t.sock")
    code = ("import sys; sys.path.insert(0, %r); from sf2.system1 import shared_laya as s; import json; "
            "c = s.connect(%r, %r, timeout=30, start_timeout=60); print(c.server_pid); "
            "print(json.dumps(c.ask('hi', %r)))" % (ROOT, sock, _cmd(sock), Q))
    procs = [subprocess.Popen([sys.executable, "-c", code], stdout=subprocess.PIPE, text=True,
                              env=_env(short, FAKE_LAYA_LOAD_S=1.0)) for _ in range(4)]
    outs = [p.communicate(timeout=60)[0].split("\n") for p in procs]
    pids = {int(o[0]) for o in outs}
    servers.extend(pids)
    assert [p.returncode for p in procs] == [0] * 4
    assert len(pids) == 1 and len(_starts(short)) == 1
    assert all(json.loads(o[1]) == {"probabilities": fake_laya.probabilities("hi", Q)} for o in outs)


# ---- failures are errors, never hangs ----

def test_a_client_whose_server_is_killed_gets_a_clear_error(short, servers):
    c = _connect(short, servers)
    os.kill(c.server_pid, signal.SIGKILL)
    t = time.time()
    with pytest.raises(RuntimeError, match="closed"):
        c.ask("x", Q)
    assert time.time() - t < 5


def test_a_server_dying_mid_request_is_an_error(short, servers):
    c = _connect(short, servers)
    with pytest.raises(RuntimeError, match="closed"):
        c.ask("DIE now", Q)


def test_a_stuck_server_times_out(short, servers):
    c = _connect(short, servers, timeout=1.0)
    t = time.time()
    with pytest.raises(RuntimeError, match="no reply"):
        c.ask("HANG", Q)
    assert time.time() - t < 5


def test_a_server_that_cannot_start_is_an_error_not_a_wait(short):
    sock = os.path.join(short, "t.sock")
    t = time.time()
    with pytest.raises(RuntimeError, match="exited"):
        shared_laya.connect(sock, [sys.executable, "-c", "import sys; sys.exit(7)"], start_timeout=30)
    assert time.time() - t < 10


@pytest.mark.parametrize("kind", ["socket", "file"])
def test_a_stale_socket_file_is_replaced(short, servers, kind):
    import socket as S
    sock = os.path.join(short, "t.sock")
    if kind == "socket":                          # left by a server that was killed: bound, nobody listening
        S.socket(S.AF_UNIX, S.SOCK_STREAM).bind(sock)
    else:
        open(sock, "w").close()
    c = _connect(short, servers)
    assert c.ask("x", Q) == {"probabilities": fake_laya.probabilities("x", Q)}
    c.close()


# ---- idle exit ----

def test_the_server_exits_when_idle_and_a_new_run_starts_another(short, servers):
    c = _connect(short, servers, idle=0.5)
    pid, sock = c.server_pid, c.sock
    time.sleep(1.5)                               # longer than idle, but a client is still connected
    assert _alive(pid) and c.ask("still here", Q)
    c.close()
    assert _wait(lambda: not _alive(pid) and not os.path.exists(sock))
    c2 = _connect(short, servers, idle=0.5)
    assert c2.server_pid != pid and len(_starts(short)) == 2
    c2.close()


def test_the_shared_server_reserves_its_memory_once_and_releases_it(short):
    sys.path.insert(0, os.path.join(ROOT, "scripts"))
    import text_laya_server

    class Ledger:
        rows = []

        def reserve(self, gb, timeout=None):
            self.rows.append(("reserve", gb))
            return "rid"

        def release(self, rid):
            self.rows.append(("release", rid))

    sock = os.path.join(short, "t.sock")
    th = threading.Thread(target=text_laya_server.main, daemon=True,
                          args=(["none", "--shared", sock, "--idle", "0.3", "--budget-gb", "6.5"],),
                          kwargs={"load": fake_laya.load, "budget": Ledger})
    th.start()
    assert _wait(lambda: os.path.exists(sock))
    c = shared_laya.Client(sock, timeout=5)
    c.ask("x", Q)
    c.close()
    th.join(10)
    assert not th.is_alive() and Ledger.rows == [("reserve", 6.5), ("release", "rid")]


# ---- the Advisor ----

class _FakeProc:
    def __init__(self, args, **kw):
        self.args = args
        self.stdout = io.StringIO('{"ready": "x"}\n')

    def poll(self):
        return 0


def test_the_advisor_uses_its_own_subprocess_by_default(monkeypatch):
    from sf2.system1 import advisor as A
    started = []
    monkeypatch.setattr(A.subprocess, "Popen", lambda args, **kw: started.append(args) or _FakeProc(args))
    monkeypatch.setattr(A.shared_laya, "connect", lambda *a, **k: pytest.fail("the shared server was used"))
    a = A.Advisor("none")
    assert a.client is None and len(started) == 1
    assert started[0][1].endswith(os.path.join("scripts", "text_laya_server.py"))
    assert started[0][2:] == ["none"]


def test_the_config_default_is_off():
    out = subprocess.run([sys.executable, "-c", "from sf2 import config as c; print(c.TEXT_LAYA_SHARED)"], cwd=ROOT,
                         env={k: v for k, v in os.environ.items() if k != "SF2_TEXT_LAYA_SHARED"},
                         capture_output=True, text=True, check=True).stdout
    assert out.strip() == "False"


def test_a_shared_advisor_asks_the_shared_server(short, servers, monkeypatch):
    from sf2.system1.advisor import Advisor
    monkeypatch.setenv("FAKE_LAYA_STARTS", os.path.join(short, "starts"))
    with Advisor("none", python=sys.executable, shared=True, server=FAKE_SERVER, cache=short, budget_gb=0) as a, \
            Advisor("none", python=sys.executable, shared=True, server=FAKE_SERVER, cache=short, budget_gb=0) as b:
        servers.append(a.client.server_pid)
        assert a.proc is None and a.client.server_pid == b.client.server_pid
        assert a.ask("x", Q) == b.ask("x", Q) == fake_laya.probabilities("x", Q)
        with pytest.raises(RuntimeError, match="text laya"):
            a.ask("x", {"type": "nonsense"})
    assert len(_starts(short)) == 1 and _alive(servers[0])       # closing a client leaves the server to its idle exit


# ---- review 2026-09-30: a slow server is not replaced, a failed start is not repeated by every waiting run ----

def test_a_live_but_slow_server_is_never_unlinked_or_doubled(short, monkeypatch):
    """A socket that accepts but does not greet in time is a busy server, not a dead one: no second server."""
    import socket as S
    sock = os.path.join(short, "t.sock")
    busy = S.socket(S.AF_UNIX, S.SOCK_STREAM)
    busy.bind(sock)
    busy.listen(8)                                # never accepts: the client's greeting read times out
    monkeypatch.setattr(shared_laya, "PROBE_S", 0.3)
    try:
        with pytest.raises(RuntimeError, match="cannot reach"):
            shared_laya.connect(sock, _cmd(sock), env=_env(short), timeout=0.3, start_timeout=10)
        assert os.path.exists(sock) and _starts(short) == []
    finally:
        busy.close()


def test_runs_waiting_on_a_failed_start_fail_fast(short):
    """Four runs start at once and the server dies while loading: one waits for it, the rest fail at once."""
    sock = os.path.join(short, "t.sock")
    code = ("import sys, time; sys.path.insert(0, %r); from sf2.system1 import shared_laya as s\n"
            "t = time.time()\ntry:\n    s.connect(%r, [sys.executable, '-c', 'import time; time.sleep(2); exit(7)'],"
            " start_timeout=60)\nexcept RuntimeError as e:\n    print(round(time.time() - t, 1), e)"
            % (ROOT, sock))
    procs = [subprocess.Popen([sys.executable, "-c", code], stdout=subprocess.PIPE, text=True) for _ in range(4)]
    outs = [p.communicate(timeout=60)[0] for p in procs]
    took = sorted(float(o.split()[0]) for o in outs)
    assert all("exited" in o or "failed" in o for o in outs), outs
    assert took[-1] < 4.0, took                   # not 4 starts x 2 s one after another
