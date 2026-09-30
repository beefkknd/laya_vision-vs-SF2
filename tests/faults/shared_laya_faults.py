"""Seeded faults for the shared text laya server (2026-09-30): each mutates one line of the code, runs the tests that
must catch it, and restores the file. Every fault must turn the tests red; exit 1 if one stays green.

    python tests/faults/shared_laya_faults.py
"""
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TESTS = ["tests/test_shared_laya.py"]
TIMEOUT = 180       # a fault that makes a test hang counts as caught (red)
FAULTS = [
    ("starters do not lock: two runs start two servers", "sf2/system1/shared_laya.py",
     "        fcntl.flock(lock, fcntl.LOCK_EX)\n", "        pass\n"),
    ("the client never times out", "sf2/system1/shared_laya.py",
     "        self.conn.settimeout(timeout)\n", "        self.conn.settimeout(None)\n"),
    ("the server leaves with a client still connected", "sf2/system1/shared_laya.py",
     "    return not st.clients and time.monotonic() - st.idle_since >= idle_s",
     "    return time.monotonic() - st.idle_since >= idle_s"),
    ("the server never leaves", "sf2/system1/shared_laya.py",
     "                if _try_leave(sock, inode, st, idle_s):", "                if False:"),
    ("replies are not the per-run helper's bytes", "sf2/system1/shared_laya.py",
     "    return json.dumps(reply) + \"\\n\"", "    return json.dumps(reply, sort_keys=True) + \"\\n\""),
    ("a stale socket file is not removed", "sf2/system1/shared_laya.py",
     "            os.unlink(sock)                     # left by", "            pass                     # left by"),
    ("a busy server's socket is replaced (a second server)", "sf2/system1/shared_laya.py",
     '        if _probe(sock) != "dead":', '        if _probe(sock) == "up":'),
    ("every waiting run repeats a failed start", "sf2/system1/shared_laya.py",
     "        if os.path.exists(failed) and os.stat(failed).st_mtime >= asked:",
     "        if False:"),
    ("a terminated server leaves its socket behind", "scripts/text_laya_server.py",
     "        signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))", "        pass"),
    ("the Advisor is shared by default", "sf2/system1/advisor.py",
     "        if TEXT_LAYA_SHARED if shared is None else shared:", "        if True if shared is None else shared:"),
    ("the shared server does not release its ledger reservation", "scripts/text_laya_server.py",
     "            ledger.release(rid)", "            pass"),
]


def run(fault) -> bool:
    name, path, old, new = fault
    full = os.path.join(ROOT, path)
    with open(full) as f:
        src = f.read()
    if src.count(old) != 1:
        raise SystemExit("fault %r: the line to mutate is not found once in %s" % (name, path))
    try:
        with open(full, "w") as f:
            f.write(src.replace(old, new))
        try:
            r = subprocess.run([sys.executable, "-m", "pytest", "-q", "-x", "-p", "no:cacheprovider"] + TESTS,
                               cwd=ROOT, capture_output=True, text=True, timeout=TIMEOUT)
            return r.returncode != 0
        except subprocess.TimeoutExpired:
            return True
    finally:
        with open(full, "w") as f:
            f.write(src)
        subprocess.run(["pkill", "-f", "fake_text_laya_server"])     # no fake server outlives a fault


def main() -> int:
    missed = 0
    for fault in FAULTS:
        red = run(fault)
        missed += not red
        print("%-5s %s" % ("RED" if red else "GREEN", fault[0]))
    print("%d of %d faults caught" % (len(FAULTS) - missed, len(FAULTS)))
    return 1 if missed else 0


if __name__ == "__main__":
    sys.exit(main())
