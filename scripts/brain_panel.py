"""The brain panel: a local page that shows both systems while scripts/learn_loop.py plays, for watching and for
recording (put it next to the Mesen window).

    python scripts/brain_panel.py            # then open http://127.0.0.1:8765

    System 1, every decision:   what she sees (in words), laya-vision's rating of each move on the shortlist,
                                text laya's pick and why (following advice / laya-vision's best / walking in)
    System 2, between rounds:   Qwen's status, the playbook and the short memory as Qwen writes them (new lines glow),
                                Qwen's last note, and every round so far

It reads the files the loop writes (out/live/decision.json, out/live/session.json, the memory folder, the session's
rounds and the loop log). Its only writes are the demo buttons, and only into a blank-start run's own memory folder
(learn_loop --fresh: memory_runs/<name>), never memory/:
    cheat   copy the seed's playbook / short memory vs this opponent into the run (default seed memory_seeds/video)
    clean   blank it
Every button press is written to the loop log, so the panel and the log show it.
"""
import argparse
import json
import os
import re
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sf2 import demo_cheat  # noqa: E402

LIVE = "out/live"
PAGE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "brain_panel.html")


def _json(path: str, default=None):
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return default


def _lines(path: str) -> List[Dict]:
    try:
        return [json.loads(x) for x in open(path) if x.strip()]
    except (OSError, ValueError):
        return []


def _memory(path: str) -> Dict:
    mem = _json(path) or {}
    return {"lessons": [x.get("text") for x in mem.get("lessons", [])],
            "mtime": os.path.getmtime(path) if os.path.exists(path) else None}


def _log(session: Dict) -> List[str]:
    """This session's lines of the loop log."""
    try:
        with open(session["log"]) as f:
            f.seek(session.get("log_offset", 0))
            return f.read().splitlines()[-400:]
    except (OSError, KeyError):
        return []


def system2(lines: List[str]) -> Dict:
    """Qwen's state from the log: what it is doing now, and its latest notes."""
    status, since, notes = "waiting for the first round", None, []
    for line in lines:
        m = re.match(r"(\d\d:\d\d:\d\d)  (.*)", line)
        if not m:
            continue
        t, msg = m.groups()
        if re.match(r"System 2: (revising|writing|reviewing)", msg):
            status, since = msg[len("System 2: "):].rstrip(" ."), t
        elif msg.startswith("System 2 (") or msg.startswith("System 2: no usable") or msg.startswith("System 2 failed"):
            status, since = "idle", t
            if msg.startswith("System 2: no usable"):          # the raw attempts are for the log, not the screen
                text = "no new %s: the evidence does not support a lesson yet" % (
                    "playbook" if "playbook" in msg else "short memory")
            else:
                text = re.sub(r"^System 2 \((\d+) s\): ", r"(\1 s) ", msg)
            notes.append({"t": t, "text": text})
        elif msg.startswith("brain panel:") or msg.startswith("short memory vs") and "outside" in msg:
            notes.append({"t": t, "text": msg})
        elif msg.startswith("System 2's new short memory"):
            notes.append({"t": t, "text": msg})
    return {"status": status, "since": since, "notes": notes[-4:]}


def state() -> Dict:
    session = _json(os.path.join(LIVE, "session.json"), {})
    decision = _json(os.path.join(LIVE, "decision.json"), {})
    me, root, opp = session.get("me", "chunli"), session.get("memory", "memory"), decision.get("opp") or "dhalsim"
    rounds = _lines(os.path.join(session.get("session", ""), "rounds.jsonl"))
    games = _lines(os.path.join(session.get("session", ""), "games.jsonl"))
    running = bool(session.get("running") and _alive(session.get("pid")))
    if not running:                                 # between takes: show what the next game will start with
        pv = demo_cheat.preview(me, session.get("seed") or demo_cheat.SEED)
        return {"session": session, "running": False, "decision": {}, "system2": {"status": "no game running"},
                "pending": demo_cheat.pending(),
                "playbook": {"lessons": pv["playbook"] or []},
                "short_by_opp": {o: {"lessons": v} for o, v in pv["short"].items()},
                "short": {"lessons": []}, "rounds": [], "games": []}
    return {"session": session, "running": True, "decision": decision, "system2": system2(_log(session)),
            "playbook": _memory(os.path.join(root, "playbook", "%s.json" % me)),
            "short": _memory(os.path.join(root, "short", "%s_vs_%s.json" % (me, opp))) if opp else {"lessons": []},
            "rounds": [{k: r.get(k) for k in ("game", "round", "opp", "result", "dealt", "taken")} for r in rounds],
            "games": [{k: g.get(k) for k in ("game", "opp", "result", "score")} for g in games]}




def _alive(pid) -> bool:
    """The loop process is still there (a killed loop never marks its session finished)."""
    try:
        os.kill(int(pid), 0)
        return True
    except (TypeError, ValueError, ProcessLookupError, PermissionError):
        return False


def action(what: str, do: str, opp: Optional[str]) -> str:
    """A demo button. With a game running: applied now (in play from the next round). Without one: saved for the next
    learn_loop --fresh run, which pushes it in before its first fight. ``opp``: the opponent picked on the page."""
    session = _json(os.path.join(LIVE, "session.json"), {})
    running = session.get("running") and _alive(session.get("pid"))
    if not running:
        return demo_cheat.add_pending(what, do, opp)
    me = session.get("me", "chunli")
    said = demo_cheat.apply(session.get("memory", ""), me, what, do, opp, session.get("seed") or demo_cheat.SEED)
    with open(session["log"], "a") as f:          # on the record: the panel shows it, the log keeps it
        f.write("%s  %s\n" % (time.strftime("%H:%M:%S"), said))
    return said + " · in play from the next round"


MAX_BODY = 1024


def button(body: bytes) -> Tuple[str, str, Optional[str]]:
    """(what, do, opp) from a POST body; ValueError for anything else (the handler answers 400)."""
    try:
        req = json.loads(body or b"{}")
    except ValueError as e:
        raise ValueError("not JSON: %s" % e)
    if not isinstance(req, dict):
        raise ValueError("not a JSON object")
    what, do, opp = req.get("what"), req.get("do"), req.get("opp")
    demo_cheat._valid(what, do, opp)
    return what, do, opp


def request_ok(host: Optional[str], origin: Optional[str], port: int) -> bool:
    """A button press only from the panel's own page: the Host is this machine (no DNS rebinding) and an Origin, if
    the browser sent one, is this server (no other web page posting here)."""
    mine = {"127.0.0.1:%d" % port, "localhost:%d" % port}
    return host in mine and (origin is None or origin in {"http://" + h for h in mine})


class Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        if self.path != "/action":
            self.send_error(404)
            return
        if not request_ok(self.headers.get("Host"), self.headers.get("Origin"), self.server.server_address[1]):
            self.send_error(403)
            return
        try:
            size = int(self.headers.get("Content-Length", 0))
            if not 0 <= size <= MAX_BODY:
                raise ValueError("body of %d bytes" % size)
            body, code = {"ok": True, "said": action(*button(self.rfile.read(size)))}, 200
        except (ValueError, OSError, KeyError) as e:
            body, code = {"ok": False, "error": str(e)}, 400
        data = json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path.startswith("/state"):
            body, kind = json.dumps(state()).encode(), "application/json"
        elif self.path in ("/", "/index.html"):
            body, kind = open(PAGE, "rb").read(), "text/html; charset=utf-8"
        else:
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header("Content-Type", kind)
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):          # quiet: the panel polls several times a second
        pass


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8765)
    args = ap.parse_args()
    print("brain panel: http://127.0.0.1:%d" % args.port, flush=True)
    ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
