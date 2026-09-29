"""A lock: the frozen inputs of a result, so the result can be re-run exactly.

    locks/<name>/manifest.json      committed: what was locked, from where, the sha256 of every file, the runs to repeat
    locks/<name>/artifacts/         git-ignored copies (APFS clones: no extra disk)
        model/  advisor/            the laya-vision checkpoint dir and text laya's adapter dir
        states/                     the savestates
        play/<log>/                 the play data used as history (actions, rounds, run.json; never the images)

A lock is written once and never overwritten. ``verify`` names every locked file that changed or went missing; a run
from a lock refuses to start unless that list is empty.
"""
import hashlib
import json
import os
import shutil
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional, Sequence

from ..data.dataset import read

ROOT = "locks"
PLAY_ROOT = "rollouts"
PLAY_FILES = ("actions.jsonl", "rounds.jsonl", "games.jsonl", "run.json")
HASH_WORKERS = 16


def _dir(name: str) -> str:
    return os.path.join(ROOT, name)


def _art(name: str) -> str:
    return os.path.join(_dir(name), "artifacts")


def _copy(src: str, dst: str) -> None:
    """An APFS clone when the file system can (cp -c), else a plain copy."""
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if subprocess.run(["cp", "-c", src, dst], capture_output=True).returncode != 0:
        shutil.copy2(src, dst)


def _sha(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()


def _plan(model: str, advisor: str, states: Sequence[str], play: Sequence[str]) -> Dict[str, str]:
    """{path under artifacts/: source file}."""
    out: Dict[str, str] = {}
    for part, src in (("model", model), ("advisor", advisor)):
        if not os.path.isdir(src):
            raise SystemExit("lock: no checkpoint dir %s" % src)
        for d, _, files in os.walk(src):
            for f in files:
                p = os.path.join(d, f)
                out[os.path.join(part, os.path.relpath(p, src))] = p
    for s in states:
        if not os.path.isfile(s):
            raise SystemExit("lock: no savestate %s" % s)
        out[os.path.join("states", os.path.basename(s))] = s
    for d in play:
        if not os.path.isfile(os.path.join(d, "actions.jsonl")):
            raise SystemExit("lock: no actions.jsonl in %s" % d)
        rel = os.path.relpath(d, PLAY_ROOT)
        for f in PLAY_FILES:
            if os.path.isfile(os.path.join(d, f)):
                out[os.path.join("play", rel, f)] = os.path.join(d, f)
    return out


def _hashes(root: str, rels: Sequence[str]) -> Dict[str, Optional[str]]:
    """{rel: sha256, or None when missing}, hashed in parallel."""
    def one(rel: str) -> Optional[str]:
        p = os.path.join(root, rel)
        return _sha(p) if os.path.isfile(p) else None
    with ThreadPoolExecutor(HASH_WORKERS) as pool:
        return dict(zip(rels, pool.map(one, rels)))


def _commit() -> str:
    r = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True)
    return r.stdout.strip() if r.returncode == 0 else "?"


def make(name: str, model: str, advisor: str, states: Sequence[str], play: Sequence[str],
         runs: Sequence[Dict], extra: Optional[Dict] = None) -> Dict:
    """Copy and hash the inputs into locks/<name>; returns the manifest. Refuses an existing lock."""
    if os.path.exists(_dir(name)):
        raise SystemExit("lock %s exists: a lock is never overwritten" % name)
    plan = _plan(model, advisor, states, play)
    art = _art(name)
    try:
        for rel, src in plan.items():
            _copy(src, os.path.join(art, rel))
        files = _hashes(art, sorted(plan))
    except BaseException:
        shutil.rmtree(_dir(name), ignore_errors=True)       # half a lock is no lock
        raise
    manifest = {"name": name, "made": time.strftime("%Y-%m-%d %H:%M:%S"), "commit": _commit(),
                "sources": {"model": model, "advisor": advisor, "states": list(states), "play": list(play)},
                "runs": list(runs), "extra": dict(extra or {}), "files": files}
    path = os.path.join(_dir(name), "manifest.json")
    with open(path + ".tmp", "w") as f:
        json.dump(manifest, f, indent=1, sort_keys=True)
    os.replace(path + ".tmp", path)
    return manifest


def manifest(name: str) -> Dict:
    path = os.path.join(_dir(name), "manifest.json")
    if not os.path.isfile(path):
        raise SystemExit("no lock %s (%s)" % (name, path))
    with open(path) as f:
        return json.load(f)


def verify(name: str) -> List[str]:
    """Every locked file that changed or is missing ([] = the lock holds)."""
    want = manifest(name)["files"]
    have = _hashes(_art(name), sorted(want))
    return ["%s: %s" % (rel, "missing" if have[rel] is None else "changed")
            for rel in sorted(want) if have[rel] != want[rel]]


def paths(name: str) -> Dict[str, str]:
    art = _art(name)
    return {part: os.path.join(art, part) for part in ("model", "advisor", "states", "play")}


def play_rows(name: str, me: str, opp: str) -> List[Dict]:
    """Her decisions against ``opp`` in the locked play data, each tagged ``log`` = its dir under rollouts/ (as
    sf2.eval.logs.load_actions tags them)."""
    root = paths(name)["play"]
    rels = sorted(os.path.dirname(r)[len("play") + 1:] for r in manifest(name)["files"]
                  if r.startswith("play" + os.sep) and r.endswith(os.sep + "actions.jsonl"))
    return [dict(a, log=rel) for rel in rels for a in read(os.path.join(root, rel, "actions.jsonl"))
            if a.get("me") == me and a.get("opp") == opp]
