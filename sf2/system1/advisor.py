"""System 1's advice follower: laya-vision rates every move from the screen, text laya (a helper process in the
laya-mlx venv, scripts/text_laya_server.py) picks one from a shortlist using System 2's short memory.

The shortlist: laya-vision's SHORTLIST best-rated moves, every move a memory lesson names (so advice can always be
followed), and forward. The text is built by sf2.system1.advice exactly as in text laya's training data.
"""
import json
import os
import subprocess
from typing import Dict, List, Optional, Sequence, Tuple

from . import shared_laya
from .advice import FORWARD, answers, prompt, question, rating, read, situation_text
from ..config import (HF_HOME, MLX_PYTHON, REPO, TEXT_LAYA_CACHE, TEXT_LAYA_IDLE_S, TEXT_LAYA_SERVER_GB,
                      TEXT_LAYA_SHARED)

SHORTLIST = 3
ROOT = REPO
SERVER = os.path.join(ROOT, "scripts", "text_laya_server.py")


class Advisor:
    """Text laya for one run: its own helper process (the default), or with ``shared`` (default
    config.TEXT_LAYA_SHARED) a client of the one server for this checkpoint (sf2.system1.shared_laya), started if
    none is running. Both give the same answers; closing a shared client leaves the server to its idle exit."""

    def __init__(self, checkpoint: str, python: str = MLX_PYTHON, shared: Optional[bool] = None,
                 server: str = SERVER, cache: str = TEXT_LAYA_CACHE, budget_gb: float = TEXT_LAYA_SERVER_GB):
        if checkpoint != "none" and not os.path.exists(os.path.join(checkpoint, "adapter.safetensors")):
            raise SystemExit("no text laya checkpoint at %s" % checkpoint)
        env = dict(os.environ, HF_HOME=HF_HOME, HF_HUB_OFFLINE="1")
        self.checkpoint = checkpoint
        self.proc, self.client = None, None
        if TEXT_LAYA_SHARED if shared is None else shared:
            sock = shared_laya.socket_path(checkpoint, cache)
            cmd = [python, server, checkpoint, "--shared", sock, "--idle", str(TEXT_LAYA_IDLE_S),
                   "--budget-gb", str(budget_gb)]
            self.client = shared_laya.connect(sock, cmd, env=env, cwd=ROOT)
            return
        self.proc = subprocess.Popen([python, server, checkpoint],
                                     cwd=ROOT, env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
        ready = self._read()
        if "ready" not in ready:
            raise RuntimeError("text laya did not start: %s" % ready)

    def _read(self) -> Dict:
        line = self.proc.stdout.readline()
        if not line:
            raise RuntimeError("text laya exited (code %s)" % self.proc.poll())
        return json.loads(line)

    def ask(self, text: str, q: Dict) -> Dict[str, float]:
        if self.client is not None:
            reply = self.client.ask(text, q)
        else:
            self.proc.stdin.write(json.dumps({"text": text, "question": q}) + "\n")
            self.proc.stdin.flush()
            reply = self._read()
        if "error" in reply:
            raise RuntimeError("text laya: %s" % reply["error"])
        return reply["probabilities"]

    def __enter__(self) -> "Advisor":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def close(self) -> None:
        """Stop the helper; kill it if it does not exit in time (never leaves it running). A shared client only
        disconnects."""
        if self.client is not None:
            self.client.close()
        elif self.proc.poll() is None:
            try:
                self.proc.stdin.close()
                self.proc.terminate()
                self.proc.wait(timeout=10)
            except (subprocess.TimeoutExpired, OSError):
                self.proc.kill()
                self.proc.wait(timeout=10)


def shortlist(scores: Dict[str, float], lessons: Sequence[str], moves: Sequence[str],
              situation: Tuple[str, str], scale: str = "p_hit", walk: float = 0.0) -> Dict[str, Optional[str]]:
    """move -> laya-vision's rating in words: the best-rated moves, the moves a "use more" / "always" lesson names
    where that lesson applies (``situation``: range, what he is doing), and forward. A named move is not added where
    its lesson does not apply: laya-vision may rate it well there, and text laya then leans to the named move
    (2026-09-29: "use more c.mk at mid range" made her play c.mk up close 10 -> 134 times). The best-rated moves skip
    those an applying avoid lesson rules out (review 2026-09-29: 485 decisions vs Honda had nothing left to pick);
    a block is rated on its own scale (``advice.rating``). ``scale`` "net": ``scores`` are the lookup table's expected
    nets (hp), rated on the net scale relative to ``walk`` (forward's value there)."""
    rng, doing = situation
    live = [les for les in (read(t, list(moves) + [FORWARD]) for t in lessons) if les.applies(rng, doing)]
    out_ruled = {les.move for les in live if les.polarity == "neg"}
    best = sorted((m for m in scores if m not in out_ruled), key=scores.get, reverse=True)[:SHORTLIST]
    named = [les.move for les in live if les.move in scores and les.move not in best and les.polarity in ("soft", "hard")]
    out = {m: rating(scores[m], m, scale, walk) for m in best + sorted(set(named))}
    out[FORWARD] = None
    return out


def applicable(lessons: Sequence[str], moves: Sequence[str], situation: Tuple[str, str]) -> List[str]:
    """The lessons that apply now (range, what he is doing), in their order; a lesson that names no move of hers
    (an opponent habit) always does. The rest are not shown to text laya: the words of a lesson about another range
    lean its choice anyway (2026-09-29: "use more c.mk at mid range" -> c.mk picked in 53% of close-range ties, 16%
    without advice)."""
    out = []
    for t in lessons:
        les = read(t, list(moves) + [FORWARD])
        if les.move is None or les.applies(*situation):
            out.append(t)
    return out


def choose(advisor: Advisor, situation: Tuple[str, str, str, str], scores: Dict[str, float],
           lessons: Sequence[str], moves: Sequence[str], scale: str = "p_hit", walk: float = 0.0) -> Dict:
    """Text laya's pick, with what it saw and what the label rule says (logged, so every game checks it). It reads
    only the lessons that apply now (``applicable``); ``scale``: what ``scores`` are (``shortlist``)."""
    lessons = applicable(lessons, moves, situation[:2])
    options = shortlist(scores, lessons, moves, situation[:2], scale, walk)
    text = prompt(situation_text(*situation), lessons)
    probs = advisor.ask(text, question(options))
    pick = max(probs, key=probs.get)
    rule, why = answers(situation[0], situation[1], options, [read(t, list(moves) + [FORWARD]) for t in lessons])
    return {"action": pick, "advice_text": text, "shortlist": options, "advice_probs": probs,
            "rule_answers": rule, "rule": why, "follows_rule": pick in rule}
