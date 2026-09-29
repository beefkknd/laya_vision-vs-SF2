"""System 1's advice follower: laya-vision rates every move from the screen, text laya (a helper process in the
laya-mlx venv, scripts/text_laya_server.py) picks one from a shortlist using System 2's short memory.

The shortlist: laya-vision's SHORTLIST best-rated moves, every move a memory lesson names (so advice can always be
followed), and forward. The text is built by sf2.advice exactly as in text laya's training data.
"""
import json
import os
import subprocess
from typing import Dict, Optional, Sequence, Tuple

from .advice import FORWARD, answers, prompt, question, rating, read, situation_text

MLX_PYTHON = os.path.expanduser("~/work/laya_mlx/.venv/bin/python")
HF_HOME = "/Volumes/ExtremeSSD/huggingface"
SHORTLIST = 3
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class Advisor:
    def __init__(self, checkpoint: str, python: str = MLX_PYTHON):
        if checkpoint != "none" and not os.path.exists(os.path.join(checkpoint, "adapter.safetensors")):
            raise SystemExit("no text laya checkpoint at %s" % checkpoint)
        env = dict(os.environ, HF_HOME=os.environ.get("HF_HOME", HF_HOME), HF_HUB_OFFLINE="1")
        self.checkpoint = checkpoint
        self.proc = subprocess.Popen([python, os.path.join(ROOT, "scripts", "text_laya_server.py"), checkpoint],
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
        """Stop the helper; kill it if it does not exit in time (never leaves it running)."""
        if self.proc.poll() is None:
            try:
                self.proc.stdin.close()
                self.proc.terminate()
                self.proc.wait(timeout=10)
            except (subprocess.TimeoutExpired, OSError):
                self.proc.kill()
                self.proc.wait(timeout=10)


def shortlist(scores: Dict[str, float], lessons: Sequence[str], moves: Sequence[str]) -> Dict[str, Optional[str]]:
    """move -> laya-vision's rating in words: the best-rated moves, every move a lesson names, and forward."""
    best = sorted(scores, key=scores.get, reverse=True)[:SHORTLIST]
    named = [les.move for les in (read(t, list(moves) + [FORWARD]) for t in lessons)
             if les.move in scores and les.move not in best]
    out = {m: rating(scores[m]) for m in best + sorted(set(named))}
    out[FORWARD] = None
    return out


def choose(advisor: Advisor, situation: Tuple[str, str, str, str], scores: Dict[str, float],
           lessons: Sequence[str], moves: Sequence[str]) -> Dict:
    """Text laya's pick, with what it saw and what the label rule says (logged, so every game checks it)."""
    options = shortlist(scores, lessons, moves)
    text = prompt(situation_text(*situation), lessons)
    probs = advisor.ask(text, question(options))
    pick = max(probs, key=probs.get)
    rule, why = answers(situation[0], situation[1], options, [read(t, list(moves) + [FORWARD]) for t in lessons])
    return {"action": pick, "advice_text": text, "shortlist": options, "advice_probs": probs,
            "rule_answers": rule, "rule": why, "follows_rule": pick in rule}
