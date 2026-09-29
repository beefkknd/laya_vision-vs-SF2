"""System 1's advice follower: laya-vision rates every move from the screen, text laya (a helper process in the
laya-mlx venv, scripts/text_laya_server.py) picks one from a shortlist using System 2's short memory.

The shortlist: laya-vision's SHORTLIST best-rated moves, every move a memory lesson names (so advice can always be
followed), and forward. The text is built by sf2.system1.advice exactly as in text laya's training data.
"""
import json
import os
import subprocess
from typing import Dict, List, Optional, Sequence, Tuple

from .advice import FORWARD, answers, prompt, question, rating, read, situation_text
from ..config import HF_HOME, MLX_PYTHON, REPO

SHORTLIST = 3
ROOT = REPO


class Advisor:
    def __init__(self, checkpoint: str, python: str = MLX_PYTHON):
        if checkpoint != "none" and not os.path.exists(os.path.join(checkpoint, "adapter.safetensors")):
            raise SystemExit("no text laya checkpoint at %s" % checkpoint)
        env = dict(os.environ, HF_HOME=HF_HOME, HF_HUB_OFFLINE="1")
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


def shortlist(scores: Dict[str, float], lessons: Sequence[str], moves: Sequence[str],
              situation: Tuple[str, str]) -> Dict[str, Optional[str]]:
    """move -> laya-vision's rating in words: the best-rated moves, the moves a "use more" / "always" lesson names
    where that lesson applies (``situation``: range, what he is doing), and forward. A named move is not added where
    its lesson does not apply: laya-vision may rate it well there, and text laya then leans to the named move
    (2026-09-29: "use more c.mk at mid range" made her play c.mk up close 10 -> 134 times)."""
    rng, doing = situation
    best = sorted(scores, key=scores.get, reverse=True)[:SHORTLIST]
    named = [les.move for les in (read(t, list(moves) + [FORWARD]) for t in lessons)
             if les.move in scores and les.move not in best and les.polarity in ("soft", "hard")
             and les.applies(rng, doing)]
    out = {m: rating(scores[m]) for m in best + sorted(set(named))}
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
           lessons: Sequence[str], moves: Sequence[str]) -> Dict:
    """Text laya's pick, with what it saw and what the label rule says (logged, so every game checks it). It reads
    only the lessons that apply now (``applicable``)."""
    lessons = applicable(lessons, moves, situation[:2])
    options = shortlist(scores, lessons, moves, situation[:2])
    text = prompt(situation_text(*situation), lessons)
    probs = advisor.ask(text, question(options))
    pick = max(probs, key=probs.get)
    rule, why = answers(situation[0], situation[1], options, [read(t, list(moves) + [FORWARD]) for t in lessons])
    return {"action": pick, "advice_text": text, "shortlist": options, "advice_probs": probs,
            "rule_answers": rule, "rule": why, "follows_rule": pick in rule}
