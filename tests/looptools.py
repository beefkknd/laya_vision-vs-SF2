"""Shared stubs for the screen-only Qwen-in-loop tests (tests/test_loop_*.py): a faithful follower text laya, a
Moment/ScreenFacts builder, and a fake emulator bridge + reader so the play loop runs with no emulator and no RAM.

Not a test module. Import it by adding this directory to sys.path first (conftest only adds the repo root)."""
import importlib.util
import os
import re
from typing import Dict, List, Optional, Sequence

import numpy as np

from sf2.screen.facts import FighterFacts, HudFacts, ScreenFacts
from sf2.system1.action_menu import CATEGORIES, CATEGORY_ORDER, DEFAULT_MOVE, category_of
from sf2.system1.advice import read as read_lesson
from sf2.system1.screen_words import Moment
from sf2.vocab import FULL_LIFE, range_of

MENU_MOVES: List[str] = [m for cat in CATEGORY_ORDER for m in CATEGORIES[cat]]
RANGE_BACK = {"up close": "close", "at mid range": "mid", "far away": "far"}

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)


def load_driver():
    """Import scripts/play_loop_screen.py as a module (it has `import _path`, so put scripts/ on sys.path first)."""
    import sys
    scripts = os.path.join(_REPO, "scripts")
    if scripts not in sys.path:
        sys.path.insert(0, scripts)
    spec = importlib.util.spec_from_file_location("play_loop_screen", os.path.join(scripts, "play_loop_screen.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class FollowerLaya:
    """A faithful stand-in for the trained text laya: it reads the advice out of the prompt TEXT and follows it over
    the offered options, returning 1.0 on the chosen one. Because it reads ONLY ``text``, a pass proves the runner put
    the situation words AND the short-memory advice into what the model is shown (the wiring under test); the trained
    model's accuracy on held-out wordings is measured elsewhere (the advice suite)."""

    def __init__(self):
        self.calls: List[Dict] = []        # every (text, options) it was asked, for assertions

    def ask(self, text: str, question: Dict) -> Dict[str, float]:
        opts = list(question["criteria"])
        self.calls.append({"text": text, "options": opts})
        rng, doing, fire = self._situation(text)
        live = [l for l in (read_lesson(t, MENU_MOVES) for t in self._advice(text)) if l.applies(rng, doing, fire)]
        ruled = {l.move for l in live if l.polarity == "neg"}

        def follow(cands: Sequence[str]) -> Optional[str]:
            for pol in ("hard", "soft"):
                hit = [l.move for l in live if l.polarity == pol and l.move in cands and l.move not in ruled]
                if hit:
                    return sorted(hit)[0]
            return None

        if opts == CATEGORY_ORDER:                                   # round 1: the category of the followed move
            move = follow(MENU_MOVES)
            choice = category_of(move) if move else "block"
        else:                                                        # round 2: the followed move in this category
            choice = follow(opts) or (DEFAULT_MOVE if DEFAULT_MOVE in opts else opts[0])
        return {o: (1.0 if o == choice else 0.0) for o in opts}

    @staticmethod
    def _situation(text: str):
        m = re.search(r"He is (up close|at mid range|far away) and (\w+)\.", text)
        return RANGE_BACK[m.group(1)], m.group(2), "A fireball is coming." in text

    @staticmethod
    def _advice(text: str) -> List[str]:
        m = re.search(r"Advice: (.*)\.\s*$", text.strip())
        if not m or m.group(1).strip() == "none":
            return []
        return [s.strip() for s in m.group(1).split(";")]


def make_moment(dx: int = 36, doing: str = "standing", my_bar: float = 1.0, his_bar: float = 1.0,
                fireball: bool = False, my_label: str = "stand", my_air: bool = False) -> Moment:
    """A neutral decision Moment (I can act): ``dx`` sets the range (range_of), ``doing`` his coarse state."""
    his_air = doing == "jumping"
    his_label = {"attacking": "attack", "stunned": "hit", "jumping": "stand"}.get(doing, "stand")
    his_crouch = doing == "crouching"
    return Moment(my_x=88, his_x=88 + dx, my_facing="right", my_label=my_label, my_air=my_air,
                  his_label=his_label, his_air=his_air, his_crouch=his_crouch,
                  my_life=int(round(my_bar * FULL_LIFE)), his_life=int(round(his_bar * FULL_LIFE)),
                  filled=(), fireball=fireball)


# --------------------------------------------------------------- fake emulator for the no-RAM play test
def _fighter(side: str, player: int, char: str, x: int, label: str, air: bool) -> FighterFacts:
    return FighterFacts(side=side, character=char, found=True, x=x, y=192, in_air=air, facing="right",
                        sprite=None, action=label, confidence=0.9, unknown=True, box=None,
                        health=1.0, player=player)


def make_facts(me: str, opp: str, dx: int = 36, over: bool = False, new_round: bool = False) -> ScreenFacts:
    left = _fighter("left", 1, me, 88, "stand", False)
    right = _fighter("right", 2, opp, 88 + dx, "stand", False)
    return ScreenFacts(left=left, right=right, projectiles=(), hud=HudFacts(health=(1.0, 1.0), timer=50,
                       bar_empty=(False, False)), round_state="over" if over else "fighting",
                       gap=dx, new_round=new_round)


class _Obs:
    def __init__(self, images: Dict[int, np.ndarray]):
        self.rams: List = []            # screen-only: never any RAM rows
        self.images = images


_FRAME = np.zeros((224, 256, 3), dtype=np.uint8)


class FakeBridge:
    """A RAM-free bridge: it returns a blank frame for the requested captures and NEVER sends RAM rows, so the real
    sf2.system1.screen_emu.ScreenEmu wraps it and raises RamForbidden the instant anything reaches for RAM."""

    def __init__(self):
        self.loaded = 0
        self.runs = 0

    def load_state(self, state: bytes) -> _Obs:
        self.loaded += 1
        return _Obs({0: _FRAME})

    def run(self, frames, caps=()) -> _Obs:
        self.runs += 1
        return _Obs({c: _FRAME for c in caps})


class _Lock:
    def __init__(self, chars):
        self.chars = tuple(chars)


class FakeReader:
    """Returns scripted ScreenFacts, ignoring the frame pixels (the no-RAM test is about the loop's data path, not
    the sprite matcher). ``script`` is a list of ScreenFacts; once exhausted it reports the round over."""

    def __init__(self, me: str, opp: str, script: List[ScreenFacts]):
        self.lock = _Lock((me, opp))
        self._script = list(script)
        self._me, self._opp = me, opp
        self.reads = 0

    def feed(self, frame) -> ScreenFacts:
        self.reads += 1
        if self._script:
            return self._script.pop(0)
        return make_facts(self._me, self._opp, over=True)
