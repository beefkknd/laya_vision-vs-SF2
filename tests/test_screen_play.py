"""Screen-only play (docs/laya_text_only_plan.md, step 3, "The rule: no RAM in play"):
- the emulator handle of screen mode exposes frames + input + savestate load only; every RAM read raises;
- the screen-play modules import no RAM row helper (an AST scan, seen red on seeded bad sources);
- the facts -> words table: the screen words equal the T0 functions' words on the equivalent RAM row."""
import ast
import os
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from sf2.screen.facts import FighterFacts, HudFacts, ScreenFacts
from sf2.system1 import screen_words as W
from sf2.system1.screen_emu import RamForbidden, ScreenEmu

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ---------------------------------------------------------------- the RAM-raising handle
class FakeBridge:
    """Answers like MesenBridge; ``ram_rows``: what a (wrong) bridge would send as RAM rows."""

    def __init__(self, ram_rows=False):
        self.ram_rows = ram_rows
        self.sent = []

    def _img(self, k):
        return np.full((224, 256, 3), k % 256, np.uint8)

    def run(self, frames, caps=()):
        self.sent.append(list(frames))
        rams = [[1, 2, 3]] * (len(frames) + 1) if self.ram_rows else []
        return SimpleNamespace(rams=rams, images={c: self._img(len(self.sent) * 10 + c) for c in caps})

    def load_state(self, data):
        return SimpleNamespace(rams=[[0]] if self.ram_rows else [], images={0: self._img(0)})

    def dump_wram(self):
        return b"\0" * 131072

    def set_vars(self, specs):
        pass

    def poke(self, writes):
        pass

    def save_state(self):
        return b"state"

    def close(self):
        pass


RAM_NAMES = ("dump_wram", "set_vars", "poke", "save_state", "watch", "reset", "rams", "nvars", "_bridge", "bridge")


@pytest.mark.parametrize("name", RAM_NAMES)
def test_every_ram_path_of_the_handle_raises(name):
    emu = ScreenEmu(FakeBridge())
    with pytest.raises(RamForbidden):
        getattr(emu, name)


def test_a_play_step_that_reads_ram_fails():
    emu = ScreenEmu(FakeBridge())
    emu.load(b"start")

    def step_reading_ram(e):                 # what a careless play step would do: look at the life byte
        e.run([[]] * 4, [0, 4])
        return e.dump_wram()[0x0C35]

    with pytest.raises(RamForbidden):
        step_reading_ram(emu)


def test_a_bridge_that_sends_ram_rows_is_refused():
    emu = ScreenEmu(FakeBridge(ram_rows=True))
    with pytest.raises(RamForbidden):
        emu.load(b"start")
    emu = ScreenEmu(FakeBridge(ram_rows=False))
    emu.load(b"start")
    emu._ScreenEmu__bridge.ram_rows = True          # the bridge starts sending RAM mid-game
    with pytest.raises(RamForbidden):
        emu.run([[]] * 4, [4])


def test_the_handle_records_every_input_and_capture_by_absolute_frame():
    emu = ScreenEmu(FakeBridge())
    f0 = emu.load(b"start")
    assert f0.shape == (224, 256, 3) and emu.frame == 0
    imgs = emu.run([[], ["right"], [], ["y"], []], [1, 5])
    assert sorted(imgs) == [1, 5]
    imgs = emu.run([["down"]] * 4, [0, 4])
    assert sorted(imgs) == [5, 9]
    assert emu.frame == 9
    assert emu.inputs == [[], ["right"], [], ["y"], [], ["down"], ["down"], ["down"], ["down"]]
    assert sorted(emu.captures) == [0, 1, 5, 9]
    assert all(len(h) == 64 for h in emu.captures.values())
    with pytest.raises(ValueError):
        emu.load(b"again")                   # one start state per handle record (a new round: a new handle record)


# ---------------------------------------------------------------- invariant: no RAM helpers in the play modules
SCREEN_PLAY = ("sf2/system1/screen_words.py", "sf2/system1/screen_play.py", "sf2/system1/screen_emu.py",
               "scripts/play_screen.py")
BANNED_MODULES = ("sf2.emu.ram", "sf2.emu.boot", "sf2.data.vs_defense",
                  "sf2.data.pairs_labels", "sf2.data.eye_v2", "sf2.data.perception", "sf2.system1.game_log",
                  "sf2.system1.opp_moves", "sf2.eval.runner", "sf2.sprites",
                  "sf2.system1.screen_replay")
# names of RAM row helpers anywhere (T0's play loop and the row functions)
BANNED_NAMES = ("_can_act", "_run", "_act", "_decide", "_close", "play_round", "situation_of_row", "view", "outcome",
                "block_outcome", "NAMES", "VARS", "GROUND_Y", "ram_entry", "action_entry", "game_entry", "open_fight",
                "set_vars", "dump_wram", "poke", "save_state", "watch", "reset")
ALLOWED_FROM = {                 # the only names a play module may take from a module that also holds row helpers
    "sf2.system1.system1": {"System1", "choices", "WAIT", "MAX_RECOVER", "MAX_FRAMES"},
    "sf2.data.vs_sweep": {"actions", "MOVEMENT"},     # the button table of each move (no RAM)
    "sf2.emu.vs": {"physical"},                       # F / B -> left / right by a facing given to it (no RAM)
    "sf2.emu.mesen": {"MesenBridge"},                 # screen_emu.py only: the wire, driven by the screen-only Lua
    "sf2.emu.headless": {"launch_argv", "find_mesen"},
}
RAMS_ALLOWED = {"sf2/system1/screen_emu.py"}          # reads .rams only to refuse a non-empty one


def _absolute(node, package):
    if not node.level:
        return node.module or ""
    parts = package.split(".")
    return ".".join(parts[:len(parts) - node.level + 1] + ([node.module] if node.module else []))


def violations(source, package="sf2.system1", rams_ok=False):
    out = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            for a in node.names:
                if any(a.name == b or a.name.startswith(b + ".") for b in BANNED_MODULES) or a.name in ALLOWED_FROM:
                    out.append("import %s" % a.name)
        elif isinstance(node, ast.ImportFrom):
            base = _absolute(node, package)
            if any(base == b or base.startswith(b + ".") for b in BANNED_MODULES):
                out.append("from %s" % base)
            for a in node.names:
                full = "%s.%s" % (base, a.name)
                if any(full == b or full.startswith(b + ".") for b in BANNED_MODULES):
                    out.append("import %s" % full)
                elif base in ALLOWED_FROM and a.name not in ALLOWED_FROM[base]:
                    out.append("from %s import %s" % (base, a.name))
                elif base == "sf2.emu" or (base.startswith("sf2.emu.") and base not in ALLOWED_FROM):
                    out.append("from %s import %s" % (base, a.name))
        elif isinstance(node, ast.Attribute):
            if node.attr in BANNED_NAMES or (node.attr == "rams" and not rams_ok):
                out.append("uses .%s" % node.attr)
        elif isinstance(node, ast.Name) and node.id in BANNED_NAMES:
            out.append("uses %s" % node.id)
    return out


def test_screen_play_modules_import_no_ram_helpers():
    bad = {}
    for rel in SCREEN_PLAY:
        with open(os.path.join(ROOT, rel)) as f:
            pkg = "sf2.system1" if rel.startswith("sf2/") else ""
            v = violations(f.read(), pkg, rams_ok=rel in RAMS_ALLOWED)
        if v:
            bad[rel] = v
    assert bad == {}


@pytest.mark.parametrize("src", [
    "from .system1 import situation",
    "from .system1 import _can_act",
    "from sf2.system1.system1 import play_round",
    "from ..data.vs_sweep import note",
    "from ..data.vs_sweep import outcome",
    "from sf2.emu.vs import GROUND_Y",
    "from ..emu.vs import view",
    "import sf2.data.vs_sweep",
    "import sf2.emu.ram",
    "from ..emu import vs",
    "from .game_log import action_entry",
    "from ..eval.runner import open_fight",
    "from .screen_replay import truth",
    "def f(b):\n    return b.dump_wram()",
    "def f(b):\n    return b.run([[]]).rams[0]",
    "def f(b):\n    b.set_vars([])",
    "from ..emu.mesen import Obs",
])
def test_scanner_catches_seeded_ram_reads(src):
    assert violations(src), src


def test_scanner_passes_clean_sources():
    assert violations("from .system1 import System1, choices\nfrom ..screen.reader import RoundReader\n"
                      "from ..vocab import bar") == []


# ---------------------------------------------------------------- the facts -> words table vs the T0 functions
from sf2.data.vs_sweep import note as ram_note  # noqa: E402  (the test, not a play module, meets both paths)
from sf2.emu.vs import GROUND_Y, view  # noqa: E402
from sf2.system1.system1 import situation as ram_situation  # noqa: E402

# (his RAM state, his y in the air?, his 7-answer label on the screen, his sprite is a crouch sprite)
STATE_CASES = [
    (0x00, False, "stand", False),
    (0x00, False, "walk", False),
    (0x02, False, "stand", True),
    (0x04, False, "jump", False),        # take-off / landing rows on the ground
    (0x04, True, "jump", False),
    (0x04, True, "attack", False),       # a jump attack: state 04, in the air
    (0x08, False, "block", False),
    (0x0A, False, "attack", False),
    (0x0C, False, "special attack", False),
    (0x0C, True, "special attack", False),   # an air special (spinning bird kick)
    (0x0E, False, "hit", False),
    (0x0E, True, "hit", False),           # knocked into the air
]


def _ram_row(my_x, his_x, his_state, his_air, my_life, his_life, my_state=0x00, my_air=False):
    return {"p1_x": my_x, "p2_x": his_x, "p1_y": GROUND_Y + (40 if my_air else 0),
            "p2_y": GROUND_Y - (60 if his_air else 0), "p1_state": my_state, "p2_state": his_state,
            "p1_life": my_life, "p2_life": his_life, "p1_facing": 0x40 if my_x < his_x else 0x00}


def _fighter(side, char, player, x, label, air, sprite="x/y", health=1.0, facing="right"):
    return FighterFacts(side, char, True, x, 190, air, facing, sprite, label, 0.9, False, None, health, player)


def _facts(my_x, his_x, label, air, crouch_sprite, my_frac, his_frac, my_label="stand", my_air=False):
    crouch_ck = next(iter(W.crouch_sprites()))
    stand_ck = "nope/stand"
    me = _fighter("?", "chunli", 1, my_x, my_label, my_air, health=my_frac,
                  facing="right" if my_x < his_x else "left")
    him = _fighter("?", "ryu", 2, his_x, label, air, sprite=crouch_ck if crouch_sprite else stand_ck,
                   health=his_frac)
    left, right = (me, him) if my_x <= his_x else (him, me)
    return ScreenFacts(replace(left, side="left"), replace(right, side="right"), (),
                       HudFacts((my_frac, his_frac), 99, (False, False)), "fighting", right.x - left.x)


def test_crouch_sprites_exist_in_the_catalog():
    chars = {ck.split("/")[0] for ck in W.crouch_sprites()}
    assert {"chunli", "ryu", "ken", "guile", "blanka", "dhalsim", "zangief"} <= chars   # honda: none in the catalog


@pytest.mark.parametrize("his_state,his_air,label,crouch", STATE_CASES)
@pytest.mark.parametrize("my_x,his_x", [(100, 140), (100, 210), (150, 40), (60, 250), (120, 174), (120, 175), (130, 130)])
def test_screen_words_equal_the_t0_words(his_state, his_air, label, crouch, my_x, his_x):
    # the bar thresholds (0.9 / 0.6 / 0.3 of 176) on both sides of each
    for my_life, his_life in ((176, 176), (150, 100), (60, 20), (0, 176), (159, 158), (106, 105), (53, 52)):
        row = _ram_row(my_x, his_x, his_state, his_air, my_life, his_life)
        side = "left" if row["p1_x"] < row["p2_x"] else "right"
        want_note = ram_note("chunli", "ryu", view(row, 1), side, version=2)
        want_sit = ram_situation(row)
        m = W.moment(_facts(my_x, his_x, label, his_air, crouch, my_life / 176, his_life / 176))
        assert W.note("chunli", m) == want_note
        assert W.situation(m) == want_sit


@pytest.mark.parametrize("my_state,my_air,my_label,can", [
    (0x00, False, "stand", True), (0x02, False, "stand", True), (0x00, False, "walk", True),
    (0x04, True, "jump", False), (0x0A, False, "attack", False), (0x0C, False, "special attack", False),
    (0x08, False, "block", False), (0x0E, False, "hit", False), (0x04, False, "jump", False),
    (0x00, True, "stand", False),          # an unknown sprite (answered "stand") in the air: not on the ground
])
def test_can_i_act_equals_t0(my_state, my_air, my_label, can):
    row = _ram_row(100, 160, 0, False, 176, 176, my_state=my_state, my_air=my_air)
    t0 = row["p1_state"] in (0, 2) and row["p1_y"] == GROUND_Y
    m = W.moment(_facts(100, 160, "stand", False, False, 1.0, 1.0, my_label=my_label, my_air=my_air))
    if my_state == 0x04 and not my_air:
        assert t0 is False and m.can_act is False      # a jump's take-off row: neither acts
    else:
        assert m.can_act == t0 == can


def test_facing_for_buttons():
    m = W.moment(_facts(100, 160, "stand", False, False, 1.0, 1.0))
    assert m.facing_right(movement=True) and m.facing_right(movement=False)
    m = W.moment(_facts(200, 160, "stand", False, False, 1.0, 1.0))
    assert not m.facing_right(movement=True) and not m.facing_right(movement=False)
    # drawn facing wins for attacks (T0: the ROM's facing byte), the x order for walks (T0: the x order)
    m = replace(m, my_facing="right")
    assert m.facing_right(movement=False) and not m.facing_right(movement=True)


def test_a_fighter_not_found_keeps_his_last_x():
    f = _facts(100, 160, "stand", False, False, 1.0, 1.0)
    first = W.moment(f)
    gone = replace(f, right=replace(f.right, found=False, x=None, action="block", unknown=True))
    m = W.moment(gone, first)
    assert m.his_x == 160 and "his_x" in m.filled
    assert W.moment(gone).his_x == W.START_X["right"]


def test_block_words_equal_t0_for_a_blocking_opponent():
    """T0 on a guarding opponent (state 08): not attacking, not crouching, doing "standing"."""
    row = _ram_row(100, 160, 0x08, False, 176, 176)
    m = W.moment(_facts(100, 160, "block", False, False, 1.0, 1.0))
    assert W.LABEL_WORDS["block"] == (0, 0, "standing")
    assert W.note("chunli", m) == ram_note("chunli", "ryu", view(row, 1), "left", version=2)
    assert W.situation(m) == ram_situation(row)


@pytest.mark.parametrize("reader_default", ["block", None])
def test_an_unknown_sprite_is_the_readers_block(reader_default):
    """Owner rule: unknown -> "block". He: T0's words for a blocking opponent. Me: block is not stand / walk, so
    I cannot act (the loop waits 4 frames, as T0 when she cannot act)."""
    f = _facts(100, 160, "stand", False, False, 1.0, 1.0)
    f = replace(f, left=replace(f.left, unknown=True, action=reader_default, confidence=0.1),
                right=replace(f.right, unknown=True, action=reader_default, sprite=next(iter(W.crouch_sprites()))))
    m = W.moment(f)
    assert m.my_label == m.his_label == "block"
    assert not m.can_act
    assert m.doing == "standing" and m.his_attacking == 0 and not m.his_crouch


def test_the_words_default_is_the_readers_default():
    from sf2.screen.reader import DEFAULT_ACTION
    assert W.DEFAULT_LABEL == DEFAULT_ACTION and DEFAULT_ACTION in W.LABEL_WORDS
