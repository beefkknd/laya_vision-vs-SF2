"""Invariant (docs/laya_text_only_plan.md, "The rule: no RAM in play"): the screen reader package sf2/screen reads the
image only. Every sf2/screen/*.py is parsed; an import of the emulator, the Lua bridge, the RAM row helpers, the OAM
snapshot or the RAM-labelled data modules - or a call of a bridge / RAM method - fails the test. The scanner itself is
seen red on seeded bad sources."""
import ast
import glob
import os
from typing import List

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PACKAGE = "sf2.screen"
BANNED_MODULES = ("sf2.emu", "mesen", "sf2.data.pairs_labels", "sf2.data.vs_sweep", "sf2.data.perception",
                  "sf2.data.action_codes", "sf2.data.eye_v2", "sf2.data.pairs_collect", "sf2.sprites.emu",
                  "sf2.sprites.oam", "sf2.sprites.collect", "sf2.sprites.build", "sf2.system1", "sf2.eval.runner",
                  "socket", "subprocess")
BANNED_NAMES = ("MesenBridge", "SnapBridge", "dump_wram", "set_vars", "load_state", "render_groups", "rams", "poke")


def _absolute(node: ast.ImportFrom, package: str) -> str:
    if not node.level:
        return node.module or ""
    parts = package.split(".")
    base = parts[:len(parts) - node.level + 1]
    return ".".join(base + ([node.module] if node.module else []))


def violations(source: str, package: str = PACKAGE) -> List[str]:
    """What in ``source`` (a module of ``package``) breaks the image-only rule."""
    out = []
    for node in ast.walk(ast.parse(source)):
        mods = []
        if isinstance(node, ast.Import):
            mods = [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            base = _absolute(node, package)
            mods = [base] + ["%s.%s" % (base, a.name) for a in node.names]
        for m in mods:
            if any(m == b or m.startswith(b + ".") for b in BANNED_MODULES):
                out.append("import %s" % m)
        if isinstance(node, ast.Attribute) and node.attr in BANNED_NAMES:
            out.append("uses .%s" % node.attr)
        if isinstance(node, ast.Name) and node.id in BANNED_NAMES:
            out.append("uses %s" % node.id)
    return out


def test_screen_package_reads_the_image_only():
    files = sorted(glob.glob(os.path.join(ROOT, "sf2", "screen", "*.py")))
    assert len(files) >= 5
    bad = {}
    for path in files:
        with open(path) as f:
            v = violations(f.read())
        if v:
            bad[os.path.basename(path)] = v
    assert bad == {}


def test_scanner_catches_seeded_bad_sources():
    seeded = [
        "from ..emu.vs import GROUND_Y",
        "from ..emu import ram",
        "import sf2.emu.mesen as m",
        "from sf2.data import pairs_labels as L",
        "from ..data.vs_sweep import actions",
        "from ..sprites.oam import render_groups",
        "def f(b):\n    return b.dump_wram()",
        "def f(b):\n    return b.run([[]]).rams[0]",
        "from ..data.perception import LAG",
    ]
    for src in seeded:
        assert violations(src), src
    assert violations("from .assets import codes15\nimport numpy as np\nfrom ..config import REPO") == []
