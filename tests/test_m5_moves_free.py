"""M5: the RAM-free carve. The future screen-only play runner must import the move menu / action vocabulary /
frame helpers with NO transitive RAM import. These tests pin:
  * each new pure module imports none of the RAM modules (checked in a fresh subprocess via sys.modules),
  * its move names + button steps match sf2.data.vs_moves.chunli() exactly,
  * its action vocabulary matches sf2.data.vs_sweep.static_actions exactly,
  * its frame helpers behave identically to sf2.data.u_data's,
  * the old modules (vs_moves / vs_sweep / eye) keep every public name they had.
"""
import subprocess
import sys

import numpy as np
import pytest

# Modules a RAM-free play import must NOT drag in (RAM map, RAM-driven emu, RAM-row builders, the table).
RAM_MODULES = (
    "sf2.emu.ram", "sf2.emu.vs", "sf2.emu.boot", "sf2.emu.headless", "sf2.emu.mesen",
    "sf2.data.vs_defense", "sf2.data.vs_moves", "sf2.data.vs_sweep", "sf2.data.perception",
    "sf2.data.u_data", "sf2.data.value_oracle", "sf2.data.pairs_labels",
)


def _imported_ram(module: str):
    """Fresh-interpreter import of ``module``; returns the RAM_MODULES that ended up in sys.modules."""
    code = (
        "import importlib, sys\n"
        "importlib.import_module(%r)\n"
        "bad = [m for m in %r if m in sys.modules]\n"
        "print('\\n'.join(bad))\n" % (module, RAM_MODULES)
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert out.returncode == 0, "import %s failed:\n%s" % (module, out.stderr)
    return [line for line in out.stdout.splitlines() if line]


@pytest.mark.parametrize("module", ["sf2.moves_free", "sf2.data.actions_free", "sf2.data.frames_free"])
def test_new_module_pulls_no_ram(module):
    assert _imported_ram(module) == [], "%s transitively imported RAM" % module


def _names_steps(moves):
    return [(m.name, m.steps) for m in moves]


def test_moves_free_matches_vs_moves_chunli():
    import sf2.moves_free as free
    from sf2.data import vs_moves

    assert _names_steps(free.chunli()) == _names_steps(vs_moves.chunli())
    assert _names_steps(free.ryu()) == _names_steps(vs_moves.ryu())
    # kind must survive too (the runner needs it to press)
    assert [(m.name, m.kind) for m in free.chunli()] == [(m.name, m.kind) for m in vs_moves.chunli()]


def test_moves_free_menu_covers_action_menu():
    import sf2.moves_free as free
    from sf2.system1 import action_menu

    menu_names = {m.name for m in free.chunli()}
    listed = {n for moves in action_menu.CATEGORIES.values() for n in moves}
    assert listed <= menu_names, "action_menu names missing mechanics: %s" % (listed - menu_names)


def test_actions_free_matches_vs_sweep():
    import sf2.data.actions_free as free
    from sf2.data import vs_sweep

    for char in ("ryu", "chunli"):
        assert free.static_actions(char) == vs_sweep.static_actions(char)
    assert free.MOVEMENT == vs_sweep.MOVEMENT


def test_frames_free_matches_u_data():
    import sf2.data.frames_free as free
    from sf2.data import u_data

    img = (np.arange(224 * 256 * 3, dtype=np.uint8) % 251).reshape(224, 256, 3)
    assert np.array_equal(free.hud_frame(img), u_data.hud_frame(img))
    assert free.eye_note("chunli") == u_data.eye_note("chunli")


def test_old_public_api_unchanged():
    from sf2.data import vs_moves, vs_sweep

    # vs_moves names its callers (vs_metrics, collect, scripts, tests) rely on
    for name in ("Move", "MOVESETS", "GAPS", "REACH_GAPS", "CONDS", "connected", "combo", "toward",
                 "seen", "life_drops", "gap", "Rows", "GUARD", "HIT", "THROWN", "SPECIAL", "ATTACK",
                 "JUMP", "BLOCK_REACTS", "movement", "normals", "blocks", "throws", "ryu", "chunli"):
        assert hasattr(vs_moves, name), "vs_moves lost %s" % name
    # vs_sweep names system1 / u_data / the gate rely on
    for name in ("MOVEMENT", "NORMALS", "SPECIALS", "static_actions", "actions", "note", "current_note",
                 "outcome", "outcome_question", "OUTCOMES", "OUTCOME_CRITERIA", "mirror_record",
                 "split_of", "TEST_INDEX", "GAPS"):
        assert hasattr(vs_sweep, name), "vs_sweep lost %s" % name
