"""Zangief becomes a PLAYABLE character: a RAM-free move menu in sf2.moves_free (owner 2026-10-06,
"get Zangief ready" -> train him next). Zangief is the first GRAPPLER player, so this pins the menu's
STRUCTURE (invariants that hold whatever the ROM supports): it loads, the 7 categories are right, the
character-general stance law does not crash on his moves, and he is a supported --me. WHICH specials
actually fire on this World-Warrior ROM (the 360 Spinning Piledriver especially) is settled empirically
by a separate on-screen verification (scripts/probe_action_moves.py), so this test pins his signature
special existing + the shared normals/blocks/throws/movement, not the verify-dependent firing.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from sf2.moves_free import menu                                           # noqa: E402
from sf2.system1.advice import available_moves, char_categories, moves_in_stance  # noqa: E402


def test_zangief_menu_loads_and_has_the_shared_move_families():
    names = [m.name for m in menu("zangief")]
    assert "walk_forward" in names and "block_high" in names             # movement + blocks
    assert any(n.startswith("s.") for n in names) and any(n.startswith("c.") for n in names)   # normals
    assert any(n.startswith("throw_F+") for n in names)                   # throws


def test_zangief_signature_special_exists():
    cats = char_categories("zangief")
    assert set(cats) >= {"move", "punch", "kick", "block", "throw", "special", "combo"}  # the 7 categories
    assert "spinning_piledriver" in cats["special"]                      # the SPD command throw
    assert "double_lariat" in cats["special"]                            # anti-air / anti-fireball spin


def test_stance_law_is_character_general_for_zangief():
    cats = char_categories("zangief")
    for stance in ("standing", "close", "crouch", "air"):
        got = moves_in_stance("punch", stance, cats)
        assert isinstance(got, list)
    assert available_moves("standing", cats)                              # non-empty followable set


def test_zangief_is_a_supported_player():
    src_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "scripts", "play_loop_screen.py")
    src = open(src_path).read()
    assert '"zangief"' in src.split("SUPPORTED_ME")[1].split(")")[0]      # zangief listed in SUPPORTED_ME
