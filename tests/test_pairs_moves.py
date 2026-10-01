"""The directed player 1's move list and cycle (sf2.data.pairs_moves): the full set of attacks (normals, crouching
normals, jumps with and without each button, throw, specials) and defence (blocks, walks, crouch, stand) per
character; each word once per seeded shuffled round; the RAM check that a move was really done."""
import random

import pytest
from ram_rows import row

from sf2.config import PAD
from sf2.data import pairs_collect as PC
from sf2.data import pairs_moves as PM
from sf2.emu.vs import GROUND_Y

CHARS = ("blanka", "chunli", "dhalsim", "guile", "honda", "ken", "ryu", "zangief")
SPECIALS = {"blanka": 2, "chunli": 2, "dhalsim": 2, "guile": 2, "honda": 2, "ken": 3, "ryu": 3, "zangief": 2}


@pytest.mark.parametrize("char", CHARS)
def test_every_character_has_the_full_set(char):
    m = PM.moves(char)
    kinds = [PM.kind(char, w) for w in m]
    assert kinds.count(PM.KIND_NORMAL) == 6
    assert kinds.count(PM.KIND_CROUCH) == 6
    assert kinds.count(PM.KIND_JUMP) == 3
    assert kinds.count(PM.KIND_JUMP_ATTACK) == 18
    assert kinds.count(PM.KIND_THROW) == 1
    assert kinds.count(PM.KIND_BLOCK) == 2
    assert kinds.count(PM.KIND_WALK) == 4
    assert kinds.count(PM.KIND_SPECIAL) == SPECIALS[char]
    assert len(m) == 40 + SPECIALS[char]


def test_walks_crouch_and_stand_are_movement_not_jumps():
    for w in ("forward", "back", "crouch", "idle"):
        assert PM.kind("ryu", w) == PM.KIND_WALK


def test_a_jump_attack_holds_the_jump_then_presses_the_button_in_the_air():
    steps = PM.moves("ryu")["jump_forward_hk"]
    assert steps == ((("U", "F"), PM.JUMP_HOLD), ((), PM.JUMP_WAIT), (("hk",), PM.PRESS), ((), PM.PRESS))
    assert PM.moves("ryu")["jump_back"] == ((("U", "B"), PM.JUMP_HOLD),)
    assert PM.jump_of("jump_forward_hk") == "jump_forward" and PM.jump_of("jump") == "jump"
    with pytest.raises(ValueError):
        PM.jump_of("forward")


def test_crouching_normals_cover_every_button():
    m = PM.moves("ken")
    for b, w in PM.CROUCH_NAMES.items():
        assert m[w][0] == (("D", b), 2)


def test_specials_are_the_system_ones():
    from sf2.data.vs_sweep import SPECIALS as S
    assert PM.moves("guile")["sonic_boom"] == S["guile"]["sonic_boom"]
    assert set(PM.specials_of("chunli")) == {"lightning_legs", "spinning_bird_kick"}


def test_unknown_word_or_character_is_refused():
    with pytest.raises(ValueError):
        PM.kind("ryu", "yoga_fire")
    with pytest.raises(ValueError):
        PM.moves("vega")


def test_cycle_uses_every_word_once_per_round_in_a_fresh_shuffle():
    words = list(PM.moves("ryu"))
    c = PM.Cycle(words, random.Random(1))
    r1 = [c.next() for _ in words]
    r2 = [c.next() for _ in words]
    assert sorted(r1) == sorted(words) == sorted(r2)
    assert r1 != r2 and r1 != words
    assert c.rounds == 2
    again = PM.Cycle(words, random.Random(1))
    assert [again.next() for _ in words] == r1
    other = PM.Cycle(words, random.Random(2))
    assert [other.next() for _ in words] != r1
    with pytest.raises(ValueError):
        PM.Cycle([], random.Random(0))


def test_press_frames_resolve_jumps_from_x_and_attacks_from_the_facing_byte():
    # player 1 is left of player 2 (x 200 < 260) but the facing byte says left (mid turn-around)
    r = row({"x": 200, "facing": 0x00})
    jf = PC.press_frames("ryu", "jump_forward", r)
    assert jf[0] == ["up", "right"]
    thr = PC.press_frames("ryu", "throw", r)
    assert thr[0] == ["left", PAD["hp"]]
    assert len(PC.press_frames("ryu", "lp", r)) >= PC.WAIT


def mv(p1, n=1, p2=None):
    base = {"aid": 0, "mclass": 0}
    return [row(dict(base, **p1), dict(base, **(p2 or {})))] * n


@pytest.mark.parametrize("word, rows, want", [
    ("hp", mv({"state": 0}) + mv({"state": 0x0A, "aid": 3}), "done"),
    ("hp", mv({"state": 0}) + mv({"state": 0x0A}), "missed"),
    ("hp", mv({"state": 0}) + mv({"state": 0x0E, "react": 2}), "interrupted"),
    ("c.mk", mv({"state": 2}) + mv({"state": 0x0A, "aid": 20}), "done"),
    ("hadoken", mv({"state": 0}) + mv({"state": 0x0C}), "done"),
    ("hadoken", mv({"state": 0}) + mv({"state": 0x0A, "aid": 2}), "missed"),
    ("throw", mv({"state": 0}) + mv({"state": 0x0A}), "done"),
    ("block_high", mv({"state": 0}) + mv({"state": 0x08}), "done"),
    ("block_low", mv({"state": 2}) + mv({"state": 0x0E, "react": 0x06}), "done"),
    ("block_high", mv({"state": 0}, 3), "held"),
    ("crouch", mv({"state": 0}) + mv({"state": 2}), "done"),
    ("idle", mv({"state": 0}, 4), "done"),
    ("idle", mv({"state": 0}) + mv({"state": 2}), "missed"),
    ("forward", mv({"x": 200}) + mv({"x": 210}), "done"),
    ("forward", mv({"x": 200}) + mv({"x": 190}), "missed"),
    ("back", mv({"x": 200}) + mv({"x": 190}), "done"),
    ("back", mv({"x": 55}) + mv({"x": 55, "state": 0x08}), "done"),                      # a block at the wall
    ("back", mv({"x": 55}) + mv({"x": 55}), "missed"),
    ("sweep", mv({"state": 0}) + mv({"state": 0x0A}, 1) + [dict(mv({"state": 0x0A})[0], result=2)], "cut"),
    ("jump", mv({"x": 200}) + mv({"state": 4, "y": GROUND_Y - 50, "x": 201}), "done"),
    ("jump_forward", mv({"x": 200}) + mv({"state": 4, "y": GROUND_Y - 50, "x": 230}), "done"),
    ("jump_forward", mv({"x": 200}) + mv({"state": 4, "y": GROUND_Y - 50, "x": 200}), "missed"),
    ("jump_back", mv({"x": 200}) + mv({"state": 4, "y": GROUND_Y - 50, "x": 170}), "done"),
    ("jump_back", mv({"x": 63}) + mv({"state": 4, "y": GROUND_Y - 50, "x": 63}), "done"),       # at the wall
    ("jump_back", mv({"x": 200}) + mv({"state": 4, "y": GROUND_Y - 50, "x": 230}), "missed"),
    ("jump_lp", mv({"x": 200}) + mv({"state": 4, "y": GROUND_Y - 50, "x": 200, "aid": 23}), "done"),
    ("jump_lp", mv({"x": 200}) + mv({"state": 4, "y": GROUND_Y - 50, "x": 200}), "missed"),
    ("jump_lp", mv({"x": 200, "aid": 3}) + mv({"state": 4, "y": GROUND_Y - 50, "x": 200}), "missed"),  # ground box
])
def test_executed_reads_the_move_from_ram(word, rows, want):
    assert PM.executed("ryu", word, rows)["status"] == want


def test_executed_lists_the_attack_ids_and_refuses_no_rows():
    out = PM.executed("ryu", "hp", mv({"state": 0x0A, "aid": 3}) + mv({"state": 0x0A, "aid": 5}))
    assert out["aids"] == [3, 5]
    with pytest.raises(ValueError):
        PM.executed("ryu", "hp", [])
