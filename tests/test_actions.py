from sf2 import actions as A


def test_every_action_has_a_macro_and_description():
    assert set(A.ACTIONS) <= set(A.MACROS) == set(A.CRITERIA)



def test_the_basic_set_has_no_special_moves():
    """World Warrior Chun-Li has no fireball and no dragon punch: those macros only gave a fierce punch."""
    assert A.ACTIONS[:11] == ["idle", "forward", "back", "jump", "jump_forward", "crouch", "lp", "hp", "lk", "hk",
                              "block"]


def test_throw_is_toward_and_fierce_on_the_same_frame():
    """On the ROM toward (or back) + fierce throws Dhalsim when he is within 42 px at the press; further out a plain
    fierce comes out. Pressing the direction first gains nothing."""
    assert "throw" in A.ACTIONS
    assert A.expand("throw") == [("F", "hp")] * 2 + [("F",)] * 2


def test_jump_forward_is_up_and_toward_the_opponent():
    assert A.expand("jump_forward") == [("U", "F")] * 4
    assert A.to_physical(("U", "F"), facing_right=True) == ["up", "right"]
    assert A.to_physical(("U", "F"), facing_right=False) == ["up", "left"]


def test_forward_follows_facing():
    assert A.to_physical(("F",), facing_right=True) == ["right"]
    assert A.to_physical(("F",), facing_right=False) == ["left"]
    assert A.to_physical(("D", "B"), facing_right=True) == ["down", "left"]
    assert A.to_physical(("hp",), True) == ["l"]  # fierce punch on the SNES pad


def test_physical_names_are_snes_buttons():
    from sf2.config import BUTTONS

    for a in A.ACTIONS:
        for tokens in A.expand(a):
            for facing in (True, False):
                assert set(A.to_physical(tokens, facing)) <= set(BUTTONS)


def test_question_is_stable():
    q = A.question()
    assert q["type"] == "choice" and list(q["criteria"]) == A.ACTIONS
    assert A.question() == q


def test_sweep_is_down_and_roundhouse_on_the_same_frame():
    """Crouching roundhouse: on the ROM it knocks Dhalsim down when it connects (reach ~70 px)."""
    assert "sweep" in A.ACTIONS
    assert A.expand("sweep") == [("D", "hk")] * 2 + [("D",)] * 2


def test_lightning_legs_is_twelve_quick_short_taps():
    """On the ROM 12 short taps, 1 frame down and 1 up, start the Legs (0C) at frame 18 from 39 of 40 standing
    starts; 8 never do, 10 miss up close; roundhouse taps never do within one macro."""
    assert "lightning_legs" in A.ACTIONS
    assert A.expand("lightning_legs") == [("lk",), ()] * 12


def test_every_character_has_the_shared_basics_and_its_own_specials():
    """docs/MOVES.md: the basics are shared by all 8; the specials are per character. Chun-Li's special is the
    Lightning Legs and the Spinning Bird Kick; a character whose specials are not in the code yet has the basics only."""
    assert A.CHARACTERS == ["ryu", "ken", "honda", "blanka", "guile", "chunli", "zangief", "dhalsim"]
    assert A.BASICS == A.ACTIONS[:13]
    assert A.moves("chunli") == A.BASICS + ["lightning_legs", "spinning_bird_kick"]
    for c in A.CHARACTERS:
        assert A.moves(c)[:13] == A.BASICS and set(A.moves(c)) <= set(A.MACROS) == set(A.CRITERIA)


def test_an_unknown_character_has_no_move_list():
    import pytest

    with pytest.raises(KeyError):
        A.moves("balrog")


def test_the_question_lists_the_characters_own_moves():
    """The model is shown its own character's move list; the default stays Chun-Li's (the existing datasets)."""
    assert A.question() == A.question("chunli")
    assert list(A.question("chunli")["criteria"]) == A.ACTIONS
    for c in A.CHARACTERS:
        assert list(A.question(c)["criteria"]) == A.moves(c)


def test_ryu_specials_are_the_sheets_motions_with_fierce_and_roundhouse():
    """docs/MOVES.md: Hadoken D, DF, F + P; Shoryuken F, D, DF + P; Hurricane Kick D, DB, B + K (verified on the ROM
    from both facings, tests/test_rom_moves.py). Facing left, forward is left."""
    assert A.moves("ryu") == A.BASICS + ["hadoken", "shoryuken", "tatsumaki"]
    assert A.expand("hadoken") == [("D",)] * 2 + [("D", "F")] * 2 + [("F", "hp")] * 2 + [()] * 2
    assert A.expand("shoryuken") == [("F",)] * 2 + [("D",)] * 2 + [("D", "F", "hp")] * 2 + [()] * 2
    assert A.expand("tatsumaki") == [("D",)] * 2 + [("D", "B")] * 2 + [("B", "hk")] * 2 + [()] * 2
    assert A.to_physical(("D", "F", "hp"), facing_right=False) == ["down", "left", "l"]
    assert A.to_physical(("D", "B"), facing_right=False) == ["down", "right"]


def test_spinning_bird_kick_holds_the_whole_down_charge_then_up_and_roundhouse():
    """docs/MOVES.md: [D]60, U + K. On the ROM the SBK needs down held at least 61 frames from standing (20 of 20
    free moments, both facings; tests/test_rom_moves.py): one macro holds 64, then up + roundhouse."""
    assert A.expand("spinning_bird_kick") == [("D",)] * 64 + [("U", "hk")] * 2 + [()] * 2
