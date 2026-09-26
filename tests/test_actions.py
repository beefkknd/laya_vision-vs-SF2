from sf2 import actions as A


def test_every_action_has_a_macro_and_description():
    assert set(A.ACTIONS) == set(A.MACROS) == set(A.CRITERIA)



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
