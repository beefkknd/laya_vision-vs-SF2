from sf2 import actions as A


def test_every_action_has_a_macro_and_description():
    assert set(A.ACTIONS) == set(A.MACROS) == set(A.CRITERIA)



def test_the_basic_set_has_no_special_moves():
    """World Warrior Chun-Li has no fireball and no dragon punch: those macros only gave a fierce punch."""
    assert A.ACTIONS == ["idle", "forward", "back", "jump", "crouch", "lp", "hp", "lk", "hk", "block"]


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
