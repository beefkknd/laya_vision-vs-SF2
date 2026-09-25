from sf2 import actions as A


def test_every_action_has_a_macro_and_description():
    assert set(A.ACTIONS) == set(A.MACROS) == set(A.CRITERIA)
    assert len(A.ACTIONS) == 12


def test_hadouken_expands_to_quarter_circle_then_release():
    seq = A.expand("hadouken")
    assert len(seq) == 12
    assert seq[0] == ("D",) and seq[3] == ("D", "F") and seq[6] == ("F", "hp") and seq[-1] == ()


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
