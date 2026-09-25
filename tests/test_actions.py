from sf2 import actions as A


def test_every_action_has_a_macro_and_description():
    assert set(A.ACTIONS) == set(A.MACROS) == set(A.CRITERIA)
    assert len(A.ACTIONS) == 12


def test_hadouken_expands_to_quarter_circle_then_release():
    seq = A.expand("hadouken")
    assert len(seq) == 12
    assert seq[0] == ("D",) and seq[3] == ("D", "F") and seq[6] == ("F", "hp") and seq[-1] == ()


def test_forward_follows_facing():
    assert A.to_physical(("F",), facing_right=True) == ["RIGHT"]
    assert A.to_physical(("F",), facing_right=False) == ["LEFT"]
    assert A.to_physical(("D", "B"), facing_right=True) == ["DOWN", "LEFT"]
    assert A.to_physical(("hp",), True) == ["Z"]


def test_to_array_uses_env_button_order():
    buttons = ["B", "A", "MODE", "START", "UP", "DOWN", "LEFT", "RIGHT", "C", "Y", "X", "Z"]
    a = A.to_array(["DOWN", "RIGHT", "Z"], buttons)
    assert [buttons[i] for i in a.nonzero()[0]] == ["DOWN", "RIGHT", "Z"]


def test_question_is_stable():
    q = A.question()
    assert q["type"] == "choice" and list(q["criteria"]) == A.ACTIONS
    assert A.question() == q
