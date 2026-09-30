"""scripts/train.py records in the checkpoint what its data taught (sf2.data.train_data.checkpoint_tags), so play reads
the right note and asks the value questions only of a checkpoint trained on them."""
import pytest

from sf2.data.train_data import checkpoint_tags

V1 = "me=ryu dist=mid side=left dx=+65 my_bar=full opp_bar=full opp_airborne=0 opp_crouch=0"
V2 = V1 + " opp_attacking=1"


def ex(text, t="choice"):
    return {"state": {"images": [], "context": text}, "q": {"t": t, "ins": "x", "crit": None}}


def test_v1_outcome_only():
    assert checkpoint_tags([ex(V1), ex(V1)]) == {"note_version": 1, "value_questions": False}


def test_v2_with_value_rows():
    assert checkpoint_tags([ex(V2), ex(V2, "score")]) == {"note_version": 2, "value_questions": True}


def test_mixed_notes_refused():
    with pytest.raises(ValueError):
        checkpoint_tags([ex(V1), ex(V2)])


def test_value_rows_on_v1_refused():
    with pytest.raises(ValueError):
        checkpoint_tags([ex(V1), ex(V1, "score")])


def test_empty_refused():
    with pytest.raises(ValueError):
        checkpoint_tags([])
