"""Value labels and note v2 for the laya-vision value fine-tune (docs/plan_laya_vision_value.md)."""
import pytest

from sf2.data.value import (NOTE_VERSIONS, VALUE_BUCKETS, expected_net, note_version, value_bucket,
                            value_probs, value_question)
from sf2.data.vs_sweep import note

R = {"a_x": 100, "d_x": 165, "a_y": 192, "d_y": 192, "a_life": 176, "d_life": 176, "d_state": 0x00}


@pytest.mark.parametrize("net,bucket", [(-176, "big_loss"), (-30, "big_loss"), (-29, "loss"), (-1, "loss"),
                                        (0, "even"), (1, "gain"), (29, "gain"), (30, "big_gain"), (99, "big_gain")])
def test_buckets(net, bucket):
    assert value_bucket(net) == bucket


def test_bucket_refuses_non_int():
    with pytest.raises(TypeError):
        value_bucket(1.5)


def test_value_question_is_an_ordered_score_over_the_buckets():
    q = value_question("throw")
    assert q["type"] == "score" and "throw" in q["instructions"]
    # a list, in bucket order, each item naming its bucket and its words (laya shows "level i: <item>")
    assert [c.split(":")[0] for c in q["criteria"]] == list(VALUE_BUCKETS)
    assert all(VALUE_BUCKETS[c.split(":")[0]] in c for c in q["criteria"])
    assert value_question("throw") == q


def test_note_v1_unchanged():
    assert note("chunli", "ryu", R, "left") == ("me=chunli dist=mid side=left dx=+65 my_bar=full opp_bar=full "
                                                "opp_airborne=0 opp_crouch=0")


@pytest.mark.parametrize("state,flag", [(0x00, 0), (0x02, 0), (0x04, 0), (0x0A, 1), (0x0C, 1), (0x0E, 0)])
def test_note_v2_adds_opp_attacking(state, flag):
    n = note("chunli", "ryu", dict(R, d_state=state), "left", version=2)
    assert n == note("chunli", "ryu", dict(R, d_state=state), "left") + " opp_attacking=%d" % flag


def test_note_unknown_version_refused():
    with pytest.raises(ValueError):
        note("chunli", "ryu", R, "left", version=3)


def test_note_version_read_from_checkpoint_config():
    assert note_version({}) == 1                       # runs/all8 and every older checkpoint
    assert note_version({"note_version": 2}) == 2
    with pytest.raises(ValueError):
        note_version({"note_version": 9})
    assert NOTE_VERSIONS == (1, 2)


def test_expected_net_weights_bucket_midpoints():
    assert expected_net({"even": 1.0}) == 0.0
    assert expected_net({"big_gain": 0.5, "big_loss": 0.5}) == pytest.approx(0.5 * 47.0 - 0.5 * 41.1)
    assert expected_net({"gain": 1.0}) > 0 > expected_net({"loss": 1.0})
    assert expected_net({"big_gain": 1.0}) > expected_net({"gain": 1.0})


def test_value_probs_maps_levels_to_buckets():
    assert value_probs({"0": 0.1, "1": 0.2, "2": 0.4, "3": 0.2, "4": 0.1}) == {
        "big_loss": 0.1, "loss": 0.2, "even": 0.4, "gain": 0.2, "big_gain": 0.1}
    with pytest.raises(ValueError):
        value_probs({"hit": 1.0})
