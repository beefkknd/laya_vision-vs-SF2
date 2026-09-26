from sf2 import labeler
from sf2.config import PAD


def frames(seq, facing=True, pad_to=16):
    out = [(list(n), facing) for n in seq]
    return out + [([], facing)] * (pad_to - len(out))


def test_a_fireball_or_dragon_punch_motion_is_just_its_punch():
    """World Warrior Chun-Li has neither special: the motion gives the plain punch."""
    seq = [["down"]] * 2 + [["down", "right"]] * 2 + [["right", PAD["hp"]]] * 2
    assert labeler.events(frames(seq)) == [(4, "hp")]
    seq = [[]] * 2 + [["right"]] * 2 + [["down"]] * 2 + [["down", "right", PAD["lp"]]]
    assert labeler.events(frames(seq)) == [(6, "lp")]


def test_plain_attack_without_motion():
    seq = [["right"]] * 3 + [["right", PAD["hk"]]] * 3
    assert labeler.events(frames(seq)) == [(3, "hk")]


def test_decision_grid_labels():
    seq = [["left", "down"]] * 4 + [["right"]] * 4 + [[]] * 4 + [["up"]] * 4
    lab = labeler.label_frames(frames(seq, pad_to=16), hold=4)
    assert [a for _, a in lab] == ["block", "forward", "idle", "jump"]


def test_every_label_is_an_action():
    from sf2.actions import ACTIONS

    seq = [[]] * 4 + [["down"]] * 2 + [["down", "right"]] * 2 + [["right", PAD["hp"]]] + [[]] * 7
    assert {a for _, a in labeler.label_frames(frames(seq), hold=4)} <= set(ACTIONS)
