from sf2 import labeler
from sf2.config import PAD


def frames(seq, facing=True, pad_to=16):
    out = [(list(n), facing) for n in seq]
    return out + [([], facing)] * (pad_to - len(out))


def test_hadouken_facing_right():
    seq = [["down"]] * 2 + [["down", "right"]] * 2 + [["right", PAD["hp"]]] * 2
    assert labeler.events(frames(seq)) == [(0, "hadouken")]


def test_hadouken_facing_left_mirrors():
    seq = [["down"]] * 2 + [["down", "left"]] * 2 + [["left", PAD["lp"]]]
    assert labeler.events(frames(seq, facing=False)) == [(0, "hadouken")]


def test_shoryuken():
    seq = [[]] * 2 + [["right"]] * 2 + [["down"]] * 2 + [["down", "right", PAD["hp"]]]
    assert labeler.events(frames(seq)) == [(2, "shoryuken")]


def test_plain_attack_without_motion():
    seq = [["right"]] * 3 + [["right", PAD["hk"]]] * 3
    assert labeler.events(frames(seq)) == [(3, "hk")]


def test_decision_grid_labels():
    seq = [["left", "down"]] * 4 + [["right"]] * 4 + [[]] * 4 + [["up"]] * 4
    lab = labeler.label_frames(frames(seq, pad_to=16), hold=4)
    assert [a for _, a in lab] == ["block", "forward", "idle", "jump"]


def test_special_labels_the_window_where_the_motion_started():
    seq = [[]] * 4 + [["down"]] * 2 + [["down", "right"]] * 2 + [["right", PAD["hp"]]] + [[]] * 7
    lab = dict(labeler.label_frames(frames(seq), hold=4))
    assert lab[4] == "hadouken"
