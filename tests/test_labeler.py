from sf2 import labeler
from sf2.config import PAD


def frames(seq, facing=True, pad_to=16):
    out = [(list(n), facing) for n in seq]
    return out + [([], facing)] * (pad_to - len(out))


def test_hadouken_facing_right():
    seq = [["DOWN"]] * 2 + [["DOWN", "RIGHT"]] * 2 + [["RIGHT", PAD["hp"]]] * 2
    assert labeler.events(frames(seq)) == [(0, "hadouken")]


def test_hadouken_facing_left_mirrors():
    seq = [["DOWN"]] * 2 + [["DOWN", "LEFT"]] * 2 + [["LEFT", PAD["lp"]]]
    assert labeler.events(frames(seq, facing=False)) == [(0, "hadouken")]


def test_shoryuken():
    seq = [[]] * 2 + [["RIGHT"]] * 2 + [["DOWN"]] * 2 + [["DOWN", "RIGHT", PAD["hp"]]]
    assert labeler.events(frames(seq)) == [(2, "shoryuken")]


def test_plain_attack_without_motion():
    seq = [["RIGHT"]] * 3 + [["RIGHT", PAD["hk"]]] * 3
    assert labeler.events(frames(seq)) == [(3, "hk")]


def test_decision_grid_labels():
    seq = [["LEFT", "DOWN"]] * 4 + [["RIGHT"]] * 4 + [[]] * 4 + [["UP"]] * 4
    lab = labeler.label_frames(frames(seq, pad_to=16), hold=4)
    assert [a for _, a in lab] == ["block", "forward", "idle", "jump"]


def test_special_labels_the_window_where_the_motion_started():
    seq = [[]] * 4 + [["DOWN"]] * 2 + [["DOWN", "RIGHT"]] * 2 + [["RIGHT", PAD["hp"]]] + [[]] * 7
    lab = dict(labeler.label_frames(frames(seq), hold=4))
    assert lab[4] == "hadouken"
