from sf2 import labeler
from sf2.config import PAD


def frames(seq, facing=True, pad_to=16):
    out = [(list(n), facing) for n in seq]
    return out + [([], facing)] * (pad_to - len(out))


def test_a_fireball_or_dragon_punch_motion_is_just_its_punch():
    """World Warrior Chun-Li has neither special: the motion gives what its last frame gives (toward + fierce: throw)."""
    seq = [["down"]] * 2 + [["down", "right"]] * 2 + [["right", PAD["hp"]]] * 2
    assert labeler.events(frames(seq)) == [(4, "throw")]
    seq = [[]] * 2 + [["right"]] * 2 + [["down"]] * 2 + [["down", "right", PAD["lp"]]]
    assert labeler.events(frames(seq)) == [(6, "lp")]


def test_plain_attack_without_motion():
    seq = [["right"]] * 3 + [["right", PAD["hk"]]] * 3
    assert labeler.events(frames(seq)) == [(3, "hk")]


def test_decision_grid_labels():
    seq = [["left", "down"]] * 4 + [["right"]] * 4 + [[]] * 4 + [["up"]] * 4 + [["up", "right"]] * 4
    lab = labeler.label_frames(frames(seq, pad_to=20), hold=4)
    assert [a for _, a in lab] == ["block", "forward", "idle", "jump", "jump_forward"]


def test_every_label_is_an_action():
    from sf2.actions import ACTIONS

    seq = [[]] * 4 + [["down"]] * 2 + [["down", "right"]] * 2 + [["right", PAD["hp"]]] + [[]] * 7
    assert {a for _, a in labeler.label_frames(frames(seq), hold=4)} <= set(ACTIONS)


def test_toward_or_back_with_fierce_or_strong_is_a_throw_and_down_roundhouse_a_sweep():
    """On the ROM toward / back + fierce or strong throws up close (a fierce further out: the throw action gives the
    same); down + roundhouse is the sweep."""
    assert labeler.events(frames([["right", PAD["hp"]]])) == [(0, "throw")]
    assert labeler.events(frames([["left", PAD["mp"]]])) == [(0, "throw")]
    assert labeler.events(frames([["left", PAD["hp"]]], facing=False)) == [(0, "throw")]
    assert labeler.events(frames([[PAD["hp"]]])) == [(0, "hp")]
    assert labeler.events(frames([["down", PAD["hk"]]])) == [(0, "sweep")]
    assert labeler.events(frames([["down", "left", PAD["hk"]]])) == [(0, "sweep")]
    assert labeler.events(frames([["down", PAD["lk"]]])) == [(0, "lk")]
    assert labeler.events(frames([["up", "right", PAD["hp"]]])) == [(0, "hp")]              # in the air: no throw
