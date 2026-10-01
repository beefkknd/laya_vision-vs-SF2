import numpy as np
import pytest

from sf2.data import action_probe as A


def rows(states, ids, cls=None):
    s = np.zeros((len(states), 0x200), np.uint8)
    s[:, A.STATE] = states
    s[:, A.ATTACK_ID] = ids
    if cls is not None:
        s[:, A.MOVE_CLASS] = cls
    return s


def test_address_stride():
    assert A.address(1, 0x3E) == 0x0C3E
    assert A.address(2, 0x3E) == 0x0E3E
    with pytest.raises(ValueError):
        A.address(3, 0)
    with pytest.raises(ValueError):
        A.address(1, 0x200)


def test_runs_and_collapse():
    assert A.runs([0, 0, 10, 10, 10, 0]) == [(0, 2), (2, 5), (5, 6)]
    assert A.runs([]) == []
    assert A.collapse([0, 46, 46, 49, 0, 45, 46]) == (46, 49, 45, 46)


def test_episode_first_id_skips_startup_zeros():
    s = rows([0, 0x0A, 0x0A, 0x0A, 0x0A, 0x0A, 0], [0, 0, 0, 5, 5, 6, 0])
    eps = A.episodes(s)
    assert [(e.start, e.end, e.state) for e in eps] == [(0, 1, 0), (1, 6, 0x0A), (6, 7, 0)]
    assert eps[1].attack_id == 5 and eps[1].ids == (5, 6)
    assert eps[0].attack_id == 0


def test_episodes_split_on_state_not_id():
    # lk (0x0A) chained into Lightning Legs (0x0C): two episodes, each its own first ID
    s = rows([0x0A] * 3 + [0x0C] * 4, [0, 7, 7, 0, 50, 51, 50])
    assert [e.attack_id for e in A.episodes(s)] == [7, 50]


def test_episodes_rejects_bad_shape():
    with pytest.raises(ValueError):
        A.episodes(np.zeros((3, 0x10), np.uint8))


def test_no_box_kind():
    s = rows([0x0A] * 5, [0] * 5, cls=[0, 6, 6, 6, 6])
    ep = A.episodes(s)[0]
    assert A.no_box_kind(s, ep) == "throw"
    s2 = rows([0x0C] * 5, [0] * 5)
    ep2 = A.episodes(s2)[0]
    assert A.no_box_kind(s2, ep2, own_shot=[0, 0, 1, 1, 1]) == "projectile"
    assert A.no_box_kind(s2, ep2, own_shot=[0] * 5) == "special_00"
    s2[:, A.SPECIAL_CLASS] = 0xFF
    assert A.no_box_kind(s2, ep2, own_shot=[0] * 5) == "cut"
    s3 = rows([0x0C] * 5, [0] * 5)
    s3[1:, A.SPECIAL_CLASS] = 0x09                      # Spinning Bird Kick hit before its box came out
    assert A.no_box_kind(s3, A.episodes(s3)[0]) == "special_09"
    s4 = rows([0x0A] * 5, [0] * 5)
    s4[1:, A.SPECIAL_CLASS] = 0x09                      # a normal (0x0A) is never classed as a special
    assert A.no_box_kind(s4, A.episodes(s4)[0]) == "cut"


def test_purity_counts_attack_states_only():
    s = rows([0x0A, 0x0A, 0, 0x0C, 0x0C, 0x0C, 0, 0x0A, 0x0A, 0x04],
             [1, 1, 0, 46, 49, 45, 0, 0, 0, 23])
    p = A.purity(A.episodes(s))
    assert p["episodes"] == 3
    assert p["with_id"] == pytest.approx(2 / 3)
    assert p["single_id"] == pytest.approx(1 / 2)
    assert A.id_counts(A.episodes(s)) == {1: 1, 46: 1, 23: 1}


def test_move_table_flags_shared_ids():
    lp_close = rows([0, 0x0A, 0x0A, 0], [0, 0, 1, 0])
    hp_far = rows([0, 0x0A, 0x0A, 0], [0, 6, 6, 0])
    throw_far = rows([0, 0x0A, 0x0A, 0], [0, 6, 6, 0])
    throw_close = rows([0, 0x0A, 0x0A, 0], [0, 0, 0, 0], cls=[0, 0xC, 0xC, 0])
    by_move, by_id = A.move_table([
        {"action": "lp", "gap": 26, "struct": lp_close}, {"action": "hp", "gap": 150, "struct": hp_far},
        {"action": "throw", "gap": 150, "struct": throw_far}, {"action": "throw", "gap": 26, "struct": throw_close}])
    assert by_move["lp"] == {"close": [1]}
    assert by_move["throw"] == {"far": [6]}
    assert by_id[6] == ["hp@far", "throw@far"]
    assert 0 not in by_id
