"""The movement-pairs labels (sf2.data.pairs_labels; docs/prereg_movement_pairs.md): movement from the state byte,
direction toward / away, facing from the facing byte, air from y, distance by the fighter's own poke band - for BOTH
fighters - and the gate's independent re-derivation agrees with them on random rows (the invariant)."""
import random

import pytest
from ram_rows import row

from sf2.data import pairs_gate as G
from sf2.data import pairs_labels as L
from sf2.emu.vs import GROUND_Y

BANDS = {"all": 72, "ryu": 50, "zangief": 82}


def prow(p1=None, p2=None, **extra):
    """A row with the attack-ID bytes too; player 2 on the right facing left by default."""
    p1 = dict({"aid": 0, "mclass": 0, "sclass": 0xFF}, **(p1 or {}))
    p2 = dict({"aid": 0, "mclass": 0, "sclass": 0xFF, "facing": 0x00}, **(p2 or {}))
    return row(p1, p2, **extra)


def still(n=5, p1=None, p2=None):
    return [prow(p1, p2) for _ in range(n)]


@pytest.mark.parametrize("p", [1, 2])
@pytest.mark.parametrize("over, want", [
    ({"state": 0x14}, "down"),
    ({"state": 0x0E, "sub": 0x04, "react": 0x02}, "down"),
    ({"state": 0x0E, "sub": 0x04, "react": 0x06}, "block"),
    ({"state": 0x0E, "react": 0x08}, "block"),
    ({"state": 0x08}, "block"),
    ({"state": 0x0E, "sub": 0x02, "react": 0x02}, "hit"),
    ({"state": 0x0C}, "special"),
    ({"state": 0x0A, "mclass": 0x08}, "special"),
    ({"state": 0x0A, "mclass": 0x02}, "attack"),
    ({"state": 0x0A, "y": GROUND_Y - 40}, "attack"),
    ({"state": 0x04, "sub": 0x06}, "jump"),
    ({"state": 0x04}, "jump"),
    ({"state": 0x00, "y": GROUND_Y - 3}, "jump"),
    ({"state": 0x02}, "crouch"),
    ({"state": 0x00}, "stand"),
    ({"state": 0x06}, "unknown"),
])
def test_movement_from_the_state_byte_for_both_fighters(p, over, want):
    rows = still(5, over if p == 1 else None, over if p == 2 else None)
    assert L.movement(rows, 4, p) == want


@pytest.mark.parametrize("p, start, end, want", [
    (1, 200, 204, "toward"), (1, 200, 196, "away"), (1, 200, 201, "none"),
    (2, 260, 256, "toward"), (2, 260, 264, "away")])
def test_walk_and_direction_relative_to_the_other_fighter(p, start, end, want):
    k = "x"
    a = prow({k: start} if p == 1 else None, {k: start} if p == 2 else None)
    rows = [a, a, a, a, prow({k: end} if p == 1 else None, {k: end} if p == 2 else None)]
    mv = "stand" if want == "none" else "walk"
    assert L.movement(rows, 4, p) == mv
    assert L.direction(rows, 4, p) == want


def test_direction_when_the_fighters_have_swapped_sides():
    # player 1 on the right (x 300) of player 2 (x 260): moving left is toward him
    rows = [prow({"x": 300})] * 4 + [prow({"x": 296})]
    assert L.direction(rows, 4, 1) == "toward"


def test_jump_direction_straight_toward_away_and_none_for_other_movements():
    air = {"state": 0x04, "y": GROUND_Y - 30}
    up = [prow(dict(air, x=200))] * 4 + [prow(dict(air, x=201))]
    fwd = [prow(dict(air, x=200))] * 4 + [prow(dict(air, x=210))]
    assert L.direction(up, 4, 1) == "none" and L.movement(up, 4, 1) == "jump"
    assert L.direction(fwd, 4, 1) == "toward"
    hit = [prow({"state": 0x0E, "react": 2, "x": 200})] * 4 + [prow({"state": 0x0E, "react": 2, "x": 190})]
    assert L.direction(hit, 4, 1) == "none"


def test_direction_unknown_on_the_same_x_or_without_the_row_four_back():
    same = [prow({"x": 250}, {"x": 260})] * 4 + [prow({"x": 260}, {"x": 260})]
    assert L.direction(same, 4, 1) == "unknown"
    assert L.movement(still(3), 2, 1) == "unknown"       # standing needs t - 4


def test_impossible_x_makes_stand_and_distance_unknown():
    rows = still(4) + [prow({"x": 65369})]
    assert L.movement(rows, 4, 1) == "unknown"
    assert L.distance(rows, 4, 1, BANDS) == "unknown"


@pytest.mark.parametrize("byte, want", [(0x40, "right"), (0x00, "left"), (0x41, "unknown")])
def test_facing_byte_for_both_fighters(byte, want):
    assert L.facing([prow({"facing": byte})], 0, 1) == want
    assert L.facing([prow(None, {"facing": byte})], 0, 2) == want


def test_air_from_y():
    assert L.air([prow({"y": GROUND_Y})], 0, 1) == "ground"
    assert L.air([prow(None, {"y": GROUND_Y - 1})], 0, 2) == "air"


def test_distance_uses_the_fighters_own_band_inclusive():
    # gap 50: inside Ryu's band (50, inclusive); gap 51 outside it but inside Zangief's (82)
    r50 = [prow({"x": 200, "char": 0}, {"x": 250, "char": 6})]
    r51 = [prow({"x": 200, "char": 0}, {"x": 251, "char": 6})]
    assert L.distance(r50, 0, 1, BANDS) == "close"
    assert L.distance(r51, 0, 1, BANDS) == "far"
    assert L.distance(r51, 0, 2, BANDS) == "close"
    r_all = [prow({"x": 200, "char": 2}, {"x": 272, "char": 6})]          # Blanka: no own band here -> "all" 72
    assert L.distance(r_all, 0, 1, BANDS) == "close"


def test_the_shipped_bands_are_the_calibrated_poke_max():
    b = L.poke_bands()
    assert b["ryu"] == 50 and b["zangief"] == 82 and b["all"] == 72


def test_labels_at_reads_the_displayed_row_one_back():
    rows = still(5) + [prow({"state": 0x0A})]
    assert L.labels_at(rows, 5, 1, BANDS)["movement"] == "stand"
    assert L.labels(rows, 5, 1, BANDS)["movement"] == "attack"


def test_episode_key_is_none_when_any_part_is_unknown():
    assert L.episode_key(still(5), 4, 1) == ("stand", "none", "right")
    assert L.episode_key(still(5), 4, 2) == ("stand", "none", "left")
    assert L.episode_key(still(5, {"facing": 0x13}), 4, 1) is None


def _random_row(rng):
    def side():
        return {"state": rng.choice([0, 0, 0, 2, 4, 6, 8, 10, 12, 14, 20]), "sub": rng.choice([0, 2, 4, 6, 8]),
                "react": rng.choice([0, 2, 6, 8, 14]), "mclass": rng.choice([0, 2, 8, 14]),
                "x": rng.choice([rng.randrange(40, 470), 65369]) if rng.random() < 0.05 else rng.randrange(40, 470),
                "y": rng.choice([GROUND_Y] * 3 + [GROUND_Y - rng.randrange(1, 80)]),
                "facing": rng.choice([0, 0x40, 0x40, 7]), "char": rng.randrange(8)}
    return prow(side(), side())


def test_the_gates_independent_labels_agree_with_the_labels_on_random_rows():
    rng = random.Random(3)
    bands = L.poke_bands()
    for _ in range(300):
        rows = [_random_row(rng) for _ in range(6)]
        # make some x moves small so stand / straight jumps occur
        if rng.random() < 0.5:
            for k in ("p1_x", "p2_x"):
                rows[1][k] = rows[5][k] - rng.choice([-1, 0, 1, 3])
        for p in (1, 2):
            for t in range(6):
                assert G.independent_labels(rows, t, p, bands) == L.labels(rows, t, p, bands), (rows, t, p)
