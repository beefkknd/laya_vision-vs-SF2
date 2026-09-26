import pytest

from sf2 import ram, ramsearch

MAP = """
# name  address  size signed
my_hp   7E0530   2    1
opp_hp  0x0730   2    1
my_x    $0522    2    0
opp_x   0722     2    0
my_y    0x0526   2    0
opp_y   0x0726   2    0
"""


def test_parse_map_accepts_bus_and_offset_hex():
    v = {x.name: x for x in ram.parse_map(MAP)}
    assert v["my_hp"].addr == 0x530 and v["my_hp"].signed
    assert v["my_x"].addr == 0x522 and v["opp_x"].addr == 0x722 and not v["opp_x"].signed
    again = {x.name: x for x in ram.parse_map(ram.format_map(list(v.values())))}
    assert again == v


def test_parse_map_requires_all_six():
    with pytest.raises(ValueError, match="opp_y"):
        ram.parse_map("\n".join(MAP.strip().splitlines()[:-1]))


def _dump(**fields):
    d = bytearray(0x2000)
    d[0x100] = 99  # a decoy timer, ticks down by 1
    for addr, v in fields.values():
        d[addr] = v & 0xFF
        d[addr + 1] = (v >> 8) & 0xFF
    return bytes(d)


def test_ramsearch_finds_the_struct():
    stride = 0x200
    base = dict(mh=(0x530, 144), oh=(0x530 + stride, 144), mx=(0x522, 80), ox=(0x522 + stride, 176),
                my=(0x526, 200), oy=(0x526 + stride, 200))

    def dump(**chg):
        f = dict(base)
        for k, v in chg.items():
            f[k] = (f[k][0], v)
        return _dump(**f)

    ph = {
        "start": [dump()],
        "right": [dump(mx=120)],
        "left": [dump(mx=82)],
        "jump": [dump(mx=82, my=170), dump(mx=82, my=140), dump(mx=82, my=175), dump(mx=82)],
        "take_hits": [dump(mx=82), dump(mx=82, mh=130), dump(mx=82, mh=130), dump(mx=82, mh=118)],
        "give_hits": [dump(mx=150, mh=118), dump(mx=150, mh=118, oh=120), dump(mx=150, mh=118, oh=104)],
    }
    chosen, report = ramsearch.analyze(ph)
    got = {v.name: v.addr for v in chosen}
    assert got == {"my_hp": 0x530, "opp_hp": 0x730, "my_x": 0x522, "opp_x": 0x722, "my_y": 0x526,
                   "opp_y": 0x726}
    assert report["stride"] == stride


def test_distance_bins_follow_the_measured_ranges():
    # random play, fixed harness (2026-09-25): Chun-Li's normals land below ~80 px, Dhalsim's hits fade past 120
    from sf2.ram import dist_bin

    assert [dist_bin(d) for d in (40, 79, 80, 119, 120, 200)] == ["close", "close", "mid", "mid", "far", "far"]


def test_cornered_means_the_stage_wall_is_close_behind_you():
    # walls measured on the ROM (tests/fixtures/walls): world x 53 and 459
    from sf2.ram import CORNER, LEFT_WALL, RIGHT_WALL, cornered

    assert (LEFT_WALL, RIGHT_WALL) == (53, 459)
    assert cornered(LEFT_WALL, facing_right=True) and cornered(LEFT_WALL + CORNER - 1, facing_right=True)
    assert not cornered(LEFT_WALL + CORNER, facing_right=True)
    assert not cornered(LEFT_WALL, facing_right=False)          # the wall is in front of her, not behind
    assert cornered(RIGHT_WALL, facing_right=False) and not cornered(RIGHT_WALL - CORNER, facing_right=False)
    assert not cornered(RIGHT_WALL, facing_right=True)


def _f(**kw):
    from sf2.ram import Fighters

    d = dict(my_hp=176, opp_hp=132, my_x=208, opp_x=304, my_y=192, opp_y=192, timer=0x99, my_state=0, opp_state=0)
    d.update(kw)
    return Fighters(**d)


def _note(f, my_air=False, opp_air=False, last="hk"):
    from sf2.ram import text_state

    return text_state(f, "chunli", "dhalsim", last, my_air, opp_air, 176)


def test_text_state_reads_like_a_player_sees_the_screen():
    assert _note(_f(opp_state=0x0A)) == \
        "me=chunli stand hp=100 opp=dhalsim attack hp=75 dist=mid facing=right corner=none time=early last=hk " \
        "fireball=none"


def test_text_state_facing_and_corner():
    from sf2.ram import LEFT_WALL, RIGHT_WALL

    assert "facing=left corner=me" in _note(_f(my_x=RIGHT_WALL, opp_x=RIGHT_WALL - 100))
    assert "facing=right corner=me" in _note(_f(my_x=LEFT_WALL + 10, opp_x=200))
    assert "facing=right corner=opp" in _note(_f(my_x=RIGHT_WALL - 90, opp_x=RIGHT_WALL))
    assert "facing=left corner=opp" in _note(_f(my_x=LEFT_WALL + 60, opp_x=LEFT_WALL))


def test_text_state_round_clock_is_coarse():
    # the clock is BCD seconds: 0x99 = 99 s
    words = [_note(_f(timer=t)).split("time=")[1].split()[0] for t in (0x99, 0x60, 0x59, 0x30, 0x29, 0x00)]
    assert words == ["early", "early", "mid", "mid", "late", "late"]


def test_text_state_words_follow_the_action_state_and_the_airborne_rule():
    def words(my_state, opp_state, my_air=False, opp_air=False):
        n = _note(_f(my_state=my_state, opp_state=opp_state), my_air, opp_air).split()
        return n[1], n[4]

    assert words(0x00, 0x02) == ("stand", "crouch")
    assert words(0x08, 0x0A) == ("block", "attack")
    assert words(0x0E, 0x0E, my_air=False, opp_air=False) == ("hit", "hit")
    assert words(0x04, 0x0A, my_air=True, opp_air=True) == ("jump", "jump")      # a jump attack is still a jump
    assert words(0x04, 0x04) == ("stand", "stand")      # jump state on the ground: take-off / landing frames
    assert words(0x12, None) == ("other", "stand")      # end-of-round poses; no state in the RAM map


def test_text_state_fireball():
    assert _note(_f()).endswith(" fireball=none")
    assert _note(_f(fireball=1, fireball_x=208 + 50)).endswith(" fireball=close")
    assert _note(_f(fireball=1, fireball_x=208 + 100)).endswith(" fireball=mid")
    assert _note(_f(fireball=1, fireball_x=208 - 150)).endswith(" fireball=far")
