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
