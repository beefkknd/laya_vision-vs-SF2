import pytest

from sf2.emu import ram

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


def test_parse_map_requires_all_six():
    with pytest.raises(ValueError, match="opp_y"):
        ram.parse_map("\n".join(MAP.strip().splitlines()[:-1]))
