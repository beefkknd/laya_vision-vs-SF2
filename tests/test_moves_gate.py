"""The ALL-8 gate (docs/TWO_SYSTEM_PLAN.md): training refuses to start until every (character, move, facing) has a
passing ROM record under the current harness stamp. No ROM needed."""
import json
import os
import sys

import pytest

from sf2 import actions as A
from sf2 import verified as V

MAP = "ram_maps/sf2_snes.txt"
ROM = "7DDCB96E0D9FEA94D9370635262AC7C28DA85214"
ALL_SPECIALS = {c: A.SPECIALS.get(c, ["special_" + c]) for c in A.CHARACTERS}   # as if every sheet were coded


@pytest.fixture
def all_coded(monkeypatch):
    for c, s in ALL_SPECIALS.items():
        for m in s:
            monkeypatch.setitem(A.MACROS, m, [((), 1)])
            monkeypatch.setitem(A.CRITERIA, m, m)
    monkeypatch.setattr(A, "SPECIALS", dict(ALL_SPECIALS))


def _complete(path, key, skip=()):
    for c in A.CHARACTERS:
        for m in A.moves(c):
            for f in V.FACINGS:
                if (c, m, f) not in skip and c not in skip:
                    V.record(c, m, f, key, "fake", path=str(path))


def test_a_complete_record_under_the_current_stamp_passes(tmp_path, all_coded):
    key = V.stamp_key(ROM, MAP)
    _complete(tmp_path / "v", key)
    V.check_moves(MAP, path=str(tmp_path / "v"))


def test_no_records_refuses(tmp_path, all_coded):
    with pytest.raises(RuntimeError, match="not verified"):
        V.check_moves(MAP, path=str(tmp_path / "missing"))


def test_a_missing_move_refuses(tmp_path, all_coded):
    _complete(tmp_path / "v", V.stamp_key(ROM, MAP), skip={("guile", "sweep", "right"), ("guile", "sweep", "left")})
    with pytest.raises(RuntimeError, match="guile/sweep/right"):
        V.check_moves(MAP, path=str(tmp_path / "v"))


def test_a_missing_facing_refuses(tmp_path, all_coded):
    _complete(tmp_path / "v", V.stamp_key(ROM, MAP), skip={("chunli", "lightning_legs", "left")})
    with pytest.raises(RuntimeError, match="chunli/lightning_legs/left"):
        V.check_moves(MAP, path=str(tmp_path / "v"))


def test_a_missing_character_refuses(tmp_path, all_coded):
    _complete(tmp_path / "v", V.stamp_key(ROM, MAP), skip={"dhalsim"})
    with pytest.raises(RuntimeError, match="dhalsim/idle/right"):
        V.check_moves(MAP, path=str(tmp_path / "v"))


def test_a_character_whose_specials_are_not_in_the_code_refuses(tmp_path, all_coded, monkeypatch):
    """Records for the basics alone do not make a character complete: every WW character has specials."""
    _complete(tmp_path / "v", V.stamp_key(ROM, MAP))
    monkeypatch.setattr(A, "SPECIALS", {c: s for c, s in ALL_SPECIALS.items() if c != "zangief"})
    with pytest.raises(RuntimeError, match="zangief: specials not in the code"):
        V.check_moves(MAP, path=str(tmp_path / "v"))


def test_a_stale_stamp_refuses(tmp_path, all_coded):
    """A record from before a harness change (another stamp) does not count, even for one move."""
    _complete(tmp_path / "v", V.stamp_key(ROM, MAP))
    V.record("ken", "sweep", "left", "0" * 64, "fake", path=str(tmp_path / "v"))
    with pytest.raises(RuntimeError, match="ken/sweep/left"):
        V.check_moves(MAP, path=str(tmp_path / "v"))
    _complete(tmp_path / "old", "1" * 64)
    with pytest.raises(RuntimeError, match="not verified"):
        V.check_moves(MAP, path=str(tmp_path / "old"))


def test_the_stamp_follows_the_harness_code_and_the_rom(tmp_path):
    assert V.stamp_key(ROM, MAP) == V.stamp_key(ROM, MAP)
    assert V.stamp_key("0" * 40, MAP) != V.stamp_key(ROM, MAP)
    other = tmp_path / "map.txt"
    other.write_text(open(MAP).read() + "\n# changed\n")
    assert V.stamp_key(ROM, str(other)) != V.stamp_key(ROM, MAP)


def test_the_rom_is_the_one_the_ram_map_names():
    assert V.map_rom_sha1(MAP) == ROM


def test_the_gate_refuses_today():
    """Only Chun-Li has specials in the code now, so the real move lists cannot be complete."""
    with pytest.raises(RuntimeError, match="specials not in the code"):
        V.check_moves(MAP, path=V.RECORDS)


def test_records_are_one_file_per_move_so_parallel_runs_do_not_collide(tmp_path):
    V.record("chunli", "lightning_legs", "right", "k" * 64, "tests/x.py::t", path=str(tmp_path))
    V.record("chunli", "sweep", "left", "k" * 64, "tests/y.py::t", path=str(tmp_path))
    assert sorted(os.listdir(tmp_path)) == ["chunli.lightning_legs.right.json", "chunli.sweep.left.json"]
    assert json.loads((tmp_path / "chunli.lightning_legs.right.json").read_text()) == {"stamp": "k" * 64,
                                                                                      "test": "tests/x.py::t"}


def test_train_refuses_before_loading_the_model(tmp_path, monkeypatch):
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))
    import train

    monkeypatch.setattr(V, "RECORDS", str(tmp_path / "missing"))
    monkeypatch.setattr(sys, "argv", ["train.py", "--data", "d", "--out", "o"])
    with pytest.raises(RuntimeError, match="not verified"):
        train.main()
