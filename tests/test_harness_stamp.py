"""Collection and play refuse to run on a harness the ROM acceptance test has not passed."""
import json

import pytest

from sf2.cli import check_harness, write_harness_stamp

ROM = "7DDCB96E0D9FEA94D9370635262AC7C28DA85214"
MAP = "ram_maps/sf2_snes.txt"


def test_a_fresh_stamp_for_this_rom_and_code_passes(tmp_path, monkeypatch):
    monkeypatch.delenv("SF2_UNVERIFIED", raising=False)
    stamp = tmp_path / "harness_ok.json"
    write_harness_stamp(ROM, MAP, str(stamp))
    check_harness(ROM, MAP, str(stamp))


def test_no_stamp_refuses(tmp_path, monkeypatch):
    monkeypatch.delenv("SF2_UNVERIFIED", raising=False)
    with pytest.raises(RuntimeError, match="tests/test_rom_harness.py"):
        check_harness(ROM, MAP, str(tmp_path / "missing.json"))


def test_another_rom_refuses(tmp_path, monkeypatch):
    monkeypatch.delenv("SF2_UNVERIFIED", raising=False)
    stamp = tmp_path / "harness_ok.json"
    write_harness_stamp(ROM, MAP, str(stamp))
    with pytest.raises(RuntimeError, match="ROM"):
        check_harness("0" * 40, MAP, str(stamp))


def test_harness_code_changed_since_the_stamp_refuses(tmp_path, monkeypatch):
    monkeypatch.delenv("SF2_UNVERIFIED", raising=False)
    stamp = tmp_path / "harness_ok.json"
    write_harness_stamp(ROM, MAP, str(stamp))
    s = json.loads(stamp.read_text())
    s["files"]["sf2/env.py"] = "0" * 64
    stamp.write_text(json.dumps(s))
    with pytest.raises(RuntimeError, match="sf2/env.py"):
        check_harness(ROM, MAP, str(stamp))


def test_override_lets_it_run_with_a_warning(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("SF2_UNVERIFIED", "1")
    check_harness(ROM, MAP, str(tmp_path / "missing.json"))
    assert "UNVERIFIED" in capsys.readouterr().out


def test_refusing_closes_the_emulator_it_started(tmp_path, monkeypatch):
    import argparse

    from sf2 import cli

    monkeypatch.delenv("SF2_UNVERIFIED", raising=False)
    monkeypatch.setattr(cli, "STAMP", str(tmp_path / "missing.json"))

    class Bridge:
        rom_sha1, closed = ROM, False

        def close(self):
            Bridge.closed = True

    monkeypatch.setattr(cli, "bridge", lambda args: Bridge())
    args = argparse.Namespace(ram_map=MAP, savestate="states/chunli_vs_dhalsim.state", capture="raw")
    with pytest.raises(RuntimeError):
        cli.make_env(args)
    assert Bridge.closed
