"""The watchable Mesen window launches with audible sound.

Mesen's saved settings default MasterVolume very low (3/100) and reduce volume when the window isn't the
focused app, so a --watch recording plays silently. The window command therefore carries explicit audio
overrides (same dotted-override mechanism as the existing --emulation.emulationSpeed). The HEADLESS launch
stays untouched. Seen RED before window_argv emitted any --audio.* flag.
"""
import os

from sf2.emu.headless import launch_argv, window_argv

_ROM = os.path.join(os.path.dirname(__file__), "fixtures", "fake.sfc")


def _rom(tmp_path):
    p = tmp_path / "fake.sfc"
    p.write_bytes(b"\x00" * 16)
    return str(p)


def test_window_sets_an_audible_master_volume(tmp_path):
    argv = window_argv(7000, rom=_rom(tmp_path))
    vol = [a for a in argv if a.startswith("--audio.masterVolume=")]
    assert vol, "window command has no --audio.masterVolume override"
    level = int(vol[0].split("=", 1)[1])
    assert level >= 50, "master volume override is not audible: %d" % level


def test_window_does_not_duck_volume_in_background(tmp_path):
    argv = window_argv(7000, rom=_rom(tmp_path))
    assert "--audio.reduceSoundInBackground=false" in argv, \
        "background volume reduction not disabled; a backgrounded Mesen window goes quiet"


def test_volume_is_clamped_to_0_100(tmp_path):
    rom = _rom(tmp_path)
    hi = [a for a in window_argv(7000, rom=rom, volume=150) if a.startswith("--audio.masterVolume=")][0]
    lo = [a for a in window_argv(7000, rom=rom, volume=-5) if a.startswith("--audio.masterVolume=")][0]
    assert hi.endswith("=100") and lo.endswith("=0")


def test_headless_launch_has_no_audio_overrides(tmp_path):
    argv = launch_argv(7000, rom=_rom(tmp_path))
    assert not any(a.startswith("--audio.") for a in argv), \
        "headless launch should not carry audio overrides"
    assert "--testrunner" in argv                       # still the headless path
