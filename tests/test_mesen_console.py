"""The --watch window opens a Mesen Script Window (the "console") for the bridge Lua. For a clean
screen-recording the owner wants it suppressible AND restorable. We cannot hide a window headlessly,
but Mesen restores each window's SAVED geometry on open, so parking the Script Window off-screen in
the settings file (inside the existing KeepMesenSettings snapshot) is the lever -- and the snapshot
restores the real settings byte-for-byte afterwards.

These tests pin the MECHANICS (the pure settings transform + the snapshot/restore), which are
script-decidable without a display. The VISUAL result (does the window actually leave the recording)
is the one thing a headless env cannot assert; it is called out for the owner in the report.

Seen RED provenance: against the pre-change headless.py there is no `park_script_window` (AttributeError)
and KeepMesenSettings takes no `hide_console`/`path` (TypeError).
"""
import json
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from sf2.emu import headless as H   # noqa: E402

OFFSCREEN_MIN = 20000               # a parked window is at least this far out


def _settings() -> dict:
    return {"Version": 1, "Debug": {"ScriptWindow": {
        "WindowLocation": {"X": 1891, "Y": 109}, "WindowSize": {"Width": 1280, "Height": 522},
        "LogWindowHeight": 100, "AutoStartScriptOnLoad": True}}}


def test_park_script_window_moves_offscreen_and_collapses_log():
    d = _settings()
    parked = H.park_script_window(d)
    sw = parked["Debug"]["ScriptWindow"]
    assert sw["WindowLocation"]["X"] >= OFFSCREEN_MIN
    assert sw["WindowLocation"]["Y"] >= OFFSCREEN_MIN
    assert sw["LogWindowHeight"] == 0
    # the bridge must still auto-run, or there is no game to watch
    assert sw["AutoStartScriptOnLoad"] is True


def test_park_script_window_does_not_mutate_input():
    d = _settings()
    before = json.dumps(d, sort_keys=True)
    H.park_script_window(d)
    assert json.dumps(d, sort_keys=True) == before      # immutability (owner coding-style)


def _write_settings_with_bom(path: str, d: dict) -> bytes:
    raw = ("﻿" + json.dumps(d, indent=2)).encode("utf-8")   # Mesen writes a UTF-8 BOM
    with open(path, "wb") as f:
        f.write(raw)
    return raw


def test_keep_settings_parks_then_restores(tmp_path):
    path = os.path.join(str(tmp_path), "settings.json")
    original = _write_settings_with_bom(path, _settings())

    with H.KeepMesenSettings(path=path, hide_console=True):
        # during the window: the Script Window is parked off-screen so Mesen opens it out of the recording
        live = json.loads(open(path, encoding="utf-8-sig").read())
        assert live["Debug"]["ScriptWindow"]["WindowLocation"]["X"] >= OFFSCREEN_MIN

    # after: the real settings are back, byte-for-byte (restorable)
    assert open(path, "rb").read() == original


def test_keep_settings_noop_when_not_hiding(tmp_path):
    path = os.path.join(str(tmp_path), "settings.json")
    original = _write_settings_with_bom(path, _settings())
    with H.KeepMesenSettings(path=path, hide_console=False):
        assert open(path, "rb").read() == original      # untouched while showing the console
    assert open(path, "rb").read() == original


def test_keep_settings_missing_file_is_safe(tmp_path):
    path = os.path.join(str(tmp_path), "absent.json")
    with H.KeepMesenSettings(path=path, hide_console=True):
        pass                                             # no file to park/restore: must not raise
    assert not os.path.exists(path)
