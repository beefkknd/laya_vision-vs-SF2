"""Command-line options every emulator-facing script shares, and the env built from them."""
import argparse
import hashlib
import json
import os

from .config import DEFAULT_RAM_MAP, DEFAULT_SAVESTATE, MESEN_PORT


def add_env_args(ap: argparse.ArgumentParser, savestate: bool = True, ram_map: bool = True) -> None:
    g = ap.add_argument_group("emulator (Mesen 2 + mesen/sf2_bridge.lua)")
    g.add_argument("--port", type=int, default=MESEN_PORT)
    g.add_argument("--launch", default=os.environ.get("SF2_MESEN_LAUNCH"),
                   help="command that starts Mesen with the ROM and the bridge script, e.g. "
                        "'/Applications/Mesen.app/Contents/MacOS/Mesen --testrunner ~/roms/sf2.sfc "
                        "mesen/sf2_bridge.lua' (default: $SF2_MESEN_LAUNCH; unset = load the script by hand)")
    g.add_argument("--headless", action="store_true",
                   help="start a windowless Mesen (--testrunner) for this run; needs --rom / $SF2_ROM and "
                        "--mesen / $SF2_MESEN (default /Applications/Mesen.app/Contents/MacOS/Mesen)")
    g.add_argument("--rom", default=os.environ.get("SF2_ROM"))
    g.add_argument("--mesen", default=os.environ.get("SF2_MESEN"))
    g.add_argument("--capture", choices=["auto", "png", "raw"], default="auto",
                   help="screenshot path; auto switches to raw when PNGs come back blank (headless)")
    g.add_argument("--seed", type=int, default=0)
    g.add_argument("--jitter", type=int, default=30,
                   help="idle 1..N frames after each savestate load, a different count per match, so repeated "
                        "matches do not replay the identical fight (0 = no idle)")
    g.add_argument("--jitter-base", type=int, default=0,
                   help="added to the idle count; parallel.py gives each worker its own range")
    if savestate:
        g.add_argument("--savestate", default=DEFAULT_SAVESTATE, help="fight-start savestate (record_human.py, F9)")
    if ram_map:
        g.add_argument("--ram-map", default=DEFAULT_RAM_MAP, help="written by scripts/find_ram.py")
    g.add_argument("--me", default="ryu")
    g.add_argument("--opp", default="ken")


def bridge(args):
    from .mesen import MesenBridge

    launch = args.launch
    if args.headless:
        from .headless import launch_argv

        launch = launch_argv(args.port, args.rom, args.mesen)
    b = MesenBridge(args.port, launch=launch)
    if args.capture != "auto":
        b.set_capture(args.capture)
    return b


def _blank(img) -> bool:
    return img is None or int(img.max()) == int(img.min())


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STAMP = os.path.join(ROOT, "out", "harness_ok.json")
HARNESS_FILES = ["sf2/env.py", "sf2/ram.py", "sf2/actions.py", "sf2/mesen.py", "mesen/sf2_bridge.lua"]


def _harness(rom_sha1: str, ram_map: str) -> dict:
    def sha(path):
        with open(path, "rb") as f:
            return hashlib.sha256(f.read()).hexdigest()

    files = {p: sha(os.path.join(ROOT, p)) for p in HARNESS_FILES}
    files["ram_map"] = sha(ram_map)
    return {"rom_sha1": rom_sha1, "files": files}


def write_harness_stamp(rom_sha1: str, ram_map: str, path: str = STAMP) -> None:
    """Written by tests/test_rom_harness.py when every acceptance check passed."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(_harness(rom_sha1, ram_map), f, indent=2)


def check_harness(rom_sha1: str, ram_map: str, path: str = STAMP) -> None:
    """Refuse to collect or play on a harness the ROM acceptance test has not passed as it is now."""
    if os.environ.get("SF2_UNVERIFIED"):
        print("WARNING: SF2_UNVERIFIED is set; running on an UNVERIFIED harness", flush=True)
        return
    fix = "run SF2_ROM=... pytest -q tests/test_rom_harness.py (or set SF2_UNVERIFIED=1)"
    if not os.path.exists(path):
        raise RuntimeError("harness not verified: %s" % fix)
    with open(path) as f:
        stamp = json.load(f)
    now = _harness(rom_sha1, ram_map)
    if stamp["rom_sha1"] != now["rom_sha1"]:
        raise RuntimeError("harness was verified on ROM %s, this is %s: %s" % (stamp["rom_sha1"], rom_sha1, fix))
    changed = [p for p, h in now["files"].items() if stamp["files"].get(p) != h]
    if changed:
        raise RuntimeError("harness changed since it was verified (%s): %s" % (", ".join(changed), fix))


def make_env(args, verified: bool = True):
    """``verified``: collection and play need a passing ROM acceptance run; diagnostics pass False."""
    from .env import FightEnv
    from .ram import load_map

    ram_map = load_map(args.ram_map)
    with open(args.savestate, "rb") as f:
        state = f.read()
    b = bridge(args)
    if verified:
        try:
            check_harness(b.rom_sha1, args.ram_map, STAMP)
        except RuntimeError:
            b.close()  # a headless Mesen we launched would otherwise outlive us
            raise
    if args.capture == "auto":
        b.set_vars([])
        if _blank(b.load_state(state).images.get(0)):
            print("screenshots came back blank: switching to the raw screen buffer", flush=True)
            b.set_capture("raw")
    return fight_env(b, ram_map, state, args)


def fight_env(backend, ram_map, state, args):
    from .env import FightEnv

    return FightEnv(backend, ram_map, state, me=args.me, opp=args.opp, jitter=args.jitter,
                     jitter_base=args.jitter_base)
