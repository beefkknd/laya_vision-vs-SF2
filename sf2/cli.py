"""Command-line options every emulator-facing script shares, and the env built from them."""
import argparse
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


def make_env(args):
    from .env import FightEnv
    from .ram import load_map

    ram_map = load_map(args.ram_map)
    with open(args.savestate, "rb") as f:
        state = f.read()
    b = bridge(args)
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
