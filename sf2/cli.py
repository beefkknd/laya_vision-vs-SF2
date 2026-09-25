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
    if savestate:
        g.add_argument("--savestate", default=DEFAULT_SAVESTATE, help="fight-start savestate (record_human.py, F9)")
    if ram_map:
        g.add_argument("--ram-map", default=DEFAULT_RAM_MAP, help="written by scripts/find_ram.py")
    g.add_argument("--me", default="ryu")
    g.add_argument("--opp", default="ken")


def bridge(args):
    from .mesen import MesenBridge

    return MesenBridge(args.port, launch=args.launch)


def make_env(args):
    from .env import FightEnv
    from .ram import load_map

    ram_map = load_map(args.ram_map)
    with open(args.savestate, "rb") as f:
        state = f.read()
    return FightEnv(bridge(args), ram_map, state, me=args.me, opp=args.opp)
