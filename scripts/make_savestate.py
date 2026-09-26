"""Make the fight-start savestate from power-on: arcade mode, Chun-Li, first opponent Dhalsim, saved on the first
frame of round 1 where holding right moves her further than idling (clock 99).

    python scripts/make_savestate.py --out states/chunli_vs_dhalsim.state      # needs $SF2_ROM

Menu inputs: title, GAME START, then Chun-Li on the select grid (down, right, jab). With these timings the arcade's
first opponent was Dhalsim; check the saved PNG next to the state.
"""
import argparse
import os

import _path  # noqa: F401
from sf2.config import DEFAULT_RAM_MAP
from sf2.dataset import save_png
from sf2.headless import launch_argv
from sf2.mesen import MesenBridge
from sf2.ram import load_map

BOOT = [("-", 1200), ("start", 2), ("-", 120), ("start", 2), ("-", 120), ("down", 2), ("-", 10),
        ("right", 2), ("-", 30), ("y", 2), ("-", 400)]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--port", type=int, default=47997)
    ap.add_argument("--rom", default=os.environ.get("SF2_ROM"))
    ap.add_argument("--mesen", default=os.environ.get("SF2_MESEN"))
    args = ap.parse_args()

    b = MesenBridge(args.port, launch=launch_argv(args.port, args.rom, args.mesen))
    b.set_capture("raw")
    ram_map = load_map(DEFAULT_RAM_MAP)
    names = [v.name for v in ram_map]
    b.set_vars(ram_map)
    b.reset()
    for inp, n in BOOT:
        b.run([[] if inp == "-" else [inp]] * n)
    for i in range(1200):  # step one frame at a time until holding right moves her more than idling does
        s = b.save_state()
        idle = dict(zip(names, b.run([[]] * 3).rams[-1]))
        b.load_state(s)
        obs = b.run([["right"]] * 3)
        before, right = dict(zip(names, obs.rams[0])), dict(zip(names, obs.rams[-1]))
        b.load_state(s)
        if right["my_x"] != idle["my_x"] and before["timer"] == 0x99 and before["my_y"] == 192:
            with open(args.out, "wb") as f:
                f.write(s)
            save_png(b.load_state(s).images[0], os.path.splitext(args.out)[0] + ".png")
            print("saved %s after %d frames: %s" % (args.out, i, before))
            break
        b.run([[]])
    else:
        print("no controllable frame found")
    b.close()


if __name__ == "__main__":
    main()
