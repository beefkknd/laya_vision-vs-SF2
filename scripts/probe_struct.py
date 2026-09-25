"""Find the real x / y inside the fighter structs once life is known (World Warrior: 0x0D12 / 0x0F12).

    python scripts/probe_struct.py --base 0x0D00 --stride 0x200 [--headless]

Loads the savestate, then walks right, walks left and jumps, dumping WRAM after each phase. Prints every 16-bit
word in [base, base+0x60) for you (and base+stride for the opponent) that changed, with its values per phase.
Your x rises on "right" and falls on "left"; your y moves on "jump" and comes back. The opponent's word at the
same offset should hold a different, plausible position. Put the pair into ram_maps/*.txt as my_x/opp_x
(or my_y/opp_y) and confirm with scripts/check_env.py.
"""
import argparse

import _path  # noqa: F401
from sf2.cli import add_env_args, bridge
from sf2.ramsearch import val


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_env_args(ap)
    ap.add_argument("--base", type=lambda s: int(s, 16), default=0x0D00)
    ap.add_argument("--stride", type=lambda s: int(s, 16), default=0x200)
    ap.add_argument("--span", type=lambda s: int(s, 16), default=0x60)
    args = ap.parse_args()

    b = bridge(args)
    b.set_vars([])
    with open(args.savestate, "rb") as f:
        b.load_state(f.read())
    phases = [("start", [[]] * 90), ("right", [["right"]] * 40), ("left", [["left"]] * 40),
              ("jump_up", [["up"]] * 2 + [[]] * 14), ("landed", [[]] * 50)]
    dumps = []
    for name, frames in phases:
        b.run(frames)
        dumps.append((name, b.dump_wram()))
    b.close()
    print("%-8s %-7s " % ("offset", "who") + " ".join("%8s" % n for n, _ in dumps))
    for off in range(0, args.span, 2):
        for who, base in (("me", args.base), ("opp", args.base + args.stride)):
            vals = [val(d, base + off, 2) for _, d in dumps]
            if len(set(vals)) > 1 or who == "opp" and len({val(d, args.base + off, 2) for _, d in dumps}) > 1:
                print("0x%04X   %-7s " % (base + off, who) + " ".join("%8d" % v for v in vals))


if __name__ == "__main__":
    main()
