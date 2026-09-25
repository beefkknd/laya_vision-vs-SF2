"""Day 1: find where your SF2 cartridge keeps life and positions, and write ram_maps/sf2_snes.txt.

    python scripts/find_ram.py --savestate states/ryu_vs_ken.state      # scripted: Python plays the phases
    python scripts/find_ram.py --manual                                 # you play each phase in Mesen

It dumps the whole 128 KiB of work RAM during phases whose effect is known (walk right, walk left, jump, get
hit, land hits) and keeps the addresses that behave like life / x / y (rules in sf2/ramsearch.py). The two
fighters share one struct layout, so the opponent's fields are yours plus the offset between the life values.

The scripted mode needs the CPU to hit you during a 10 s wait and your fierce punches to land while walking in;
if it finds no opponent life, use --manual. Either way, confirm with scripts/check_env.py afterwards.
You start on the LEFT in both modes.
"""
import argparse
import os
import threading

import _path  # noqa: F401
from sf2 import ramsearch
from sf2.cli import add_env_args, bridge
from sf2.ram import format_map


def scripted(b, state):
    if state:
        b.load_state(state)
    idle = lambda n: b.run([[]] * n)  # noqa: E731
    ph = {}
    idle(90)
    ph["start"] = [b.dump_wram()]
    b.run([["right"]] * 40)
    ph["right"] = [b.dump_wram()]
    b.run([["left"]] * 40)
    ph["left"] = [b.dump_wram()]
    b.run([["up"]] * 2)
    ph["jump"] = []
    for _ in range(15):
        b.run([[]] * 4)
        ph["jump"].append(b.dump_wram())
    ph["take_hits"] = []
    for _ in range(20):
        idle(30)
        ph["take_hits"].append(b.dump_wram())
    ph["give_hits"] = []
    b.run([["right"]] * 60)
    for _ in range(20):
        b.run(([["right", "l"]] * 2 + [["right"]] * 6) * 3)
        ph["give_hits"].append(b.dump_wram())
    return ph


def manual(b):
    steps = [("start", "Stand still at the start of a round (you on the left)", False),
             ("right", "Walk RIGHT toward the opponent a little (no jumping, no attacks)", False),
             ("left", "Walk LEFT back about as far", False),
             ("jump", "Jump straight up ONCE and land", True),
             ("take_hits", "Let the opponent hit you 2-3 times; do not attack", True),
             ("give_hits", "Hit the opponent 2-3 times", True)]
    ph = {}
    for name, prompt, often in steps:
        done = threading.Event()
        print("\n>>> %s, then press Enter here." % prompt, flush=True)
        threading.Thread(target=lambda: (input(), done.set()), daemon=True).start()
        dumps = []
        while not done.is_set():
            b.watch(8, 0)
            if often:
                dumps.append(b.dump_wram())
        dumps.append(b.dump_wram())
        ph[name] = dumps
    return ph


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_env_args(ap)
    ap.add_argument("--manual", action="store_true")
    ap.add_argument("--force", action="store_true", help="overwrite an existing RAM map")
    args = ap.parse_args()
    if os.path.exists(args.ram_map) and not args.force:
        ap.error("%s exists; pass --force to overwrite" % args.ram_map)
    state = None
    if not args.manual and os.path.exists(args.savestate):
        with open(args.savestate, "rb") as f:
            state = f.read()
    elif not args.manual:
        print("no savestate at %s: starting from wherever Mesen is now (be at the start of a fight)"
              % args.savestate)

    b = bridge(args)
    b.set_vars([])
    try:
        ph = manual(b) if args.manual else scripted(b, state)
    finally:
        b.close()
    print("analysing %d dumps..." % sum(len(v) for v in ph.values()), flush=True)
    chosen, report = ramsearch.analyze(ph)
    for k in ("my_hp", "opp_hp", "my_x", "my_y"):
        print("%-7s candidates: %s" % (k, ", ".join("0x%04X" % c[0] + ("/8bit" if len(c) == 3 and c[1] == 1 else "")
                                                    for c in report[k]) or "none"))
    if not chosen or len(chosen) < 6:
        print("\nCould not pin down all six variables%s. Try --manual, or read them off Mesen's memory viewer "
              "(Debug > Memory Tools) and write %s by hand (format in sf2/ram.py)."
              % (" (found: %s)" % ", ".join(v.name for v in chosen) if chosen else "", args.ram_map))
        return
    os.makedirs(os.path.dirname(args.ram_map) or ".", exist_ok=True)
    with open(args.ram_map, "w") as f:
        f.write("# found by scripts/find_ram.py for ROM sha1 %s (%s); player struct stride 0x%X\n"
                % (b.rom_sha1, b.rom_name, report["stride"]))
        f.write(format_map(chosen))
    print("\nwrote %s:\n%s" % (args.ram_map, format_map(chosen)))
    print("Next: python scripts/check_env.py  (x must move on forward/back, y on jump)")


if __name__ == "__main__":
    main()
