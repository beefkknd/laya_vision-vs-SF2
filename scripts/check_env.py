"""Day 1: drive Mesen from Python, send each of the 12 actions, sanity-check the RAM map.

    python scripts/check_env.py            # then load mesen/sf2_bridge.lua in Mesen's Script Window

Writes out/check/*.png. What to look for in the printout:
  * my_x grows while "forward" runs when you are on the left (and shrinks on the right).
  * my_y changes during "jump" and comes back ("me=... jump" in the note in between).
  * lp / hp / hadouken screenshots show the move.
If x/y never move, the RAM map is wrong: rerun scripts/find_ram.py or fix ram_maps/*.txt by hand.
"""
import argparse
import os

import _path  # noqa: F401
from sf2 import actions as A
from sf2.cli import add_env_args, make_env
from sf2.dataset import save_png


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_env_args(ap)
    ap.add_argument("--out", default="out/check")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    env = make_env(args, verified=False)
    img = env.reset()
    print("frame shape:", img.shape, " full life:", env.full_hp, " start:", env.f)
    print("text:", env.text())
    env.run_frames([[]] * 60)  # let the round intro play
    save_png(env.frame, os.path.join(args.out, "00_start.png"))

    for i, a in enumerate(A.ACTIONS, 1):
        before = env.f
        xs, ys = [], []
        res = env.act(a, on_frame=lambda f: (xs.append(f.my_x), ys.append(f.my_y)))
        for f in env.run_frames([[]] * 24):  # let the move play out
            xs.append(f.my_x)
            ys.append(f.my_y)
        save_png(env.frame, os.path.join(args.out, "%02d_%s.png" % (i, a)))
        print("%-10s frames=%2d  x %4d->%4d (min %d max %d)  y %4d->%4d (min %d max %d)  dealt=%d taken=%d  %s"
              % (a, res.frames, before.my_x, env.f.my_x, min(xs), max(xs), before.my_y, env.f.my_y, min(ys),
                 max(ys), res.dmg_for, res.dmg_against, env.text()))
    for _ in range(20):
        env.act("forward")
    print("after walking forward 20x:", env.f, env.text())
    print("screenshots in", args.out)
    env.close()


if __name__ == "__main__":
    main()
