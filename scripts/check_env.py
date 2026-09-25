"""Day 1: hook the emulator, take screenshots, send each of the 12 actions, sanity-check the RAM map.

    python scripts/check_env.py                 # headless; writes out/check/*.png
    python scripts/check_env.py --watch         # also opens the emulator window

What to look for in the printout:
  * my_x grows while "forward" runs when you are on the left (and shrinks on the right).
  * my_y changes during "jump" and returns afterwards (airborne=1 in between).
  * lp / hp / hadouken frames show the move in out/check/.
If x/y never move, the position addresses in sf2/ram.py are wrong for your ROM revision: fix them before
collecting data, because the teacher and the text note both depend on them.
"""
import argparse
import os

import _path  # noqa: F401
from sf2 import actions as A
from sf2.config import DEFAULT_STATE
from sf2.dataset import save_png
from sf2.env import FightEnv


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--state", default=DEFAULT_STATE)
    ap.add_argument("--watch", action="store_true")
    ap.add_argument("--out", default="out/check")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    env = FightEnv(args.state, render=args.watch)
    obs = env.reset()
    print("buttons:", env.buttons)
    print("frame shape:", obs.shape, " start:", env.f, " text:", env.text())
    for _ in range(60):  # let the round intro play
        env.step_frame([])
    save_png(env.frame, os.path.join(args.out, "00_start.png"))

    for i, a in enumerate(A.ACTIONS, 1):
        before = env.f
        xs, ys = [], []
        res = env.act(a, on_frame=lambda e: (xs.append(e.f.my_x), ys.append(e.f.my_y)))
        for _ in range(24):  # let the move play out, idle
            env.step_frame([])
            xs.append(env.f.my_x)
            ys.append(env.f.my_y)
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
