"""Day 1-2: play SF2 yourself in a window and log every frame's pad state (rung 1 of the teacher ladder).
Also the way to make your own savestates (Ryu vs Ken, a training-mode setup, ...).

    python scripts/record_human.py --session s1                 # play; Esc quits; writes human/s1/
    python scripts/label_human.py --session human/s1 --name human_s1   # log -> dataset rows
    python scripts/record_human.py --no-log --save-state-to states/ryu_vs_ken.state   # F5 saves the state

Keyboard: arrows move; A S D = light / medium / fierce punch; Z X C = short / forward / roundhouse kick;
Enter = start; F5 = save savestate (with --save-state-to); Esc = quit.
Gamepad (Xbox layout): stick/d-pad move; X Y RB = punches; A B LB = kicks; Start = start.

Play clean: 10-20 minutes of the things you want copied (fireball at mid range, anti-air shoryuken, block
jump-ins). Every 4th frame's screen is saved; the labeler turns your inputs into the 12 options.
"""
import argparse
import gzip
import json
import os

import _path  # noqa: F401
from sf2.config import DEFAULT_STATE, FPS, HOLD, PAD
from sf2.dataset import save_png
from sf2.env import FightEnv


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--state", default=DEFAULT_STATE)
    ap.add_argument("--session", default="s1")
    ap.add_argument("--out", default="human")
    ap.add_argument("--scale", type=int, default=3)
    ap.add_argument("--no-log", action="store_true")
    ap.add_argument("--save-state-to", default=None)
    args = ap.parse_args()

    import pygame

    keys = {pygame.K_UP: "UP", pygame.K_DOWN: "DOWN", pygame.K_LEFT: "LEFT", pygame.K_RIGHT: "RIGHT",
            pygame.K_a: PAD["lp"], pygame.K_s: PAD["mp"], pygame.K_d: PAD["hp"],
            pygame.K_z: PAD["lk"], pygame.K_x: PAD["mk"], pygame.K_c: PAD["hk"], pygame.K_RETURN: "START"}
    pad_buttons = {2: PAD["lp"], 3: PAD["mp"], 5: PAD["hp"], 0: PAD["lk"], 1: PAD["mk"], 4: PAD["hk"], 7: "START"}

    env = FightEnv(args.state)
    env.reset()
    pygame.init()
    h, w = env.frame.shape[:2]
    screen = pygame.display.set_mode((w * args.scale, h * args.scale))
    pygame.display.set_caption("SF2 - record (Esc quits)")
    joy = None
    if pygame.joystick.get_count():
        joy = pygame.joystick.Joystick(0)
        joy.init()
    clock = pygame.time.Clock()

    d = os.path.join(args.out, args.session)
    log = None
    if not args.no_log:
        os.makedirs(os.path.join(d, "images"), exist_ok=True)
        log = open(os.path.join(d, "log.jsonl"), "a")

    running = True
    while running and not env.done:
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT or (ev.type == pygame.KEYDOWN and ev.key == pygame.K_ESCAPE):
                running = False
            if ev.type == pygame.KEYDOWN and ev.key == pygame.K_F5 and args.save_state_to:
                os.makedirs(os.path.dirname(args.save_state_to) or ".", exist_ok=True)
                with gzip.open(args.save_state_to, "wb") as fh:
                    fh.write(env.env.em.get_state())
                print("saved state ->", args.save_state_to)
        pressed = pygame.key.get_pressed()
        names = {n for k, n in keys.items() if pressed[k]}
        if joy is not None:
            x, y = joy.get_axis(0), joy.get_axis(1)
            hx, hy = joy.get_hat(0) if joy.get_numhats() else (0, 0)
            if x < -0.5 or hx < 0:
                names.add("LEFT")
            if x > 0.5 or hx > 0:
                names.add("RIGHT")
            if y < -0.5 or hy > 0:
                names.add("UP")
            if y > 0.5 or hy < 0:
                names.add("DOWN")
            names |= {n for b, n in pad_buttons.items() if b < joy.get_numbuttons() and joy.get_button(b)}

        f = env.f
        if log is not None:
            my_air, opp_air = env.airborne()
            rec = {"frame": env.frame_no, "names": sorted(names), "facing_right": f.facing_right,
                   "my_hp": f.my_hp, "opp_hp": f.opp_hp, "my_x": f.my_x, "opp_x": f.opp_x, "my_y": f.my_y,
                   "opp_y": f.opp_y, "my_air": my_air, "opp_air": opp_air}
            if env.frame_no % HOLD == 0:  # the screen the player saw when choosing this frame's input
                rec["image"] = "images/f%07d.png" % env.frame_no
                save_png(env.frame, os.path.join(d, rec["image"]))
            log.write(json.dumps(rec) + "\n")
        env.step_frame(sorted(names))
        surf = pygame.surfarray.make_surface(env.frame.swapaxes(0, 1))
        screen.blit(pygame.transform.scale(surf, screen.get_size()), (0, 0))
        pygame.display.flip()
        clock.tick(FPS)
    if log is not None:
        log.close()
        print("log ->", d)
    env.close()
    pygame.quit()


if __name__ == "__main__":
    main()
