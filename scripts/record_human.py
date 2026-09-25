"""Play SF2 yourself in Mesen while Python logs every frame (rung 1 of the teacher ladder). Also where
savestates come from: press F9 in Mesen at the start of a fight.

    python scripts/record_human.py --session s1                       # play; Ctrl-C here to stop
    python scripts/label_human.py --session human/s1 --name human_s1  # log -> dataset rows
    python scripts/record_human.py --no-log --save-state-to states/ryu_vs_ken.state   # F9 = save

Use Mesen normally (your own pad / keyboard mapping, normal speed). Pick characters and stage, and press F9 in
Mesen (or Enter in this terminal) at "FIGHT!" to write the savestate every other script starts from. Every frame's buttons and RAM variables go to
human/<session>/log.jsonl, every 4th frame's screen to images/.
Play clean: 10-20 minutes of what you want copied (fireball at mid range, anti-air shoryuken, block jump-ins).
"""
import argparse
import json
import os
import threading

import _path  # noqa: F401
from sf2.cli import add_env_args, bridge
from sf2.config import HOLD
from sf2.dataset import save_png
from sf2.ram import load_map

CHUNK = 60


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_env_args(ap, savestate=False)
    ap.add_argument("--session", default="s1")
    ap.add_argument("--out", default="human")
    ap.add_argument("--no-log", action="store_true")
    ap.add_argument("--save-state-to", default="states/ryu_vs_ken.state")
    args = ap.parse_args()

    ram_map = load_map(args.ram_map) if os.path.exists(args.ram_map) else []
    if not ram_map and not args.no_log:
        ap.error("no RAM map at %s: run scripts/find_ram.py first, or use --no-log to just make a savestate"
                 % args.ram_map)
    b = bridge(args)
    b.set_vars(ram_map)
    names = [v.name for v in ram_map]
    d = os.path.join(args.out, args.session)
    log = None
    if not args.no_log:
        os.makedirs(os.path.join(d, "images"), exist_ok=True)
        log = open(os.path.join(d, "log.jsonl"), "a")
    frame, saves = 0, 0
    enter = threading.Event()

    def stdin_loop():
        while True:
            input()
            enter.set()

    threading.Thread(target=stdin_loop, daemon=True).start()
    print("recording - play in Mesen; F9 in Mesen (or Enter here) saves a savestate; Ctrl-C here stops", flush=True)

    def save(data):
        nonlocal saves
        path = args.save_state_to if saves == 0 else args.save_state_to.replace(".state", "_%d.state" % saves)
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "wb") as f:
            f.write(data)
        saves += 1
        print("savestate ->", path, flush=True)

    try:
        while True:
            if enter.is_set():
                enter.clear()
                save(b.save_state())
            obs = b.watch(CHUNK, HOLD)
            if obs.state:
                save(obs.state)
            if log is None:
                frame += len(obs.inputs)
                continue
            for i, pressed in enumerate(obs.inputs):
                rec = {"frame": frame, "names": pressed, "ram": dict(zip(names, obs.rams[i]))}
                if i in obs.images:  # the screen the player saw when choosing this frame's input
                    rec["image"] = "images/f%07d.png" % frame
                    save_png(obs.images[i], os.path.join(d, rec["image"]))
                log.write(json.dumps(rec) + "\n")
                frame += 1
            log.flush()
    except KeyboardInterrupt:
        pass
    finally:
        if log is not None:
            log.close()
            print("%d frames -> %s" % (frame, d))
        b.close()


if __name__ == "__main__":
    main()
