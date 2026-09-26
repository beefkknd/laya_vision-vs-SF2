"""Record a RAM trace from the real ROM for the harness tests (tests/trace_mesen.py replays it).

    python scripts/record_trace.py --headless --plan walk --out tests/fixtures/walk.jsonl.gz \
        --savestate states/chunli_vs_dhalsim.state --me chunli --opp dhalsim

Every frame stores the buttons held and the raw bytes of a few WRAM windows (both fighter structs and the
low page), so a trace checks any RAM map whose addresses fall inside them. Plans that walk choose the physical
direction from the fighters' world x (0x0D18 / 0x0F18), never from the RAM map under test.

Plans:
  walk       idle 60 frames, walk left 80 decisions, right 150, left 60 (crosses the opponent once)
  facing     toward x8, away x8, idle until first hit, idle 90 frames, toward x8, away x8
  start      hold toward for 150 frames straight after the savestate loads
  ko_round2  idle until the round ends and the bars refill, then hold toward 400 frames (unchecked)
  knockdown  stand, crouch, one jump, walk in, jump in place until hit in the air, 240 more frames
  walls      walk left until x stops (the left wall), idle 30, walk right through Dhalsim until x stops, idle 30,
             walk left 300 decisions (Dhalsim jumps over her; she backs into the left wall), idle 60
  fireball   seeded random back/idle/block/jump/forward until two of Dhalsim's Yoga Fires have come and gone,
             then 60 idle frames. From the Chun-Li savestate, --seed 1 gets them in round 2
  timeover   a random-policy round until it ends, then 600 idle frames; inputs are not checked on replay.
             From the Chun-Li savestate, --seed 4 --jitter 30 runs out the clock (33 vs 12)
"""
import argparse
import os
import random
import subprocess

import sys

import _path  # noqa: F401
from sf2.actions import ACTIONS
from sf2.cli import add_env_args, make_env
from sf2.mesen import Obs
from sf2.ram import Var, load_map

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tests"))
from trace_mesen import save  # noqa: E402

# low page, fighters (0x0D00 / 0x0F00), timer, fighter action state (0x0C00 / 0x0E00), projectile slot (0x1050)
WINDOWS = [[0x0000, 0x200], [0x0D00, 0x80], [0x0F00, 0x80], [0x1AC0, 0x10], [0x0C00, 0x80], [0x0E00, 0x80],
           [0x1000, 0x80]]
MY_WX, OPP_WX = 0x0D18, 0x0F18
DECISION = 4


class Recorder:
    """Wraps a MesenBridge: adds one 1-byte var per window byte and keeps every frame's inputs + bytes."""

    def __init__(self, inner, check_inputs=True):
        self.inner, self.check_inputs, self.rows, self.n = inner, check_inputs, [], 0
        self.extra = [Var("w%04X" % a, a, 1, False) for s, n in WINDOWS for a in range(s, s + n)]

    def __getattr__(self, name):
        return getattr(self.inner, name)

    def set_vars(self, specs):
        self.n = len(specs)
        self.inner.set_vars(list(specs) + self.extra)

    def _mem(self, values):
        b, out, i = values[self.n:], [], 0
        for _, n in WINDOWS:
            out.append(bytes(b[i:i + n]).hex())
            i += n
        return out

    def _cut(self, obs):
        return Obs([r[:self.n] for r in obs.rams], obs.inputs, obs.images, obs.state)

    def load_state(self, data):
        obs = self.inner.load_state(data)
        self.rows = [{"in": [], "mem": self._mem(obs.rams[0])}]
        return self._cut(obs)

    def run(self, frames, caps=()):
        obs = self.inner.run(frames, caps)
        for f, values in zip(frames, obs.rams[1:]):
            self.rows.append({"in": sorted(f) if self.check_inputs else None, "mem": self._mem(values)})
        return self._cut(obs)

    def world_x(self):
        mem = bytes.fromhex(self.rows[-1]["mem"][1]), bytes.fromhex(self.rows[-1]["mem"][2])
        return int.from_bytes(mem[0][0x18:0x1A], "little"), int.from_bytes(mem[1][0x18:0x1A], "little")


def toward(rec):
    me, opp = rec.world_x()
    return "right" if opp > me else "left"


def away(rec):
    return "left" if toward(rec) == "right" else "right"


def hold(env, direction_fn, decisions):
    for _ in range(decisions):
        env.run_frames([[direction_fn()]] * DECISION)


def plan_walk(env, rec):
    env.run_frames([[]] * 60)
    for d, n in (("left", 80), ("right", 150), ("left", 60)):
        hold(env, lambda: d, n)


def plan_facing(env, rec):
    hold(env, lambda: toward(rec), 8)
    hold(env, lambda: away(rec), 8)
    for _ in range(400):
        if env.f.my_hp < env.full_hp:
            break
        env.run_frames([[]] * DECISION)
    env.run_frames([[]] * 90)
    hold(env, lambda: toward(rec), 8)
    hold(env, lambda: away(rec), 8)


def plan_start(env, rec):
    for _ in range(150):
        env.run_frames([[toward(rec)]])


def plan_ko_round2(env, rec):
    while not env.act("idle").round_over:
        pass
    while not env.f.my_hp == env.f.opp_hp == env.full_hp:
        env.run_frames([[]], capture=False)
    rec.check_inputs = False  # replay idles through the round intro here; the ROM ignores input until "FIGHT!"
    for _ in range(400):
        env.run_frames([[toward(rec)]], capture=False)


def plan_knockdown(env, rec):
    env.run_frames([[]] * 30)                              # standing
    env.run_frames([["down"]] * 32)                        # crouching
    env.run_frames([["up"]] * 4 + [[]] * 60)               # one jump, landed
    while env.f.dx > 90:                                   # walk in
        env.run_frames([[toward(rec)]] * DECISION)
    for _ in range(40):                                    # jump in place until knocked out of the air
        hp = env.f.my_hp
        env.run_frames([["up"]] * 4 + [[]] * 56)
        if env.f.my_hp < hp and env.f.my_y != env.ground[0]:
            break
    env.run_frames([[]] * 240)                             # falling, down, getting up


def _to_wall(env, rec, direction, max_decisions):
    """Hold ``direction`` until world x has not changed for 32 frames of walking (state 00: not blocking or hit)."""
    xs = []
    for _ in range(max_decisions):
        env.run_frames([[direction]] * DECISION)
        walking = bytes.fromhex(rec.rows[-1]["mem"][4])[3] == 0
        xs = xs + [rec.world_x()[0]] if walking else []
        if len(xs) >= 8 and len(set(xs[-8:])) == 1:
            break


def plan_walls(env, rec):
    _to_wall(env, rec, "left", 200)                        # back to the left wall, Dhalsim in front
    env.run_frames([[]] * 30)
    _to_wall(env, rec, "right", 400)                       # through Dhalsim to the right wall
    env.run_frames([[]] * 30)
    hold(env, lambda: "left", 300)                         # toward him; from the ROM he jumps over her and
    env.run_frames([[]] * 60)                              # she ends up backed into the left wall


def plan_fireball(env, rec, seed):
    rng = random.Random(seed)
    flying = seen = 0
    for _ in range(3000):
        if env.act(rng.choice(["back", "back", "idle", "block", "jump", "forward"])).round_over:
            if not env.next_round():
                break
        on = bytes.fromhex(rec.rows[-1]["mem"][6])[0x50]      # 0x1050: projectile slot in use
        seen += flying and not on
        flying = on
        if seen == 2:
            break
    env.run_frames([[]] * 60)


def plan_timeover(env, rec, seed):
    rng = random.Random(seed)
    while not env.act(rng.choice(ACTIONS)).round_over:
        pass
    env.run_frames([[]] * 600)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_env_args(ap)
    ap.add_argument("--plan", required=True, choices=["walk", "facing", "start", "ko_round2", "knockdown", "walls", "fireball", "timeover"])
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    if args.plan != "timeover":
        args.jitter = 0

    env = make_env(args, verified=False)
    rec = Recorder(env.backend, check_inputs=args.plan != "timeover")
    env.backend = rec
    rec.set_vars(load_map(args.ram_map))
    env.reset()
    if args.plan in ("timeover", "fireball"):
        globals()["plan_" + args.plan](env, rec, args.seed)
    else:
        globals()["plan_" + args.plan](env, rec)
    rom = env.backend.inner.rom_sha1
    env.close()

    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    header = {"windows": WINDOWS, "rom_sha1": rom, "savestate": args.savestate, "plan": args.plan,
              "seed": args.seed, "jitter": args.jitter, "commit": commit, "frames": len(rec.rows) - 1}
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    save(args.out, {"header": header, "rows": rec.rows})
    print("wrote %s: %d frames" % (args.out, len(rec.rows) - 1))


if __name__ == "__main__":
    main()
