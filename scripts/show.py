"""One game for the arcade console: laya plays Chun-Li in a Mesen window from power-on, forever.

    python scripts/show.py --model runs/r1/best --port 47801

The cartridge boots from reset: Capcom logo, the intro, the title, GAME START, Chun-Li, and laya takes over on the
first frame of round 1 (sf2/boot.py). Then it is the real arcade game: after a win the game moves on to the next
opponent (bonus stages included), after a loss it continues (START, like a coin) into the rematch; nothing is
reloaded (sf2/boot.next_fight). One line per decision on stdout; scripts/arcade.py runs several of these and
colors them. Nothing is written to data/ or rollouts/. Ctrl-C (or closing the console) stops it and its Mesen.
"""
import argparse
import os
from contextlib import nullcontext

import _path  # noqa: F401
from sf2 import ram
from sf2.boot import CHARACTERS, P2_CHAR, boot, next_fight
from sf2.cli import STAMP, add_env_args, check_harness, fight_env
from sf2.headless import bridge_for_port, find_mesen, launch_argv
from sf2.loop import play
from sf2.mesen import MesenBridge
from sf2.policy import LayaPolicy
from sf2.ram import load_map

SHORT = {"idle": "idle", "forward": "fwd", "back": "back", "jump": "jump", "jump_forward": "jump-fwd",
         "crouch": "crouch", "lp": "LP", "hp": "HP", "lk": "LK", "hk": "HK", "block": "block", "throw": "throw",
         "sweep": "sweep", "lightning_legs": "legs"}


def opponent(env) -> int:
    return env.backend.dump_wram()[P2_CHAR]  # player 2's character id, once per fight


def clock(timer) -> str:
    return "--" if timer is None else "%02d" % ((timer >> 4) * 10 + (timer & 0xF))  # BCD seconds


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="runs/r1/best")
    ap.add_argument("--matches", type=int, default=0, help="stop after this many fights (0 = forever)")
    add_env_args(ap)
    ap.set_defaults(me="chunli", opp="dhalsim")  # arcade.py gives each game its own --jitter-base: different fights
    args = ap.parse_args()
    say = lambda s: print(s, flush=True)  # noqa: E731

    say("loading %s" % args.model)
    pol = LayaPolicy(args.model)
    ram_map = load_map(args.ram_map)
    # a windowed Mesen with this port's copy of the bridge script; it closes when we exit
    launch = launch_argv(args.port, args.rom, args.mesen) if args.headless else \
        [find_mesen(args.mesen), os.path.expanduser(args.rom), bridge_for_port(args.port)]
    b = MesenBridge(args.port, launch=launch)
    try:
        check_harness(b.rom_sha1, args.ram_map, STAMP)
        say("power on: intro, title, GAME START, Chun-Li")
        env = fight_env(b, ram_map, boot(b), args)
        run(env, pol, ram_map, args, say)
    finally:
        b.close()


def run(env, pol, ram_map, args, say):
    fights = {"n": 0, "won": 0}
    reset = env.reset

    def next_match():
        if env.episode >= 0:
            say("-- fight %d over (%d won so far); the game moves on --" % (fights["n"], fights["won"]))
            env.savestate = next_fight(env.backend, ram_map=ram_map)  # the current frame: loading it changes nothing
        img = reset()
        fights["n"] += 1
        env.opp = CHARACTERS.get(opponent(env), env.opp)  # the note names the real opponent
        say("== FIGHT %d vs %s: round 1 ==" % (fights["n"], env.opp.upper()))
        return img

    env.reset = next_match
    act = env.act

    def act_and_report(action, on_frame=None):
        res = act(action, on_frame)
        if res.dmg_for:  # booked on the decision the hit lands in, often a move or two after the attack
            say("      >> hit lands: -%d to him" % res.dmg_for)
        if res.dmg_against:
            say("      << she takes %d" % res.dmg_against)
        if res.round_over:
            who = {"me": "CHUN-LI WINS", "opp": "chun-li loses", "draw": "draw"}.get(res.winner, res.winner)
            say("== round %d: %s (wins %d-%d) ==" % (env.round + 1, who, env.wins["me"], env.wins["opp"]))
            if env.wins["me"] >= 2:
                fights["won"] += 1
        return res

    env.act = act_and_report

    def choose(env, prev, cur, text):
        keep_alive = getattr(env.backend, "keep_alive", None)
        with keep_alive() if keep_alive else nullcontext():
            a, probs = pol.act(prev, cur, text)
        f = env.f
        say("f%d r%d %ss  hp %3d:%-3d  %-8s %3.0f%%"
            % (fights["n"], env.round + 1, clock(f.timer), max(0, f.my_hp), max(0, f.opp_hp), SHORT[a],
               100 * probs[a]))
        return a, {}

    while not args.matches or fights["n"] < args.matches:
        play(env, choose, 1, log_every=0)  # one fight at a time, so a night of play does not pile up rows


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
