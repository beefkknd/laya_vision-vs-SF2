"""Harness behaviour checked against ROM traces (tests/fixtures, recorded by scripts/record_trace.py).

The env runs with the real RAM map in ram_maps/sf2_snes.txt. The facing trace was recorded pressing the
direction that pointed at the opponent's world position, so replaying it through ``env.act("forward")``
fails on the first frame the env's idea of facing is wrong.
"""
import os

from sf2.env import FightEnv
from trace_mesen import TraceMesen, fixture_map, load

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MAP = fixture_map(os.path.join(ROOT, "ram_maps", "sf2_snes.txt"))


def _env(name):
    t = load(os.path.join(ROOT, "tests", "fixtures", name + ".jsonl.gz"))
    env = FightEnv(TraceMesen(t), MAP, b"", me="chunli", opp="dhalsim")
    env.reset()
    return env, t


def test_walk_across_the_stage_brings_the_fighters_together_and_past_each_other():
    env, t = _env("walk")
    fs = env.run_frames([r["in"] for r in t["rows"][1:]], capture=False)
    assert fs[0].dx > 80                           # the savestate starts about 100 px apart
    assert min(f.dx for f in fs) < 20              # Chun-Li walks through Dhalsim's position
    assert max(abs(b.opp_x - a.opp_x) for a, b in zip(fs, fs[1:])) <= 12   # positions never jump


def test_facing_flips_only_where_the_fighters_cross():
    env, t = _env("walk")
    fs = env.run_frames([r["in"] for r in t["rows"][1:]], capture=False)
    flips = [i for i in range(1, len(fs)) if fs[i].facing_right != fs[i - 1].facing_right]
    assert flips and all(fs[i].dx < 20 for i in flips)


def test_forward_and_back_press_toward_and_away_from_the_opponent():
    env, t = _env("facing")                        # recorded pressing toward / away from the world positions
    n = len(t["rows"]) - 1
    moved = []

    def step(action):
        s0, x0, o0 = env.f.my_state, env.f.my_x, env.f.opp_x
        env.act(action)                            # fails at once if the env presses another direction
        if s0 == env.f.my_state == 0:              # neutral throughout: not hit, blocking or pushed
            moved.append((action, (env.f.my_x - x0) * (1 if o0 > x0 else -1)))

    for a in ["forward"] * 8 + ["back"] * 8:
        step(a)
    while env.frame_no < n - 64 - 90:              # the recorder idled until the first hit, then 90 frames
        env.act("idle")
    env.run_frames([[]] * 90, capture=False)
    for a in ["forward"] * 8 + ["back"] * 8:
        step(a)
    assert env.frame_no == n
    toward = [d for a, d in moved if a == "forward"]
    assert len(toward) >= 8 and all(d > 0 for d in toward)
    assert all(d <= 0 for a, d in moved if a == "back")


def test_ground_level_comes_from_the_savestate_not_the_end_of_the_start_jitter():
    from trace_mesen import make_trace

    rows = [{"my_hp": 176, "opp_hp": 176, "my_x": 200, "opp_x": 384, "my_y": 192, "timer": 0x79, "my_state": 0,
             "fireball": 0, "fireball_x": 0, "result": 0,
             "opp_y": 140 if 1 <= i <= 40 else 192, "opp_state": 4 if 1 <= i <= 40 else 0}
            for i in range(101)]                                               # Dhalsim mid-jump early on
    env = FightEnv(TraceMesen(make_trace(MAP, rows)), MAP, b"", jitter=20)
    env.reset()                                    # the jitter idles while he is in the air
    for _ in range(12):
        env.act("idle")                            # he has landed by now
    assert env.airborne() == (False, False)


def test_time_over_goes_to_the_higher_life_without_counting_the_zeroed_bars_as_damage():
    t = load(os.path.join(ROOT, "tests", "fixtures", "timeover.jsonl.gz"))   # timer ran out at 33 vs 12
    env = FightEnv(TraceMesen(t), MAP, b"", jitter=t["header"]["jitter"])
    env.reset()
    while True:
        res = env.act("idle")                      # inputs are not checked in this trace
        if res.round_over:
            break
    assert res.winner == "me" and env.wins == {"me": 1, "opp": 0}
    assert (res.dmg_for, res.dmg_against) == (0, 0)


def test_time_over_is_judged_when_the_rom_judges_it():
    """The clock shows 00 for a last 30 frames and hits still count in them; then the ROM sets its round result
    (0x1ACF: 1 Chun-Li, 2 Dhalsim, FF draw). ~480 frames later it zeroes both bars."""
    t = load(os.path.join(ROOT, "tests", "fixtures", "timeover.jsonl.gz"))
    timer_zero = next(i for i, r in enumerate(t["rows"]) if r["mem"][3][16:18] == "00")   # 0x1AC8
    judged = next(i for i, r in enumerate(t["rows"]) if r["mem"][3][30:32] != "00")      # 0x1ACF
    assert judged == timer_zero + 30
    env = FightEnv(TraceMesen(t), MAP, b"", jitter=t["header"]["jitter"])
    env.reset()
    while not env.act("idle").round_over:
        pass
    assert judged <= env.backend.t < judged + 4


def test_first_decision_of_the_match_moves():
    env, t = _env("start")
    x0 = env.f.my_x
    env.act("forward")
    assert env.f.my_x != x0


def test_ko_then_the_first_decision_of_round_two_moves():
    env, t = _env("ko_round2")
    while True:
        res = env.act("idle")
        if res.round_over:
            break
    assert res.winner == "opp"
    assert env.next_round() and env.round == 1
    assert (env.f.my_hp, env.f.opp_hp) == (176, 176)
    x0 = env.f.my_x
    env.act("forward")                             # before "FIGHT!" the ROM ignores it
    assert env.f.my_x != x0


def test_round_two_starts_on_its_first_controllable_frame():
    """On the ROM, holding forward from the refill first moves her 183-185 frames after it (6 round ends: KO
    won / lost, time over, rounds 1-2 and 2-3). The env used to idle 22-51 frames into the fight."""
    env, t = _env("ko_round2")
    while not env.act("idle").round_over:
        pass
    ko = env.backend.t
    assert env.next_round()
    life = [(int(r["mem"][1][0x24:0x28], 16), int(r["mem"][2][0x24:0x28], 16)) for r in t["rows"]]   # 0x0D12, 0x0F12
    refill = next(i for i in range(ko, len(life)) if life[i] == (0xB000, 0xB000))     # 176 little-endian
    assert env.backend.t == refill + 182


def test_parallel_workers_and_matches_all_start_differently():
    import argparse
    import sys

    from fake_mesen import MAP as FAKE_MAP, FakeMesen
    from sf2.cli import add_env_args, fight_env

    sys.path.insert(0, os.path.join(ROOT, "scripts"))
    import parallel

    ap = argparse.ArgumentParser()
    add_env_args(ap)
    idles = []
    for i in range(4):                             # parallel.py --workers 4 ... --seed 4242 --matches 12
        env = fight_env(FakeMesen(), FAKE_MAP, b"", ap.parse_args(parallel.worker_seed_args(4242, i)))
        for _ in range(3):
            env.reset()
            idles.append(env.backend.t)            # frames idled before the first decision
    assert len(set(idles)) == 12, sorted(idles)


def test_knocked_into_the_air_is_not_airborne_but_a_jump_is():
    env, t = _env("knockdown")
    state = [int(r["mem"][4][6:8], 16) for r in t["rows"]]        # Chun-Li's action state, 0x0C03
    jump, knocked = [], []
    for i, r in enumerate(t["rows"][1:], 1):
        env.run_frames([r["in"]], capture=False)
        off_ground = env.f.my_y < 185
        if off_ground and state[i] == 0x04:
            jump.append(env.airborne()[0])
        if off_ground and state[i] == 0x0E:
            knocked.append(env.airborne()[0])
    assert len(jump) > 100 and all(jump)
    assert len(knocked) > 50 and not any(knocked)


def test_thrown_is_not_airborne():
    """Dhalsim lifts her to y 136 (state 00) and throws her across the screen (state 14): not a jump."""
    env, t = _env("close")
    thrown = []
    for r in t["rows"][1:]:
        env.run_frames([r["in"]], capture=False)
        if env.f.my_state == 0x14 or (env.f.my_state == 0 and env.f.my_y == 136):
            thrown.append(env.airborne()[0])
    assert len(thrown) > 80 and not any(thrown)


def test_walking_back_stops_at_the_measured_walls():
    from sf2.ram import LEFT_WALL, RIGHT_WALL

    env, t = _env("walls")
    fs = env.run_frames([r["in"] for r in t["rows"][1:]], capture=False)
    assert min(f.my_x for f in fs) == LEFT_WALL and max(f.my_x for f in fs) == RIGHT_WALL
    # the walls hold whoever is on the other side: she is pinned there facing both ways in this trace
    pinned = {(f.my_x, f.facing_right) for f in fs if f.my_x in (LEFT_WALL, RIGHT_WALL)}
    assert pinned == {(LEFT_WALL, True), (RIGHT_WALL, False)}


def test_cornered_on_both_walls_and_not_in_mid_stage():
    env, t = _env("walls")
    fs = env.run_frames([r["in"] for r in t["rows"][1:]], capture=False)
    at_wall = [f for f in fs if f.my_x in (53, 459)]
    mid = [f for f in fs if 150 <= f.my_x <= 360]
    assert len(at_wall) > 500 and all(f.my_cornered for f in at_wall)
    assert len(mid) > 300 and not any(f.my_cornered for f in mid)
    assert not any(f.opp_cornered for f in fs)                    # Dhalsim stays between 148 and 377 here


def test_dhalsim_cornered_at_the_left_wall():
    env, t = _env("knockdown")                                     # he backs to the left wall late in this trace
    fs = env.run_frames([r["in"] for r in t["rows"][1:]], capture=False)
    near = [f for f in fs if f.opp_x <= 60 and f.opp_x < f.my_x]   # his back to the wall, she is in front
    assert len(near) > 10 and all(f.opp_cornered and not f.my_cornered for f in near)


def _notes(name):
    """(Fighters, note fields, note words, airborne, recorded state bytes) after every frame of a trace; a state
    byte is (0E, reaction) in 0E and (04, sub-state) in 04."""
    env, t = _env(name)
    out = []
    for r in t["rows"][1:]:
        env.run_frames([r["in"] or []], capture=False)
        words = env.text().split()
        out.append((env.f, dict(kv.split("=") for kv in words if "=" in kv), words, env.airborne(),
                    tuple((b[3], b[0x4A]) if b[3] == 0x0E else (b[3], b[4]) if b[3] == 0x04 else b[3]
                          for b in (bytes.fromhex(r["mem"][4]),
                                                                              bytes.fromhex(r["mem"][5])))))
    return out


WORD = {0x00: "stand", 0x02: "crouch", 0x04: "stand", 0x08: "block", 0x0A: "attack", 0x0E: "hit"}


def test_note_corner_and_facing_on_the_walls_trace():
    notes = _notes("walls")
    at_left = [n for f, n, *_ in notes if f.my_x == 53]
    at_right = [n for f, n, *_ in notes if f.my_x == 459]
    assert at_left and all(n["corner"] == "me" and n["facing"] == "right" for n in at_left)
    assert at_right and all(n["corner"] == "me" and n["facing"] == "left" for n in at_right)
    assert all(n["corner"] == "none" for f, n, *_ in notes if 150 <= f.my_x <= 360)
    facing = [n["facing"] for _, n, *_ in notes]
    flips = [i for i in range(1, len(notes)) if facing[i] != facing[i - 1]]
    assert len(flips) >= 2 and all(notes[i][0].dx < 25 for i in flips)       # only where they cross
    assert all(n["facing"] == ("right" if f.my_x <= f.opp_x else "left") for f, n, *_ in notes)


def test_note_state_words_match_the_recorded_state_bytes():
    """An attack from the air is 0A for Dhalsim, and 04 with sub-state 06 for Chun-Li (checked on the ROM)."""
    seen, her_air_attacks = set(), 0
    for name in ("walls", "knockdown", "timeover", "ko_round2"):
        for f, n, words, airs, states in _notes(name):
            for word, air, state in zip((words[1], words[4]), airs, states):
                if isinstance(state, tuple) and state[0] == 0x0E:
                    want = "block" if state[1] in (6, 8) else "hit"
                elif air:
                    want = "jumpattack" if state in (0x0A, (0x04, 0x06)) else "jump"
                else:
                    want = WORD.get(state[0] if isinstance(state, tuple) else state, "other")
                assert word == want, (name, f, words)
                seen.add(word)
            her_air_attacks += airs[0] and states[0] == (0x04, 0x06)
    assert seen == {"stand", "crouch", "jump", "jumpattack", "block", "attack", "hit", "other"}
    assert her_air_attacks > 50


def test_opponent_attacking_matches_the_recorded_state_byte():
    for name in ("walls", "timeover"):
        n = 0
        for f, _, _, _, (_, his) in _notes(name):
            assert f.opp_attacking == (his == 0x0A)
            n += f.opp_attacking
        assert n > 500


def test_note_sees_dhalsim_cornered_and_hit():
    assert any(n["corner"] == "opp" for f, n, *_ in _notes("knockdown") if f.opp_x <= 60 and f.opp_x < f.my_x)
    assert any(words[4] == "hit" for _, _, words, _, _ in _notes("timeover"))


def test_note_clock_runs_early_to_late_in_a_timed_out_round():
    t = load(os.path.join(ROOT, "tests", "fixtures", "timeover.jsonl.gz"))
    env = FightEnv(TraceMesen(t), MAP, b"", me="chunli", opp="dhalsim", jitter=t["header"]["jitter"])
    env.reset()
    seen = []
    while True:
        seen.append((env.f.timer, env.text().split("time=")[1].split()[0]))
        if env.act("idle").round_over:
            break
    words = [w for _, w in seen]
    assert words[0] == "early" and words[-1] == "late" and "mid" in words
    assert words == sorted(words, key=["early", "mid", "late"].index)       # never goes back
    assert all(w == "late" for t, w in seen if t < 0x30) and all(w == "early" for t, w in seen if t >= 0x60)


def test_note_at_the_start_of_round_two():
    env, t = _env("ko_round2")
    while not env.act("idle").round_over:
        pass
    assert env.next_round()
    words = env.text().split()
    n = dict(kv.split("=") for kv in words if "=" in kv)
    assert (words[1], words[4], n["facing"], n["corner"], n["time"]) == ("stand", "stand", "right", "none", "early")
    assert "hp=100" in env.text() and n["last"] == "idle"


def _stuns(t, base):
    """0E episodes of the fighter whose state byte is at ``base`` + 3 (0x0C00 Chun-Li, 0x0E00 Dhalsim), from the
    recorded bytes: (first frame, frame after, blocked). Blocked = entered from guard (08, or 0A on the frame an
    attack came) and cost at most a Yoga Fire's chip (8) of the true life at +0x35; hits cost 10 or more."""
    w = [s for s, _ in t["header"]["windows"]].index(base)
    b = [bytes.fromhex(r["mem"][w]) for r in t["rows"]]
    out, i = [], 1
    while i < len(b):
        if b[i][3] == 0x0E and b[i - 1][3] != 0x0E:
            j = i
            while j < len(b) and b[j][3] == 0x0E:
                j += 1
            out.append((i, j, b[i - 1][3] in (0x08, 0x0A) and b[i - 1][0x35] - min(x[0x35] for x in b[i:j]) <= 8))
            i = j
        i += 1
    return out


def test_block_stun_reads_block_and_hit_stun_reads_hit():
    """0E is hit stun and block stun alike; the note tells them apart (for her and for him)."""
    n = {("me", True): 0, ("me", False): 0, ("opp", True): 0, ("opp", False): 0}
    for name in ("walls", "walk", "facing", "fireball", "timeover", "close", "win", "knockdown", "ko_round2"):
        t = load(os.path.join(ROOT, "tests", "fixtures", name + ".jsonl.gz"))
        notes = _notes(name)
        for who, base, k in (("me", 0x0C00, 1), ("opp", 0x0E00, 4)):
            for i, j, blocked in _stuns(t, base):
                words = {notes[f - 1][2][k] for f in range(i, j)}
                assert words == {"block"} if blocked else "hit" in words <= {"hit", "dizzy"}, (name, who, i, words)
                n[who, blocked] += 1
    assert n[("me", True)] >= 5 and n[("opp", True)] >= 15 and min(n.values()) >= 5, n


def test_block_stun_is_controllable_hit_stun_is_not():
    """On the ROM holding down in block stun switches her to a crouching guard (tests/test_rom_harness.py); in hit
    stun no input changes anything."""
    for name in ("walls", "walk", "facing", "timeover"):
        env, t = _env(name)
        stuns = _stuns(t, 0x0C00)
        ctl = []
        for r in t["rows"][1:]:
            env.run_frames([r["in"] or []], capture=False)
            ctl.append(env.controllable())
        for i, j, blocked in stuns:
            assert all(ctl[f - 1] == blocked for f in range(i, j)), (name, i, blocked)
        assert any(b for *_, b in stuns) and not all(b for *_, b in stuns)


def test_knocked_down_is_not_controllable():
    env, t = _env("knockdown")
    hit = []
    for r in t["rows"][1:]:
        env.run_frames([r["in"]], capture=False)
        assert env.controllable() == (env.f.my_state != 0x0E)
        hit.append(not env.controllable())
    assert sum(hit) > 200


def test_held_thrown_and_end_of_round_poses_are_not_controllable_but_dizzy_is():
    """Checked on the ROM (tests/test_rom_harness.py): no input changes anything while Dhalsim holds her up for a
    throw (state 00 at y 136) or throws her (14), or in the winner's / loser's poses (10 / 12); mashing shortens
    a dizzy (0E, sub-state 08, flag +0x89), so a dizzy is controllable and the note says so."""
    seen = {"held": 0, "thrown": 0, "pose": 0, "dizzy": 0, "down": 0}
    for name in ("close", "win", "timeover", "ko_round2", "knockdown"):
        env, t = _env(name)
        for r in t["rows"][1:]:
            env.run_frames([r["in"] or []], capture=False)
            b = bytes.fromhex(r["mem"][4])                      # 0x0C00: state +03, sub-state +04, dizzy +89
            f = env.f
            kind = ("held" if b[3] == 0 and f.my_y == 136 else "thrown" if b[3] == 0x14 else
                    "pose" if b[3] in (0x10, 0x12) else "dizzy" if (b[3], b[4], b[0x89]) == (0x0E, 8, 1) else
                    "down" if b[3] == 0x0E and b[4] in (4, 6) else None)
            if kind:
                seen[kind] += 1
                assert env.controllable() == (kind == "dizzy"), (name, kind, f)
                assert (env.text().split()[1] == "dizzy") == (kind == "dizzy")
    assert seen["held"] > 30 and seen["thrown"] > 80 and seen["pose"] > 100 and seen["dizzy"] > 100, seen


def test_uncontrollable_decisions_stay_in_the_rollout_but_not_in_the_dataset(tmp_path):
    from sf2.dataset import Writer, read
    from sf2.loop import play
    from sf2.rollout import gate

    t = load(os.path.join(ROOT, "tests", "fixtures", "knockdown.jsonl.gz"))
    for r in t["rows"]:
        r["in"] = None                             # replay the ROM's frames under idle decisions

    def run(writer):
        env = FightEnv(TraceMesen(t), MAP, b"", me="chunli", opp="dhalsim")
        return play(env, lambda *a: ("idle", {"actor": "idle"}), 1, writer=writer, max_decisions=500, log_every=0)

    rows, rounds = run(None)
    w = Writer(str(tmp_path), "ctx", source="ctx", val_every=0, skip_uncontrollable=True)
    rows_w, rounds_w = run(w)
    w.close()
    off = [r for r in rows if r["meta"]["controllable"] is False]
    assert len(rows) == 500 and len(off) > 40
    assert gate(rows_w, rounds_w) == gate(rows, rounds)             # the gate still counts every decision
    written = read(str(tmp_path / "ctx" / "train.jsonl"))
    assert len(written) == 500 - len(off) and all(r["meta"]["controllable"] for r in written)
    assert len(os.listdir(tmp_path / "ctx" / "images")) <= 2 * len(written)


def _flights(fs):
    """Stretches of consecutive frames with a projectile on screen."""
    out, cur = [], []
    for f in fs:
        if f.fireball:
            cur.append(f)
        elif cur:
            out.append(cur)
            cur = []
    return out + ([cur] if cur else [])


def test_yoga_fire_flies_from_dhalsim_toward_chun_li():
    for name, n in (("fireball", 2), ("walls", 1)):
        env, t = _env(name)
        fs = env.run_frames([r["in"] for r in t["rows"][1:]], capture=False)
        flights = _flights(fs)
        assert len(flights) == n
        for fl in flights:
            toward = 1 if fl[0].my_x > fl[0].opp_x else -1
            assert 30 < (fl[0].fireball_x - fl[0].opp_x) * toward < 70         # spawns just in front of him
            steps = [(b.fireball_x - a.fireball_x) * toward for a, b in zip(fl, fl[1:])]
            assert all(2 <= s <= 4 for s in steps[:-1]) and 0 <= steps[-1] <= 4   # ~3 px/frame toward her;
                                                                                   # stops on the frame it hits
    env, t = _env("fireball")
    fs = env.run_frames([r["in"] for r in t["rows"][1:]], capture=False)
    last = _flights(fs)[-1][-1]
    end = fs.index(last)
    assert fs[end + 1].my_state == 0x0E and fs[end + 1].my_hp < fs[end - 1].my_hp   # the second one hits her


def test_note_shows_the_fireball_by_distance_to_her():
    from sf2.ram import dist_bin

    for name in ("fireball", "walls"):
        for f, n, *_ in _notes(name):
            assert n["fireball"] == (dist_bin(abs(f.fireball_x - f.my_x)) if f.fireball else "none")
    words = {n["fireball"] for _, n, *_ in _notes("fireball")}
    assert words == {"none", "close", "mid"}


def test_dhalsim_reels_in_state_0E_when_her_hits_land_and_blocks_in_08_then_0E():
    """sf2.ram's state table on the close trace: 0E is hit stun and block stun alike; only a hit costs life."""
    t = load(os.path.join(ROOT, "tests", "fixtures", "close.jsonl.gz"))
    opp = [bytes.fromhex(r["mem"][5]) for r in t["rows"]]                   # 0x0E00: state +03, life +35
    hits = [i for i in range(1, len(opp)) if 0 < opp[i][0x35] < opp[i - 1][0x35]]     # 0: cleared between rounds
    assert len(hits) > 10 and all(opp[i][3] == 0x0E for i in hits)
    stuns = [i for i in range(1, len(opp) - 5) if opp[i][3] == 0x0E and opp[i - 1][3] != 0x0E]
    blocked = [i for i in stuns if opp[i - 1][3] == 0x08 and opp[i + 5][0x35] == opp[i - 1][0x35]]
    assert len(blocked) > 10


def _play_close(name, seed):
    """Replay record_trace.py's close plan through the env (the trace checks every input)."""
    import random

    env, t = _env(name)
    rng = random.Random(seed)
    winners, ends, dealt, taken = [], [], [0], [0]
    for _ in range(3000):
        res = env.act("forward" if env.f.dx > 30 else rng.choice(["idle", "idle", "crouch", "lp", "forward"]))
        dealt[-1] += res.dmg_for
        taken[-1] += res.dmg_against
        if res.round_over:
            winners.append(res.winner)
            ends.append(env.backend.t)
            if not env.next_round():
                break
            dealt.append(0)
            taken.append(0)
    rom = [(bytes.fromhex(r["mem"][4])[0xD0], bytes.fromhex(r["mem"][5])[0xD0]) for r in t["rows"]]  # round wins
    won = [i for i in range(1, len(rom)) if rom[i] != rom[i - 1] and rom[i] != (0, 0)]
    return env, winners, ends, rom, won, list(zip(dealt, taken))


def test_rounds_end_when_the_rom_ends_them_and_go_to_the_side_it_gives_them_to():
    """The ROM counts round wins at 0x0CD0 / 0x0ED0 on the KO frame; the life bar only drains to zero up to ~35
    frames later. A KO'd fighter has lost his whole bar."""
    env, winners, ends, rom, won, dmg = _play_close("win", 13)                   # won, lost, won
    assert winners == ["me", "opp", "me"] and env.round == 2
    assert env.wins == {"me": 2, "opp": 1} == dict(zip(("me", "opp"), rom[won[-1]]))
    assert all(0 <= e - w < 4 for e, w in zip(ends, won)) and [d if w == "me" else a for (d, a), w in zip(dmg, winners)] == [176] * 3
    env, winners, ends, rom, won, dmg = _play_close("close", 1)                  # 1-1, then round 3 lost
    assert winners == ["me", "opp", "opp"] and env.round == 2
    assert env.wins == {"me": 1, "opp": 2} == dict(zip(("me", "opp"), rom[won[-1]]))
    assert all(0 <= e - w < 4 for e, w in zip(ends, won)) and [d if w == "me" else a for (d, a), w in zip(dmg, winners)] == [176] * 3


def test_a_match_ends_after_the_fourth_round_even_on_draws():
    """On the ROM (draws forced by writing equal life and a 1 s clock) the 4th round is the "FINAL ROUND"; after
    a draw in it there is no 5th round, the game goes back to the title screen."""
    from trace_mesen import make_trace

    rows = [{"my_hp": 176, "opp_hp": 176, "my_x": 200, "opp_x": 304, "my_y": 192, "opp_y": 192, "timer": 0,
             "my_state": 0, "opp_state": 0, "fireball": 0, "fireball_x": 0,
             "result": 0xFF if 100 <= i % 300 < 110 else 0} for i in range(1600)]     # a draw every 300 frames
    env = FightEnv(TraceMesen(make_trace(MAP, rows)), MAP, b"")
    env.reset()
    results = []
    while True:
        res = env.act("idle")
        if res.round_over:
            results.append(res.winner)
            if not env.next_round():
                break
    assert results == ["draw"] * 4


def test_next_round_never_waits_into_the_next_opponents_fight():
    """On the ROM the bars refill 600-940 frames after a round ends. After a won match the map screen and "VS"
    follow, and the next opponent's bars refill ~1354 frames after the last round end: that is not a new round."""
    from trace_mesen import make_trace

    rows = [{"my_hp": 176 if i < 50 or i >= 1404 else 100, "opp_hp": 176 if i < 40 or i >= 1404 else 0,
             "my_x": 200, "opp_x": 304, "my_y": 192, "opp_y": 192, "timer": 0x50, "my_state": 0, "opp_state": 0,
             "fireball": 0, "fireball_x": 0, "result": 1 if 50 <= i < 1404 else 0} for i in range(3000)]
    env = FightEnv(TraceMesen(make_trace(MAP, rows)), MAP, b"")
    env.reset()
    while not env.act("idle").round_over:
        pass
    assert env.wins == {"me": 1, "opp": 0} and not env.next_round()


def test_a_dizzy_lasts_while_she_stays_in_its_sub_state_after_the_flag_clears():
    """On the ROM (seed 12 of an attack-spamming close plan) the dizzy flag +0x89 cleared 24 frames into a dizzy
    that went on for 250 more frames in 0E / 08, and mashing still shortened it (318 -> 144 frames). The 16-frame
    08 at the end of getting up, never flagged, is not a dizzy."""
    from trace_mesen import make_trace

    base = {"my_hp": 100, "opp_hp": 176, "my_x": 200, "opp_x": 260, "my_y": 192, "opp_y": 192, "timer": 0x80}
    phases = ([(0x00, 0, 0)] * 10 + [(0x0E, 4, 0)] * 20 + [(0x0E, 6, 0)] * 10 + [(0x0E, 8, 0)] * 16   # got up
              + [(0x00, 0, 0)] * 10 + [(0x0E, 4, 1)] * 20 + [(0x0E, 8, 1)] * 24 + [(0x0E, 8, 0)] * 200  # dizzy
              + [(0x00, 0, 0)] * 10)
    rows = [dict(base, my_state=s, my_sub=sub, my_dizzy=flag) for s, sub, flag in phases]
    env = FightEnv(TraceMesen(make_trace(MAP, rows)), MAP, b"")
    env.reset()
    got = []
    for _ in phases[1:]:
        env.run_frames([[]], capture=False)
        got.append((env.text().split()[1], env.controllable()))
    assert got[39:55] == [("hit", False)] * 16                   # the end of getting up
    assert got[84] == ("hit", False) and got[85:309] == [("dizzy", True)] * 224
    assert got[-1] == ("stand", True)


def test_a_hits_damage_lands_on_the_decision_it_hit_in():
    """The life bars (0x0D12 / 0x0F12) drain 1 per frame after a hit, so read per decision they smear a hit over
    4-9 decisions. The true life at +0x35 of 0x0C00 / 0x0E00 drops by the whole hit on the hit frame."""
    import random

    env, t = _env("close")
    true = [tuple(0 if x > 176 else x for x in (bytes.fromhex(r["mem"][4])[0x35], bytes.fromhex(r["mem"][5])[0x35]))
            for r in t["rows"]]                    # a KO blow can wrap it to 255
    drops = lambda k, a, b: sum(max(0, true[i - 1][k] - true[i][k]) for i in range(a + 1, b + 1))  # noqa: E731
    rng = random.Random(1)
    hits = []
    for _ in range(3000):
        f0 = env.backend.t
        res = env.act("forward" if env.f.dx > 30 else rng.choice(["idle", "idle", "crouch", "lp", "forward"]))
        if res.round_over:
            if not env.next_round():
                break
            continue
        assert (res.dmg_against, res.dmg_for) == (drops(0, f0, env.backend.t), drops(1, f0, env.backend.t))
        if res.dmg_against:
            hits.append(res.dmg_against)
    assert len(hits) >= 8 and min(hits) >= 10          # whole hits, not 1-3 life points of a draining bar
