"""Harness behaviour checked against ROM traces (tests/fixtures, recorded by scripts/record_trace.py).

The env runs with the real RAM map in ram_maps/sf2_snes.txt. The facing trace was recorded pressing the
direction that pointed at the opponent's world position, so replaying it through ``env.act("forward")``
fails on the first frame the env's idea of facing is wrong.
"""
import os

from sf2.env import FightEnv
from sf2.ram import load_map
from trace_mesen import TraceMesen, load

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MAP = load_map(os.path.join(ROOT, "ram_maps", "sf2_snes.txt"))


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
             "fireball": 0, "fireball_x": 0,
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


def test_round_ends_when_the_timer_reaches_zero():
    t = load(os.path.join(ROOT, "tests", "fixtures", "timeover.jsonl.gz"))
    timer_zero = next(i for i, r in enumerate(t["rows"]) if r["mem"][3][16:18] == "00")   # 0x1AC8
    env = FightEnv(TraceMesen(t), MAP, b"", jitter=t["header"]["jitter"])
    env.reset()
    while not env.act("idle").round_over:
        pass
    assert timer_zero <= env.backend.t < timer_zero + 4   # not ~480 frames later, when the ROM zeroes the bars


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
    """(Fighters, note fields, note words, airborne, recorded state bytes) after every frame of a trace."""
    env, t = _env(name)
    out = []
    for r in t["rows"][1:]:
        env.run_frames([r["in"] or []], capture=False)
        words = env.text().split()
        out.append((env.f, dict(kv.split("=") for kv in words if "=" in kv), words, env.airborne(),
                    (int(r["mem"][4][6:8], 16), int(r["mem"][5][6:8], 16))))
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
    seen = set()
    for name in ("walls", "knockdown", "timeover", "ko_round2"):
        for f, n, words, airs, states in _notes(name):
            for word, air, state in zip((words[1], words[4]), airs, states):
                want = ("hit" if state == 0x0E else ("jumpattack" if state == 0x0A else "jump") if air
                        else WORD.get(state, "other"))
                assert word == want, (name, f, words)
                seen.add(word)
    assert seen == {"stand", "crouch", "jump", "jumpattack", "block", "attack", "hit", "other"}


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


def test_knocked_down_is_not_controllable():
    env, t = _env("knockdown")
    hit = []
    for r in t["rows"][1:]:
        env.run_frames([r["in"]], capture=False)
        assert env.controllable() == (env.f.my_state != 0x0E)
        hit.append(not env.controllable())
    assert sum(hit) > 200


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
