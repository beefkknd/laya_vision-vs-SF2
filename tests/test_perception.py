"""The U arm's perception labels, questions 1-7 (sf2.data.perception; docs/prereg_u_perception.md).

Rows are built with tests/ram_rows.row (p1 = me, p2 = him) and, where the label must hold on real frames, cut from the
his_moves probe excerpts (tests/fixtures/opp_moves: real RAM, Chun-Li vs Ryu / Ken / Honda). Every label is read at
t = n - LAG: the image at frame n shows the RAM of frame n - 1 (measured, see LAG's docstring)."""
import json
from pathlib import Path

import pytest

from ram_rows import row
from sf2.data import perception as P
from sf2.emu.vs import GROUND_Y, NAMES

TH = {"lag": 1, "throw_max": {"chunli": 47, "all": 40}, "poke_max": {"chunli": 87, "all": 90}, "mid_max": 120,
      "trend_eps": 2, "k": 6, "walls": [53, 459], "corner_d": 14, "near_px": 120}
CHUNLI = 5


def mk(n_rows, f, **over):
    """n_rows rows; f(i) -> (p1 overrides, p2 overrides, extra) for row i."""
    out = []
    for i in range(n_rows):
        p1, p2, extra = f(i)
        out.append(row(dict({"char": CHUNLI}, **p1), p2, **extra))
    return out


def still(n_rows=40, p1=None, p2=None, **extra):
    return mk(n_rows, lambda i: (dict(p1 or {}), dict(p2 or {}), extra))


def at(rows, n=None, th=TH):
    n = len(rows) - 31 if n is None else n
    return P.labels(rows, n, th)


# ---- the record and the moment ----

def test_labels_are_read_one_frame_before_the_decision_row():
    rows = still(p1={"x": 100}, p2={"x": 130})
    n = 9
    rows[n] = row({"x": 100, "char": CHUNLI}, {"x": 300})          # only row n is far: the screen still shows 30 px
    assert at(rows, n)["range"] == "throw"
    assert at(rows, n, dict(TH, lag=0))["range"] == "far"


def test_every_question_is_answered_from_a_plain_window():
    lab = at(still(p1={"x": 100}, p2={"x": 170}))
    assert lab == {"range": "poke", "trend": "steady", "phase": "neutral", "air": "grounded", "projectile": "none",
                   "me_can_act": "free", "him_can_act": "free", "corner": "neither", "my_bar": "full",
                   "his_bar": "full"}
    assert set(lab) == set(P.QUESTIONS)


def test_too_few_rows_give_unknown_not_a_guess():
    rows = still(p1={"x": 100}, p2={"x": 170})
    lab = P.labels(rows[:3], 2, TH)          # no row t - 4 for the trend
    assert lab["trend"] == "unknown" and lab["range"] == "poke"


# ---- 1: range band and trend ----

@pytest.mark.parametrize("gap,band", [(20, "throw"), (47, "throw"), (48, "poke"), (87, "poke"), (88, "mid"),
                                      (119, "mid"), (120, "far"), (300, "far")])
def test_range_bands_use_my_characters_thresholds(gap, band):
    assert at(still(p1={"x": 100}, p2={"x": 100 + gap}))["range"] == band


def test_range_falls_back_to_the_pooled_band_for_a_character_without_one():
    rows = still(p1={"x": 100, "char": 0}, p2={"x": 144})               # Ryu (id 0): all = 40
    for r in rows:
        r["p1_char"] = 0
    assert at(rows)["range"] == "poke"


@pytest.mark.parametrize("d,trend", [(-8, "closing"), (-3, "closing"), (-2, "steady"), (2, "steady"),
                                     (3, "opening"), (9, "opening")])
def test_trend_is_the_gap_change_over_the_four_frames_between_the_images(d, trend):
    n = 20
    t = n - 1
    rows = mk(40, lambda i: ({"x": 100}, {"x": 200 + (d if i >= t else 0)}, {}))
    assert at(rows, n)["trend"] == trend


# ---- 2: his phase ----

def attack_rows(start, dur, contact_at=None, n_rows=60):
    """He attacks on frames [start, start + dur); I am hit (hit stun) from contact_at for 12 frames."""
    def f(i):
        p2 = {"x": 160, "state": 0x0A if start <= i < start + dur else 0}
        hit = contact_at is not None and contact_at <= i < contact_at + 12
        p1 = {"x": 100, "state": 0x0E if hit else 0, "react": 0x02 if hit else 0}
        return p1, p2, {}
    return mk(n_rows, f)


def test_an_attack_that_has_not_touched_me_and_lasts_k_more_frames_is_recovering_after_a_miss():
    assert at(attack_rows(5, 30), 16)["phase"] == "recovering after a miss"


def test_an_attack_that_ends_within_k_frames_is_attacking():
    assert at(attack_rows(5, 14), 16)["phase"] == "attacking"          # t = 15, ends at 18 < 15 + 6


def test_an_attack_that_touched_me_earlier_in_the_episode_is_attacking():
    assert at(attack_rows(5, 30, contact_at=10), 20)["phase"] == "attacking"


def test_an_attack_that_touches_me_after_t_plus_k_is_attacking_not_a_miss():
    assert at(attack_rows(5, 30, contact_at=28), 16)["phase"] == "attacking"


def test_a_life_drop_counts_as_contact():
    rows = attack_rows(5, 30)
    for r in rows[25:]:
        r["p1_life"] = 150
    assert at(rows, 16)["phase"] == "attacking"


def test_an_attack_whose_end_is_beyond_the_rows_is_unknown():
    rows = attack_rows(5, 200, n_rows=40)
    assert P.labels(rows, 16, TH)["phase"] == "unknown"


def test_an_attack_whose_start_is_before_the_rows_is_unknown():
    rows = attack_rows(0, 30)
    assert P.labels(rows, 3, TH)["phase"] == "unknown"


def test_contact_already_on_me_before_his_attack_is_not_his_contact():
    rows = attack_rows(10, 30, contact_at=0)          # I reel 0-11 from an earlier hit; his attack starts at 10
    assert at(rows, 20)["phase"] == "recovering after a miss"


@pytest.mark.parametrize("p2,phase", [({"state": 0x08}, "blocking"), ({"state": 0x0E, "react": 0x06}, "blocking"),
                                      ({"state": 0x0E, "react": 0x08}, "blocking"),
                                      ({"state": 0x0E, "react": 0x02}, "being hit"),
                                      ({"state": 0x14}, "being hit"), ({"state": 0x02}, "neutral"),
                                      ({"state": 0x04, "y": 150}, "neutral")])
def test_his_other_phases(p2, phase):
    assert at(still(p1={"x": 100}, p2=dict({"x": 200}, **p2)))["phase"] == phase


# ---- 3: his air state ----

def jump_rows(dx, land_at=None, n_rows=60):
    def f(i):
        air = land_at is None or i < land_at
        return {"x": 100}, {"x": 200 + dx * i, "y": 150 if air else GROUND_Y, "state": 0x04 if air else 0}, {}
    return mk(n_rows, f)


def test_air_grounded_at_me_away_and_landing():
    assert at(still(p1={"x": 100}, p2={"x": 200}))["air"] == "grounded"
    assert at(jump_rows(-2), 20)["air"] == "jumping at me"
    assert at(jump_rows(2), 20)["air"] == "jumping away or straight up"
    assert at(jump_rows(0), 20)["air"] == "jumping away or straight up"
    assert at(jump_rows(-2, land_at=24), 20)["air"] == "landing"       # t = 19, lands at 24 <= 19 + 6
    assert at(jump_rows(-2, land_at=27), 20)["air"] == "jumping at me"


def test_air_direction_is_toward_me_wherever_i_am():
    rows = mk(60, lambda i: ({"x": 400}, {"x": 200 + 2 * i, "y": 150, "state": 4}, {}))
    assert at(rows, 20)["air"] == "jumping at me"


def test_airborne_with_no_landing_seen_before_the_rows_end_is_unknown_only_if_t_plus_k_is_missing():
    rows = jump_rows(-2, n_rows=22)
    assert P.labels(rows, 20, TH)["air"] == "unknown"


# ---- 4: projectile ----

def shot_rows(start_x, v, me_x=100, slot="shot2", n_rows=40):
    return mk(n_rows, lambda i: ({"x": me_x}, {"x": 400}, {slot: 1, slot + "_x": start_x + v * i}))


def test_projectile_far_near_and_none():
    assert at(still(p1={"x": 100}, p2={"x": 400}))["projectile"] == "none"
    assert at(shot_rows(400, -3), 9)["projectile"] == "far"            # t = 8: x 376, 276 px from me
    assert at(shot_rows(230, -3), 9)["projectile"] == "near"           # x 206: 106 px
    assert at(shot_rows(150, 3), 9)["projectile"] == "none"            # flying away from me


def test_my_own_projectile_is_not_his():
    assert at(shot_rows(230, -3, slot="shot1"), 9)["projectile"] == "none"


def test_a_projectile_just_spawned_is_coming_at_me_from_his_side():
    rows = shot_rows(230, -3)
    for r in rows[:8]:
        r["shot2"] = 0
    assert at(rows, 9)["projectile"] == "near"


def test_real_fireball_frames_are_coming_at_me():
    """Ken's real fireball (probe rows, full RAM, Chun-Li vs Ken): once out, every frame is far or near, never none;
    and Chun-Li's own slot (shot1) stays empty: his projectile is shot2."""
    fx = json.loads((Path(__file__).parent / "fixtures" / "perception" / "ken_fireball.json").read_text())
    rows = [dict(zip(fx["names"], v)) for v in fx["rows"]]
    out = [i for i in range(1, len(rows)) if rows[i]["shot2"] and rows[i - 1]["shot2"]]
    seen = [P.labels(rows, i + 1, TH)["projectile"] for i in out]
    assert len(seen) > 20 and all(s in ("far", "near") for s in seen) and "near" in seen
    assert not any(r["shot1"] for r in rows)


# ---- 5: who can act ----

def stun_rows(who, subs, flag_at=None):
    """``who`` (1 or 2) runs through (state, sub, frames) phases, then stands."""
    seq = [(st, sb) for st, sb, k in subs for _ in range(k)]
    def f(i):
        st, sb = seq[i] if i < len(seq) else (0, 0)
        me = {"state": st, "sub": sb, "x": 100, "dizzy": int(flag_at is not None and i >= flag_at and st == 0x0E)}
        free = {"x": 100 if who == 2 else 200}
        p = dict(me, x=100 if who == 1 else 200)
        return (p, free, {}) if who == 1 else (free, p, {})
    return mk(len(seq) + 40, f)


KNOCKDOWN = [(0x0E, 0, 1), (0x0E, 2, 40), (0x0E, 4, 10), (0x0E, 6, 16)]


@pytest.mark.parametrize("who,key", [(1, "me_can_act"), (2, "him_can_act")])
def test_can_act_free_stunned_knocked_down_dizzy(who, key):
    reel = stun_rows(who, [(0x0E, 2, 14)])
    assert P.labels(reel, 6, TH)[key] == "stunned"
    assert P.labels(stun_rows(who, [(0x14, 2, 20)] + KNOCKDOWN[2:]), 6, TH)[key] == "knocked down"
    down = stun_rows(who, KNOCKDOWN)
    assert P.labels(down, 6, TH)[key] == "knocked down"               # still in the air, the landing is ahead
    assert P.labels(down, 60, TH)[key] == "knocked down"              # getting up
    dz = stun_rows(who, KNOCKDOWN + [(0x0E, 8, 80)], flag_at=40)
    assert P.labels(dz, 80, TH)[key] == "dizzy"
    assert P.labels(stun_rows(who, [(0, 0, 5)]), 3, TH)[key] == "free"


def test_sub_8_without_the_flag_at_its_start_is_not_dizzy():
    rows = stun_rows(2, [(0x0E, 2, 10), (0x0E, 8, 12)])
    assert P.labels(rows, 15, TH)["him_can_act"] == "stunned"


def test_a_stun_whose_end_is_beyond_the_rows_without_a_landing_is_unknown():
    rows = stun_rows(2, [(0x0E, 2, 200)])[:40]
    assert P.labels(rows, 20, TH)["him_can_act"] == "unknown"


# ---- 6: corner ----

@pytest.mark.parametrize("mx,hx,corner", [(53, 120, "me"), (67, 120, "me"), (68, 120, "neither"),
                                          (300, 459, "him"), (300, 445, "him"), (300, 444, "neither"),
                                          (459, 420, "me")])
def test_corner_is_within_d_of_the_observed_walls(mx, hx, corner):
    assert at(still(p1={"x": mx}, p2={"x": hx}))["corner"] == corner


# ---- 7: bars ----

def test_bars_follow_vocab_bar_on_life():
    lab = at(still(p1={"x": 100, "life": 176}, p2={"x": 200, "life": 40}))
    assert (lab["my_bar"], lab["his_bar"]) == ("full", "low")
    lab = at(still(p1={"x": 100, "life": 255}, p2={"x": 200, "life": 100}))     # a KO wraps life to 255
    assert (lab["my_bar"], lab["his_bar"]) == ("low", "half")


def test_every_answer_is_in_the_locked_answer_set():
    import random
    rng = random.Random(0)
    for _ in range(300):
        rows = mk(45, lambda i: ({"x": rng.randrange(40, 470), "state": rng.choice([0, 2, 4, 0x0A, 0x0E, 0x14]),
                                  "sub": rng.choice([0, 2, 4, 6, 8]), "y": rng.choice([GROUND_Y, 150]),
                                  "react": rng.choice([0, 2, 6, 8]), "dizzy": rng.choice([0, 1]),
                                  "life": rng.randrange(0, 256)},
                                 {"x": rng.randrange(40, 470), "state": rng.choice([0, 2, 4, 0x0A, 0x0E, 0x14]),
                                  "sub": rng.choice([0, 2, 4, 6, 8]), "y": rng.choice([GROUND_Y, 150]),
                                  "react": rng.choice([0, 2, 6, 8]), "life": rng.randrange(0, 256)},
                                 {"shot2": rng.choice([0, 1]), "shot2_x": rng.randrange(0, 500)}))
        lab = P.labels(rows, rng.randrange(0, 45), TH)
        for q, a in lab.items():
            assert a in P.QUESTIONS[q] + (P.UNKNOWN,), (q, a)


def test_contact_seen_decides_attacking_even_when_the_episode_runs_past_the_rows():
    """Smoke game: his 65+ frame attack that hit me ran past the rows; the contact already decides it."""
    rows = attack_rows(5, 200, contact_at=12, n_rows=40)
    assert P.labels(rows, 16, TH)["phase"] == "attacking"


def test_an_attack_ending_before_t_plus_k_is_attacking_even_if_it_started_before_the_rows():
    rows = attack_rows(0, 8)              # t = 3, ends at 7 < 3 + 6
    assert P.labels(rows, 4, TH)["phase"] == "attacking"
