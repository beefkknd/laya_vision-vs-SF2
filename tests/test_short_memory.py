"""The SIMPLE live early-game policy (sf2/system2/short_memory.py): while losing, change one line; while
winning, freeze. Two states (trying/kept), one counter (rounds-in-play), one threshold (SWAP_AFTER=2).

These are the RED-first tests for the deadlock the old timer tangle caused (playbooks/chun/round_03_ryu:
12 straight losses, every Coach claim refused 'already 2 claims in test', zero change). The pure policy
has no since/idx coordinate and no test-slot cap, so that class of bug cannot recur -- proven here.
"""
from sf2.system1.advice import followable as _real_followable
from sf2.system2 import short_memory as SM


def _chunli_fol(move, rng):
    return _real_followable(move, rng, "chunli")


def _trying_at(move, rounds, range_=None, when="standing"):
    return dict(SM._trying_entry(_claim(move, range_=range_, when=when)), rounds=rounds)


# --- a deterministic injected scorer so graduation/weakest tests don't need 20-row CI fixtures -------
def _scorer(verdict_by_move):
    """A fake condition scorer: move -> {cls, tries, diff}. Mirrors lessons.condition_evidence's shape."""
    def score(rows, claim):
        return verdict_by_move.get(claim["move"], {"cls": "few", "tries": 0, "diff": 0.0})
    return score


def _claim(move, kind="use_more", range_=None, when="standing"):
    return {"kind": kind, "move": move, "range": range_, "when": when, "view": None}


def _trying(move, rounds, **ev):
    e = SM._trying_entry(_claim(move))
    return dict(e, rounds=rounds, evidence=ev)


def _kept(move):
    e = SM._trying_entry(_claim(move))
    return dict(e, state="kept", rounds=9)


def _wl(seq):
    """'LLW' -> per-round [{won,lost}] the loop records (lost round = lost>won)."""
    return [{"won": 0, "lost": 1} if c == "L" else {"won": 1, "lost": 0} for c in seq]


def _full(trying=(), kept=()):
    """A FULL short memory (len == SM.MAX_LINES). Pads with distinct filler trying lines so the room/swap
    branch is exercised whatever SM.MAX_LINES is. In-play entries need not be real moves (not validated)."""
    reg = [_kept(m) for m in kept] + [_trying(m, rounds=5) for m in trying]
    i = 0
    while len(reg) < SM.MAX_LINES:
        reg.append(_trying("fill%d" % i, rounds=5))
        i += 1
    assert len(reg) == SM.MAX_LINES
    return reg


# --------------------------------------------------------------------------- the one signal
def test_loss_streak_counts_trailing_lost_rounds():
    assert SM.loss_streak(_wl("WLLL")) == 3
    assert SM.loss_streak(_wl("LLW")) == 0          # a win breaks the streak
    assert SM.loss_streak([]) == 0


# --------------------------------------------------------------------------- growth (room) vs freeze (full)
def test_room_admits_a_coach_claim_even_while_winning():
    # early-game growth: with free slots, a valid Coach claim fills one (win or lose). Not a swap -- nothing dropped.
    reg = [_kept("s.mk"), _trying("s.hk", rounds=9)]                 # 2 in play, room for 3 more
    out, ev = SM.step(reg, _wl("WW"), claims=[_claim("throw_F+hp")], rows=[], moves={"s.mk", "s.hk", "throw_F+hp"},
                      scorer=_scorer({}))
    assert ev["added"] and "throw_F+hp" in ev["added"][0]
    assert ev["removed"] == []                                      # grown, not swapped


def test_full_memory_while_winning_freezes():
    # FULL (MAX_LINES) and winning: nothing changes -- stop churning once the established set is winning.
    reg = _full(kept=["s.mk"])                                      # all slots filled, all immune/aged
    reg = [dict(r, state="kept") for r in reg]                      # make every slot immune so none can be swapped
    out, ev = SM.step(reg, _wl("WW"), claims=[_claim("lightning_legs")], rows=[], moves={"lightning_legs"},
                      scorer=_scorer({}))
    assert ev["added"] == [] and ev["removed"] == []
    assert [r["line"] for r in out] == [r["line"] for r in reg]     # memory unchanged


# --------------------------------------------------------------------------- the deadlock-breaker
def test_losing_streak_forces_one_swap_even_when_slots_are_full():
    # the round_03_ryu shape: the short memory is FULL (MAX_LINES) of aged trying lines, she is losing, the
    # Coach offers a fresh move. Old code refused it ('already 2 claims in test'); the simple policy MUST
    # drop the weakest trying line and bring the fresh one in -- the deadlock cannot recur (no slot cap).
    reg = _full(kept=["s.mk"])                                    # FULL (SM.MAX_LINES), one kept + filler trying
    coach = [_claim("lightning_legs", range_="far")]
    out, ev = SM.step(reg, _wl("LL"), claims=coach, rows=[], moves={"lightning_legs"}, scorer=_scorer({}))
    assert ev["added"] and "lightning_legs" in ev["added"][0], "a fresh line must be admitted while losing"
    assert ev["removed"], "a trying line must be dropped to make room"
    lines = [r["line"] for r in out if r["state"] in SM.IN_PLAY_STATES]
    assert any("lightning_legs" in l for l in lines)              # the new memory really changed
    assert len(lines) <= SM.MAX_LINES                             # never grows past the cap
    assert any("s.mk" in l for l in lines)                        # the kept line survived


def test_fair_chance_window_no_double_swap_right_after_one():
    # FULL memory, losing, but the newest trying line was just admitted (rounds=0): it gets one SWAP_AFTER
    # window of chance before it (or any line) can be swapped again -- no thrashing. Same single threshold.
    reg = _full(trying=["young"])                                   # FULL; make 'young' the just-admitted one
    reg = [dict(r, rounds=0) if r["line"] == "use more young when he stands" else r for r in reg]
    out, ev = SM.step(reg, _wl("LLL"), claims=[_claim("throw_F+hp")], rows=[], moves={"throw_F+hp"},
                      scorer=_scorer({}))
    assert ev["added"] == [] and ev["removed"] == []


def test_blank_start_admits_on_a_losing_streak_with_no_trying_lines():
    out, ev = SM.step([_kept("s.mk")], _wl("LL"), claims=[_claim("throw_F+hp")], rows=[],
                      moves={"s.mk", "throw_F+hp"}, scorer=_scorer({}))
    assert ev["added"] and "throw_F+hp" in ev["added"][0]
    assert ev["removed"] == []                                   # room free (1 line) -> add, not swap


# --------------------------------------------------------------------------- retention (advisory)
def test_trying_line_graduates_to_kept_when_the_scorer_says_clearly_good():
    reg = [_trying("s.hk", rounds=3)]
    sc = _scorer({"s.hk": {"cls": "better", "tries": 25, "diff": 12.0}})
    out, ev = SM.step(reg, _wl("W"), claims=[], rows=[], moves={"s.hk"}, scorer=sc)
    assert out[0]["state"] == "kept", out[0]["why"]


def test_graduation_is_advisory_only_unproven_lines_stay_trying():
    reg = [_trying("s.hk", rounds=3)]
    sc = _scorer({"s.hk": {"cls": "unclear", "tries": 25, "diff": 1.0}})
    out, _ = SM.step(reg, _wl("W"), claims=[], rows=[], moves={"s.hk"}, scorer=sc)
    assert out[0]["state"] == "trying"


def test_kept_line_is_never_the_swap_victim():
    # FULL of kept + exactly one trying: losing must drop the TRYING one, never a kept line.
    reg = [_kept("k%d" % i) for i in range(SM.MAX_LINES - 1)] + [_trying("victim", rounds=5)]
    out, ev = SM.step(reg, _wl("LL"), claims=[_claim("fresh")], rows=[], moves={"fresh"}, scorer=_scorer({}))
    assert ev["removed"] and "victim" in ev["removed"][0]
    assert all(r["state"] == "kept" for r in out if "victim" not in r["line"] and "fresh" not in r["line"])


# --------------------------------------------------------------------------- the live cap (owner: 10)
def test_live_cap_is_ten_and_shows_only_that_many():
    assert SM.MAX_LINES == 10
    reg = [_kept("k%d" % i) for i in range(4)] + [_trying("t%d" % i, rounds=1) for i in range(8)]  # 12 in play
    assert len(SM.in_play(reg)) == 10                                # capped at the live MAX_LINES
    # the offline lessons cap is independent (unchanged) so the book/scoring path is untouched
    assert SM.L.MAX_LINES == 5


# --------------------------------------------------------------------------- tone adjustment (between rounds)
def test_retone_escalates_use_more_to_always_in_place():
    reg = [_trying("s.mk", rounds=4)]                               # "use more s.mk when he stands"
    coach = [_claim("s.mk", kind="always")]                         # same move+situation, harder tone
    out, ev = SM.step(reg, _wl("W"), claims=coach, rows=[], moves={"s.mk"}, scorer=_scorer({}))
    assert ev.get("retoned") is True
    assert out[0]["line"] == "always s.mk when he stands"           # tone changed in place
    assert out[0]["state"] == "trying"                              # state preserved (not re-admitted)
    assert out[0]["rounds"] == 5                                    # its clock kept (aged once this round)
    assert ev["removed"] == ["use more s.mk when he stands"] and ev["added"] == ["always s.mk when he stands"]


def test_retone_can_reverse_a_rule_to_avoid():
    reg = [_kept("s.mk")]
    out, ev = SM.step(reg, _wl("W"), claims=[_claim("s.mk", kind="avoid")], rows=[], moves={"s.mk"},
                      scorer=_scorer({}))
    assert ev.get("retoned") is True
    assert out[0]["line"] == "avoid s.mk when he stands" and out[0]["state"] == "kept"


def test_retone_is_the_one_change_per_round_and_skips_admission():
    # a retone AND a novel claim offered the same round: only the retone happens (one change per round).
    reg = [_trying("s.mk", rounds=4)]
    coach = [_claim("s.mk", kind="always"), _claim("lightning_legs")]
    out, ev = SM.step(reg, _wl("W"), claims=coach, rows=[], moves={"s.mk", "lightning_legs"}, scorer=_scorer({}))
    assert ev.get("retoned") is True
    assert not any("lightning_legs" in r["line"] for r in out)      # the second claim was NOT also admitted


def test_no_retone_when_the_coach_repeats_the_same_tone():
    reg = [_trying("s.mk", rounds=4)]
    out, ev = SM.step(reg, _wl("W"), claims=[_claim("s.mk", kind="use_more")], rows=[], moves={"s.mk"},
                      scorer=_scorer({}))
    assert not ev.get("retoned")                                    # identical kind -> nothing to retone
    assert [r["line"] for r in out] == ["use more s.mk when he stands"]


# --------------------------------------------------------------------------- enforceability (Step 1A)
def test_unfollowable_coach_claim_is_not_admitted():
    # 's.lp up close' can never be played (close offers only cl.*): refused at admission. Not losing -> no pool.
    reg = [_kept("s.mk")]
    bad = _claim("s.lp", range_="close", when="jumping")
    out, ev = SM.step(reg, _wl("W"), claims=[bad], rows=[], moves={"s.mk", "s.lp"},
                      scorer=_scorer({}), followable=_chunli_fol)
    assert ev["added"] == [] and not any("s.lp" in r["line"] for r in out)


def test_carried_unfollowable_line_is_dropped_each_round():
    reg = [_kept("s.mk"), _trying_at("s.lp", 5, range_="close", when="jumping")]   # a dead carried rule
    out, ev = SM.step(reg, _wl("W"), claims=[], rows=[], moves={"s.mk", "s.lp"},
                      scorer=_scorer({}), followable=_chunli_fol)
    assert any("s.lp" in l for l in ev["removed"])
    assert not any("s.lp" in r["line"] for r in out)


def test_drop_unfollowable_is_targeted_and_immutable():
    reg = [_kept("s.mk"), _trying_at("s.lp", 5, range_="close", when="jumping")]
    kept, dropped = SM.drop_unfollowable(reg, _chunli_fol)
    assert dropped == ["use more s.lp up close when he jumps"]
    assert [r["line"] for r in kept] == ["use more s.mk when he stands"]
    assert len(reg) == 2                                         # input list untouched


def test_default_allows_all_when_no_predicate_given():
    # pure callers without a character pass no predicate -> enforceability is off (synthetic-move tests still valid)
    reg = [_kept("s.mk")]
    out, ev = SM.step(reg, _wl("W"), claims=[_claim("s.lp", range_="close")], rows=[], moves={"s.mk", "s.lp"},
                      scorer=_scorer({}))
    assert ev["added"] and "s.lp" in ev["added"][0]


# --------------------------------------------------------------------------- graduation via pooled stats (Step 1B)
def test_pooled_stats_graduate_a_trying_line_without_block_local_evidence():
    # the entry carries POOLED stats saying clearly 'better' (>= MIN_TRIES both sides); block-local scorer is blind
    # (returns 'few'). _ev prefers the pooled stats, so the rule graduates trying -> kept. This is the Step-1B fix.
    good = {"n_mine": 25, "sum_mine": 25 * 30, "sumsq_mine": 25 * 900,
            "n_rest": 25, "sum_rest": 25 * -10, "sumsq_rest": 25 * 100,
            "applicable": 50, "followed": 25, "rounds_fired": 5, "wins_fired": 3, "rounds_idle": 0, "wins_idle": 0}
    e = dict(_trying("s.mk", 5), stats=good)
    out, ev = SM.step([e], _wl("W"), claims=[], rows=[], moves={"s.mk"}, scorer=_scorer({}))
    assert out[0]["state"] == "kept", out[0]["why"]
