"""The SIMPLE live early-game policy (sf2/system2/short_memory.py): while losing, change one line; while
winning, freeze. Two states (trying/kept), one counter (rounds-in-play), one threshold (SWAP_AFTER=2).

These are the RED-first tests for the deadlock the old timer tangle caused (playbooks/chun/round_03_ryu:
12 straight losses, every Coach claim refused 'already 2 claims in test', zero change). The pure policy
has no since/idx coordinate and no test-slot cap, so that class of bug cannot recur -- proven here.
"""
from sf2.system2 import short_memory as SM


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


# --------------------------------------------------------------------------- the one signal
def test_loss_streak_counts_trailing_lost_rounds():
    assert SM.loss_streak(_wl("WLLL")) == 3
    assert SM.loss_streak(_wl("LLW")) == 0          # a win breaks the streak
    assert SM.loss_streak([]) == 0


# --------------------------------------------------------------------------- freeze while winning
def test_no_swap_while_not_losing_enough():
    reg = [_kept("s.mk"), _trying("s.hk", rounds=9)]
    out, ev = SM.step(reg, _wl("WL"), claims=[_claim("throw_F+hp")], rows=[], moves={"throw_F+hp"},
                      scorer=_scorer({}))
    assert ev["added"] == [] and ev["removed"] == []
    assert [r["line"] for r in out] == [r["line"] for r in reg]   # memory unchanged


# --------------------------------------------------------------------------- the deadlock-breaker
def test_losing_streak_forces_one_swap_even_when_slots_are_full():
    # the round_03_ryu shape: the short memory is FULL (MAX_LINES) of aged trying lines, she is losing, the
    # Coach offers a fresh move. Old code refused it ('already 2 claims in test'); the simple policy MUST
    # drop the weakest trying line and bring the fresh one in -- the deadlock cannot recur (no slot cap).
    moves = {"s.mk", "walk_forward", "spinning_bird_kick", "s.hk", "s.mp", "lightning_legs"}
    reg = [_kept("s.mk"),
           _trying("walk_forward", rounds=5), _trying("spinning_bird_kick", rounds=5),
           _trying("s.hk", rounds=5), _trying("s.mp", rounds=5)]        # 1 kept + 4 trying = MAX_LINES (5)
    coach = [_claim("lightning_legs", range_="far")]
    out, ev = SM.step(reg, _wl("LL"), claims=coach, rows=[], moves=moves, scorer=_scorer({}))
    assert ev["added"] and "lightning_legs" in ev["added"][0], "a fresh line must be admitted while losing"
    assert ev["removed"], "a trying line must be dropped to make room"
    lines = [r["line"] for r in out if r["state"] in SM.IN_PLAY_STATES]
    assert any("lightning_legs" in l for l in lines)              # the new memory really changed
    assert len(lines) <= SM.L.MAX_LINES                           # never grows past the cap
    assert any("s.mk" in l for l in lines)                        # the kept line survived


def test_fair_chance_window_no_double_swap_right_after_one():
    # just swapped in a fresh trying line (rounds=0); losing again must NOT immediately swap it out --
    # one SWAP_AFTER window of chance. Uses the SAME single threshold, no extra knob.
    moves = {"s.mk", "lightning_legs", "throw_F+hp"}
    reg = [_kept("s.mk"), _trying("lightning_legs", rounds=0)]
    out, ev = SM.step(reg, _wl("LLL"), claims=[_claim("throw_F+hp")], rows=[], moves=moves, scorer=_scorer({}))
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
    # full of kept + one trying: losing must drop the TRYING one, never a kept line.
    moves = {"s.mk", "s.hk", "throw_F+hp", "lightning_legs", "walk_forward", "s.mp"}
    reg = [_kept("s.mk"), _kept("s.hk"), _kept("throw_F+hp"), _kept("walk_forward"),
           _trying("lightning_legs", rounds=5)]
    out, ev = SM.step(reg, _wl("LL"), claims=[_claim("s.mp")], rows=[], moves=moves, scorer=_scorer({}))
    assert ev["removed"] == ["use more lightning_legs when he stands"] or "lightning_legs" in (ev["removed"] or [""])[0]
    assert all(r["state"] == "kept" for r in out if "lightning_legs" not in r["line"] and "s.mp" not in r["line"])
