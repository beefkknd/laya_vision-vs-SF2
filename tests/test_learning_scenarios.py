"""Catalog of the learning process: the Qwen-driven lifecycle events as named scenarios, each with a
predeclared trajectory. Seeded from REAL Coach proposals captured against the live server
(out/.../probe_guile, probe_honda, 2026-10-03) so the fixtures are evidence, not imagination.

Each scenario threads the observed LEARNING CONTEXT (verdict -> escalate/consolidate, his_cells, rule
tracking) into the real decide() gate. The events covered:
  SUGGEST+PROMOTE+CONFIRM   a situational rule beats the incumbent on dev + held, terminal replicates
  KEEP (inconclusive)       dev CI spans 0 -> incumbent stands
  KEEP (worse)              candidate significantly worse -> incumbent stands
  KEEP (coverage miss)      fires below the floor -> no grip
  CEILING                   fires + followed but loses -> executor limit, not the memory
  WINNER'S CURSE            dev-significant but held does not replicate -> not promoted
  PROMOTE then TERMINAL-FAIL a dev+held promotion that the untouched terminal block does NOT confirm
  DEMOTE                    a rule gone negative in the context is dropped by promoting the set without it
  MODE                      verdict drives escalate (losing) vs consolidate (winning), the real rule
  NOISE control             many candidates in a round, at most one promoted (run_round)
"""
from sf2.system2.learning_sim import Context, RuleTrack, Suggestion, simulate
from sf2.system2.outcome_loop import Playbook, run_round
from sf2.system2.promotion import BlockStat

# --- real-seeded playbooks (ids/lines from the probes) ---
BOOK = Playbook("book", ("use more throw up close",))
GUILE_WALK = Playbook("guile+walk", ("use more throw up close", "use more walk_forward at mid range when he stands"))
HONDA_JUMP = Playbook("honda+smk_jump", ("use more throw up close", "use more s.mk at mid range when he jumps"))
HONDA_STAND = Playbook("honda+smk_stand", ("use more throw up close", "use more s.mk at mid range when he stands"))

# --- real-seeded contexts (his_cells + verdict from the probe scouts) ---
LOSING_GUILE = Context("losing", (("mid/standing", 8, 27), ("mid/attacking", 6, 20), ("far/standing", 4, 13)))
WINNING_HONDA = Context("winning", (("mid/standing", 11, 28), ("mid/jumping", 7, 18), ("mid/attacking", 4, 10)))
# round-2 Honda context: the jumping rule has gone negative (real: s.mk mid/jumping net -0.3)
HONDA_JUMP_WENT_BAD = Context("winning", (("mid/standing", 22, 31), ("mid/jumping", 17, 24)),
                             rules=(RuleTrack("use more throw up close", 6, 4.1, True),
                                    RuleTrack("use more s.mk at mid range when he jumps", 8, -0.3, False)))

# --- measurements ---
STRONG_DEV = BlockStat(60.0, 30.0, 90.0, 10, 12, 0.80, 0.95)
STRONG_HELD = BlockStat(45.0, 15.0, 75.0, 9, 12, 0.82, 0.95)
WEAK_HELD = BlockStat(18.0, -22.0, 58.0, 7, 12, 0.80, 0.95)       # spans 0: does not replicate
INCONCLUSIVE = BlockStat(-3.0, -93.0, 87.0, 7, 12, 0.55, 0.90)    # CI spans 0
WORSE = BlockStat(-84.7, -139.2, -30.1, 3, 12, 0.65, 1.0)        # Guile real rewrite
COVERAGE_MISS = BlockStat(40.0, 10.0, 70.0, 8, 12, 0.20, 1.0)    # positive-looking but barely fires
CEILING_MEASURE = BlockStat(-40.0, -80.0, -5.0, 3, 12, 0.65, 1.0)  # fires + followed + loses
TERMINAL_OK = BlockStat(40.0, 10.0, 70.0, 9, 12, 0.80, 0.95)
TERMINAL_FAIL = BlockStat(10.0, -30.0, 50.0, 6, 12, 0.80, 0.95)  # untouched block does NOT confirm


def _one(ctx, kind, cand, dev, held=None):
    return [Suggestion(context=ctx, kind=kind, candidate=cand, dev=dev, held=held)]


# ---------------------------------------------------------------------------
def test_suggest_promote_confirm():
    r = simulate(BOOK, _one(LOSING_GUILE, "suggest", GUILE_WALK, STRONG_DEV, STRONG_HELD), terminal=TERMINAL_OK)
    assert r.rounds[0].mode == "escalate"           # losing -> escalate (find offense/movement)
    assert r.rounds[0].verdict == "promote"
    assert r.final.id == "guile+walk"
    assert r.confirmed is True                       # terminal untouched block replicated


def test_keep_inconclusive():
    r = simulate(BOOK, _one(WINNING_HONDA, "suggest", HONDA_JUMP, INCONCLUSIVE), terminal=TERMINAL_OK)
    assert r.rounds[0].verdict == "keep" and r.final.id == "book"
    assert r.confirmed is None                        # nothing promoted -> no terminal test


def test_keep_worse():
    r = simulate(BOOK, _one(LOSING_GUILE, "suggest", GUILE_WALK, WORSE))
    assert r.rounds[0].verdict == "keep" and r.final.id == "book"


def test_keep_coverage_miss():
    r = simulate(BOOK, _one(WINNING_HONDA, "suggest", HONDA_JUMP, COVERAGE_MISS))
    assert r.rounds[0].verdict == "keep"
    assert "fire" in r.rounds[0].reason.lower() and r.rounds[0].ceiling is False


def test_ceiling_executor_limit():
    r = simulate(BOOK, _one(LOSING_GUILE, "suggest", GUILE_WALK, CEILING_MEASURE))
    assert r.rounds[0].verdict == "keep"
    assert r.rounds[0].ceiling is True               # fires + followed + no win -> executor, not memory


def test_winners_curse_not_promoted():
    r = simulate(BOOK, _one(WINNING_HONDA, "suggest", HONDA_JUMP, STRONG_DEV, WEAK_HELD))
    assert r.rounds[0].verdict == "keep" and r.final.id == "book"
    assert "replicate" in r.rounds[0].reason


def test_promote_then_terminal_fails():
    # dev + held both say promote, but the UNTOUCHED terminal block does not confirm -> caught false positive
    r = simulate(BOOK, _one(LOSING_GUILE, "suggest", GUILE_WALK, STRONG_DEV, STRONG_HELD), terminal=TERMINAL_FAIL)
    assert r.final.id == "guile+walk"                # it was promoted in-loop
    assert r.confirmed is False                       # ...but the terminal test refuses to confirm it


def test_demote_rule_gone_negative():
    # round 0: add s.mk/jumping (promoted). round 1: context shows it went negative -> demote by promoting
    # the set WITHOUT it (and with the standing rule instead). Final must not carry the jumping rule.
    rounds = [
        Suggestion(WINNING_HONDA, "suggest", HONDA_JUMP, STRONG_DEV, STRONG_HELD),
        Suggestion(HONDA_JUMP_WENT_BAD, "demote", HONDA_STAND, STRONG_DEV, STRONG_HELD),
    ]
    r = simulate(HONDA_JUMP.__class__("honda_book", ("use more throw up close",)), rounds)
    assert [x.kind for x in r.rounds] == ["suggest", "demote"]
    assert r.final.id == "honda+smk_stand"
    assert "use more s.mk at mid range when he jumps" not in r.final.rules   # the dead rule is gone
    assert "use more s.mk at mid range when he stands" in r.final.rules
    # the context that drove the demote correctly carried the negative-tracking rule
    assert any(not rt.good for rt in HONDA_JUMP_WENT_BAD.rules)
    assert HONDA_JUMP_WENT_BAD.negative_rules()[0].line == "use more s.mk at mid range when he jumps"


def test_mode_follows_verdict():
    assert LOSING_GUILE.mode == "escalate"
    assert WINNING_HONDA.mode == "consolidate"


def test_coverage_gated_near_miss_is_flagged_in_the_ledger():
    # r1-style: significant gain (CI excludes 0) but fires below the 40% floor -> kept, but flagged as a
    # near-miss so a high-significance low-coverage rule is visible, not silently dropped.
    sbk = BlockStat(136.7, 69.5, 203.8, 4, 12, 0.31, 1.0)
    cand = Playbook("sbk", ("use more throw up close", "use more spinning_bird_kick at mid range when he stands"))

    def measure(incumbent, candidate, seeds):
        return sbk

    rr = run_round(BOOK, [cand], measure, (0, 1), (2, 3))
    row = rr.rows[0]
    assert row["verdict"] == "keep"
    assert row["dev_significant"] is True
    assert row["coverage_gated"] is True
    # a plain inconclusive candidate is NOT a near-miss
    def measure2(i, c, s):
        return BlockStat(-3.0, -40.0, 34.0, 6, 12, 0.6, 0.9)
    row2 = run_round(BOOK, [Playbook("x", ("use more throw up close", "r"))], measure2, (0, 1), (2, 3)).rows[0]
    assert row2["coverage_gated"] is False


def test_noise_control_one_promotion_per_round():
    # many candidates in one round; at most one is promoted (the best held delta). Real Coach is capped
    # to a few, but the engine must hold the line even if more arrive.
    cands = [Playbook("c%d" % i, ("use more throw up close", "r%d" % i)) for i in range(4)]

    def measure(incumbent, candidate, seeds):
        # only c2 is a real winner; the rest are inconclusive noise
        if candidate.id == "c2":
            return STRONG_DEV if tuple(seeds) == (0, 1) else STRONG_HELD
        return INCONCLUSIVE

    rr = run_round(BOOK, cands, measure, (0, 1), (2, 3))
    assert rr.new_incumbent.id == "c2"
    assert sum(1 for row in rr.rows if row["verdict"] == "promote") == 1
