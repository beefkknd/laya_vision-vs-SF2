"""Keep/drop core of the outcome-driven loop (step C), test-first.

A candidate playbook replaces the incumbent ONLY IF it beats it on a DEV block AND replicates on a
FRESH held-out block (winner's-curse guard, plan principle 3), after passing the effective-coverage
gate (it must actually FIRE, principle 5). "Fires + followed but no win" raises the CEILING diagnostic
(executor/move-set limit, not the memory; plan principle 1 / the KILL criterion).

Fixtures are the real step-B measurements (provenance: out/measure run ba500pefp, 2026-10-03):
  KEN winning set  DEV delta +72.7 CI[+33.0,+112.3] wins 11/12   -> promote (if held replicates)
  GUILE rewrite    DEV delta -84.7 CI[-139.2,-30.1] wins 3/12 fires .65 follows 1.0 -> keep + CEILING
  HONDA rewrite    DEV delta  -3.2 CI[ -93.5,+87.2] wins 7/12     -> keep (inconclusive, CI spans 0)
  KEN seeded-red   DEV delta -68.7 CI[-104.2,-33.2] wins 0/12 fires ~0 -> keep (never fired, not ceiling)
"""
from sf2.system2.promotion import BlockStat, Cfg, decide, warrants_held


def _ken_dev_win(**kw):
    d = dict(delta=72.7, lo=33.0, hi=112.3, cand_wins=11, n=12, fire_rate=0.90, follows=0.95)
    d.update(kw)
    return BlockStat(**d)


# --- PROMOTE: beats incumbent on dev AND replicates on a fresh held-out block ---
def test_promote_when_dev_significant_and_held_replicates():
    dev = _ken_dev_win()
    held = _ken_dev_win(delta=48.0, lo=12.0, hi=84.0, cand_wins=9)  # fresh block, still clearly positive
    dec = decide(dev, held)
    assert dec.verdict == "promote", dec.reason
    assert dec.ceiling is False


# --- WINNER'S CURSE GUARD: dev-significant but held-out does NOT replicate -> keep ---
def test_keep_when_dev_significant_but_held_spans_zero():
    dev = _ken_dev_win()
    held = _ken_dev_win(delta=18.0, lo=-22.0, hi=58.0, cand_wins=7)  # CI spans 0 on the fresh block
    dec = decide(dev, held)
    assert dec.verdict == "keep", dec.reason


def test_keep_when_dev_significant_but_no_held_block_given():
    # cannot confirm on dev alone; refusing to promote is the winner's-curse guard
    dec = decide(_ken_dev_win(), None)
    assert dec.verdict == "keep"
    assert "held" in dec.reason.lower()


# --- KEEP: candidate is significantly WORSE on dev (Guile rewrite) + CEILING diagnostic ---
def test_keep_and_ceiling_when_fires_and_followed_but_loses():
    guile = BlockStat(delta=-84.7, lo=-139.2, hi=-30.1, cand_wins=3, n=12, fire_rate=0.65, follows=1.0)
    dec = decide(guile, None)
    assert dec.verdict == "keep", dec.reason
    assert dec.ceiling is True, "fires 65%% + follows 100%% + no win must raise the executor-ceiling flag"


# --- KEEP: inconclusive, CI spans 0 (Honda rewrite) ---
def test_keep_when_dev_inconclusive():
    honda = BlockStat(delta=-3.2, lo=-93.5, hi=87.2, cand_wins=7, n=12, fire_rate=0.55, follows=0.9)
    dec = decide(honda, None)
    assert dec.verdict == "keep", dec.reason


# --- KEEP (not ceiling): the seeded-red control re-keyed the throw so it NEVER FIRES ---
def test_keep_and_not_ceiling_when_candidate_never_fires():
    control = BlockStat(delta=-68.7, lo=-104.2, hi=-33.2, cand_wins=0, n=12, fire_rate=0.02, follows=0.0)
    dec = decide(control, None)
    assert dec.verdict == "keep", dec.reason
    assert dec.ceiling is False, "a rule that never fires is a coverage miss, not an executor ceiling"
    assert "fire" in dec.reason.lower()


# --- EFFECTIVE-COVERAGE GATE: a positive-looking delta is rejected if it barely fires ---
def test_keep_when_below_fire_floor_even_if_delta_positive():
    flaky = BlockStat(delta=50.0, lo=10.0, hi=90.0, cand_wins=8, n=12, fire_rate=0.20, follows=1.0)
    dec = decide(flaky, _ken_dev_win())  # even with a good held block, the dev fire floor gates first
    assert dec.verdict == "keep", dec.reason
    assert dec.ceiling is False


# --- Config is honoured (immutable override) ---
def test_fire_floor_is_configurable():
    flaky = BlockStat(delta=50.0, lo=10.0, hi=90.0, cand_wins=8, n=12, fire_rate=0.20, follows=1.0)
    held = BlockStat(delta=40.0, lo=8.0, hi=72.0, cand_wins=8, n=12, fire_rate=0.22, follows=1.0)
    dec = decide(flaky, held, Cfg(min_fire=0.15))
    assert dec.verdict == "promote", dec.reason


# --- FAIL/edge: malformed stats are rejected at the boundary ---
def test_rejects_empty_block():
    import pytest
    with pytest.raises(ValueError):
        decide(BlockStat(delta=0.0, lo=0.0, hi=0.0, cand_wins=0, n=0, fire_rate=0.0, follows=0.0), None)


def test_rejects_ci_not_bracketing_delta():
    import pytest
    with pytest.raises(ValueError):
        decide(BlockStat(delta=72.7, lo=80.0, hi=112.3, cand_wins=11, n=12, fire_rate=0.9, follows=0.9), None)


# --- warrants_held gates the EXPENSIVE held-out run: only when dev would otherwise promote ---
def test_warrants_held_true_only_when_dev_passes_fire_and_significance():
    assert warrants_held(_ken_dev_win()) is True
    # below fire floor: don't spend held-out games
    assert warrants_held(_ken_dev_win(fire_rate=0.2)) is False
    # dev not significantly better: don't spend held-out games
    assert warrants_held(_ken_dev_win(delta=-10.0, lo=-50.0, hi=30.0)) is False


def test_decide_agrees_with_warrants_held():
    # if warrants_held is False, decide keeps regardless of a (hypothetical) held block
    dev = _ken_dev_win(fire_rate=0.2)
    assert warrants_held(dev) is False
    assert decide(dev, _ken_dev_win()).verdict == "keep"


# --- INVARIANT: the decision core stays PURE (no games/IO/Qwen leak into it) ---
def test_promotion_core_is_pure():
    import pathlib
    src = pathlib.Path(__file__).resolve().parents[1] / "sf2" / "system2" / "promotion.py"
    text = src.read_text()
    for banned in ("subprocess", "import os", "import json", "import qwen", "requests", "open("):
        assert banned not in text, "pure keep/drop core must not reference %r" % banned
