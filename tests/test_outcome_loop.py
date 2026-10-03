"""Outcome-driven loop orchestration (step C), test-first and PURE.

The loop mechanics are deterministic and must be provable WITHOUT running a game or calling Qwen, so
the measurer (runs games -> BlockStat) and the proposer (Qwen -> candidate playbooks) are injected.
Here they are fakes with canned numbers. What is under test:
  - a candidate is promoted only when decide() says promote (dev-significant + held replicates);
  - the expensive held-out block is measured ONLY when the dev result warrants it (noise/cost control);
  - at most ONE promotion per round, the best by held-out delta (noise control);
  - held-out seed blocks ROTATE across rounds and the TERMINAL block is never touched in-loop
    (winner's-curse guard, plan principle 3);
  - every (round, candidate) writes an evidence-ledger row (system-of-record, not prose).
"""
from sf2.system2.promotion import BlockStat
from sf2.system2.outcome_loop import (Playbook, SeedBlocks, blocks_for_round, run_round, run_session)

INC = Playbook("incumbent", ("use more throw up close",))
WIN = Playbook("winner", ("use more s.mk at mid when he attacks",))
WIN2 = Playbook("winner2", ("use more s.hk at mid when he attacks",))
BAD = Playbook("bad", ("use more c.mk far away when he jumps",))

GOOD_DEV = BlockStat(72.7, 33.0, 112.3, 11, 12, 0.90, 0.95)
GOOD_HELD = BlockStat(48.0, 12.0, 84.0, 9, 12, 0.90, 0.95)
BETTER_HELD = BlockStat(70.0, 30.0, 110.0, 11, 12, 0.92, 0.97)
WEAK_HELD = BlockStat(18.0, -22.0, 58.0, 7, 12, 0.90, 0.95)   # spans 0 -> does not replicate
BAD_DEV = BlockStat(-84.7, -139.2, -30.1, 3, 12, 0.65, 1.0)


class FakeMeasure:
    """table: {candidate_id: (dev_BlockStat, held_BlockStat)}. Returns dev on dev_seeds, held otherwise."""
    def __init__(self, dev_seeds, held_seeds, table):
        self.dev, self.held, self.table, self.calls = tuple(dev_seeds), tuple(held_seeds), table, []

    def __call__(self, incumbent, candidate, seeds):
        self.calls.append((candidate.id, tuple(seeds)))
        dev_stat, held_stat = self.table[candidate.id]
        return dev_stat if tuple(seeds) == self.dev else held_stat


def test_promote_winner_and_keep_incumbent_when_none_improve():
    m = FakeMeasure((0, 1), (2, 3), {"winner": (GOOD_DEV, GOOD_HELD)})
    rr = run_round(INC, [WIN], m, (0, 1), (2, 3))
    assert rr.promoted and rr.new_incumbent.id == "winner", rr.rows

    m2 = FakeMeasure((0, 1), (2, 3), {"bad": (BAD_DEV, None)})
    rr2 = run_round(INC, [BAD], m2, (0, 1), (2, 3))
    assert not rr2.promoted and rr2.new_incumbent.id == "incumbent"


def test_held_block_measured_only_when_dev_warrants_it():
    m = FakeMeasure((0, 1), (2, 3), {"bad": (BAD_DEV, None), "winner": (GOOD_DEV, GOOD_HELD)})
    run_round(INC, [BAD, WIN], m, (0, 1), (2, 3))
    held_calls = [c for c in m.calls if c[1] == (2, 3)]
    assert [c[0] for c in held_calls] == ["winner"], "held-out run only for the dev-significant candidate"
    assert ("bad", (0, 1)) in m.calls, "the bad candidate still gets a dev run"


def test_one_promotion_per_round_best_by_held_delta():
    m = FakeMeasure((0, 1), (2, 3), {"winner": (GOOD_DEV, GOOD_HELD), "winner2": (GOOD_DEV, BETTER_HELD)})
    rr = run_round(INC, [WIN, WIN2], m, (0, 1), (2, 3))
    assert rr.new_incumbent.id == "winner2", "promote the larger held-out gain"


def test_dev_significant_but_held_does_not_replicate_keeps_incumbent():
    m = FakeMeasure((0, 1), (2, 3), {"winner": (GOOD_DEV, WEAK_HELD)})
    rr = run_round(INC, [WIN], m, (0, 1), (2, 3))
    assert not rr.promoted and rr.new_incumbent.id == "incumbent"
    row = rr.rows[0]
    assert row["verdict"] == "keep" and "replicate" in row["reason"]


def test_ledger_row_per_candidate_carries_stats_and_verdict():
    m = FakeMeasure((0, 1), (2, 3), {"winner": (GOOD_DEV, GOOD_HELD), "bad": (BAD_DEV, None)})
    rr = run_round(INC, [WIN, BAD], m, (0, 1), (2, 3))
    assert len(rr.rows) == 2
    ids = {r["candidate"] for r in rr.rows}
    assert ids == {"winner", "bad"}
    for r in rr.rows:
        assert set(("candidate", "verdict", "ceiling", "reason", "dev_delta", "dev_fire")) <= set(r)


# --- seed discipline -------------------------------------------------------
def test_blocks_rotate_and_terminal_is_untouched():
    sb = SeedBlocks(dev=((0, 1), (2, 3)), held=((10, 11), (12, 13)), terminal=(90, 91, 92))
    seen_held = set()
    for k in range(3):
        dev, held = blocks_for_round(sb, k)
        assert set(dev).isdisjoint(held), "dev and held must never share a seed (no peeking leakage)"
        assert set(dev).isdisjoint(sb.terminal) and set(held).isdisjoint(sb.terminal)
        seen_held.add(held)
    assert len(seen_held) >= 2, "held-out block must rotate across rounds (no winner's curse)"


def test_seedblocks_rejects_overlap():
    import pytest
    with pytest.raises(ValueError):
        SeedBlocks(dev=((0, 1), (2, 3)), held=((3, 4),), terminal=(90,))  # 3 is in both dev and held
    with pytest.raises(ValueError):
        SeedBlocks(dev=((0, 1),), held=((2, 3),), terminal=(3, 4))        # 3 overlaps a dev/held seed


# --- full session ----------------------------------------------------------
def test_run_session_promotes_over_rounds_and_logs():
    sb = SeedBlocks(dev=((0, 1), (2, 3), (4, 5)), held=((10, 11), (12, 13), (14, 15)), terminal=(90, 91))
    dev_blocks = set(sb.dev)  # which seed tuples are dev blocks

    def proposer(opp, incumbent, k):
        return [WIN] if k == 0 else [BAD]  # round 0 offers a winner; later rounds only noise

    def measure(incumbent, candidate, seeds):
        measure.seeds_used.add(tuple(seeds))
        dev_stat, held_stat = {"winner": (GOOD_DEV, GOOD_HELD), "bad": (BAD_DEV, None)}[candidate.id]
        return dev_stat if tuple(seeds) in dev_blocks else held_stat
    measure.seeds_used = set()

    ledger = []
    final = run_session("guile", INC, proposer, measure, sb, n_rounds=3, ledger_write=ledger.append)
    assert final.id == "winner", "the winner from round 0 stays the incumbent"
    assert all("round" in r and r["opp"] == "guile" for r in ledger)
    assert {r["round"] for r in ledger} == {0, 1, 2}
    assert sb.terminal not in measure.seeds_used, "terminal block never measured in-loop"
