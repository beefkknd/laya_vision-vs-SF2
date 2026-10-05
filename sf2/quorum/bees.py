"""The bee roster: one declarative source of truth for which voters the quorum runs, each one's ROLE, when it fires,
and where its proposal comes from. The decider (decider.py) wires these; table_study.py reads this to phrase its
bee suggestions; a test pins BEES == config.VOTERS so the two never drift.

Three roles organise the swarm:
  base     the generalist proposer -- today's full two-stage text-laya pick; votes every decision and is the fallback.
  exploit  reads the thick trunk -- votes the cell's best CONFIDENT, POSITIVE move; abstains when the table is unsure.
  explore  fills gaps -- votes the least-sampled followable move where the table is thin/blind, loud where it is
           uncertain and fading as a winner accrues (frontier); the fireball bee is the same, gated to the fb=1 slice.

RETIRED 2026-10-05 (run A_20261005_084309): the category-FORCING bees defend/punish/combo. Forcing a category the
trunk already owned made the quorum spam it (combos ~6x/round) and cost win-rate (65% vs the ~71% baseline). The
flavour machinery (voters.py) stays available for a config that sets its own ``flavors``, but no bee forces a category
by default. The lesson: organise bees to FILL what the table lacks, not to push a category it already has.
"""
BEES = {
    "laya":     {"role": "base",    "fires": "every decision",               "source": "voters.base_proposal"},
    "table":    {"role": "exploit", "fires": "cell has a confident+ move",    "source": "tally.table_proposal"},
    "frontier": {"role": "explore", "fires": "always (loud where thin/blind)", "source": "frontier.frontier_proposal"},
    "fireball": {"role": "explore", "fires": "a fireball is out (fb=1)",      "source": "frontier.fireball_proposal"},
    "pressure": {"role": "explore", "fires": "opponent is attacking",        "source": "frontier.pressure_proposal"},
}

ROLES = ("base", "exploit", "explore")


def by_role(role: str):
    """The bee names with this role, e.g. by_role('explore') -> ['frontier', 'fireball']."""
    return [name for name, b in BEES.items() if b["role"] == role]
