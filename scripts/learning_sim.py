"""Readable simulation of the self-learning loop as ONE narrative: a playbook's whole lifecycle across
rounds, with the observed learning context shown at each step. Deterministic, no games/network - it runs
the same engine (sf2.system2.learning_sim) the catalog tests gate. For eyeballing what the loop does.

    python scripts/learning_sim.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sf2.system2.learning_sim import Context, RuleTrack, Suggestion, simulate  # noqa: E402
from sf2.system2.outcome_loop import Playbook  # noqa: E402
from sf2.system2.promotion import BlockStat  # noqa: E402

# A coherent Chun-Li-vs-Guile story, seeded from the real probe: turtling+losing -> learn walk_forward
# to close distance -> sharpen -> try a rule that fires+follows but can't win (executor ceiling, kept) ->
# a rule that went negative gets demoted -> terminal untouched confirm.
B = Playbook("book", ("use more throw up close",))
WALK = Playbook("+walk", ("use more throw up close", "use more walk_forward at mid range when he stands"))
WALK2 = Playbook("+walk+poke", WALK.rules + ("use more s.mk at mid range when he attacks",))
DROP = Playbook("-walk", ("use more throw up close", "use more s.mk at mid range when he attacks"))

STRONG = BlockStat(60.0, 30.0, 90.0, 10, 12, 0.80, 0.95)
HELD = BlockStat(45.0, 15.0, 75.0, 9, 12, 0.82, 0.95)
CEIL = BlockStat(-30.0, -70.0, -2.0, 3, 12, 0.66, 1.0)
INCONC = BlockStat(-3.0, -40.0, 34.0, 6, 12, 0.6, 0.9)
TERMINAL = BlockStat(38.0, 9.0, 67.0, 9, 12, 0.80, 0.95)

LOSING = Context("losing", (("mid/standing", 8, 27), ("mid/attacking", 6, 20), ("far/standing", 4, 13)))
WINNING = Context("winning", (("mid/standing", 13, 24), ("mid/jumping", 6, 16)),
                  rules=(RuleTrack("use more walk_forward at mid range when he stands", 9, 12.0, True),))
WALK_WENT_BAD = Context("winning", (("mid/standing", 6, 16), ("close/standing", 3, 8)),
                        rules=(RuleTrack("use more walk_forward at mid range when he stands", 7, -10.0, False),
                               RuleTrack("use more s.mk at mid range when he attacks", 5, 6.0, True)))

STORY = [
    Suggestion(LOSING, "suggest", WALK, STRONG, HELD, "losing by turtling -> close distance with walk_forward (top cell mid/standing)"),
    Suggestion(WINNING, "sharpen", WALK2, INCONC, None, "add a mid poke when he attacks - but it doesn't clearly beat the set"),
    Suggestion(WINNING, "suggest", WALK2, CEIL, None, "the poke fires and is followed, yet still loses hp (executor ceiling)"),
    Suggestion(WALK_WENT_BAD, "demote", DROP, STRONG, HELD, "walk_forward went net -10 -> drop it, keep the poke"),
]


def bar(ctx):
    cells = ", ".join("%s %d%%" % (c, p) for c, _, p in ctx.his_cells[:3])
    trk = "; ".join("%s net%+.0f%s" % (r.line.split(" at ")[0].replace("use more ", ""), r.net,
                                        "" if r.good else " (DEAD)") for r in ctx.rules)
    return "verdict=%-7s mode=%-11s cells[%s]%s" % (ctx.verdict, ctx.mode, cells, (" | tracking: " + trk) if trk else "")


def main():
    print("SELF-LEARNING LOOP - simulated trajectory (Chun-Li vs Guile). text-laya plays; the loop keeps")
    print("a short-memory playbook only by measured OUTCOME.\n")
    res = simulate(B, STORY, terminal=TERMINAL)
    inc = "book"
    for i, (sug, log) in enumerate(zip(STORY, res.rounds)):
        print("round %d  [%s]" % (i, sug.kind.upper()))
        print("  context : %s" % bar(sug.context))
        print("  propose : %s   (%s)" % (sug.candidate.id, sug.why))
        dev = sug.dev
        print("  measured: dev %+.1f CI[%+.1f,%+.1f] wins %d/%d fire %.0f%% follows %.0f%%%s"
              % (dev.delta, dev.lo, dev.hi, dev.cand_wins, dev.n, 100 * dev.fire_rate, 100 * dev.follows,
                 ("  held %+.1f CI[%+.1f,%+.1f]" % (sug.held.delta, sug.held.lo, sug.held.hi)) if sug.held else ""))
        tag = "PROMOTE" if log.verdict == "promote" else ("KEEP+CEILING" if log.ceiling else "KEEP")
        print("  DECIDE  : %-12s %s" % (tag, log.reason))
        if log.incumbent_after != inc:
            print("  -> incumbent now: %s  rules=%s" % (log.incumbent_after, list(sug.candidate.rules)))
            inc = log.incumbent_after
        print()
    print("FINAL incumbent: %s  rules=%s" % (res.final.id, list(res.final.rules)))
    print("TERMINAL untouched confirm: %s" % ({True: "CONFIRMED", False: "NOT confirmed", None: "n/a"}[res.confirmed]))


if __name__ == "__main__":
    main()
