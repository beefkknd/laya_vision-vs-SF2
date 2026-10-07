"""Pre-flight RED/GREEN gate for a counter-bee -- computed from the TABLE, no emulator, no training.

A bee votes k/(k + n of its move in the cell), so it is loud ONLY where that move is under-sampled,
and it is useless (or harmful) where the move is already thick or is a known loser. Those are facts
in the table, so a bee should be ADMITTED by this gate before any 90-minute round is spent on it.
This mechanizes the recipe's bee-SELECTION principles (owner 2026-10-07): aim at under-sampled cells
with a plausibly-good move; never push a move the thick trunk already owns.

RED reasons: gate never fires; MUTE (move thick in every target cell); LOSER (loud only where the
move is clearly worse than the cell's covered best). GREEN otherwise, naming the cell it helps.
"""
from typing import Dict, List, Sequence, Tuple

from .config import QuorumConfig

COVERED_N = 20       # "confident" sample bar for the covered-best comparison
LOUD_CONF = 0.3      # a bee is "loud" when k/(k+n) >= this (n <= ~19 at k=8): genuinely under-sampled
LOSER_MARGIN = 2.0   # HP below the covered best that counts as "clearly worse" (a loser)
LOSER_FLOOR = -12.0  # an absolute floor: a move this negative is a loser even with no covered best to compare


def _cells_for(cells: Dict[str, List], rng: str, posture: str):
    """Target cells: every 'when' whose range+posture match the bee gate (fb / his_label splits included)."""
    out = []
    for when, cell in cells.items():
        p = when.split("|")
        if len(p) > 1 and p[0] == rng and p[1] == posture:
            out.append((when, cell))
    return out


def check_bee(cells: Dict[str, List], spec: Sequence[str], cfg: QuorumConfig) -> Tuple[str, str]:
    """(verdict, reason) for one [range, posture, move] bee against ``cells`` (value_table cells)."""
    rng, posture, move = spec
    hits = _cells_for(cells, rng, posture)
    if not hits:
        return "RED", "gate %s|%s never fires (no such cell in the table)" % (rng, posture)
    loud = []                                                  # cells where the bee is loud (it fires here)
    for when, cell in hits:
        s = cell.get(move)
        n = int(s[0]) if s else 0
        if cfg.k / (cfg.k + n) >= LOUD_CONF:                   # under-sampled here -> the bee can act
            m = (s[1] / s[0]) if (s and s[0]) else None        # None = blind (no samples)
            cov = [c[1] / c[0] for c in cell.values() if c[0] >= COVERED_N]
            best = max(cov) if cov else None
            loud.append((when, n, cfg.k / (cfg.k + n), m, best, sum(int(c[0]) for c in cell.values())))
    if not loud:
        nmin = min(int(cell.get(move, [0])[0] or 0) for _, cell in hits)
        return "RED", "MUTE: '%s' already thick (min n=%d -> conf %.3f) in every %s|%s cell" % (
            move, nmin, cfg.k / (cfg.k + nmin), rng, posture)
    # the bee fires in EVERY loud cell its gate matches, so it is RED if the move is a LOSER in ANY of
    # them (it would spend samples confirming a loss there -- how walk_forward hurt in mid|standing|block).
    harmful = [(w, m, best) for (w, n, c, m, best, v) in loud if m is not None
               and ((best is not None and m < best - LOSER_MARGIN) or (best is None and m < LOSER_FLOOR))]
    if harmful:
        w, m, best = harmful[0]
        return "RED", "LOSER: '%s' is loud at %s but %.1f%s -- it would waste samples confirming a loss" % (
            move, w, m, "" if best is None else " vs covered best %.1f" % best)
    good = sorted((t for t in loud if t[3] is None or t[3] > 0
                   or (t[4] is not None and t[3] >= t[4] - LOSER_MARGIN) or (t[4] is None and t[3] >= LOSER_FLOOR)),
                  key=lambda t: -t[5])
    if good:
        when, n, conf, m, best, visits = good[0]
        kind = ("blind (unsampled -> coverage)" if m is None else "positive %.1f" % m if m > 0
                else "least-bad %.1f (no covered best)" % m if best is None else "least-bad %.1f vs best %.1f" % (m, best))
        return "GREEN", "loud at %s (n=%d, conf %.2f, %d visits): '%s' %s" % (when, n, conf, visits, move, kind)
    return "RED", "no useful loud cell for '%s' in %s|%s" % (move, rng, posture)


def check_specs(cells: Dict[str, List], specs: Sequence[Sequence[str]], cfg: QuorumConfig):
    return [(spec, *check_bee(cells, spec, cfg)) for spec in specs]


def main(argv=None):
    import argparse, json, sys
    ap = argparse.ArgumentParser(description="RED/GREEN pre-flight gate for counter-bees")
    ap.add_argument("--table", required=True)
    ap.add_argument("--specs", required=True, help='JSON [[range,posture,move],...]')
    a = ap.parse_args(argv)
    cells = json.load(open(a.table))["cells"]
    rows = check_specs(cells, json.loads(a.specs), QuorumConfig())
    red = 0
    for spec, verdict, reason in rows:
        print("%-5s %-34s %s" % (verdict, "|".join(spec), reason))
        red += verdict == "RED"
    return 1 if red else 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
