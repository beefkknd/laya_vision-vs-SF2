# Overnight self-learning loop run - 2026-10-02 (Chun-Li, all 6 opponents)

## Verdict: the loop works. Chun-Li gains knowledge against 4 of 6 opponents.
Screen-only play (no RAM in play), two-stage text laya (cat_v1 condition-off fixed + move_v1), seeded from book.json,
Qwen (threebody) learns after every game, graded by the table (G5). 12 games x 2 rounds per opponent (honda 10).

| Opponent | hp/round first5 -> last5 | Delta | rounds W-L | follows_rule | Qwen rules good/bad (G5) |
|---|---|---|---|---|---|
| honda   | -49 -> +27 | **+76** | 8-12  | 82% | 23/2  |
| ryu     | -28 -> -1  | **+27** | 5-19  | 69% | 12/11 |
| ken     | -25 -> +1  | **+26** | 12-12 | 77% | 12/7  |
| zangief | -57 -> -31 | +26     | 6-18  | 82% | 12/10 |
| guile   | -45 -> -47 | -2      | 5-19  | 72% | 12/10 |
| dhalsim | -39 -> -50 | -12     | 4-20  | 76% | 17/0  |

## What this shows
- **4/6 improve** (honda, ryu, ken, zangief); ken reaches parity (12-12). honda is the clearest (+76, winning rounds).
- **Qwen learns opponent-specific tactics**, e.g. vs ken "use more hp up close when he attacks"; vs guile "c.mk at
  mid range when he stands" + "avoid block_high at mid"; vs ryu "block_low far away when he attacks"; vs dhalsim/
  zangief "throw up close". It ADMITS and RETIRES rules each game (rotation works).
- **text laya follows** 69-82% and climbs within a run (honda hit 100% mid-run); the condition-off fix held (she
  blocks when a rule's condition is off instead of spamming one move).
- **The table (G5) grades the rules**, and the grade explains the misses:
  - dhalsim: rules were ALL table-good (17/0) yet she still lost - good rules, but blocking is not enough against
    stretchy zoning. A STRATEGY ceiling, not a loop failure.
  - guile / ryu / zangief: ~half the admitted rules are table-BAD - Qwen's rule quality has room; tightening the
    System-2 prompt or the admission check would help.

## The two failures are informative, not broken plumbing
The loop ran clean end to end for all 6 (seed -> play -> screen evidence -> Qwen -> short memory -> play), no RAM in
play, Qwen failover armed (not needed - threebody stayed up). The zoners (dhalsim, guile) are the known-hard Chun-Li
matchups; the system correctly learned defensive rules but Chun-Li's answer set is not enough to get in. That is the
next thing to work on (better offense rules / anti-zoning), and it is a real finding the loop surfaced.

## Data
rollouts/loop_screen/overnight_chunli_<opp>/ (trace.jsonl, verdict.json, per-round dirs, loop_report.json).
Per-game detail: scripts/loop_report.py <dir>.

## Caveats
- Scored from the screen (--no-score): round result/hp from the health bars (reader gated vs RAM at .98+; the replay
  RAM referee is available but off for speed). A final confirmation run with --score would cross-check with RAM truth.
- "better" = last-5 vs first-5 mean hp/round, 12 games/opponent, 1 seed - exploratory, not the pre-registered verdict
  (docs/prereg_loop_run.md wants more seeds + a FROZEN control arm). This run shows the loop FUNCTIONS and gains
  knowledge; the pre-registered run is the formal verdict.
