# Two-stage Qwen (System 2): Scout + Coach (plan, 2026-10-03, owner's design)

## Why (from the 30-round session)
One Qwen prompt does two different jobs (observe + strategize) and defaults to the safe one: it kept adding BLOCK rules
(almost every opponent's final rules are block_low/high + throw, offense nearly absent), churn was tiny (14 added / 1
removed in 26 rounds), and she turtled to a flat ~23% win rate (hit rate ~85/round, never climbs). The owner: split
Qwen into two rounds, like the text-laya category/move split, so each side can be prompted and fine-tuned for its own
job.

## The split (both run BETWEEN games, in System 2 - never in play; no hard-gate impact)
### Stage 1 - SCOUT (summarize the game). Grounded, temperature 0.
Input: the game's screen-evidence record (per decision: his action/range, her action, dealt/taken; round results;
win/loss; what she did MOST; what LANDED vs whiffed; which current rules fired and whether they tracked good/bad
outcomes). Output: a structured SUMMARY (factual, no strategy): her dominant actions, offense actually used, total
dealt/taken, his key patterns/threats, per-rule "fired N, tracked damage up/down", and a winning/losing verdict.

### Stage 2 - COACH (decide rule changes). Creative, higher temperature.
Input: the Scout summary + the current in-play rules + the win/loss & hp trend over recent games. Output: rule changes
in the advice grammar (so text laya can follow), in one of two MODES:
- **Consolidate (winning/stable):** PROMOTE the rules that keep working so they STICK (a sticky/verified state review
  won't easily drop); retire noise. (Fixes the low-churn, nothing-sticks problem.)
- **Escalate (losing):** DO NOT add defense. Find offense - approach, anti-air, punish, or a CREATIVE combo/exploit
  not tried before. The prompt is given the list of what has already been tried (from the Scout) so it does not
  repeat, and is explicitly forbidden from answering "block more" when she is already losing by blocking.

## Guardrails / mechanics
- Output stays in the advice grammar (char_menu_moves(me) + ranges + his states + fireball), validated as today, so a
  proposed rule is followable; an unparseable one is dropped with a reason.
- "Stick" = a rule promoted after it keeps correlating with good outcomes gets a verified/sticky state; a rule that
  correlates with losses is retired. This is the churn fix.
- Exploration: vary the per-visit start seed so games are not identical replays (today ken v2=v3=v4 are byte-identical,
  so Qwen sees nothing new). A deterministic replay teaches nothing.
- Keep the single-prompt path as a FALLBACK for A/B (does two-stage beat one?).

## Why two prompts help the fine-tuning (owner's point)
Each stage has one job, so each can be tuned/grader-scored on its own:
- Scout graded on FAITHFULNESS (does its summary match the screen/RAM record?).
- Coach graded on STRATEGIC QUALITY (G5 table-grade of its proposed rules + did the win rate / hit rate actually rise
  after its changes?).
One prompt mixes a temperature-0 observer with a creative strategist; neither is tunable without hurting the other -
the same reason the text-laya category/move split worked.

## Also on the list (separate, after the Qwen split): the block-biased cat_v1
cat_v1 over-reaches for block (we trained condition-off->block hard to kill the lightning_legs spam). Rebalance it with
a saner block-vs-attack mix once the Qwen layer is pushing offense - otherwise even good offense rules get blocked over.

## Build order
1. Two-stage Qwen (Scout + Coach) in System 2, single-prompt fallback kept. Tested with a mock Qwen (the plumbing) and
   a short live session (does she attack more / win more?).
2. Per-visit seed variation (exploration) in the session driver.
3. (Later) rebalance cat_v1; grade Scout faithfulness + Coach quality as the two tuning signals.
