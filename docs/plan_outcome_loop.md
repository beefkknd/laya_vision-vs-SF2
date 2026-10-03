# Outcome-driven loop — confirmed plan (Fable + gpt6 + owner, 2026-10-03)

text-laya (the model) is the PLAYER; short memory (Qwen's rules) in its prompt is the intelligence; the oracle is
scaffolding (labels, follows_rule, prompt routing) and must NOT become the policy (memory: feedback_textlaya_is_the_player).
Goal: good short memory -> text-laya wins reliably. This plan is what two independent reviews (Fable, gpt6/codex)
converged on, refined by the owner.

## Agreed principles (both reviews)
1. Firing + following is NECESSARY but NOT SUFFICIENT - a rule can fire, be followed, and still lose. Keep rules by
   OUTCOME, and hold an explicit KILL criterion: fires >=50% + followed >=0.7 but no win => the ceiling is the 40-frame
   executor / move set, NOT the short memory (a finding, not a reason to touch text-laya).
2. Evaluate whole PLAYBOOKS, not accumulated single rules (individually-good rules conflict). Promote a complete memory
   version against the incumbent; allow deletion/replacement; occasional removal tests to see which rules contribute.
3. Statistical discipline: 12 seeds x 1 round ~= +-28pp CI. A loop that re-uses held-out seeds across rounds = winner's
   curse. Rotate held-out seed blocks per round; keep a TERMINAL untouched test; "learned" counts only on a fresh block.
4. Do NOT retrain text-laya first. Routing showed it follows a clean prompt. follows_rule is a DIAGNOSTIC that routes
   failures: fired-but-not-followed -> text-laya debt (retrain queue, never rewritten as a rule); never-fired -> Coach
   debt. Retrain only if a valuable tactic repeatedly can't be followed.
5. Keep rules by EFFECTIVE COVERAGE = fire-rate x follows x hp-delta, not hp-delta alone. Free offline gate before any
   seeds: reject a candidate set whose predicted fire-rate on last-session traces is < ~40%.

## Owner refinements (override where they differ)
A. "when" IS IMPORTANT - rules stay SITUATIONAL (keyed to his state). Do NOT drop the when-requirement / allow blanket
   range-only rules. A final winning memory can have MORE rules, but every one must be situational; COVERAGE comes from
   MANY situational rules covering the occurring (range,state) cells, and routing surfaces the applicable one each
   moment. (So the Coach fix is prompt+coverage, not a grammar relaxation. The open empirical question for the
   experiment: can a situational set keyed to the top-occurring cells match/beat the range-only Expert set? Owner bets
   yes.)
B. NOISE CONTROL is first-class for ALL next-phase testing. When Qwen proposes/researches by having text-laya try
   rules, strictly LIMIT how many rules are tried at once (a few, not many) or the results are noise. Few candidates,
   enough seeds each, playbook-level changes - never a flood.

## Plan (reordered by the reviews; owner-refined)
A. COACH COVERS OCCURRING SITUATIONS WITH SITUATIONAL RULES. Feed Qwen the Scout's (range,state) histogram + each
   rule's fire-count and follows_rule; have it propose SITUATIONAL rules that cover the top-occurring cells (keep the
   when-requirement). Cap positives (~3) so routing never shows competing positives. [Coach prompt, not grammar.]
B. DECISIVE FIRING/COVERAGE EXPERIMENT (before building any loop; ~2-3 runs, no new training). On Guile + Honda (the
   never-fired failures): compare (a) the existing Expert set vs (b) a rewritten <=5-rule SITUATIONAL set keyed to the
   top-2 occurring (range,state) cells from existing traces. Report fire-rate, follows_rule, hp-margin, wins; dev 0-11,
   held-out 12-23. CONTROL (Fable's seeded-red): re-key Ken's winning throw-up-close to "when he is stunned" (rare); if
   Ken does NOT fall back toward 2/12, firing is not the causal variable and the premise is a coincidence.
   Predeclared verdicts:
   - CONFIRM: fire-rate >=50% AND hp-CI excludes 0 on dev AND replicates held-out AND wins >=9/12 on one opponent.
   - KILL: fires + followed but no win -> control-authority ceiling (executor/move set), not memory.
   - NOISE: 6-8/12, overlapping CIs -> re-key once and rerun; do not iterate.
   (Separate LEARNING from RELIABILITY: 80% supports learning but misses the ~100% target; Ken 11/12 is not near-100%.)
C. ONLY IF B CONFIRMS: build the outcome-driven loop RIGHT. Qwen writes a situational playbook -> text-laya plays it
   (routing on) -> engine keeps the PLAYBOOK by effective coverage on dev AND a fresh held-out block -> per-opponent
   verified playbook; measure win-rate climb across a session. Noise control throughout: few candidates/round, enough
   seeds each, rotating held-out + a terminal untouched test. A successful search = one playbook; repeat with fresh
   seeds + independent Coach sessions before claiming the PROCESS reliably learns.

## Baseline this builds on (shipped today)
Movement channel fixed; oracle routing fixed the multi-rule collapse (B1) - baselines jumped (Honda 0->6/12, Guile
3->6/12) with no retrain; first real win Ken 11/12 (game-guide set); block stays the no-rule default (D1). See
docs/blockages.md.
