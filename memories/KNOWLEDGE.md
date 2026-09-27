# System 2 knowledge

What System 2 (Claude now, Qwen later) has learned about advising laya. It is written for the next adviser: read it
before writing rules. Every line is a measured result, with its source run; nothing here is a guess.
Plan: docs/TWO_SYSTEM_PLAN.md. Runs: `out/results.jsonl`.

## How the hands (laya r2) behave vs Ryu
- r2 alone beats Ryu narrowly: +40.7 net damage per round, 69% of rounds, 37/40 matches (eval). The scripted
  teacher gets +89.3, 87% of rounds, 40/40. (`ryu_A_eval`, `ryu_T_eval`)
- **laya's mistakes are confident ones.** Decisions where laya is unsure (top-2 margin in the lowest 5%) lead to a
  hit in the next 0.5 s *less* often than confident ones: 10.0% vs 15.5%. Don't look for mistakes where laya
  hesitates; look where it gets hit. (`ryu_A_dev`, scripts/moments.py)
- **laya disagrees strongly with anti-air.** When Ryu jumps in at mid range, laya wants `jump_forward` (41–54%) and
  gives fierce `hp` only 1–10%. (prototype, dev opening 16)

## What the note can and cannot say
- It can: each fighter's state word, life, distance (close < 80 px, mid < 120 px, far), facing, corner, clock, last
  move, fireball distance.
- It cannot: jump height (so no "kick at the top of the jump"), exact distance (the throw works within 42 px, but
  "close" means under 80 px), whether he is closing in, or which move he is doing.
- A rule that needs something the note cannot say will misfire. Prefer rules whose trigger the note states exactly.

## What rules do
- **Rules carry real knowledge on their own.** Playbook v1 on a random player: −61.5 → −23.9 net damage per round,
  +35.2 paired (95% CI +16.5 to +53.9). (`ryu_R_dev`, `ryu_D1_dev`)
- **Rules cover few situations.** v1's 13 rules match 11–13% of controllable decisions.
- **A rule that contradicts a confident laya does nothing at a low tau.** At τ = 0.3 the anti-air rule fired 12
  times and changed 0 moves.
- **Pushed hard, the teacher's tactics hurt laya.** v1 at τ = 0.75 on r2: −18.3 vs r2 alone +40.2, a paired
  **−62.6** (−84 to −42). It changed only 4% of decisions. (`ryu_C1t075_dev`, Studio)
- **The 0.5 s outcome of a changed move does not predict its cost.** In that run, every rule's changed moves looked
  neutral or positive over 0.5 s (air kick +2.0, anti-air fierce up close +24), except mid-range anti-air roundhouse
  (−6.1), yet the match was 62 points worse. Judge a rule by paired play on the dev openings (ablation), never by
  its local window.

- **The air kick is part of the harm, not all of it.** v1 without `me_state=jump -> hk` at τ = 0.75: −0.2, which is
  +16.9 over full v1 (−1 to +35, inconclusive) but still −45.7 vs r2 alone (−61 to −31). (`ryu_C1noair_t075_dev`)
- **For v1, τ above 0.45 changes nothing.** Of 1,245 fired decisions, 793 already had the rule's move as laya's top
  move (nothing to change); the other 452 all had a gap of at most 0.45 and changed at every τ ≥ 0.5. τ = 0.5, 0.75 and
  1 (a full override) play identically: −18.3 ± 7.7. The harm comes from *which* moves the rules pick, not from how
  hard they push. (`ryu_C1w05_dev` = `ryu_C1t075_dev` = `ryu_C1t1_dev`; an earlier note here claiming "gaps above
  0.75" was wrong and is corrected.)

- **Forward selection: no group of the teacher's rules helps r2.** Each of v1's six groups alone on r2, paired vs r2
  alone on dev: anti-air −54.0, air kick −25.2, throw −22.5 (all worse); guard, jump-in and walk-in-from-far 0.0
  (laya already does them, so they never change a move). r2 learned this teacher's tactics plus two DAgger rounds of
  corrections, so the teacher's own rules can only repeat laya or override its better choices. **A System 2 that
  helps must bring knowledge laya does not already have**: look at the surprised moments, where laya gets hit, not at
  the teacher. (`ryu_fs_*_dev`, 2026-09-27)

## Method that works
- Change one thing, re-run the 20 dev openings (about 6 min on the Studio, 8 on the Mac Pro), and compare paired
  with `scripts/paired.py`. Verdicts only on the eval openings.
- Build a playbook the way the teacher was built: add one rule group at a time to an empty memory, and keep a group
  only if the paired dev gain is positive. Don't start from a full playbook and cut: v1 as a whole was −62.6.
