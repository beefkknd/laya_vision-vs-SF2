# Blind review of README.md by Fable (2026-10-02), verbatim

Reviewer: Claude Fable, blind (given only README.md, the repo, and the owner's statement of the project). Reviewed
README.md at feat/lv-value 3b4afab.

## 1. Claims
| Claim | Verdict | Evidence |
|---|---|---|
| Screen reader `sf2/screen/` reads who/where/action/air/facing/health/fireball/round-over, no RAM | VERIFIED | sf2/screen/facts.py:10-59; tests/test_screen_invariant.py bans sf2.emu/mesen/RAM imports in sf2/screen (222 tests pass, 44 s) |
| Words `sf2/system1/screen_words.py` facts -> sentence | VERIFIED | screen_words.py:163-172 (note/situation); sentence form = advice.situation_text:73 |
| Text laya `runs/text_laya/advice_v1`, trained on a fixed rule over words, never on outcomes | VERIFIED | adapter.json (data test_data/advice); build_advice_data.py:1-12 targets = advice.answers (label rule); test.json tuned 0.992, held-out wordings 0.987 (the "own test set" of diag 2b exists) |
| "without advice it decides nothing useful" | WRONG as stated | advice.answers():161-179 falls to "vision"/"walk": with no lesson it picks the best-RATED option. It is not nothing; it is the rating (table/laya-vision) deciding. And play_screen.py:81 runs exactly that, today |
| Table `lessons/value_oracle_v1.json`, from RAM-labelled games | VERIFIED | 96 cells keyed (MY char, range, opp_attacking, opp_airborne); docs/prereg_value_oracle.md:7-10 (from rollouts/lv_value RAM logs) |
| Table "NOT in play" | WRONG for the code; true only as intent | play_screen.py:53,70,80 loads it into System1(oracle=table); system1._by_table:155-168 ranks every decision with it. The only screen-play entry point is table+text laya, "Advice: none", qwen off (run.json: advice none, qwen off) |
| Replay `scripts/replay_score.py` replays with RAM, refuses on drift, pixel-exact | VERIFIED | screen_replay.py:119-132 (sha256 + np.array_equal of PNGs, ReplayMismatch); replay_score.py:72-76,93-95 exit 1 |
| `follows_rule` logged per decision | VERIFIED but semantics off | advisor.choose:127 `follows_rule = pick in rule`; screen_play.py:42 logs it. With no lessons `rule` is "vision"/"walk", so it measures following the RATING, not "the advice" (README step 3) |
| Gate "position .98-.99, action at catalog limit, health as drawn 1.0, 7 ms/frame" | VERIFIED numbers, MISLEADING framing | out/screen_gate/s303/gate.json: x .9895/.9843, action .893/.832 vs ceiling .899/.831, health 1.0, 6.95 ms. But `pass: False` in all three gate.json (s303 set a round_over 0.25, 12 early time-overs); docs/laya_text_only_plan.md:116-121 records it as "accepted gate-wording defect, owner: move on". README says "gated" and never says the gate script FAILS. Also health vs RAM life is .97/.94, and at decisions the my_bar WORD agrees with RAM only .74/.63 (rollouts/screen_play/smoke_20261002_ryu*/score.json) |
| No RAM in play: bridge without RAM + raising handle | VERIFIED | mesen/sf2_bridge_screen.lua:1-14,177; screen_emu.py:44-50 (__getattr__ -> RamForbidden, RAM rows refused) |
| Unknown sprite -> "block" + logged | VERIFIED | reader.py:40,160-163; unknown_log.py:50; out/screen_reader_unknown/gate/*/unknown.jsonl. (Stale docstrings still say "stand": screen_play.py:12, laya_text_only_plan.md:99,113) |
| 847 ROM poses | VERIFIED | out/sprite_rom/index.json poses sum = 847 |
| laya-vision eye parked on threebody `runs/eye3_q3_512` | UNVERIFIABLE here | not in this worktree's runs/; only in memory notes (session_2026-10-02.md:119-120) |
| Qwen OFF on threebody by owner's order (G6) | UNVERIFIABLE from repo | memory note only |

## 2. Against the owner's statement
- The owner: "testing text laya with the table is useless". The repo's ONLY runnable screen-only loop (play_screen.py arm S0) is precisely that, and "Already done: screen-only play mode" lists it as an achievement without saying the mode is the rejected arm. The README's table says "dropped", the code says it is the whole thing.
- "Text laya decides nothing useful without advice" misstates the mechanism: it follows the ratings. Correct framing per owner: an unguided score is meaningless, not "it decides nothing".
- Nothing contradicts "RAM only to check the conversion" in intent; in fact Qwen's whole input path is RAM today (see G3) and the README admits it only inside the checklist.
- Measures section promises per-opponent scoring of rules by the table; the table has no opponent axis (cell = my char, range, attacking, airborne) and Chun-Li vs Guile is held out: a "vs Honda" rule cannot be scored per opponent, only generically. Not stated.

## 3. Gap checklist
- G1 REAL. advice.answers needs every option to carry one of 3 rating words (`RATING[options[m]]` KeyError otherwise); training data built with laya-vision's rated top-3 (build_advice_data.py:1-4). Fix is sound but understated: it is a retrain of text laya (new question shape), not a wiring change, and the "walk in" default is already the rule's own fallback (answers:177).
- G2 REAL. No "web" source exists; lessons.py:283-288 only knows view "book"/"players' tip". Fix plausible via from_book-style registry entries.
- G3 REAL and UNDERSTATED. Qwen's evidence = game_log.action_entry (p1_life/p2_life/p2_state RAM, game_log.py:66-80) and opp_moves.py (RAM-only: fireball from shot slot, uppercut from y, slap from p2_sub). lessons.condition_evidence:97-99 and lessons.cause:115-119 need dealt/taken/actual; character_prompt threats need opp_move. The screen record of G3 gives hp drops + a 7-label action: no hit/whiff/blocked, no fireball/uppercut/throw naming -> the "HIS THREATS" and "if you see X" views and `cause` must be rebuilt, not just re-fed. screen_play's decisions.jsonl has no dealt/taken at all today.
- G4 REAL. advice.py:100-110 WHEN = 5 states + range; reader does see projectiles (facts.projectiles) but words never say so. (Reader also has no "crouch" label; screen_words derives crouching from crouch sprites, screen_words.py:109 - OK.)
- G5 REAL. Today table ranks in play (_by_table). "no fireball cell" correct (value_oracle cell = range/attacking/airborne). Add: no opponent cell either.
- G6/G7/G8: plausible; G7 has no code yet (no per-game report script).
- MISSING gaps: (M1) no screen-only play path WITH an advisor+lessons (qwen_lessons.py and learn_loop.py are RAM-game loops; play_screen hardcodes advice "none", qwen "off"); the Qwen-in-loop runner must be written. (M2) the gate's own FAIL (round_over) is unacknowledged: a time-over round ending 1 frame early changes which decision is last; needs a stated waiver. (M3) health-bar words at decisions agree .63-.74 in the smoke run: damage-from-bar-drops (G3) must be validated at decision granularity, not per frame. (M4) lesson retirement (lessons.review/stop:212-250) keys on per-game hp from RAM game entries; needs the screen record too.

## 4. Confusing / self-contradictory
- "Table NOT in play ... dropped" vs "Already done: screen-only play mode" (which is the table in play). Diagnosis 1 says the table scores rules, but a rule's situation ("when he jumps up close vs Ken") is scorable only without the opponent.
- Step 4 "Qwen reads the screen-based record" is written as present tense though G3 says it does not exist.

## Verdict
Numbers are honest; the narrative is not. The README presents an intended architecture as if the pieces exist and omits that (a) the only working play mode is the one the owner called useless, (b) the gate it cites reports FAIL, (c) Qwen's entire evidence path (including lesson admission/retirement and his-move naming) is RAM-built and must be rewritten, not re-pointed. Needs a rewrite of "Already done", the gate line, G3, and an added "what does not exist yet: the Qwen-in-loop screen runner".
