# Harness and component ledger: every defect found, its root cause, and which component owns it

Owner (2026-09-30): "learn the boundary of each component ... understand the root cause of the problem, then decide
which is caused by vision laya, which is text laya, which is Qwen." This ledger holds the defects found so far (P0/P1
code-review items are in docs/qwen_learning.md section 0). Component: **H** harness (runner, budget, logs, data
hygiene), **S** statistics / analysis, **V** verifier (sf2/system2/lessons.py), **P** Qwen's prompt, **Q** Qwen,
**R** label rule / System 1 glue (sf2/system1/advice.py, advisor.py), **LV** laya-vision, **TL** text laya.

| # | date | symptom | root cause | comp. | fix (commit) | guard |
|---|---|---|---|---|---|---|
| 1 | 09-29 | the Mac crashed (Jetsam, watchdog panic) | 38 parallel jobs used ~208 GB | H | machine-wide memory budget ledger (bc1a575) | budget tests; peak 150 GB seen |
| 2 | 09-29 | runs launched from zsh failed | zsh does not word-split `$ARGS` | H | arguments written out | - |
| 3 | 09-29 | test rollouts counted as her play history | `rollouts/qwen_moves` was not a test dir | H | TEST_DIRS + run.json "test" | test_logs |
| 4 | 09-29 | two runs vs one opponent wrote one console log | log name was `<opp>_<arm>` | H | a log folder per run (3acb927) | test_character_prompt |
| 5 | 09-29 | 2 of 24 runs died | budget queue timeout (NoRoom, 30 min) with 36 runs queued | H | launch in waves of 12 | - (procedure) |
| 6 | 09-29 | last runs used ports past the range | 36 runs x 2 ports > 64 | H | range 96 (837bdb1) | - |
| 7 | 09-30 | every job of a batch failed at once | arm name `qwen` is reserved in ab_memory | H | renamed `qwenset` before any game (8381fa8) | parse_fixed refuses it (by design) |
| 8 | 09-30 | two 2-round smoke runs counted as evidence | nothing marks a smoke run | H | runs under 3 games skipped and listed (84c7dd5) | test_track_record, test_compare_prompts |
| 9 | 09-29 | per-opponent intervals too narrow; Honda "-13.6 hurts" | rounds used as the unit; rounds of a run share its lessons | S | run as the unit (review; dbe6eec) | test_stats run_level |
| 10 | 09-29 | "Qwen holds 50-60% vs chance 9%" | chance drew from empty cells; Qwen reads the verifier's labels | S | table-aware chance: Qwen ~2x (dbe6eec) | test_qwen_lessons verdict |
| 11 | 09-29 | "use more forward" vs Honda passed yet rounds lost | the verifier compares moves within one situation; where a lesson takes her (more time up close) is invisible to it | V | track record (round outcome across runs); shown, not refused | test_track_record |
| 12 | 09-29 | --track switched the loop off vs Honda | refusing on a record whose attribution is confounded (lessons in play together) | V | show, never refuse (merge after review) | test_track_record |
| 13 | 09-29 | early stop fired 27x vs Ryu (6 Ken, 1 Honda) | threshold set on all opponents pooled; Ryu's rounds swing more | V | open | - |
| 14 | 09-29 | Qwen's claims refused as malformed (16) | Qwen copied a line's words ("jumps") into JSON | Q/P | parser reads the words (2ebea64) | test_track_record |
| 15 | 09-29 | Qwen re-proposes refused/known lessons (20-45% duplicates) | Qwen does not use the registry it is shown | Q | open (measured each run) | verdict refused_duplicate_share |
| 16 | 09-29 | a block never rated above "likely fails" | block score = P(blocked), one threshold with P(hit) | R/LV | block scale in code (7d85eb7) | test_block_scale |
| 17 | 09-29 | 485 "nothing_left" decisions vs Honda | avoid lessons ruled out the whole shortlist | R | shortlist skips avoided moves (7d85eb7); 18 left (forward avoided too) | test_block_scale |
| 18 | 09-30 | "use more block_low far away when he attacks" changed nothing | a soft lesson cannot override "likely fails" (by design) | R | none needed: "always" is the hard form | Ryu batch |
| 19 | 09-30 | Qwen never found "use more throw up close" vs Ryu (+42 hp/round when tested) | she threw 13 times in 2,259 decisions (laya-vision rarely rates it high), so the data Qwen reads never showed it; the what-if never tried it | LV (exploration) -> Q | players' tips, verified by A/B (docs/prereg_ryu_expert.md) | fixed_arms_report |
| 20 | 09-29 | a laya fine-tune looked needed (soft lessons "followed 48%") | the metric mixed text laya with laya-vision's ratings; text laya follows its rule 98-100% | S | follows_rule reported (dbe6eec) | test_laya_evidence |
| 21 | 09-30 | the Ken expert batch lost (529 broken log lines) | two ab_memory batches launched in the same second shared one rollouts/ab/<stamp> folder and interleaved their writes | H | folder <stamp>_s<seed>, an existing one refused | test_ab_root |
| 22 | 09-30 | the players' main advice vs Honda (keep away, walk back) cannot be given | System 1 picks only attacks, blocks and forward: no walk back | R (action set) | open - an owner decision | Honda batch |
| 23 | 09-30 | "avoid spinning_bird_kick" vs Zangief lost 44.7 hp/round (players, Qwen and the verifier all backed it) | ruling out her one non-failing move leaves only "likely fails" moves, so the rule falls back to walking in; the verifier compares within a situation and cannot see the substitute | R / V | open | Zangief batch, boundary trace |

Text laya (TL): follows its label rule 98-100% where it was trained; two untrained cases found 2026-09-30 (lessons naming
forward, nothing left) - docs/component_boundaries.md.
