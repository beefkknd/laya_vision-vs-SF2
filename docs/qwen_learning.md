# Helping Qwen learn Street Fighter II

*2026-09-28. Chun-Li vs the SNES CPU. System 1 = laya-vision (sees) + text laya (picks, following advice).
System 2 = Qwen (`Jundot--Qwen3.8-27B-oQ4e-mtp` on omlx) writes the advice.*

## 0. Prerequisites before the next Qwen round (code review, 2026-09-29)

Verified first: after the other session's changes (brain panel, video, `seed_memory`, `demo_cheat`, `--fresh`/`--seed`/`--live` in the loop), the suite passes (101 passed, 3 skipped). Then two separate reviewers read the code, one for correctness and one for structure. I checked their main claims myself before listing them here.

### P0: must fix first (they affect whether results can be trusted, or can damage memory)

| # | Problem | Where | Fix |
|---|---|---|---|
| 1 | **Proofs 1–2 may be inflated.** The code coach counted *every* log, including earlier A/B and notebook runs played from the **same savestates with the same seed** as the proof run. 33–70% of its data per opponent came from such runs (Ryu 70%, Zangief 60%, Ken 56%, Honda 53%, Dhalsim 33%). Guile and Blanka had no such runs. | `sf2/code_coach.py` `attacks()`, `scripts/ab_memory.py` | The coach counts only non-test logs (exclude `rollouts/ab`, `rollouts/notebook`, `--fresh` runs). The test uses a **new seed**. The summary records the sources. **Re-run the proofs; until then they are provisional.** |
| 2 | **A/B statistics too generous.** Rounds of one opponent are pooled as if independent. Pairing is by index, so a crashed arm is silently truncated. A verdict still prints when a job failed. | `ab_memory.summarize`, `_ci` | Require equal round counts. Show the CI per opponent, and pool with the opponent as the unit (bootstrap over opponents). No verdict if any job failed. |
| 3 | **The brain panel can write production `memory/`** (breaks the rule that only Qwen writes it). `opp` from the POST body goes unchecked into a file path (`../../memory/...`). There's no Origin check, so a web page could post to it. | `sf2/demo_cheat.py:14-31`, `scripts/brain_panel.py` | Whitelist `opp`, `what` and `do`. Assert `realpath(dest)` is inside `memory_runs/<name>`. Reject non-localhost Origins. Validate the `--fresh` name. |
| 4 | **Memory writes aren't atomic.** The loop can read a half-written file and crash (`JSONDecodeError`), or report System 2's own write as "changed outside", skipping its checks. | `learn_loop.save_memory`, `demo_cheat.apply`, `learn_loop` mtime reload | Write to a temp file, then `os.replace`. Look for outside changes only while System 2 is idle. A failed reload keeps the old memory and logs why. |
| 5 | **Runner robustness.** Ctrl-C in `ab_memory` or `notebook_run` leaves child runs alive. Log files aren't closed. A reply from Qwen that is valid JSON but not an object kills the learn arm. | `ab_memory.py:180`, `notebook_run.py:57,142` | `try/finally` terminates the children. Close the files. The reflect step guards `isinstance(reply, dict)`. |
| 6 | **Small logic slips.** `tries_allowed` uses Python's `round`, so one loss in 4 gives 0 experiments. `notebook.check`'s docstring claims a budget it doesn't enforce (the trim does). | `sf2/notebook.py` | Explicit rule (e.g. `ceil`), and fix the docstring. |
| 7 | **Test gaps** on the new logic. | `ab_memory` stats, `demo_cheat`, loop reload, `_trim` | A failing test first for each of 2–6 (the testing rules). |

**P0 status (2026-09-29): done**, each with a test seen red first.
1. `sf2/eval/logs.py` decides what counts as play data, and both the coach and `learn_loop.history()` use it. Every run now writes `run.json` (memory root, fresh name, seed). The 13 past demo sessions were backfilled `fresh` from `logs/learn_chunli.log` ("earlier rounds: {}"). `ab_memory` and `notebook_run` pick a new seed per run, and `rollouts/ab/<stamp>/run.json` records the seed and the coach's source logs.
2. `sf2/eval/stats.py`: unequal pairs are an error. Pooling uses the opponent as the unit (a two-level bootstrap). A failed job gives NO VERDICT. **Added after the smoke:** fewer than 2 opponents, or fewer than 10 paired rounds per opponent, gives TOO FEW. A 1-round smoke had printed "HURTS".
3. `demo_cheat` checks `what`, `do` and `opp` against fixed lists and resolves every path inside one `memory_runs/<name>`, so symlinks and `..` can't reach `memory/`. The panel's POST checks Host and Origin (it returns 403 otherwise; checked with curl). `--fresh` names are validated before anything is created.
4. `memory.save` writes atomically. `memory.OutsideWatch` picks up outside edits only while System 2 is idle, and a bad file keeps the memory in play and logs why.
5. `sf2/eval/runner.fan_out` stops every child on Ctrl-C or an error and closes the logs. `notebook.apply_reply` ignores a reply that isn't a JSON object.
6. `tries_allowed` rounds up (ceil): one loss in 4 now earns an experiment. The `check` docstring was fixed.

### P1: cleanup, so the code stops growing

**P1 status (2026-09-29): done**, one commit per step (3d7239a .. e75bf0e), each after pytest plus a headless smoke. Where
code moved without meaning to change, the output was compared to a golden taken before the change: report goldens,
the dataset build reproduced byte for byte, same-seed play runs, the audit, and a seeded text-laya training run.
Refactors that found real bugs (each now covered by a test):
- memory_churn listed changes in set order, so the churn report reshuffled on every run;
- the default character list in the audit, eval_outcome and verify_replay took `test_data/advice/` as a character
  and crashed;
- moving modules one folder down would have broken their repo-root paths, so `config.REPO` is now the only place
  that finds the checkout (with an invariant test).
`verify_replay` failed on live-play rows (`KeyError: 'live'`, since a4fe0db: they have no savestate recipe). Fixed in
a separate session (bc3b66d, merged f8569ac): it skips and counts them. Not done: `train.py`'s main is still long, because it has no smoke mode to
prove a split. The review's list below is kept as it was written.

**Delete the legacy teacher/DAgger pipeline**, after tagging it `legacy-dagger`. Nothing live imports it; I checked the import graph.
- Modules: `sf2/{env,teacher,loop,rollout,labeler,cli,ramsearch}.py`.
- Scripts: `scripts/{play_student,play_teacher,collect_teacher,relabel,gate,label_human,record_human,find_ram,check_env}.py` and `dagger_round.sh`.
- Tests: `tests/{test_labeler,test_pipeline,test_actions}.py`, and the FightEnv / ramsearch parts of `fake_mesen.py`, `test_ram.py` and `test_lua_bridge.py`. Keep the bridge-protocol checks.

**Trim the partly live modules:**
- `policy.py`: keep `make_state` only.
- `dataset.py`: keep `save_png`, `read`, `write_jsonl`.
- `ram.py`: keep `Var`, `load_map`, `CLOSE`, `MID`.
- `config.py`: drop `DEFAULT_RAM_MAP`, `DEFAULT_SAVESTATE`, `HOLD`, `NEXT_WINDOW`, `WHIFF_WINDOW`.
- `actions.py`: goes once `policy` and `dataset` no longer use it.

**Remove duplication:**
- **Headless arm runner.** The same steps (savestate → Mesen → character check → seeded rounds → jsonl → close) are copied in `play_system1`, `ab_memory` and `notebook_run`. The `--one` fan-out is copied in 5 scripts. → `sf2/eval/runner.py` (`open_fight()` context manager plus `fan_out()`).
- **Vocabulary.** Range naming exists twice, `RANGES` 3 times, character lists 5 times; state names and bars are scattered. → one `sf2/vocab.py`.
- **Action-log loading** (6 places) → `sf2/eval/logs.py`. **Stats** (`_ci`, `slope`, per-arm summaries) → `sf2/eval/stats.py`.
- **`sys.path` hacks** in 4 scripts → use `scripts/_path.py`.

**Target package layout.** Move one subpackage per commit, rewrite imports with no compatibility shims, and after each step run pytest plus a 1-round headless `ab_memory` smoke.
```
sf2/config.py, vocab.py
sf2/emu/      mesen, headless, boot, vs, ram
sf2/data/     vs_sweep, vs_moves, vs_defense, vs_metrics, frames, dataset, train_data, lora
sf2/system1/  system1, policy, advisor, advice, text_laya, mlx_lora, game_log
sf2/system2/  system2, prompts, checks, qwen, memory, memory_churn, notebook, notebook_prompts, round_facts, code_coach
sf2/eval/     runner, logs, stats
sf2/demo/     demo_cheat          (make_video.py outside the package: it has its own .venv-media)
tests/        mirrors the layout; split test_vs_moves.py (it also tests memory, system2, game_log)
```
Order: tag and delete legacy → `vocab.py` → `eval/logs` + `eval/stats` → `eval/runner`, then the 3 runners become thin CLIs → move emu → data → system2 → system1.

**Machine specifics go into `sf2/config.py`,** each overridable by an environment variable:
- the laya-mlx venv path and `HF_HOME` (`advisor.py:14-15`);
- Mesen's app and `settings.json` paths (`headless.py`);
- the text-laya base model;
- the video output folder;
- the Qwen URL and model, currently duplicated in `make_video.py`;
- the ports, currently scattered over 8 scripts (47990, 47991, 48001, 48401, 48901, 49101, 49501, 8765);
- default paths repeated in 4+ scripts (`runs/all8/best`, `runs/text_laya/advice_v1`, `memory_runs`, `memory_seeds/video`, `out/live/pending.json`).

**Split the big functions:**
- `learn_loop.main` (~158 lines): split into setup, fight loop and shutdown; move `System2` into `sf2/system2/async_runner.py`; stop mutating the globals `MEM_ROOT` and `LOG`.
- `vs_dataset.py` (433 lines): move `build` and `collect` into `sf2/data/`.
- Also: `train_text_laya.main`, `audit_dataset.audit_char`, `system1.play_round`, `boot.boot`.

**Hygiene:**
- Rename `scripts/test_system2_prompts.py` → `check_system2_prompts.py`: it's a live-Qwen harness that pytest may collect. Add `testpaths = ["tests"]` to pyproject.
- Delete the stale `sf2/__pycache__/short_memory*.pyc`.
- Merge `learning_gaps.py` and `memory_churn.py` into one `report.py` with subcommands.
- Archive the probes (`probe_memory_words.py`, `probe_text_laya.py`).

**Fix docs drift:**
- README: the intro, the diagram (through `env.py`), the layout table, the "Day 1" sections and "20 tests" still describe the teacher/DAgger pipeline.
- `pyproject` description: same.
- `PLAN.md`: entirely the old plan. Mark it historical and point to this file.
- `config.py` comments name deleted scripts.

### Owner decisions (2026-09-29)
1. **The memory format follows text laya's fine-tuning.** The contract is what text laya was trained to read: at most 5 plain advice lines in the `sf2/advice.py` grammar (use more / avoid / always + move + range / "when he ..."). However Qwen keeps its knowledge (notebook or playbook), what reaches System 1 is those lines. The formats merge into one: Qwen's notes plus the advice lines. Code coach, notebook plan and short memory all produce lines in that grammar, validated by `advice.read`. A new wording or condition Qwen needs means extending text laya's training data, not loosening the checks.
2. **Legacy pipeline: delete** (after tagging it `legacy-dagger`).
3. **Order:** P0 → P1 cleanup → clean re-run of proofs 1–2 → Stage A.

## 0b. The lesson loop (2026-09-29): Qwen proposes, code verifies

**Design** (Chun-Li, one opponent at a time; `scripts/qwen_lessons.py`, `sf2/system2/{lessons,lesson_prompt}.py`).
A lesson is one text-laya line (use more / avoid + move + range + optionally "when he ..."). After every 3-round game,
code reviews the registry, Qwen proposes at most 2 new claims (seeing the registry with verdicts, her moves overall with
code's classes, and the last game by move, range and his state), and code judges each claim on the rounds inside its
own condition: an avoid at once, a use more tried in play for up to 3 games; registered lessons retire when their
evidence stops holding; at most 5 lines in play. Qwen only proposes; code keeps every book.

**How we got here.** A first loop gave Qwen the whole job (pick and keep 5 lines from a table). Q1 (identify) passed
20/20 per opponent once code supplied the classes, but Q2 (keep over games) needed six prompt adjustments, each fixing
one case and exposing another; the last two moved failures around (replays 18/20, 0/20). Every failure was bookkeeping
(counting lines, an avoided move looking cheap, dropping a line when its class changed), which code does exactly. The
trace also found a System 1 leak: a lesson's move reached the shortlist, and its words leaned text laya's ties, in
ranges the lesson did not name (fixed: `advisor.applicable`; c.mk up close 50 -> 28 on the same seed).

**Results** (10 games x 3 rounds, paired no-advice arm, same seed; history = her play data vs him):

| Run | Invariant violations | Qwen claims that hold | Random claims that hold | Won loop / none | Hit points / round vs none |
|---|---|---|---|---|---|
| Ken 1 | 0 | 43% | 12% | 6 / 2 | +22.9 [−13.9, +59.7] |
| Ken 2 | 0 | 38% | 12% | 9 / 3 | **+42.9 [+14.8, +70.9]** |
| Ryu 1 | 0 | 50% | 10% | 11 / 5 | **+42.6 [+5.7, +79.5]** |
| Ryu 2 | 0 | 64% | 10% | 10 / 8 | −13.4 [−51.2, +24.4] |
| Honda 1 | 0 | 50% | 12% | 0 / 1 | +12.3 [−10.0, +34.6] |
| Honda 2 | 0 | 67% | 9% | 2 / 1 | −6.7 [−31.5, +18.1] |

Per opponent (60 paired rounds): Ken +32.9 [+9.8, +56.0], Ryu +14.6 [−12.6, +41.7], Honda +2.8 [−13.9, +19.5];
pooled with the opponent as the unit +16.7 [−2.1, +35.9]: not shown.

**Reading.**
1. *Can Qwen propose what is worth learning?* Yes: its claims hold on the data 38–67% of the time vs 9–12% for random
   claims over the same moves, ranges and situations, in every run. Most use a condition code does not enumerate
   ("avoid c.hp up close when he jumps", "use more lp up close when he is stunned").
2. *Does the loop keep the good and drop the bad, game after game?* Yes, by construction and checked: 0 violations in
   66 updates; lessons were registered, and "use more" lessons retired when their evidence faded. It also found good
   moves by testing them ("use more mp at mid range" +7.5, "use more c.mk at mid range when he stands" +5.1).
3. *Does it win more?* Against Ken, yes; against Ryu and Honda, not shown. The lessons are mostly "avoid" lines: they
   stop losses but do not tell her what to do instead.
4. Lessons are reproducible: 12 were registered in both runs of their opponent. All are in `lessons/chunli.json`
   (`python scripts/qwen_lessons.py --export`), for reuse.

## 0c. Two views, a relative yardstick (2026-09-29), Ken only so far

**Why.** In the 12 lesson-loop arms she took 92% of her damage while attacking, and chose a block in **0** of 3853
decisions where he attacked (laya-vision rates blocks "likely fails"; a "use more" lesson does not override that, an
"always" lesson does). A fixed-lesson A/B (`ab_memory.py --fixed`, Ken, 30 rounds): "always block_low when he attacks"
-> 120 blocks, 78 blocked cleanly, 4.3 taken per block vs ~10 otherwise, yet total taken unchanged (166 vs 169): the
blocks replaced cheap walking, not her attacks into his (15.6 taken each; most damage is both starting at once, when
he is still standing at her decision).

**Changes.** Each claim is judged against her own average in the same situation (so a block at -4 is "better" when
her average there is -10); a third kind "always"; Qwen sees an ATTACK view (her attacks vs her average there) and a
DEFENSE view (damage taken by cause: traded / punished / stuffed / caught, and what she chose when he attacked) plus
her record, and proposes one claim per view; exceptions (a narrower opposite claim) are judged instead of refused;
a "use more" / "always" may name a move she never used.

| Ken run | Loop version | Qwen holds (random) | Lessons | Hit points vs none | Damage taken vs none |
|---|---|---|---|---|---|
| 1 | one view, absolute (0b) | 43% (12%) | 6 | +22.9 [−13.9, +59.7] | – |
| 2 | one view, absolute (0b) | 38% (12%) | 7 | +42.9 [+14.8, +70.9] | – |
| 3 | two views, relative | 29% (13%) | 3 | +2.5 [−28.6, +33.7] | −6.7 [−2.9, +16.2] |
| 4 | two views, relative | 44% (12%) | 3 | +9.6 [−25.5, +44.7] | −4.9 [−5.9, +15.6] |
| 5 | + exceptions, untried moves | 50% (13%) | 3 | +18.8 [−5.5, +43.2] | **−5.5 [+1.9, +9.1]** |
| 6 | + exceptions, untried moves | 56% (12%) | 3 | +0.7 [−28.8, +30.1] | −2.9 [−5.3, +11.1] |

(damage-taken column: how much less she took per round than the no-advice arm; interval of that difference)

**Trend.** Qwen's claims keep holding at 3-5x chance from both views; damage taken tends down (significant once);
hit points are not better than the one-view loop: the relative yardstick registers about half as many lessons (it
rejects avoids that are bad in absolute terms but no worse than her other options there). Qwen proposed a block once
in 20 defense claims (as "use more", which cannot override the rating) and never "always"; the defense view shows raw
damage, so Qwen twice re-proposed "avoid hp up close when he jumps" (punished for 221) although hp is her best option
there (-1.0 vs -10.5, an anti-air).

## 0d. "What if?" when she is stuck (2026-09-29), 3 opponents x 2 seeds

**Change.** When she lost (or won) each of the last 3 games and no registered lesson changed, Qwen gets a third slot,
"what_if": losing -> a move she rarely or never used where she keeps losing; winning -> a different option where she
wins, to confirm. Its own test slot, judged like any claim. The defense view now says, per damage row, whether that
move is better or worse than her other moves there, and names the defensive moves she never used.

| Run | Qwen holds (random) | Stuck -> what-ifs | Registered | Won loop / none | Hit points vs none | Taken less vs none |
|---|---|---|---|---|---|---|
| Ken 1 | 41% (10%) | 1 | 5 | 1 / 3 | +14.6 [−11.4, +40.7] | +1.4 [−4.2, +6.9] |
| Ken 2 | 28% (8%) | 3 | 4, incl. **always block_high up close when he jumps** | 4 / 1 | **+67.3 [+40.1, +94.5]** | **+16.9 [+2.4, +31.4]** |
| Ryu 1 | 53% (10%) | 0 (never stuck) | 7 | 10 / 5 | +21.7 [−18.6, +62.1] | +0.5 [−17.6, +18.6] |
| Ryu 2 | 54% (10%) | 0 | 7 | 8 / 9 | −1.9 [−37.4, +33.6] | +0.2 [−15.1, +15.6] |
| Honda 1 | 41% (12%) | 3 | 3 | 2 / 2 | −6.9 [−42.2, +28.4] | −3.1 [−12.5, +6.3] |
| Honda 2 | 36% (7%) | 3 | 2 | 1 / 2 | +13.6 [−8.0, +35.1] | −4.1 [−8.7, +0.5] |

Per opponent (60 paired rounds): **Ken hit points +41.0 [+21.1, +60.8], taken 9.1 less [+1.2, +17.1]**; Ryu +9.9
[−16.9, +36.7]; Honda +3.4 [−17.3, +24.0]. Pooled with the opponent as unit: +18.1 [−3.4, +40.8], not shown. 0
invariant violations in every run.

**What the what-ifs did.** Stuck runs made Qwen reach for defense with "always" for the first time ("always back /
crouch / jump_back at mid range when he attacks", "always block_low ...") and one became a verified lesson (block_high
vs Ken's jump-ins). But back, crouch and jump_back got **0 tries**: System 1 only picks among laya-vision's rated moves
(attacks, blocks) and walking in, so advice about other movement cannot be followed. Fixed: the loop offers and
accepts only `system1.choices(me)`.

**Second round, with the moves fix** (`system1.choices`: attacks, blocks, walking in):

| Run | Qwen holds (random) | What-ifs -> result | Registered | Won loop / none | Hit points vs none | Taken less |
|---|---|---|---|---|---|---|
| Ken 1 | 32% (8%) | 2: 1 rejected, 1 in test | 4 | 3 / 1 | +10.7 [−15.5, +36.8] | +2.7 [−1.8, +7.3] |
| Ken 2 | 61% (8%) | 0 (never stuck) | 4 | 5 / 0 | +30.9 [−0.7, +62.6] | +9.6 [−1.6, +20.7] |
| Ryu 1 | 53% (11%) | 0 | 8 | 11 / 8 | **+46.0 [+11.9, +80.1]** | **+18.0 [+0.3, +35.6]** |
| Ryu 2 | 50% (11%) | 3, in test at the end | 6 | 5 / 7 | −5.4 [−51.6, +40.8] | −13.1 [−34.2, +8.0] |
| Honda 1 | 74% (12%) | 1 refused (already known) | 5, incl. **always block_high at mid range when he attacks** | 2 / 4 | +11.6 [−17.5, +40.7] | +1.0 [−7.3, +9.3] |
| Honda 2 | 41% (11%) | 1: **always block_low at mid range when he attacks -> registered** | 4 | 1 / 0 | −0.1 [−26.9, +26.8] | −4.7 [−9.3, −0.2] |

Round 2 pooled (opponent as unit): hit points **+15.6 [+0.0, +31.0], helps**. **Both rounds, 12 runs, 120 paired
rounds per opponent: Ken +30.9 [+16.6, +45.2] (taken 7.6 less [+2.7, +12.6]), Ryu +15.1 [−4.7, +34.9], Honda +4.6
[−9.6, +18.8]; pooled +16.8 [+2.3, +32.4]: helps.** 0 invariant violations in all 12. Three "always block" lessons
are now verified (Ken's jump-ins; Honda at mid range, high and low), two of them found by a what-if. laya-vision is
left as is: its block score (P(blocked)) is low by construction (70 of 336 block training rows ended "blocked"), so
blocks read "likely fails" and only an "always" lesson makes her block.

## 0e. Lock `lesson_loop_v1` (2026-09-29): the 0d round 2 frozen, git tag `lesson-loop-v1`

`scripts/lock.py make` copied (APFS clones) and hashed the inputs of the 6 round-2 runs of 0d into
`locks/lesson_loop_v1/` (manifest committed, copies git-ignored): laya-vision `runs/all8/best`, text laya
`runs/text_laya/advice_v1`, 3 savestates, 9 play logs (her history: identical rows to the live read, Ken 2552, Ryu
2259, Honda 2041); the Qwen model, Hugging Face base revisions and ROM hash are recorded. `scripts/lock.py verify` checks
it; `qwen_lessons.py --lock lesson_loop_v1 --run N` repeats run N from the copies only (refused if anything changed),
output under `rollouts/locked/` (never play data).

Repeat of run 0 (Ken, seed 11688), `scripts/compare_runs.py`:

| | original | from the lock |
|---|---|---|
| no-advice arm (30 rounds, 649 decisions) | | byte-identical |
| loop: rounds 0-17 | | identical |
| loop: Qwen's reply after game 1 | | differs (Qwen at temperature 0 is not repeatable) |
| lessons in play differ from | | game 6 |
| registered at end | 4 avoid | the same 4 + "use more hp up close" |
| hp/round vs none | +10.7 [-15.5, +36.8] | +13.7 [-12.7, +40.1] |
| Qwen claims that hold | 32% | 37% |

So the lock reproduces the game, laya and the code exactly; the only source of drift is Qwen. A repeat is comparable
in distribution, not byte for byte.

## 0f. The character prompt (2026-09-29): one opponent, one moment

Owner: "what opponent specific move always win against me, or, if I see X, what works, what doesn't work... the
prompts now narrow down from general game plays to specific turn, specific character." No laya is retrained (owner):
where the grammar falls short, the gap is written down here and Qwen compensates.

`qwen_lessons.py --prompt character` (`sf2/system2/character_prompt.py`; the default `views` is the 0c/0d prompt the
lock ran with). After each game Qwen sees, for this opponent only:

- **his threats** - what he did that hurt her, by her range at the decision: he jumps in / attacks on the ground / hits
  her from afar (damage with no attack of his in the window: a fireball already on the way), damage and share, and
  the moves of hers they caught;
- **if you see X** - per situation of his when she decides (what he is doing x range, the 5 most damaging): her
  answers ranked against her other moves there (works / fails / unclear / too few) and the moves she never tried there;
- the record, the registry, refusals, and "what if?" when stuck, as before.

It answers `{"answer": use more / always, "stop": avoid}`; every claim must name what he is doing (a claim without
"when" is a problem, not a claim). Verified exactly as before; the chance baseline in the verdict also always names a
"when".

### The gap: his specific moves

| What the owner asks | What exists | Gap |
|---|---|---|
| which of his moves beat her | his state per frame: stand, crouch, jump, attack, guard, hit stun; in the air or not | the CPU's specials are logged as a plain attack (state 0C "special" never appears in 6,852 rows vs Ken/Ryu/Honda); no move id; his fireball slot (0x1050) is in the RAM map but not in the logs |
| "if I see a fireball / an uppercut" | text laya reads "when he jumps / crouches / attacks / stands / is stunned" + a range | no word for a specific move; adding one needs new text-laya training data (not done, owner) |

Compensation on the Qwen side: the prompt tells Qwen to say his moves by where they happen - a fireball is "when he
attacks far away" (or at mid range), an uppercut or a jump-in kick close by is "when he jumps" / "when he attacks up
close". Coarse: at mid range "when he attacks" mixes a fireball with a sweep. Cheapest way to narrow it later without
touching laya: log the fireball slot and prove his move-id address (probably near 0x0F80, by symmetry with hers at
0x0D80; unverified), so the views (not the grammar) can name his move.

What the frozen play data already shows (share of her damage): Ken 59% from jump-ins; Ryu and Honda 56-57% from
ground attacks, which cost her 11.7-12.3 per hit (Ken's 5.2).

### First result: the lock's 6 seeds, character vs two views

`qwen_lessons.py --lock lesson_loop_v1 --run N --prompt character`, N = 0..5, output
`rollouts/locked/lesson_loop_v1/*_character`. All 6 no-advice arms byte-identical to the lock's, so both prompts are
paired against the same baseline (60 rounds per opponent). 0 invariant violations, 0 failed jobs.

| hp per round vs no advice | two views (lock) | character | character - two views |
|---|---|---|---|
| Ken | +20.8 [+0.3, +41.3] | **+51.8 [+30.6, +72.9]** | **+31.0 [+8.8, +53.1]** |
| Ryu | +20.3 [-8.9, +49.5] | +17.1 [-10.6, +44.9] | -3.1 [-29.7, +23.5] |
| Honda | +5.8 [-13.9, +25.5] | -10.6 [-27.3, +6.1] | -16.4 [-33.3, +0.6] |
| pooled (opponent as the unit) | +15.6 [+0.0, +31.0] HELPS | +19.4 [-11.3, +50.4] not shown | +3.8 [-20.6, +30.5] not shown |

Qwen claims that hold: character 33 / 40 / 75 / 67 / 54 / 47% (two views 32 / 61 / 53 / 50 / 74 / 41%; chance 6-11%).
Ken: both seeds registered the same 4 lessons around his top threat (use more hp, avoid sweep and c.hp up close when
he jumps; avoid spinning_bird_kick at mid range when he stands); damage taken 13.8 per round less (two views: 6.2).

Honda seed 11703 (-31.0): the loss sits in games 4-6 (-118, -76, -65 per round), while "always block_low at mid
range when he attacks" was on test next to two "use more forward at mid range" lessons (walking in: 732 decisions vs
421 without advice). The block itself measured -2.0 per decision vs her -4.5 there (rejected after 3 games as not
clearly better; "use more block_low" there was then registered as clearly better). The per-decision yardstick saw no
harm while the rounds were lost - it only counts damage up to her next decision. Confounded (the lines changed
together), not attributed. Open: a test is not stopped early however badly the rounds go.

## 0g. How players describe the game (2026-09-29): `--prompt character_fgc`

Owner: see whether players have a standard way to describe SF2, to phrase Qwen and (later) text laya. Research notes
with sources: `docs/sf2_world_warrior_notes.md` (arcade World Warrior sources; the SNES port may differ).

`--prompt character_fgc` = the character prompt (0f) plus, in `sf2/system2/character_prompt.py`:
- each situation of his named the way players do (`term()`): jumping = anti-air, attacking far away = his zoning
  (fireball), attacking closer = his attack (block or beat it), standing at mid/far = footsies, standing up close =
  throw range, stunned = punish;
- a primer, framed as ideas for code to test: Chun-Li's WW tools (normals, walk speed, long throw; slow unsafe specials;
  anti-airs standing mk/hk/hp, c.mk vs far jumps), no reversals / throw escape / meter, the CPU reacting to the move she
  commits to and getting more aggressive as the clock runs down, and notes on this opponent only (Ken/Ryu: Shoryuken,
  Hurricane Kick, Hadoken; Honda: Headbutt punishable on block, jab beats it, keep-away).
The plain `character` prompt is byte-identical to before (checked on the lock's play data, 3 opponents x 2 modes).

Agreement with what the loop already verified: "avoid spinning_bird_kick" is registered in most runs (players: slow,
unsafe in WW); Ken's "use more hp up close when he jumps" is an anti-air.

### For a later laya fine-tune: general gaps only

Owner (2026-09-29): a laya fine-tune (text laya or laya-vision) is for general game play, never to improve one or a
couple of characters; Qwen is the second brain and does the character learning. So what counts as fine-tune evidence
is a gap that holds across opponents (counted per opponent, pooled), and a grammar word only when it is a general game
concept that applies to every opponent (a projectile, a missed attack, a knockdown) - never a named character's move.
His specific moves (`opp_move`, sf2/system1/opp_moves.py) feed Qwen's views; they are not laya training data.

Candidate words players need that the grammar lacks (all general):

| Player's term | Needed in the logs | Needed in the grammar |
|---|---|---|
| whiff punish / punish a blocked special | his state split: starting vs recovering from an attack | "when he misses", "after he is blocked" |
| zoning, fireball | projectile slot 0x1050 (read per frame, not logged) | "when a fireball is coming" |
| tick throw, pressure | her previous action / "he blocks" (state guard exists, `opp_doing` maps it to standing) | "when he blocks" |
| throw loop, meaty, wake-up | a knocked-down / getting-up state | "when he gets up" |
| jump-in, cross-up | none (System 1 cannot pick a jump) | a jump move in `choices` |
| CPU aggression late in the round | `clock` is logged per decision | "late in the round" |

Evidence so far (`scripts/laya_evidence.py`, 31 runs; still to be split per opponent to show it is general): soft lessons are followed 48% vs 10% without advice, 65% when
laya-vision has the move in its top 3 vs 41% when not; hard 96%, avoid 98%. Qwen named his specific move in 2 of 618
claims (both "fireball", Ryu).

## 1. What is proven

### Clean re-run (2026-09-29): proofs 1–2 are NOT reproduced

Run `rollouts/ab/20260929-084920`: seed 86160 (new); the coach built from play data only (13 logs: no A/B, notebook or
demo runs); the same arms, opponents and 30 rounds; the P0 statistics (pairs must line up, opponent as the unit).

| Arm | Opponents | Paired rounds | vs none (hp / round) | 95% CI, opponent as unit | Verdict |
|---|---|---|---|---|---|
| code_short | 5 (no play data vs Guile, Blanka) | 150 | +15.6 | −13.4 to +43.3 | not shown |
| code_playbook | 7 | 210 | +9.6 | −11.0 to +29.8 | not shown |

Per opponent, vs none: code_short Ryu +2, Honda +15, Ken +59, Zangief +25, Dhalsim −24; code_playbook Ryu +20,
Honda +39, Ken +21, Zangief −7, Dhalsim −27, Guile +3, Blanka +17.

**Which change did it** (same data, both statistics):

| Run | Arm | Rounds as independent | Opponent as unit |
|---|---|---|---|
| 2026-09-28 (leaky coach, seed 0) | code_short | +28.8 [+14.3, +43.3] | +28.8 [+1.3, +64.0] helps |
| | code_playbook | +31.8 [+18.4, +45.1] | +31.8 [+1.5, +63.5] helps |
| 2026-09-29 (clean, seed 86160) | code_short | +15.6 [+1.0, +30.2] | +15.6 [−13.4, +43.3] not shown |
| | code_playbook | +9.6 [−4.5, +23.6] | +9.6 [−11.0, +29.8] not shown |

So the leak roughly doubled the effect: with clean data it is about half, and too small to show over 7 opponents ×
30 rounds. The stricter statistics alone would still have passed the leaky run. Counted advice may help a little
(the mean is positive for both arms and most opponents), but that is **not proven**; against Dhalsim both arms hurt.
Everything below this section was written from the 2026-09-28 run and is kept as it was.

### Three seeds pooled (2026-09-29): still not shown

Pre-registered before running: 2 more seeds (31337, 52718) with the same frozen memories, pooled with seed 86160
per opponent (90 paired rounds each), opponent as the unit (`ab_memory.py --pool`, runs `rollouts/ab/20260929-084920`,
`-111807`, `-111809`).

| Arm | Opponents | Paired rounds | vs none | 95% CI | Verdict |
|---|---|---|---|---|---|
| code_short | 5 | 450 | +12.4 | −7.6 to +36.8 | not shown |
| code_playbook | 7 | 630 | +14.1 | −1.1 to +29.4 | not shown |

Per opponent, vs none (hp / round): code_short Ryu −8, Honda +4, Ken +56, Zangief +19, Dhalsim −9; code_playbook
Ryu +14, Honda +14, Ken +44, Zangief +15, Dhalsim −18, Guile +6, Blanka +24. Rounds won of 90, none → playbook:
Ryu 32→32, Honda 10→20, Ken 10→30, Zangief 16→31, Dhalsim 32→25, Guile 46→54, Blanka 11→14.

**Reading:** the playbook (general knowledge of herself) is positive against 6 of 7 opponents and the interval only just
touches 0; the short memory (this opponent's rounds) is driven by Ken and hurts against Ryu and Dhalsim. Tripling the
rounds narrowed little: the spread is **between opponents**, not round noise, so more rounds of the same 7 will not
settle it. Counted advice is a small, opponent-dependent effect, not yet a general "helps".

### The 2026-09-28 run (superseded: the coach learned from the test's own replays)

Every test is headless and paired. The opponent is locked by a savestate. Every arm plays the same rounds with the same start delays; only the advice differs. The score is hit points per round (damage dealt minus taken), compared with "Advice: none" round by round. The criteria were fixed before the run: an arm **helps** when the 95% confidence interval of its pooled difference is above 0.

| Arm | What the advice is | Paired rounds | vs none (hit points / round) | 95% CI | Verdict |
|---|---|---|---|---|---|
| code_short | written by code from her rounds against **this** opponent | 150 | **+28.8** | +14.3 to +43.3 | **helps** |
| code_playbook | written by code from her rounds against **other** opponents | 210 | **+31.8** | +18.4 to +45.1 | **helps** |
| qwen | System 2's current short memory | 150 | +1.1 | −14.2 to +16.4 | not shown |
| qwen − code_short | Qwen vs counting, same rounds | 150 | **−27.7** | −42.0 to −13.5 | **worse** |

Rounds won out of 30 (run `rollouts/ab/20260928-091745`):

| Opponent | none | code_short | code_playbook | qwen |
|---|---|---|---|---|
| Ken | 3 | 17 | 16 | 4 |
| Zangief | 6 | 6 | 14 | 4 |
| Ryu | 6 | 10 | 10 | 0 |
| Dhalsim | 8 | 9 | 4 | 13 |
| E. Honda | 1 | 1 | 2 | 0 |
| Blanka (never met) | 3 | – | 11 | – |
| Guile (never met) | 16 | – | 16 | – |

**So:**
1. **A short memory works:** advice from her own numbers wins rounds.
2. **The playbook helps:** general knowledge of herself carries over, even to opponents she has never met.
3. **The path from memory to action works:** text laya follows advice 98% of the time.
4. **Qwen is the weak link.** Its advice is no better than none, and clearly worse than simple counting over the same logs.

Other runs that point the same way:
- Qwen revising the short memory after every lost round: 45–71% of the memory changes per rewrite.
- The per-round notebook against Ryu came out −27 hit points per round vs none (CI −52 to −2). All 20 notebook lines were problems; none said what works.

## 2. Why Qwen fails: the evidence

What Qwen told her against Ryu, next to what her logged rounds say (net hit points per try):

| Qwen's lesson | The counts | Verdict |
|---|---|---|
| avoid c.mk at mid: it misses | c.mk@mid: 405 tries, hits 37%, **+2.2 net** | wrong: a winning move, banned for missing |
| avoid c.hp at mid: it misses | c.hp@mid: 109 tries, hits 39%, **+2.6 net** | wrong, same reason |
| use lp when he jumps | lp@close: 574 tries, **−3.1 net** | a losing move, advised |
| stop spinning_bird_kick | −6.5 at mid, −18.4 up close | right |
| (nothing) | throw@close: 100 tries, **+10.0 net**, her best move | missed |

Against Ken the pattern repeats: "avoid c.hp at mid: it whiffs", while c.hp@mid is net positive.

**The causes, mostly in what we gave Qwen:**
1. **Wrong yardstick.** System 2's claim words (`sf2/system2_prompts.py` `CLAIMS`) are *lands / whiffs / blocked / punished*: rates, not results. A move that misses 60% of the time but wins more than it loses is "whiffs", so Qwen bans it. Nothing in its input says "net hit points".
2. **Negativity.** The digest leads with failures. Its lessons are 4 avoids to 1 use; the notebook was 20 problems and 0 strengths. Avoiding moves makes her passive, and passivity loses (with advice she attacked 59% of decisions, without it 84%).
3. **Recency.** Revising after every lost round means it learns from the last 60 seconds. It flip-flops: "c.mk at mid" went use → avoid → use 4 times in 4 minutes.
4. **Counting in words.** An LLM estimating rates from a long digest is worse at arithmetic than code, and it doesn't know which differences are noise.

The comparison that matters: plain counting (sort by net, keep the top 3, avoid the bottom 2) beat Qwen by 28 hit points per round. Qwen doesn't need to beat nothing; it needs to add something on top of counting.

## 3. Principles

1. **Code counts, Qwen reasons.** Every number (tries, hit %, punished %, net hit points per move × range × opponent, and later × what he was doing) comes from code. Qwen never estimates a rate.
2. **Start from the counting baseline.** Qwen gets code's lines as the starting advice. A change is allowed only with a stated reason from the table (a condition, a habit, an experiment), and code checks the reason against the numbers.
3. **One yardstick: net hit points.** A lesson that calls a net-positive move bad, or a net-negative move good, is rejected by code.
4. **Balanced.** The self notes and the plan must contain what works, not only what fails.
5. **Stable by design.** The long-term notes move slowly (the change budget). The per-round plan changes at most 1–2 lines, plus experiments.
6. **Prove every step with the paired A/B,** criteria written first. Keep what beats the baseline; drop what doesn't.

## 4. Plan: three stages, each with its own test

### Stage A: Qwen with the right inputs, fixed advice
- **Give Qwen** the net-hit-points table (the one `sf2/system2/code_coach.py` computes) and code's own lines as a draft.
- **Ask for** at most 5 lines. Each line either keeps a draft line or replaces it with a reason naming a row of the table.
- **Code checks** that every line's direction matches the sign of its row's net, and that at least 2 lines are "use more".
- **Test:** a `qwen_table` arm in `scripts/ab_memory.py`, same opponents and rounds. **Pass:** not worse than code_short (CI upper bound ≥ 0). **Goal:** better than code_short.
- *If it fails even here, the honest outcome is to let code write the advice and give Qwen a narrower job (Stage B only).*

### Stage B: add what counting can't
Counting has a blind spot: it averages over what the opponent is doing. Qwen can add:
- **conditions:** "when he jumps, use X". The table gains a column for what he was doing (jumping, attacking, standing), which the round facts already record;
- **opponent habits:** he jumps a lot at far, he attacks when I walk in;
- **the unmet opponent:** which of my general strengths to lead with.

**Test:** a `qwen_plus` arm must beat code_playbook (CI above 0) on the same rounds.

### Stage C: learning over rounds
- **The notebook loop** (`scripts/notebook_run.py`), starting from the code baseline instead of an empty book, with the Stage A/B checks.
- **Every round:** facts → Qwen updates the notes (budget) and the plan (1–2 line changes).
- **Experiments are hypotheses:** "try throw up close: expect a positive net". The next round's facts say whether it held, and a confirmed experiment enters the notes.
- **Test:** against a locked opponent over 40+ rounds, the learn arm must beat a fixed code_short arm in the last third of the rounds (paired, CI above 0), and its hit points must rise over rounds more than the control's.

### Stage D: back into the arcade loop
Only after C passes. `scripts/learn_loop.py` uses the notebook, the code baseline and the async System 2, and the churn and gap reports keep watching it.

## 5. Measurement protocol

- **Harness:** `scripts/ab_memory.py` (fixed advice, many opponents) and `scripts/notebook_run.py` (learning over rounds).
- **Pairing:** same savestate and same seed in every arm. An arm with no advice replays the control exactly; checked in the smoke run.
- **Sample size:** 30 rounds per arm per opponent gives a pooled CI of about ±15 hit points. That's enough to see ±25; smaller effects need more rounds.
- **Criteria written into the script docstring before running.** The script prints the verdict.
- **Product-LLM rule:** Qwen's prompts get live tests with repeats and variations (PASS/FAIL/FLIP), never retry-until-green.

## 6. Open questions and risks

- **Selection bias.** The counts come from moves System 1 chose in the situations where it chose them. "throw@close is +10" means "when laya thought a throw fit". Forcing it everywhere may not keep +10. The A/B measures the real effect, so trust it over the table.
- **Dhalsim is the exception:** the playbook hurt there (−28) and Qwen helped (+17). Opponent-specific knowledge seems to matter more against him.
- **Spinning bird kick against Ryu is negative in the counts**, yet the no-advice arm won rounds leaning on it. Big swings can win rounds even with a negative average. Maybe round wins and net hit points should both be tracked.
- **E. Honda:** nothing helps yet (1–2 rounds won of 30 in every arm). Maybe a defence problem (she almost never blocks) that advice about attacks can't fix.
- **Does Stage B need Qwen at all?** Code could count by what he was doing too. Qwen's clearest value may be only experiments and reasoning about opponents she has never met. Stage A/B results will tell.

## 7. Files

| What | Where |
|---|---|
| counting coach | `sf2/system2/code_coach.py` |
| what counts as play data (test runs excluded) | `sf2/eval/logs.py` (`run.json` per run) |
| A/B statistics (pairs, opponent-level bootstrap, verdicts) | `sf2/eval/stats.py` |
| fixed-advice A/B (proofs 1–3) | `scripts/ab_memory.py` → `rollouts/ab/<stamp>/{run.json,summary.json}` |
| per-round facts, notebook, prompts, runner | `sf2/system2/{round_facts,notebook,notebook_prompts}.py`, `scripts/notebook_run.py` |
| advice → move | `sf2/system1/advice.py` (label rule), `sf2/system1/advisor.py`, `runs/text_laya/advice_v1` |
| System 2 beside the game | `sf2/system2/async_runner.py` (used by `scripts/learn_loop.py`) |
| where the loop loses, memory churn | `scripts/report.py gaps`, `scripts/report.py churn` |
