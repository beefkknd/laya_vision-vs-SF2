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

## 1. What is proven *(provisional until P0-1: re-run on clean data with a new seed)*

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
- **Give Qwen** the net-hit-points table (the one `sf2/code_coach.py` computes) and code's own lines as a draft.
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
| counting coach | `sf2/code_coach.py` |
| fixed-advice A/B (proofs 1–3) | `scripts/ab_memory.py` → `rollouts/ab/<stamp>/summary.json` |
| per-round facts, notebook, prompts, runner | `sf2/round_facts.py`, `sf2/notebook.py`, `sf2/notebook_prompts.py`, `scripts/notebook_run.py` |
| advice → move | `sf2/advice.py` (label rule), `sf2/advisor.py`, `runs/text_laya/advice_v1` |
| where the loop loses, memory churn | `scripts/learning_gaps.py`, `scripts/memory_churn.py` |
