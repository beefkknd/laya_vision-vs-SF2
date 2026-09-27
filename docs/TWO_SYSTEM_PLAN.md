# Two-System Fighter: plan

A Street Fighter II player built like a human player:

- **System 1 (the hands):** laya-vision, the trained r2 checkpoint, picks a move every 4 frames from what it sees.
- **System 2 (the mind):** an adviser that reviews what went wrong and writes **situation rules** into a memory.
  The hands consult the memory before every move. The first adviser is Claude (the coding agent). Qwen replaces
  it last. You can override either.

**Hypothesis:** rules in a memory, with no retraining, make r2 play better against an opponent it was not trained
for. Then: an adviser that writes those rules during play makes it improve match by match.

**End goal (the demo):** start weak, then watch it get better live.
- **System 1** plays arcade Street Fighter II continuously (the Mesen windows and console from `scripts/arcade.sh`).
- **System 2** reviews moments and refines the memory *while System 1 keeps playing*.
- **One screen shows all three:** System 1 playing; System 2's learning (which rules it wrote, why, and whether they
  fired and helped); and a curve of win rate against each opponent over time.

The live demo *shows* learning. The verdicts (CP2, CP3) still come from frozen runs on eval openings.

**Status:** planning, v3 (v2 plus the two reviews in §6). Branch `two-system`. Nothing runs until you approve.

## Checkpoints

Each checkpoint is a result you can see. The bars are set before any run.

- [x] **CP1: we know the gap (Phase 1).** On the eval openings, measure against Ryu:
  - r2 alone (arm A), the starting point;
  - the scripted teacher, the ceiling for a playbook built from its tactics.

  If r2 already wins nearly everything, pick a harder opponent.
- [ ] **CP2: Chun-Li beats Ryu (Phase 2).** With the frozen playbook (arm C):
  - she wins most matches;
  - her net damage per round has a 95% confidence interval above 0;
  - she beats r2 alone by the +15 effect size, paired;
  - C beats D, so laya itself still matters.
- [ ] **CP3: she learns (Phase 5).** A memory written during play, starting empty, beats r2 alone on openings it never
  saw, in at least 2 separate sessions: first with Claude as the adviser, then with Qwen.

- [ ] **CP4: a second character.** When Chun-Li (r2 plus memory) wins at least 95% of eval matches against Ryu (38 of
  40), start a second playable character and train it against Ryu (§7-style tutorial: its own action set, verified
  special-move macros, a base checkpoint, then the same two-system loop).

**Track B checkpoints (the all-character System 1):**
- [x] **CPB0: the Studio is a verified machine** (2026-09-27: 30/30, and the replay identical in every field and all 41 rounds). The ROM suite passes 30/30 there, and the Dhalsim replay reproduces
  +93.1 exactly.
- [ ] **CPB1: every character is playable.** For each of the 8 characters as player 1:
  - a fight-start savestate vs Ryu that passes the same 5 checks as Chun-Li's;
  - every move in its action set, specials included, verified on the ROM.
- [ ] **CPB2: every character has labels worth copying.** For each character, its label source (a teacher, filtered
  exploration, or System 2) beats random play vs Ryu by at least +15 net damage per round, paired, on the eval
  openings. A character without this cannot be trained yet.
- [ ] **CPB3: one checkpoint plays all 8.** On the eval openings vs Ryu, each character:
  - beats random play by at least +15, paired;
  - reaches at least half of its label source's margin over random.

  And Chun-Li is no worse than r2: the paired interval against r2 must reach above −15.

What is known now:
- The teacher beat Ryu during claw's arcade exploration (9 opponents in a row, mostly 2–0).
- The scripted teacher vs Ryu (measured 2026-09-27, eval openings): **40 of 40 matches, 87% of rounds, +89.3 ± 5.8**
  net damage per round. The 95% match bar for CP4 is reachable: the teacher's tactics already clear it.
- laya r1 has both beaten and lost to Ryu (1 win and 2 losses, far too few to mean anything).

## Two tracks

- **Track A (this plan's phases, Mac Pro):** build and prove the two-system loop with Chun-Li, the character whose
  System 1 already exists.
- **Track B (the real System 1, Mac Studio):** a laya checkpoint that plays **every playable character** and
  understands their moves. The Chun-Li loop is the setup; the all-character checkpoint is the real task (owner,
  2026-09-27). laya's question lists its options as text, so one checkpoint can take a different move list per
  character.

### Track B tasks (Mac Studio, `~/work/hobby/laya_vision_vs_SF2`)
- [x] **B0 setup:** repo, a venv pinned to the Mac Pro's exact package versions, the same Mesen build and settings,
  the ROM, the savestates, the base model. The ROM suite passes 30/30 and the Dhalsim replay reproduces +93.1.
- [x] **B0 measure:** training 138 s step 0 to final and 6.2 s per eval (Mac Pro 163 s and 10 s); play 36.2 decisions/s on 4 workers, 11.0 on 1. Measured with omlx loaded but not answering; a run with Qwen answering is still to do.
- [x] **B1 pick any character:** boot and select for each of the 8 playable World Warrior characters (Ryu, E. Honda,
  Blanka, Guile, Ken, Chun-Li, Zangief, Dhalsim) as player 1; a fight-start savestate per character vs Ryu, verified
  (2026-09-27, Studio). `scripts/make_savestate.py --out states/p1_<me>_vs_<opp>.state --me <me> --opp <opp> [--via ken]`.
  All 8 pass the 5 checks (start at the first controllable frame; note words for both; a match plays to its end;
  walls; Ryu's (Ken's) fireball in the note's field). Found on the ROM:
  - Each character has only 3-4 possible first opponents, never itself. Ryu is never the first opponent of Ryu,
    Honda, Guile or Zangief. **Ryu's state is vs Ken** (Ryu continuing as Ryu against Ryu gets Ken instead).
  - **Honda, Guile and Zangief meet Ryu after a continue** (`--via ken`: Ken vs Ryu, lose, continue, pick them; the
    continue keeps the opponent). Their fight shows BATTLE 02 and "1 WIN" instead of BATTLE 01 (owner: acceptable?).
  - The first-opponent idle count is searched each boot: loading a state shifts which frame an input lands on.
  - Walls depend on the character (width), not only the stage; left + right is always 512: Chun-Li 53/459,
    Ryu/Ken/Guile/Dhalsim 55/457, Zangief 59/453, Blanka 63/449, Honda 64/448. `ram.py` LEFT_WALL/RIGHT_WALL and
    `cornered()` are Chun-Li's.
- [ ] **B2 moves:** an action set per character: the shared basics plus that character's specials as verified macros
  (motion inputs: fireball, dragon punch, hurricane kick, spinning pile driver; charge inputs: sonic boom, flash kick,
  rolling attack, headbutt; mashes: hundred hand slap, electricity). Each one is checked on the ROM, as Lightning Legs
  was.
- [ ] **B2 note:** the RAM note and state words hold for each character (the per-character checks JOURNAL.md lists)
- [x] **B3a baseline labels (first, to get the Studio GPU training):** the existing scripted teacher is already
  character-neutral in code (it ignores the character and never picks Lightning Legs), so for a **baseline
  checkpoint** every character uses the shared basic moves and this teacher. Gate it per character vs random play on
  the dev openings first. Collect only for characters where it wins by at least +15 (CPB2), with the same number of
  decisions per character (`collect_teacher --greedy --eps 0.1`), then train from the base model on the Studio.
  Specials come in the next round (B2).
- [ ] **B3 labels (the hard part):** choose the label source per character. Options: a generic teacher built from
  per-character move data (range, speed, which move anti-airs); outcome-filtered exploration (`relabel.py --mode
  filter`); System 2 as the teacher. Prototype the cheapest one on one character first.
- [x] **B3a result (2026-09-27):** the character-neutral teacher beats random play for all 8, paired on dev (Ryu +85.9,
  Honda +104.5, Blanka +88.9, Guile +44.9, Ken +41.3, Chun-Li +188.8, Zangief +27.7 [CI +11 to +44, the weakest],
  Dhalsim +94.2). Collected on the training openings (121–600): 30,630 rows, 3,543–4,089 per character. The balance
  gate refused the first mix twice on real data (Zangief, then Guile, 11% under); both were topped up with new
  openings and a new seed, then it passed.
- [x] **B4 baseline v0 trained** on the Studio in 54 min (from the base model, 256 px, 2 epochs, 7,657 steps):
  `runs_local/base_all_v0/best` (Studio worktree `laya_vision_vs_SF2_trackA`; a copy at `runs/base_all_v0` on the Mac
  Pro). It agrees with the teacher on 94.8% of held-out frames, evenly across characters (Chun-Li 92.2%, the
  lowest, to Honda 96.8%). Agreement does not predict play, so CPB3 gates it per character on the eval openings.
- [x] **No v1 training now (owner, 2026-09-27):** the loop improves play through memory, not retraining. v0 (fought
  only Ryu, and Ken as Ryu) is System 1: a novice against every other opponent, which fits "start weak". B3b's
  330,853 balanced rows over 47 pairings stay on the Studio for a later clean-up round (folding learned memory and
  broader opponents into the weights), not as a prerequisite.
- [ ] **CPB3 gate for v0:** v0, the teacher and random play for all 8 pairings on the 40 eval openings, plus r2 on the
  same Chun-Li state
- [ ] **B3b more opponents (running while the baseline trains):** the other 48 player-1 × opponent pairings among the
  8 playable characters (no mirror matches): a savestate each with automated start checks, Guile's sonic boom
  checked in the note, the teacher-vs-random check per pairing (kept at +15 or more), 5,000 decisions each on the
  training openings, and a balance gate over characters **and opponents** before the next training run
- [ ] **B4 train:** one multi-character checkpoint trained **from the base model** (`thaitea/laya-vision-smolvlm-256m`,
  not r2, so no character or opponent is built in), with a per-character move list in the question.
- [ ] **No Chun-Li focus (owner rule).** Chun-Li is treated like every other character: her labels are collected
  fresh, from the same label source, against the same opponents, and in the same amount. claw's old Chun-Li data
  (about 62k decisions, all vs Dhalsim) is **not** imported. It may be used only in a separate, labelled experiment.
- [x] **Balance gate [SCRIPT]:** before any training run, a script counts the training rows per character and per
  opponent, and refuses to train if any is more than ±10% off an equal share. It must be seen failing on a skewed
  set first.
- [ ] **Gates per character** vs Ryu on the saved openings; the verdict needs all 8 (CPB3), and Chun-Li no worse than
  r2 is one of the 8 bars, not the headline.
- [ ] **B5:** the two-system loop from Track A on top of the all-character checkpoint

---

## 1. The idea

| | What it holds | Changes |
|---|---|---|
| **Checkpoint** (r2 LoRA weights) | *Familiarity*: reading the screen, what each move does, spacing, timing | Slowly, by training (not in this plan) |
| **Memory** (situation rules) | *Knowledge*: "vs Ryu, when he jumps in from mid range, fierce punch" | Between rounds |

The game only ever waits for System 1. System 2 never blocks a frame.

---

## 2. Design

### 2.1 Where things run

Qwen fallback when the Studio is busy training: the same model on claw (Mac mini) at `http://192.168.1.199:8800/v1`,
same id `qwen38-27b-oq4e-mtp` (verified 2026-09-27; only the base URL changes, key in claw's `~/.omlx/settings.json`).

Everything runs on the Mac Pro (M4 Max). Qwen on the Mac Studio's omlx is used only from Phase 4: model id
`qwen38-27b-oq4e-mtp` (the Jundot Qwen3.8-27B oQ4e MTP build) at `http://192.168.1.216:8000/v1`. The API key
lives in the Studio's `~/.omlx/settings.json`. Verified from the Mac Pro on 2026-09-27: thinking off answered in
about 20 s (first load); thinking on generated about 36 tokens/s.

### 2.2 One decision with a memory

```
screen + RAM note ──▶ laya r2 ──▶ p (14 moves), top move t
RAM note ──▶ fields ──▶ memory: the most specific matching rule ──▶ move m
play m  if  p[t] − p[m] ≤ τ,   else play t        (τ = 0.3 by default, predeclared)
```

- **A nudge, not an override.** A rule wins only where laya already half-agrees. `τ = 1` would be an override;
  `τ = 0` changes nothing. τ is fixed before each test and never tuned on evaluation openings.
- **Every use is logged twice:** *fired* (a rule matched) and *changed* (the move played differs from laya's top
  move). Only changed decisions can affect play.
- **Matching:** the rule with the most conditions wins; a tie goes to the owner's rules, then Claude's, then
  Qwen's. The author comes from **which file the rule is in** (`owner.txt`, `claude.txt`, `qwen.txt`), never from
  the line itself.
- **Loop guard:** a rule may not condition on `last=X` and advise `X` (it could repeat itself forever).
- **The memory changes only at round boundaries,** and each round logs the exact memory snapshot it played with.

### 2.3 Moments: what System 1 hands to System 2

- A moment record (`sf2/contract.py`): the note, the second of notes before it, probabilities, move played,
  damage in the next second, and the two frames.
- **Flags, one per episode:**
  - `surprised`: the last controllable decision before she takes a hit. One per hit; decisions within 30 frames of
    it are not flagged again.
  - `unsure`: the top-2 margin is below a threshold that flags about 5% of decisions.
  - `audit`: a random 1% sample, drawn independently of the other flags.

  A lost round is summarised once, not flagged decision by decision.
- Moments are appended to `out/moments/<session>.jsonl`. System 2 reads the file.

### 2.4 Tests (predeclared)

- **Openings are split up front.** `dev` openings are for writing and debugging rules; `eval` openings are for
  verdicts only. Each is a saved schedule of start delays, identical across arms.
- **Pairing comes from the opening schedule, not the worker layout.** Each opening is a saved start delay, played
  by whichever worker picks it up. A frozen memory never changes during a run, so arms A, C and D can use many
  workers. A learning session (arm B) needs one worker per memory (§2.5).
- **The statistic:** the per-opening *paired difference* in net damage per round (arm − control), with its 95%
  confidence interval. It replaces the unpaired `sqrt(se_a² + se_b²)`.
- **Sample size:** the first gate of each comparison is a 20-opening **pilot**. It estimates the variance of the
  paired difference. The verdict run uses as many eval openings as needed to detect **+15 net damage per round**
  (the practical effect size). A confidence interval that includes both 0 and +15 means **inconclusive**, not a
  failure.

| Arm | Player | Memory | Question it answers |
|---|---|---|---|
| A | r2 | empty | control |
| C | r2 | frozen playbook | Do rules help the hands? |
| D | random policy | the same frozen playbook | Does laya matter, or do the rules just replace it? C must beat D |
| B | r2 | grows during the session (adviser) | Does it learn? The final frozen memory vs A on unseen eval openings, repeated over at least 2 sessions |

### 2.5 Execution: headless and parallel

Experiments run headless (no Mesen window), several games at once. Windowed play (`scripts/arcade.sh`) is only
for watching.

| Run | Workers | Estimate on the Mac Pro |
|---|---|---|
| Frozen arm (A, C, D), 20 openings | 4 play workers (the GPU saturates near 27 decisions/s) | about 10 min |
| Arm D (random policy, no model) | up to 12 | a few minutes |
| Learning sessions (arm B) | 1 worker per session, 2–4 sessions side by side | each session runs at single-worker speed (about 12 decisions/s) |
| Qwen (System 2) | on the Mac Studio | does not compete with play for the GPU |

**Logs, kept for learning curves.** Every run's workers write, timestamped and named, into one file you can follow
with `tail -F out/live.log`. Every finished run appends a row (the run, model, opponent, openings and gate numbers)
to `out/results.jsonl`. Nothing is overwritten, so a curve can be plotted from the ledger at any time.

**Why this beats fine-tuning.** One LoRA round took about 100 minutes of training plus 12–15 minutes of gate play on
claw. One memory iteration (edit rules, then re-run a 20-opening dev pilot) should take about 10 minutes, and it
needs no data collection and no checkpoint. So we can try roughly ten playbook variants in the time of one training
round, and keep the checkpoint fixed so every difference comes from the memory.

---

## 3. Decisions

- [x] Qwen runs on the Mac Studio's omlx, from Phase 4; everything else runs on the Mac Pro.
- [x] Build the mechanism with any System 2 first; Qwen is plugged in last.
- [x] The first System 2 is Claude. You can override it at any time, and your rules outrank Claude's.
- [x] Stay lockstep for every test; real-time play is out of scope.
- [x] Nudge, not override: τ = 0.3 unless you choose otherwise.
- [x] No automatic rule dropping in v1: the grading ledger is diagnostics only.
- [x] **System 2 learns in batches (owner, 2026-09-27):** play a batch of openings, Qwen reviews the moments, the
  memory updates, and the next batch plays. Live comes later, only if batches show Qwen helps.
- [x] **The playbook starts empty:** Qwen gets the game's description (the note's fields and values, the moves, the
  rule format) and the moments, and no tactics. System 1 should be a novice that knows how to play; if laya is
  very strong, something is wrong.
- [ ] First opponent: Ryu (scan other opponents only if Ryu leaves no room to improve).
- [ ] The first playbook's source: TEACHER.md's proven tactics, written as rules (default), or Claude's reading of moments alone.

---

## 4. Tasks

### Phase 0: shared contract (done, `d435553`)
- [x] Note → 12 named fields, validated
- [x] Situation rules: parse, validate, match; playbook errors name the line
- [x] Moment record, validated; confidence and flag order
- [x] 34 test cases; two seeded faults caught

### Phase 1: a verified baseline against Ryu
- [x] Copy `states/arcade_chunli_vs_ryu.state` (and the Dhalsim state) from claw
- [x] **Verify Ryu's RAM:** round start, controllability, round transitions, damage accounting, the walls, and the
  projectile slot for his fireball, as JOURNAL.md lists per opponent. Fix the harness first if any check fails.
- [x] **Verify this machine:** replay the Dhalsim gate with r2 and reproduce about +93 net damage per round
- [x] Save the opening schedules: 20 `dev` and at least 40 `eval` start delays, drawn at random
- [x] Parallel runner plays a saved opening schedule (openings assigned explicitly, not by worker index) with a frozen memory file
- [x] **Measure:** the wall time of a 20-opening headless pilot with 4 workers: 7.6 min (the Dhalsim replay)
- [x] Control A: r2 vs Ryu on the dev openings: 20/20 matches, 69% of rounds, +40.2 ± 10.9
- [x] The teacher vs Ryu on the same openings: dev 20/20 matches, +93.8; eval 40/40 matches, 87% of rounds, +89.3
- [x] Every run is logged: `out/live.log` (tail) and `out/results.jsonl` (append-only)
- [x] **CP1:** on eval, r2 +40.7 ± 7.5 (37/40 matches, 69% of rounds) vs the teacher +89.3 ± 5.8; paired gap −50.0 (−68, −32). In PROGRESS.md.
- [x] Check: A leaves room to improve: yes (31% of rounds lost, 50 points below the teacher), so Ryu stays
- [x] **Verify** the Studio's omlx is reachable from the Mac Pro: yes, as `qwen38-27b-oq4e-mtp` (fixed on the Studio side 2026-09-27)

### Phase 2: the memory, with a frozen playbook (arms C and D)
- [x] **Prototype first:** with an empty memory, play is identical to r2: all 489 decisions on dev opening 16
- [x] **Prototype:** `opp_state=jump dist=mid -> hp` fired 12 times and changed **0**. laya gives fierce 1–10% when Ryu jumps (it wants jump_forward, 41–54%), so every gap (0.32–0.66) exceeds τ = 0.3. A rule that contradicts the hands never applies at a global τ of 0.3.
- [ ] **Decide τ:** per rule (the rule's weight is its τ; owner rules may use 1 = override), fixed on dev before any eval run (recommended), or a single global τ
- [ ] Memory in the play loop: most-specific match, the τ nudge, author from the file, the loop guard, a snapshot per round
- [ ] Playbook v1 (Claude): TEACHER.md's Stage 3/4 tactics written as rules, each with its source, and adapted to Ryu where the tactic was Dhalsim-specific
- [ ] You review the playbook: keep, edit or veto
- [ ] Pilot C vs A on the dev openings: the fired and changed rates, and the paired difference
- [ ] **Verdict C vs A** on eval openings, sized from the pilot, at +15 net damage per round
- [ ] **Verdict C vs D:** the same playbook on a random policy. C must beat D, or laya adds nothing.
- [ ] If C is inconclusive or fails: check the changed rate first (rules that never change a move do nothing), then τ, then whether the note lacks a field the rules need. Add one note field only if the evidence names it.

- [x] **Forward selection:** none of v1's six groups helps r2 (three are worse, three change nothing). See PROGRESS.md.
- [x] **Stop rule for the Chun-Li study (owner, 2026-09-27):** triggered by "no rule group helps" (not 80%).
  Track A stopped; waiting for Track B's baseline.
  The rule as set: stop Track A once the harness is verified and Chun-Li with memory wins at least 80% of rounds vs
  Ryu (r2 alone 69%, the teacher 87%), or stop and report if forward selection finds no rule group that helps; then
  wait for Track B's baseline all-character checkpoint and wire it into the loop for a new round.

### Phase 3: moments
- [x] Moments from any student rollout: `scripts/moments.py rollouts/<name>` → `out/moments/<name>.jsonl` (one flag per episode, as in §2.3); r2 vs Ryu dev: 341 surprised, 693 unsure, 130 audit
- [x] Tune the `unsure` margin on dev play so about 5% of decisions are flagged: margin 0.036
- [x] **Verify:** do `unsure` decisions precede damage more often? **No: 0.65×** (hit within 0.5 s: 10.0% unsure vs 15.5% confident). laya's mistakes are confident ones. Per the predeclared rule, System 2 gets only `surprised` and `audit`.
- [ ] A terminal listing per moment: the note timeline, the probabilities, the outcome and the frame paths
- [ ] Claude revises the playbook from dev-opening moments only (v2); pilot it on dev and give a verdict on fresh eval openings

### Phase 4: Qwen as System 2 (offline)
- [ ] Prompt v1: a moment as text → rules in the Step-0 format
- [ ] Score on recorded dev moments without playing: the valid-format rate, agreement with Claude's rules on the same moments, and latency with thinking on and off
- [ ] Check: at least 90% valid and mostly agreeing, before it writes to a live memory

### Phase 5: learning during a session (arm B)
- [ ] Adviser loop: reads moments (surprised first), writes to its rules file, which is merged at round boundaries, with the delay logged
- [ ] Session: a sequence of dev openings with the memory growing; save the memory snapshot after each round
- [ ] Run 2–4 sessions side by side (one worker and one memory each): this gives the repeated sessions the verdict needs
- [ ] **Verdict:** the final frozen memory vs A on unseen eval openings, paired; repeat over at least 2 sessions from an empty memory
- [ ] B with Claude as the adviser first, then Qwen. If the Claude session passes and the Qwen one fails, the loop works and Qwen's advice is the problem.
- [ ] Write up in PROGRESS.md and the journal

### Phase 6: the live demo (the end goal)
- [ ] Decide the weak start: r2 with an empty memory against opponents it has never beaten (the default), or an
  earlier checkpoint (r0)
- [ ] Live loop: the arcade console plays continuously with the memory on. System 2 runs beside it (Qwen as a
  worker, or Claude through a headless `claude -p` call per batch of moments) and writes rules that are merged at
  round boundaries.
- [ ] A per-round ledger for live play: the time, opponent, win or loss, net damage, and the memory version it
  played with (appended, like `out/results.jsonl`)
- [ ] A live view (a local page that refreshes itself, beside the Mesen windows): a win-rate curve per opponent over
  rounds; a timeline of the memory (each rule added or dropped, its reason, how often it fired and changed a move);
  and the latest flagged moments
- [ ] **Verify:** the curve on the live view matches the ledger, and every rule on the timeline appears in a memory
  snapshot
- [ ] Record a session (Mesen, console and live view side by side) for the video

- [ ] **Qwen leads laya across characters:** once Track B's all-character checkpoint exists, Qwen is System 2 for
  every character, not only Chun-Li. It starts from `memories/KNOWLEDGE.md` (what System 2 has learned, with
  measured sources) and keeps adding to it.

### Later (not in this plan)
- [ ] A weak-adviser run: the same session protocol with a deliberately weaker System 2 (a thin playbook, or the
  smaller `qwen38-flashnext-oq4e-mtp`), to see a slow early start in the learning curve from `out/results.jsonl`
- [ ] Automatic grading and rule dropping (needs changed-decision counts, a window of at least 1 s, distinct episodes, and a 2-SE bar)
- [ ] A situation-vector lookup from laya's hidden state (only if a note field cannot express what the rules need)
- [ ] Other characters, folding memory into the weights, real-time play

---

## 5. Risks and guards

| Risk | Guard |
|---|---|
| The rules are just a scripted bot and laya adds nothing | Arm D (the same rules on a random policy) |
| A rule changes nothing | The *changed* rate is logged and read before any verdict |
| Rules are tuned on the openings they are judged on | Dev and eval openings are split up front |
| An online session is read as independent samples | Session verdicts use the final frozen memory on unseen openings, repeated |
| Ryu's RAM differs from Dhalsim's | Phase 1 verifies it before any Ryu number counts |
| The advice is wrong | Your review, your rules outrank others', and frozen verdicts |
| The 0.5 s window misses slow tactics | Diagnostics use at least 1 s; no automatic dropping in v1 |

---

## 6. Review log (2026-09-27)

Plan v2 was reviewed blind by GPT-6 (gpt-6-astra, via codex) and by Claude Fable. Both saw only the plan and the
public repo.

**Accepted (both reviewers):** the blend was an override at λ ≥ 0.5, so it is now the τ nudge; paired per-opening
statistics with a pilot and a predeclared effect size; frozen-playbook efficacy separated from session learning;
one sequential worker per memory; no automatic grading in v1; one flag per episode; Ryu's RAM verified first; the
opponent scan, HTML page, hidden-state lookup and moment advice cut.

**Accepted (one reviewer):**
- dev and eval openings, and saved opening schedules (GPT-6);
- arm D and the loop guard (Fable);
- the author set by file (both, in different forms);
- moving the memory before moments, because rules don't need moments (Fable).

**Not accepted:** sampling from p instead of argmax (Fable, as one option). Upstream's Atari work found that the top
move beats sampling, and here sampling the teacher's soft distribution played far worse than its top choice (+12 vs
+91). The τ nudge keeps argmax.

The raw reviews are kept in `docs/reviews/`.
