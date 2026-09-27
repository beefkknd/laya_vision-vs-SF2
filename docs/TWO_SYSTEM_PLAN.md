# Two-System Fighter: plan

A Street Fighter II player built like a human player:

- **System 1 (the hands):** laya-vision, the trained r2 checkpoint, picks a move every 4 frames from what it sees.
- **System 2 (the mind):** an adviser that reviews what went wrong and writes **situation rules** into a memory.
  The hands consult the memory before every move. The first adviser is Claude (the coding agent). Qwen replaces
  it last. You can override either.

**Hypothesis:** rules in a memory, with no retraining, make r2 play better against an opponent it was not trained
for. Then: an adviser that writes those rules during play makes it improve match by match.

**Status:** planning, v3 (v2 plus the two reviews in §6). Branch `two-system`. Nothing runs until you approve.

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

Everything runs on the Mac Pro (M4 Max). Qwen on the Mac Studio's omlx is used only from Phase 4.

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
- [ ] Copy `states/arcade_chunli_vs_ryu.state` (and the Dhalsim state) from claw
- [ ] **Verify Ryu's RAM:** round start, controllability, round transitions, damage accounting, the walls, and the
  projectile slot for his fireball, as JOURNAL.md lists per opponent. Fix the harness first if any check fails.
- [ ] **Verify this machine:** replay the Dhalsim gate with r2 and reproduce about +93 net damage per round
- [ ] Save the opening schedules: 20 `dev` and at least 40 `eval` start delays, drawn at random
- [ ] Parallel runner plays a saved opening schedule (openings assigned explicitly, not by worker index) with a frozen memory file
- [ ] **Measure:** the wall time of a 20-opening headless pilot with 4 workers (estimate about 10 min). Correct §2.5 if it's off.
- [ ] Control A: r2 vs Ryu on the dev openings, one sequential worker. Record the paired-difference variance.
- [ ] Check: A leaves room to improve (not 40/40). If it doesn't, pick the next weakest opponent from a 3-opponent scan.
- [ ] **Verify** the Studio's omlx is reachable from the Mac Pro (it failed once), or set up an SSH tunnel. This is 10 minutes of work that decides Phase 4.

### Phase 2: the memory, with a frozen playbook (arms C and D)
- [ ] **Prototype first:** with an empty memory, play must be identical to r2, decision for decision, on 1 dev opening
- [ ] **Prototype:** one rule that must fire (`opp_state=jump dist=mid -> hp`). The log shows *fired* and *changed*, and the move changes only within τ.
- [ ] Memory in the play loop: most-specific match, the τ nudge, author from the file, the loop guard, a snapshot per round
- [ ] Playbook v1 (Claude): TEACHER.md's Stage 3/4 tactics written as rules, each with its source, and adapted to Ryu where the tactic was Dhalsim-specific
- [ ] You review the playbook: keep, edit or veto
- [ ] Pilot C vs A on the dev openings: the fired and changed rates, and the paired difference
- [ ] **Verdict C vs A** on eval openings, sized from the pilot, at +15 net damage per round
- [ ] **Verdict C vs D:** the same playbook on a random policy. C must beat D, or laya adds nothing.
- [ ] If C is inconclusive or fails: check the changed rate first (rules that never change a move do nothing), then τ, then whether the note lacks a field the rules need. Add one note field only if the evidence names it.

### Phase 3: moments
- [ ] Confidence and flags in the play loop; write `out/moments/*.jsonl` (one flag per episode, as in §2.3)
- [ ] Tune the `unsure` margin on dev play so about 5% of decisions are flagged
- [ ] **Verify:** do `unsure` decisions precede damage more often than confident ones (predeclared: at least 1.5×)? If not, System 2 gets only `surprised` and `audit`.
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

### Later (not in this plan)
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
