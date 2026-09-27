# Two-System Fighter: plan

A Street Fighter II player built like a human player:

- **System 1 (the hands):** laya-vision acts every 4 frames from what it sees.
- **System 2 (the mind):** Qwen reviews the moments that went wrong, thinks between rounds, and leaves instructions in a **memory** that the hands read before every move.

The player should get better match by match against an opponent, **without retraining the model in between**.

Written 2026-09-27. Start with Chun-Li. The design is meant to generalise to any character.

---

## 1. The idea in one table

| | What it holds | How it changes |
|---|---|---|
| **Checkpoint** (System 1's LoRA weights) | *Familiarity*: reading the screen, what each move does, ranges, timing, walls, when you can act, action → reaction | Slowly, by training |
| **Memory** (written by System 2 or by you) | *Knowledge*: "vs Ryu, when he jumps in from mid range, anti-air with fierce" | Instantly, between rounds |

- The game only ever waits for System 1. System 2 works asynchronously, in its own process, and never blocks a frame.
- System 1's weights are fixed during play, so **the memory is the only way to give it instructions.**
- Training (a LoRA round) is an occasional clean-up that folds memory into the weights. It is not part of the loop.

---

## 2. What already exists (do not rebuild)

Repo: **github.com/beefkknd/laya_vision-vs-SF2** (main at `32f310e` or later). The local copy at `~/work/hobby/laya_vision-vs-SF2` is far behind GitHub, so clone or update first.

| Piece | Where | State |
|---|---|---|
| laya-vision (SmolVLM-256M + typed-decision head, LoRA r16 on text layers) | `sf2/lora.py`, `sf2/policy.py`; base `thaitea/laya-vision-smolvlm-256m` | r2: **+93.1 net damage/round, 40/41 rounds vs Dhalsim** |
| Lockstep Mesen bridge | `mesen/sf2_bridge.lua`, `sf2/mesen.py` | The game waits for Python every decision; it renders every frame |
| Fight env, RAM map, actions (14) | `sf2/env.py`, `ram_maps/sf2_snes.txt`, `sf2/actions.py` | RAM note: `me=chunli stand hp=100 opp=dhalsim attack dist=mid facing=right corner=none time=early last=forward fireball=none` |
| `controllable` flag | `sf2/ram.py` | Hit / throw / knockdown frames are scored but not trained |
| Scripted teacher | `sf2/teacher.py`, `TEACHER.md` | +105.5 with about 5 rules (baseline and ablation only) |
| Gate | `scripts/gate.py`, `PROGRESS.md` | 20 paired matches (`--workers 4 --seed 4242`), net damage per round, "better" = more than 2 combined SE |
| ROM acceptance suite | `tests/test_rom_harness.py` | 30 checks. It writes a stamp; collection and play refuse to run without it |
| DAgger / LoRA training | `scripts/train.py`, `scripts/loop/*.sh`, `sf2/vision_cache.py` | Frozen vision features cached (2.4× faster) |
| Replay viewer, arcade mode | `scripts/show.py`, `scripts/arcade.sh`, `sf2/boot.py` | Plays from power-on |
| Lessons learned | `README.md`, `docs/lessons.html`, `JOURNAL.md` | Read these first |

### Where the assets are

- **Trained checkpoints, data, rollouts:** on **claw** (the Mac mini, M4 Pro, `ssh` via the `claw` shell function), in `~/work/hobby/laya_vision-vs-SF2/`:

  | Folder | Size | Contents |
  |---|---|---|
  | `runs/` | 11 GB | `r2`, `r1`, `r0_256`, … |
  | `data/` | 18 GB | training data |
  | `rollouts/` | 5.9 GB | recorded play |

  Only `runs/r2` and the savestates are needed to start.
- **ROM:** `~/Downloads/Street Fighter II (USA).zip` on this Mac (the M3 Ultra).
- **Emulator:** `/Applications/Mesen.app` (Mesen 2).
- **Qwen (System 2):** the local omlx server. It is OpenAI-compatible at `http://127.0.0.1:8000/v1/chat/completions`, model `Jundot--Qwen3.8-27B-oQ4e-mtp`, and thinking is toggled with `chat_template_kwargs {"enable_thinking": ...}`.
  - Thinking-on answers take about 2–60 s.
  - The `agent_harness` project shows prompts and budgets (`harness/model_client.py`). Talk to Qwen over HTTP only; do not import harness code.

### Known gaps this plan targets

- It only knows Dhalsim. It has beaten Ryu, and lost to him.
- Its only teacher is a hand-written script.
- It cannot improve during a session.
- On the M4 Pro it takes about 72 ms per decision against a 67 ms real-time budget.

---

## 3. Step 0: the shared contract (half a day, before the lanes split)

Both lanes plug into two formats, so define and test them first.

### The moment record

Written by System 1. One JSON line per flagged decision, on `controllable` frames only:

```json
{"match": 3, "round": 2, "frame": 1432, "why": "surprised",
 "notes_before": ["...RAM note, 1 s before..."], "note": "me=chunli stand ... opp=ryu jump dist=mid ...",
 "probs": {"hp": 0.31, "hk": 0.29, "block": 0.12},
 "played": "hk", "dealt": 0, "taken": 18, "frames": ["t-4.png", "t.png"]}
```

It is flagged for one of three reasons (`why`):

| `why` | Trigger |
|---|---|
| `unsure` | The top-2 margin is below a threshold, tuned to flag about 5% of decisions |
| `surprised` | Damage taken in the next 0.5 s (30 frames), or a round lost |
| `audit` | A random 1% of confident decisions, to find blind spots |

### The instruction format

Written by you or System 2.

- **Situation rule** over the RAM note's fields. Validated: unknown fields or moves are rejected.
  ```
  opp=ryu opp_state=jump dist=mid -> hp        # optional: weight, reason, author
  ```
- **Moment advice:** `{moment id -> move, reason}`.

---

## 4. Lane 1: System 1 plays, and you are the first System 2

1. **Set up this Mac.**
   - Clone or update the repo to GitHub main.
   - `uv pip install -e '.[model,dev]'`.
   - Copy `runs/r2` and the savestates from claw.
   - Unzip the ROM and point `SF2_ROM` at it.
   - Check `ram_maps/sf2_snes.txt` against this ROM (USA). `scripts/check_env.py` does this, and `scripts/find_ram.py` can repair it.
2. **Verify the harness.**
   - `SF2_ROM=... pytest -q tests/test_rom_harness.py`: all 30 checks must pass, and it writes the stamp.
   - Replay the Dhalsim gate with r2 (the `PROGRESS.md` command). It should reproduce **about +93 net damage per round**. If it does not, stop and find out why before going further.
   - Measure decision latency on the M3 Ultra.
3. **Log moments.** Add the situation vector (the pooled hidden state before the decision head) and the confidence (top-2 margin, entropy) to the decision call, which needs no retraining, and write moment records. Check that `unsure` actually marks where damage is taken. If it does not, the hands cannot ask the right questions.
4. **Play Chun-Li vs Ryu** (a savestate is needed; `scripts/make_savestate.py` exists). Review flagged moments with a readable replay (`scripts/show.py`).
5. **Build the memory, and you instruct System 1.**
   - **Lookup:** match the symbolic fields first, then cosine similarity on the situation vector. Keep it under 1 ms.
   - **Blend:** `p = (1-λ)·p_laya + λ·p_advice`, where λ rises with similarity and with the entry's track record.
   - **Grade:** track the net damage in the 0.5 s after each use. An entry that does worse than Laya alone is dropped.
   - **Persist:** save the memory per session.
   - **Instruct:** you write rules in the Step-0 format, and they apply live.
6. **Measure:** the same paired openings with your rules on and off.

---

## 5. Lane 2: how Qwen handles these questions (offline, in parallel)

1. Feed Qwen recorded moments: claw's existing rollouts to start, then Lane 1's moment logs.
2. Prompt work:
   - how a moment is described (note timeline, options with probabilities, outcome);
   - thinking on or off and the reasoning budget;
   - the playbook it gets (yours, a guide, or none);
   - its own notes file (as Qwen writes its own rules in agent_harness).
3. Score its advice **without playing**:
   - valid Step-0 format;
   - agreement with your rules on the same moments;
   - whether it points away from moves that led to damage;
   - latency.
4. Done when its advice is valid and mostly agrees with yours.

---

## 6. Merge: the two systems together

- Qwen replaces you as the memory's writer, running as a background worker that reads the moment queue: surprises first, then unsure moments, then audits.
- Advice that fails grading goes back to Qwen: "your advice failed here".
- You can still override. Your rules outrank Qwen's.

### The session test: does it learn like a human?

Chun-Li vs Ryu, 20 matches on the paired openings, played in order.

| Arm | Memory | Advice from |
|---|---|---|
| A: control | off | none |
| B: two systems | on | Qwen |
| C: mechanics check | on | the scripted teacher, or your rules |

**Predeclared rule:**

- B beats A on matches 11–20 (the same openings, paired) by at least 2 combined SE of net damage per round.
- B's per-match curve rises.
- If C passes and B does not, the memory works and the advice is the problem.

**Also read:** flagged moments per round (these should fall as the hands take over), memory hit rate, the share of advice surviving grading, and Qwen's queue delay.

---

## 7. Any character: the tutorial phase

- The checkpoint is Chun-Li's familiarity. It carries over to new **opponents**, but not to new **characters**, who have different moves, reach, speed and specials.
- The memory can correct a competent player; it cannot make an incompetent one competent.

For each new character:

1. Build an action set with special-move macros (for example Ryu's fireball), and verify each one on the ROM (as Stage 4 did for Lightning Legs).
2. Check that the RAM note's state words fit the character.
3. Make savestates for each matchup.
4. **Tutorial:** exploratory play plus a simple teacher, or Qwen and the memory, to collect basic labels. Train a base checkpoint (this is where LoRA consolidation comes back).
5. Then run the two-system loop for its matchups.

A shared "general" System 1 across characters is a later, bigger step.

---

## 8. Risks and guards

| Risk | Guard |
|---|---|
| Advice is wrong and the hands copy it (the student inherits the teacher's blind spots) | Outcome grading; the audit sample; your corrections outrank Qwen |
| Memory matches the wrong situation | Symbolic fields must match first; a similarity threshold; small λ at first |
| `unsure` does not mean unsure | Check it in Lane 1 step 3 before building on it |
| Qwen and laya-vision share the GPU | Lockstep keeps results correct; for speed tests, run Qwen on claw |
| Real time: 72 ms vs a 67 ms budget | Measure on the M3 Ultra; the memory adds under 1 ms; stay lockstep until it fits |
| Identical fights hide effects | Paired openings plus start jitter; check `distinct_matches == matches` |
| A new machine behaves differently (it happened before: cache paths, Mesen from a background shell) | The acceptance suite plus the Dhalsim replay before anything else |

---

## 9. Lessons carried over from the Othello work (laya_othello, 2026-09-23 to 27)

- **The student becomes a best response to its teachers.** Laya was trained against one bot family and was then wiped out by a different style (Pinocchio). Vary the opponents and check against an outsider.
- **Verify assumptions with cheap play before long runs.** It changed the plan twice.
- **Fix the rule before the run, then trust the gate.** Frame accuracy and loss do not predict play.
- **Ten games can mislead in both directions** (3–7 became 30–10 over 40). Size tests to their noise.
- **Measure speed, don't assume it.** The Othello oracle was secretly pure Python; moving it to C made it about 200× faster.
- **A new signal gives a jump; repeating the same loop gives nothing.** When the gate stalls, change the signal.

---

## 10. Decisions for you

- **Playbook:** you write it, an existing Chun-Li vs Ryu guide, or Qwen starts with nothing.
- **First opponent:** Ryu (the known weakness), or another.
- **Where Qwen runs:** this Mac, or claw.
- **How far instructions go:** blend only, or may a certain rule override the hands?
- **Real time:** stay lockstep until latency fits 67 ms?

## Rough effort

| Step | Time |
|---|---|
| Step 0 | half a day |
| Lane 1 to "you instructing live" | about 2 days |
| Lane 2 | 1–2 days, in parallel |
| Merge and session test | about 1 day |
| Each new character | a few days |

Related:

- **Spec page:** https://claude.ai/artifact/LzgPuZDmNmJNF3CLXn2Mw5
- **Journals:** `~/work/me/journals/laya_othello/2026-09-26.md` and `2026-09-27.md`
