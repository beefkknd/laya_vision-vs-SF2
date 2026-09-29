# laya_vision-vs-SF2

This project teaches **laya-vision** to play **SNES Street Fighter II**, running in **Mesen 2** on a Mac Studio. It follows the plan in [PLAN.md](PLAN.md):

1. A frame-level teacher labels frames.
2. laya-vision learns to copy it.
3. The student plays, and the teacher labels the student's own frames (DAgger).
4. The model trains again.

Round wins and damage decide whether a round of training worked; validation loss does not.

## How the pieces connect

```
Mesen 2 (your SNES ROM)                                Python (this repo)
  mesen/sf2_bridge.lua  ── TCP 127.0.0.1:47800 ──▶  sf2/emu/mesen.py  →  sf2/env.py  →  teacher / student / recorder
  every input poll:                                   RUN n frames of input
    apply the next planned input, or                  ◀── RAM values for every frame
    report and wait for Python                        ◀── screenshots (4 frames before the end, and the end)
```

- **The emulator never runs ahead of Python.** At each input poll, the Lua script applies the next planned input. When the plan runs out, it blocks until Python sends the next command. How long the model takes to decide never changes the fight.
- **Special moves are macros.** The model picks `hadouken`, and the glue sends the 12-frame quarter-circle + fierce input as one run.
- **Python writes every file** (savestates, screenshots, logs). The Lua script only needs network access.
- **The script can stay loaded all day.** When a Python script ends, it disconnects. The Lua script reconnects to the next one within a second.

## The learning loop (System 1 + System 2), as it runs now

```
screen ─▶ laya-vision (runs/all8/best)      rates every move: "likely works / may work / likely fails"
            │  top 3 + moves the memory names + forward
            ▼
          text laya (runs/text_laya/advice_v1, MLX)   reads the moment + Qwen's short memory, picks one move
            │
          game log (rollouts/learn/<me>/<session>/)
            ▼
          System 2: Qwen on omlx, in a background thread   revises the short memory after a lost round,
                                                           the playbook after a game (memory/)
```

- **laya-vision** stays general: what a move does now, from the picture and a note (no opponent, no advice).
- **Text laya** (ModernBERT-large via `laya-mlx`, runs in `~/work/laya_mlx/.venv`) is fine-tuned only to FOLLOW
  advice: polarity words (use more / avoid / never / always / drop …) and conditions (up close, when he jumps …).
  The label rule is `sf2/advice.py`; data `scripts/build_advice_data.py`; training `scripts/train_text_laya.py`.
  System 1 talks to it through a helper process (`sf2/advisor.py`, `scripts/text_laya_server.py`).
- **Qwen** judges which advice is good; only Qwen writes `memory/`.

| Run / measure | Command |
|---|---|
| Play arcade mode in a window, learn until stopped | `python scripts/learn_loop.py` (`--minutes 10`, `--headless`, `--advisor off`, `--ab-advice`) |
| Does the short memory help? (headless, opponent locked, paired) | `python scripts/ab_memory.py --rounds 30` |
| Where the loop loses (see / choose / advice / walk / defend / lessons) | `python scripts/report.py gaps` |
| How much each memory rewrite changed (churn, flips) | `python scripts/report.py churn` |
| Does a model read advice words at all? | the probes `probe_memory_words.py` / `probe_text_laya.py`, removed 2026-09-29; in git at 7b5e280 |

Start omlx first (`~/work/omlx/start`). Findings so far and the next plan: `~/work/me/journals/laya_vision-vs-SF2/`.

## Setup details (what this exact setup used)

Everything below ran on one Mac Studio (M3 Ultra, 32 cores, 256 GB). Three separate Python environments / servers:

| Piece | What | Where it runs | Settings that matter |
|---|---|---|---|
| Game | *Street Fighter II (USA)* SNES ROM, sha1 `7ddcb96e0d9fea94d9370635262ac7c28da85214`, in `roms/` (never committed) | Mesen 2 (MesenCE app 1.0, `/Applications/Mesen.app`), headless `--testrunner` or a window | controller 2 plugged in with `--snes.port2.type=SnesController`; window at 150% speed (`--speed`); the loop restores Mesen's `settings.json` afterwards (a window writes its overrides back) |
| laya-vision (System 1: sees) | base [`thaitea/laya-vision-smolvlm-256m`](https://huggingface.co/thaitea/laya-vision-smolvlm-256m) (SmolVLM-256M backbone) + our LoRA `runs/all8/best` | this repo's `.venv`: Python 3.12, PyTorch 2.14 on Apple MPS, transformers 5.17 | **256x256 native pixels, no upscale**: the 256x224 SNES frame is padded to 256x256, HUD rows 0-61 blanked, `image_interpolation: nearest`, GPU preprocess, no image splitting; two frames 4 apart + a one-line note (bars as words, no opponent name). LoRA r=16, alpha=32, 2 epochs, batch 8, lr 2e-4 (adapters) / 1e-4 (head), always from the base checkpoint; 22 actions per character, answer = hit / whiff / blocked / none / got_hit |
| Text laya (System 1: decides) | [`aac6fef/laya-mlx`](https://huggingface.co/aac6fef/laya-mlx): the MLX conversion of [`convaiinnovations/laya`](https://huggingface.co/convaiinnovations/laya), ModernBERT-large, 421M + our LoRA `runs/text_laya/advice_v1` | **laya-mlx** (Apple MLX runtime, not omlx): its own venv `~/work/laya_mlx/.venv` (uv, Python 3.12, mlx 0.32.2, laya-mlx 0.1.0); started as a helper process by `sf2/advisor.py` | fp32; LoRA r=16, alpha=32 on every layer's `Wqkv`/`Wo`/`Wi` plus the decision head; AdamW lr 1e-4 cosine, batch 32, early stop (600 steps, ~6 min); max 512 tokens in; ~12 ms per decision. `HF_HOME=/Volumes/ExtremeSSD/huggingface`, `HF_HUB_OFFLINE=1` |
| Qwen (System 2: learns) | `Jundot--Qwen3.8-27B-oQ4e-mtp` (Qwen3.8 27B, 4-bit oQ4e quant with MTP) | the **omlx** server, OpenAI-compatible at `http://127.0.0.1:8000/v1`; start with `~/work/omlx/start` | thinking **off** (with thinking a review took ~5 min; without ~9-11 s), max 8192 tokens, temperature 0 (`sf2/config.py`); runs in a background thread so the game never waits |

## What laya-vision actually is

- **It is not `laya-mlx`.** [`laya-mlx`](https://github.com/mizorewww/laya-mlx) (the Apple MLX runtime) is text-only. laya-vision is a separate research fork, [r33drichards/laya-vision](https://github.com/r33drichards/laya-vision). It uses a **SmolVLM-256M** image backbone with Laya's typed-decision head and the same `predict(state, questions)` API. It is **PyTorch** and runs on Apple **MPS**.
- **Checkpoint:** [`thaitea/laya-vision-smolvlm-256m`](https://huggingface.co/thaitea/laya-vision-smolvlm-256m). It was trained on photo questions and knows no games.
- **No LoRA upstream.** Its trainer only freezes whole layers. `sf2/data/lora.py` adds rank-r adapters to the text layers and merges them back before saving, so a trained run is an ordinary laya-vision checkpoint.
- **Upstream already tried this loop on Atari** ([game-training.md](https://github.com/r33drichards/laya-vision/blob/main/docs/game-training.md)):
  - two frames beat one (median score 0.201 vs 0.131)
  - one DAgger round lifted the median from 0.201 to 0.310
  - taking the top action beat sampling
  - frame accuracy did not predict play strength

  So this project uses **two frames** (4 frames ago and now) plus a text note, and the gate is real play.

## Layout

| Plan piece | Here |
|---|---|
| Emulator | `mesen/sf2_bridge.lua` (inside Mesen) + `sf2/emu/mesen.py` (Python end) |
| Fight env | `sf2/env.py`. One call = one decision. It tracks rounds and matches from the life values; an episode is one match from your savestate |
| RAM map (per cartridge) | `ram_maps/sf2_snes.txt`, found by `scripts/find_ram.py` (`sf2/ramsearch.py`) |
| Action set (12) | `sf2/actions.py`: `idle forward back jump crouch lp hp lk hk block hadouken shoryuken` |
| Text state | `me=ryu opp=ken dist=mid my_hp=80 opp_hp=45 last=hadouken airborne=0 opp_airborne=1` |
| Teacher: you | `scripts/record_human.py` (you play in Mesen), then `scripts/label_human.py` → `sf2/labeler.py` recognises fireball and dragon-punch motions |
| Teacher: scripted dummy | `sf2/teacher.py`. RAM rules that return a distribution, used as a soft target |
| LoRA | `scripts/train.py` + `sf2/data/lora.py`. Always starts from base laya-vision 256M; early stopping on `val.jsonl`, or 5% of train |
| VS BATTLE (both pads) | `sf2/emu/vs.py`: boot to a 2-player fight, place the fighters at a gap, record an exchange. `scripts/vs_moves.py`: every move of both fighters, checked and measured, reach sweeps |
| Stage-1 data | `sf2/data/vs_sweep.py` + `scripts/vs_dataset.py`: 20 actions x 3 ranges per character vs a still dummy, labelled hit / whiff / blocked / none from RAM; `scripts/audit_dataset.py`, `scripts/verify_replay.py` check it |
| What the model sees | `sf2/data/frames.py`: every frame has its HUD (rows 0-61) blanked, at train and play time |
| Student plays / relabel / gate | `scripts/play_student.py`, `scripts/relabel.py` (`dagger` or `filter`), `scripts/play_teacher.py`, `scripts/gate.py`. `scripts/dagger_round.sh N` runs one turn of the loop |

Directions are relative: `forward` is toward the opponent, `back` is away, and `block` is down-back. Buttons follow SF2's default SNES layout: **Y X L = jab / strong / fierce, B A R = short / forward / roundhouse**. If your in-game button config differs, change `PAD` in `sf2/config.py`.

## Setup (Mac Studio)

1. **Mesen 2.** Install it from [mesen.ca](https://www.mesen.ca/) or [GitHub releases](https://github.com/SourMesen/Mesen2/releases), and open your SF2 ROM.
2. **Allow the script to use the network.** Open Debug → Script Window → Settings → Restrictions, and tick **"Allow network access"**. To override the port with `SF2_BRIDGE_PORT`, also tick "Allow access to I/O and OS functions".
3. **Python.** In this repo:
   ```bash
   uv venv -p 3.12 && source .venv/bin/activate
   uv pip install -e '.[model,dev]'
   uv pip install "laya @ git+https://github.com/r33drichards/laya-vision@568feeeada793f70f736756b0f3a7643d1e75910"
   pytest -q      # 20 tests. Needs no ROM or model; the Lua-bridge tests run only if lua5.4 + LuaSocket are installed
   ```
4. **Load the bridge.** In Mesen's Script Window: Open → `mesen/sf2_bridge.lua` → Run. It shows "waiting for a Python script". Leave it loaded.
5. **Speed.** For recording yourself, play at normal speed. For teacher collection and student play, set Mesen's emulation speed to maximum; the bridge still waits for Python on every decision.

Every emulator script starts with *"waiting for Mesen on 127.0.0.1:47800"* and continues once the bridge connects.

## Day 1: savestate, RAM map, check

```bash
# 1. Pick Ryu vs Ken (vs mode, or wherever you like). At "FIGHT!" press F9 in Mesen (or Enter here), then Ctrl-C.
python scripts/record_human.py --no-log --save-state-to states/ryu_vs_ken.state

# 2. Find where your cartridge keeps life and positions. Writes ram_maps/sf2_snes.txt
python scripts/find_ram.py                     # scripted: Python walks, jumps, waits to get hit, punches
python scripts/find_ram.py --manual --force    # if that fails: you play each phase in Mesen when prompted

# 3. Send each of the 12 actions. x must move on forward/back, y on jump. Screenshots in out/check/
python scripts/check_env.py
```

**Why the RAM finder exists.** World Warrior, Turbo and Super SF2 (and each region) keep life and positions at different addresses, and I couldn't verify any of them without your ROM. `find_ram.py` dumps the full 128 KiB of work RAM during known phases: walk right, walk left, jump, get hit, land hits. It keeps the addresses that behave like life, x and y. You can also read them off Mesen's memory viewer (Debug → Memory Tools) and write `ram_maps/sf2_snes.txt` by hand; the format is in `sf2/emu/ram.py`. Commit the map once `check_env.py` looks right.

## The rest of the week

```bash
# Optional: your own play as teacher (rung 1). Play clean fireballs, anti-airs and blocks for 10–20 minutes
python scripts/record_human.py --session s1          # Ctrl-C to stop
python scripts/label_human.py --session human/s1 --name human_s1

# Day 2: seed set from the scripted teacher (epsilon-expert, soft targets) + reference lines for the gate
python scripts/collect_teacher.py --name seed_teacher --decisions 30000 --eps 0.25
python scripts/play_teacher.py --name teacher --matches 10
python scripts/play_teacher.py --name random --policy random --matches 10

# Day 3: first LoRA; the student plays the CPU; compare
python scripts/train.py --data data/seed_teacher --out runs/r0          # + --data data/human_s1 if recorded
python scripts/play_student.py --model runs/r0/best --name r0 --matches 10
python scripts/gate.py rollouts/random rollouts/teacher rollouts/r0

# Days 4–6: DAgger rounds. The student plays, the teacher labels its frames, retrain, gate
scripts/dagger_round.sh 1
scripts/dagger_round.sh 2

# Cheap alternative to DAgger: keep only student actions that won the next 0.5 s
python scripts/relabel.py --rollout rollouts/r0 --name filter_r1 --mode filter
```

On day 7, compare `scripts/gate.py rollouts/teacher rollouts/r0 rollouts/r1 rollouts/r2`. If win rate and damage per round have not moved, the labels are too coarse. Fix the teacher or add a macro; don't collect more frames.

**Headless runs.** Mesen 2 can run without a window: `Mesen --testrunner <rom> mesen/sf2_bridge.lua`. Pass that as `--launch "<command>"` (or set `SF2_MESEN_LAUNCH`), and Python starts Mesen itself and ends it when done. I haven't confirmed that screenshots work in test-runner mode. If `check_env.py --launch ...` saves blank images, use the windowed setup above.

## Stage 1: still-opponent data for all 8 characters

Python plays both pads in VS BATTLE (Mesen's saved settings leave port 2 empty; `sf2/emu/headless.py` plugs a pad in for
the run). Each character stands on the left, facing right, at 10 gaps per range (close < 55 <= mid < 120 <= far px)
against a dummy that stands, crouches or crouch-blocks, and presses each of its 20 actions. RAM gives the outcome.

```bash
python scripts/vs_dataset.py run --pairs ryu:chunli,ken:guile,honda:blanka,zangief:dhalsim   # ~5 min, 48 headless Mesens
python scripts/audit_dataset.py        # ~96k mechanical checks per character; exit 1 on any violation
python scripts/verify_replay.py        # re-records a random sample from each record's boot; must match byte for byte
python scripts/train.py --out runs/all8 --data test_data/ryu --data test_data/ken --data test_data/chunli \
    --data test_data/guile --data test_data/honda --data test_data/blanka --data test_data/zangief --data test_data/dhalsim
```

`test_data/<char>/` (local only, git-ignored):

| file | rows | what |
|---|---|---|
| `train.jsonl` | 2520 | what `train.py` reads: `train_real` + `train_mirrored` |
| `train_real.jsonl` | 1260 | real, left side: 20 actions x 3 ranges x 7 gaps x 3 postures |
| `train_mirrored.jsonl` | 1260 | the same, flipped: frame mirrored, left/right buttons swapped, `side`/`dx` flipped |
| `test_real_left.jsonl`, `test_real_right.jsonl` | 540 each | real frames at the 3 held-out gaps per range, on each side (the mirroring check) |

Each record: two model frames (4 frames apart, HUD blanked), the RAM note, the question
`sf2.data.vs_sweep.outcome_question(action)` (choice: hit / whiff / blocked / none) and its `label`, plus the measurements
(damage, frames until the fighter can act again, travel) and the boot savestate it came from. The special-move timings
are the ROM-verified ones from the move tests; charge moves charge on down-back so the gap does not change.

## Check these on day 1 (the likely breakpoints)

1. **RAM map.** `check_env.py` must show x moving on `forward`/`back` and y on `jump`. The teacher, the text note and facing all depend on them.
2. **Buttons (`PAD`).** The `lp`/`hk` screenshots should show a jab and a roundhouse.
3. **Macro timing (`sf2/actions.py`).** `out/check/11_hadouken.png` should show a fireball. If it doesn't, lengthen each motion step from 3 frames to 4.
4. **Estimates to tune:** `INTRO_SKIP` (`sf2/env.py`), and `CLOSE`/`MID` (`sf2/emu/ram.py`, SNES pixels).
5. **KO detection.** A round ends when a life value goes negative, or when both bars refill (time over, or a cart that stops at 0). If rounds never end in the logs, look at the life values around a KO in `check_env.py`.

## Status

Working end to end in Mesen: stage-1 data for all 8 characters, laya-vision LoRA `runs/all8/best`, text laya
`runs/text_laya/advice_v1`, the arcade learning loop with async System 2, and the measurement scripts above.
Known gaps (2026-09-28): laya-vision's live move ratings do not rank well against a moving CPU, she almost never
blocks, and the short memory changes a lot between rounds. The headless A/B shows the memory makes her attacks
safer but has no net effect overall (helps vs Honda/Ken/Dhalsim, hurts vs Ryu). Next: a distilled playbook.
