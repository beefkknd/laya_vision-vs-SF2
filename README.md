# laya_vision-vs-SF2

This project teaches **laya-vision** to play **SNES Street Fighter II**, running in **Mesen 2** on a Mac Studio. **System 1** sees the screen (laya-vision) and decides which move to try (text laya), following advice. **System 2** (Qwen on omlx) writes and refines that advice from game logs.

The learning loop is in `scripts/learn_loop.py` (`--minutes`, `--games`, `--headless`, `--advisor off`, `--ab-advice`, `--fresh`). Measurement scripts measure whether the advice helps: `scripts/ab_memory.py` (many opponents, fixed advice, paired), `scripts/notebook_run.py` (learning over rounds), and `scripts/report.py {gaps,churn}` (where losses happen and how much memory changes).

The old teacher/DAgger pipeline (before 2026-09-29) is archived at git tag `legacy-dagger`.

## How the pieces connect

```
Mesen 2 (your SNES ROM)                          Python (this repo)
  mesen/sf2_bridge.lua  ── TCP 127.0.0.1:47800 ──▶  sf2/emu/mesen.py  →  game_log
  every input poll:                                   RUN n frames of input
    apply the next planned input, or                  ◀── RAM values for every frame
    report and wait for Python                        ◀── screenshots (4 frames before the end, and the end)

System 1 (one decision at a time)                System 2 (async, no blocking)
  screen  →  laya-vision  →  text laya  →  action        Qwen  writes  memory/  (short & playbook)
              (rates all moves)  (picks one, following advice)             (after every lost round + game end)
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
  The label rule is `sf2/system1/advice.py`; data `scripts/build_advice_data.py`; training `scripts/train_text_laya.py`.
  System 1 talks to it through a helper process (`sf2/system1/advisor.py`, `scripts/text_laya_server.py`).
- **Qwen** judges which advice is good; only Qwen writes `memory/`.

| Run / measure | Command |
|---|---|
| Play arcade mode in a window, learn until stopped | `python scripts/learn_loop.py` (`--minutes 10`, `--headless`, `--advisor off`, `--ab-advice`) |
| Does the short memory help? (headless, opponent locked, paired) | `python scripts/ab_memory.py --rounds 30` |
| Where the loop loses (see / choose / advice / walk / defend / lessons) | `python scripts/report.py gaps` |
| How much each memory rewrite changed (churn, flips) | `python scripts/report.py churn` |

Start omlx first (`~/work/omlx/start`). Findings and next steps: `docs/qwen_learning.md` and `~/work/me/journals/laya_vision-vs-SF2/`.

## Setup details (what this exact setup used)

Everything below ran on one Mac Studio (M3 Ultra, 32 cores, 256 GB). Three separate Python environments / servers:

| Piece | What | Where it runs | Settings that matter |
|---|---|---|---|
| Game | *Street Fighter II (USA)* SNES ROM, sha1 `7ddcb96e0d9fea94d9370635262ac7c28da85214`, in `roms/` (never committed) | Mesen 2 (MesenCE app 1.0, `/Applications/Mesen.app`), headless `--testrunner` or a window | controller 2 plugged in with `--snes.port2.type=SnesController`; window at 150% speed (`--speed`); the loop restores Mesen's `settings.json` afterwards (a window writes its overrides back) |
| laya-vision (System 1: sees) | base [`thaitea/laya-vision-smolvlm-256m`](https://huggingface.co/thaitea/laya-vision-smolvlm-256m) (SmolVLM-256M backbone) + our LoRA `runs/all8/best` | this repo's `.venv`: Python 3.12, PyTorch 2.14 on Apple MPS, transformers 5.17 | **256x256 native pixels, no upscale**: the 256x224 SNES frame is padded to 256x256, HUD rows 0-61 blanked, `image_interpolation: nearest`, GPU preprocess, no image splitting; two frames 4 apart + a one-line note (bars as words, no opponent name). LoRA r=16, alpha=32, 2 epochs, batch 8, lr 2e-4 (adapters) / 1e-4 (head), always from the base checkpoint; 22 actions per character, answer = hit / whiff / blocked / none / got_hit |
| Text laya (System 1: decides) | [`aac6fef/laya-mlx`](https://huggingface.co/aac6fef/laya-mlx): the MLX conversion of [`convaiinnovations/laya`](https://huggingface.co/convaiinnovations/laya), ModernBERT-large, 421M + our LoRA `runs/text_laya/advice_v1` | **laya-mlx** (Apple MLX runtime, not omlx): its own venv `~/work/laya_mlx/.venv` (uv, Python 3.12, mlx 0.32.2, laya-mlx 0.1.0); started as a helper process by `sf2/system1/advisor.py` | fp32; LoRA r=16, alpha=32 on every layer's `Wqkv`/`Wo`/`Wi` plus the decision head; AdamW lr 1e-4 cosine, batch 32, early stop (600 steps, ~6 min); max 512 tokens in; ~12 ms per decision. `HF_HOME=/Volumes/ExtremeSSD/huggingface`, `HF_HUB_OFFLINE=1` |
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
| Game state | `sf2/data/frames.py` (every frame has HUD rows 0-61 blanked); `sf2/vocab.py` (fighters, ranges, health bars, what the opponent is doing) |
| RAM values | `ram_maps/sf2_snes.txt`, format in `sf2/emu/ram.py` (`Var`, `load_map`, `CLOSE`, `MID`) |
| Action set (22 per char) | `sf2/data/vs_sweep.py` (`actions(char)`: 20 moves + block_high / block_low; special moves are macros with the ROM-tested timings from `sf2/data/vs_moves.py`); button names in `sf2/config.py` (`PAD`) |
| VS BATTLE (both pads) | `sf2/emu/vs.py`: boot to a 2-player fight. `sf2/data/vs_sweep.py`: place fighters at gaps, sweep all moves. `scripts/vs_dataset.py`: build the dataset for all 8 characters. |
| Stage 1 data | `sf2/data/vs_sweep.py` + `scripts/vs_dataset.py`: 20 actions x 3 ranges per character vs a still dummy, labelled hit / whiff / blocked / none from RAM; `scripts/audit_dataset.py`, `scripts/verify_replay.py` check it. Data is in `test_data/<char>/` (train/test splits with mirroring). |
| System 1: laya-vision | `sf2/system1/system1.py` plays a round and calls laya-vision; `sf2/system1/policy.py` builds its input (two frames + the note from `sf2/data/vs_sweep.py`). |
| System 1: text laya | `sf2/system1/advisor.py` starts the text-laya helper process (runs `scripts/text_laya_server.py` in the laya-mlx venv). `sf2/system1/text_laya.py` loads the fine-tuned model. `sf2/system1/advice.py` is the label rule (the grammar of valid advice lines). |
| System 1: game log | `sf2/system1/game_log.py` saves actions (what text laya read and picked, what move happened, did it follow the rule). |
| Learning loop | `scripts/learn_loop.py`: System 1 plays arcade mode, System 2 (async) learns. `sf2/system2/async_runner.py` manages the Qwen background thread. |
| System 2: prompts & checks | `sf2/system2/prompts.py` (what Qwen is asked), `sf2/system2/checks.py` (constraints on Qwen's reply). |
| System 2: memory | `sf2/system2/memory.py` (short memory per opponent, playbook); `sf2/system2/memory_churn.py` (how much it changed). |
| System 2: Qwen | `sf2/system2/qwen.py` calls the omlx server. `sf2/system2/notebook.py`: the notebook Qwen keeps between rounds (`scripts/notebook_run.py`). |
| System 2: code baseline | `sf2/system2/code_coach.py` writes advice lines from net hit points per try of each (move, range), no LLM. |
| Measurement: A/B | `scripts/ab_memory.py`: paired headless runs (same savestate, same seed, different advice arms). |
| Measurement: notebook | `scripts/notebook_run.py`: learning over one opponent for 40+ rounds. |
| Measurement: reports | `scripts/report.py {gaps,churn}`: where losses happen and how memory changes over time. |
| Evaluation | `sf2/eval/runner.py` (headless arm runner, common to all scripts); `sf2/eval/logs.py` (what counts as play data); `sf2/eval/stats.py` (A/B stats with bootstrap CI). |
| Video | `tools/make_video.py` (captions + audio sync, runs in `.venv-media`). Needs `~/Desktop/laya_video/work/` inputs. |

Directions are relative: `forward` is toward the opponent, `back` is away, and `block` is down-back. Buttons follow SF2's default SNES layout: **Y X L = jab / strong / fierce, B A R = short / forward / roundhouse**. If your in-game button config differs, change `PAD` in `sf2/config.py`.

## Stage 1: still-opponent data for all 8 characters

Python plays both pads in VS BATTLE (Mesen's saved settings leave port 2 empty; `sf2/emu/headless.py` plugs a pad in for
the run). Each character stands on the left, facing right, at 10 gaps per range (close < 55 <= mid < 120 <= far px)
against a dummy that stands, crouches or crouch-blocks, and presses each of its 20 actions. RAM gives the outcome.

```bash
python scripts/vs_dataset.py run --pairs ryu:chunli,ken:guile,honda:blanka,zangief:dhalsim   # ~5 min, 48 headless Mesens
python scripts/audit_dataset.py        # ~96k mechanical checks per character; exit 1 on any violation
python scripts/verify_replay.py        # re-records a random sample from each record's boot; must match byte for byte
                                       # (open issue: it fails on live-play rows, which have no boot to replay)
python scripts/train.py --out runs/all8 --data test_data/ryu --data test_data/ken --data test_data/chunli \
    --data test_data/guile --data test_data/honda --data test_data/blanka --data test_data/zangief --data test_data/dhalsim
```

`test_data/<char>/` (local only, git-ignored): `train.jsonl` (what `train.py` reads: `train_real` + `train_mirrored`, the same rows flipped), `test_real_left.jsonl` / `test_real_right.jsonl` (real frames at the held-out gaps, on each side: the mirroring check); row counts in `stats.json`. Besides the still-opponent rows there are block rows and capped live-play rows (`vs_dataset.py import-live`).

Each record: two model frames (4 frames apart, HUD blanked), the RAM note, the question
`sf2.data.vs_sweep.outcome_question(action)` (choice: hit / whiff / blocked / none) and its `label`, plus the measurements
(damage, frames until the fighter can act again, travel) and the boot savestate it came from. The special-move timings
are the ROM-verified ones from the move tests; charge moves charge on down-back so the gap does not change.

## Setup (Mac Studio)

1. **Mesen 2.** Install it from [mesen.ca](https://www.mesen.ca/) or [GitHub releases](https://github.com/SourMesen/Mesen2/releases), and open your SF2 ROM.
2. **Allow the script to use the network.** Open Debug → Script Window → Settings → Restrictions, and tick **"Allow network access"**. To override the port with `SF2_BRIDGE_PORT`, also tick "Allow access to I/O and OS functions".
3. **Python.** In this repo:
   ```bash
   uv venv -p 3.12 && source .venv/bin/activate
   uv pip install -e '.[model,dev]'
   uv pip install "laya @ git+https://github.com/r33drichards/laya-vision@568feeeada793f70f736756b0f3a7643d1e75910"
   .venv/bin/python -m pytest -q      # needs no ROM or model; the Lua-bridge tests run only if lua5.4 + LuaSocket are installed
   ```
4. **Load the bridge.** In Mesen's Script Window: Open → `mesen/sf2_bridge.lua` → Run. It shows "waiting for a Python script". Leave it loaded.
5. **Speed.** For learning and testing, set Mesen's emulation speed to maximum; the bridge still waits for Python on every decision.

Every emulator script starts with *"waiting for Mesen on 127.0.0.1:47800"* and continues once the bridge connects.

## Status

Working end to end: stage-1 data for all 8 characters, laya-vision LoRA `runs/all8/best`, text laya `runs/text_laya/advice_v1`, arcade learning loop with async System 2, and A/B measurement. Known gaps (2026-09-28): laya-vision's live move ratings do not rank well against a moving CPU, she almost never blocks, and Qwen's advice needs both the right inputs (net hit points per move, not miss rates) and a narrower scope (Stage A: fixed advice from counted data). See `docs/qwen_learning.md` for the plan.
