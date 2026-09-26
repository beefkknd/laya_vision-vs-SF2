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
  mesen/sf2_bridge.lua  ── TCP 127.0.0.1:47800 ──▶  sf2/mesen.py  →  sf2/env.py  →  teacher / student / recorder
  every input poll:                                   RUN n frames of input
    apply the next planned input, or                  ◀── RAM values for every frame
    report and wait for Python                        ◀── screenshots (4 frames before the end, and the end)
```

- **The emulator never runs ahead of Python.** At each input poll, the Lua script applies the next planned input. When the plan runs out, it blocks until Python sends the next command. How long the model takes to decide never changes the fight.
- **Special moves are macros.** The model picks `hadouken`, and the glue sends the 12-frame quarter-circle + fierce input as one run.
- **Python writes every file** (savestates, screenshots, logs). The Lua script only needs network access.
- **The script can stay loaded all day.** When a Python script ends, it disconnects. The Lua script reconnects to the next one within a second.

## What laya-vision actually is

- **It is not `laya-mlx`.** [`laya-mlx`](https://github.com/mizorewww/laya-mlx) (the Apple MLX runtime) is text-only. laya-vision is a separate research fork, [r33drichards/laya-vision](https://github.com/r33drichards/laya-vision). It uses a **SmolVLM-256M** image backbone with Laya's typed-decision head and the same `predict(state, questions)` API. It is **PyTorch** and runs on Apple **MPS**.
- **Checkpoint:** [`thaitea/laya-vision-smolvlm-256m`](https://huggingface.co/thaitea/laya-vision-smolvlm-256m). It was trained on photo questions and knows no games.
- **No LoRA upstream.** Its trainer only freezes whole layers. `sf2/lora.py` adds rank-r adapters to the text layers and merges them back before saving, so a trained run is an ordinary laya-vision checkpoint.
- **Upstream already tried this loop on Atari** ([game-training.md](https://github.com/r33drichards/laya-vision/blob/main/docs/game-training.md)):
  - two frames beat one (median score 0.201 vs 0.131)
  - one DAgger round lifted the median from 0.201 to 0.310
  - taking the top action beat sampling
  - frame accuracy did not predict play strength

  So this project uses **two frames** (4 frames ago and now) plus a text note, and the gate is real play.

## Layout

| Plan piece | Here |
|---|---|
| Emulator | `mesen/sf2_bridge.lua` (inside Mesen) + `sf2/mesen.py` (Python end) |
| Fight env | `sf2/env.py`. One call = one decision. It tracks rounds and matches from the life values; an episode is one match from your savestate |
| RAM map (per cartridge) | `ram_maps/sf2_snes.txt`, found by `scripts/find_ram.py` (`sf2/ramsearch.py`) |
| Action set (12) | `sf2/actions.py`: `idle forward back jump crouch lp hp lk hk block hadouken shoryuken` |
| Text state | `me=chunli stand hp=80 opp=dhalsim jump hp=45 dist=mid facing=right corner=opp time=late last=hk` (`sf2/ram.py`: each fighter's state word stand/crouch/jump/block/attack/hit/other, whose back is to a wall, round clock early/mid/late) |
| Teacher: you | `scripts/record_human.py` (you play in Mesen), then `scripts/label_human.py` → `sf2/labeler.py` recognises fireball and dragon-punch motions |
| Teacher: scripted dummy | `sf2/teacher.py`. RAM rules that return a distribution, used as a soft target |
| LoRA | `scripts/train.py` + `sf2/lora.py`. Early stopping uses held-out teacher frames |
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
   pytest -q      # 22 tests. Needs no ROM or model; the Lua-bridge tests run only if lua5.4 + LuaSocket are installed
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

**Why the RAM finder exists.** World Warrior, Turbo and Super SF2 (and each region) keep life and positions at different addresses, and I couldn't verify any of them without your ROM. `find_ram.py` dumps the full 128 KiB of work RAM during known phases: walk right, walk left, jump, get hit, land hits. It keeps the addresses that behave like life, x and y. You can also read them off Mesen's memory viewer (Debug → Memory Tools) and write `ram_maps/sf2_snes.txt` by hand; the format is in `sf2/ram.py`. Commit the map once `check_env.py` looks right.

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

## Headless and parallel runs

Every emulator script takes `--headless`. It then starts its own windowless Mesen (`Mesen --testrunner <rom> <bridge>`) on its own port, so no window or script loading is needed. Set `SF2_ROM` (and `SF2_MESEN` if Mesen isn't in `/Applications`). Screenshots switch to Mesen's raw screen buffer automatically if the headless PNGs come back blank.

`scripts/parallel.py` runs N of those at once, each with a different seed and a random idle start, so the workers don't replay the same fight. It splits `--decisions` / `--matches` between them and merges the results into the usual `data/<name>` / `rollouts/<name>`:

```bash
export SF2_ROM=~/roms/sf2.sfc
python scripts/parallel.py --workers 4 collect_teacher --name seed_teacher --decisions 40000 --eps 0.25
python scripts/parallel.py --workers 4 play_student --model runs/r0/best --name r0 --matches 12
WORKERS=4 scripts/dagger_round.sh 1
```

Parallel workers make **collection and evaluation** faster. Training stays one process on the GPU. [AGENTS.md](AGENTS.md) is the runbook for coding agents, including worker counts for a Mac mini M4.

## Check these on day 1 (the likely breakpoints)

1. **RAM map.** `check_env.py` must show x moving on `forward`/`back` and y on `jump`. The teacher, the text note and facing all depend on them.
2. **Buttons (`PAD`).** The `lp`/`hk` screenshots should show a jab and a roundhouse.
3. **Macro timing (`sf2/actions.py`).** `out/check/11_hadouken.png` should show a fireball. If it doesn't, lengthen each motion step from 3 frames to 4.
4. **Estimates to tune:** `INTRO_SKIP` (`sf2/env.py`), and `CLOSE`/`MID` (`sf2/ram.py`, SNES pixels).
5. **KO detection.** A round ends when a life value goes negative, or when both bars refill (time over, or a cart that stops at 0). If rounds never end in the logs, look at the life values around a KO in `check_env.py`.

## Status

The unit tests cover the actions, the labeler, LoRA merging and the dataset → laya-vision loader. They also run the **real `sf2_bridge.lua`** in stock Lua 5.4 + LuaSocket against a mock of Mesen's `emu` API: RUN/WATCH/DUMP/savestates, the full environment, the scripted RAM finder, and reconnecting. None of it has run inside Mesen itself, or with your ROM or the real checkpoint yet. Day 1 is the first real test.
