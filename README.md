# laya_vision-vs-SF2

This project teaches **laya-vision** to play **Street Fighter II: Special Champion Edition** (Sega Genesis) on a Mac Studio. It follows the plan in [PLAN.md](PLAN.md). A frame-level teacher labels frames. laya-vision learns to copy it. The student plays, the teacher labels the student's own frames (DAgger), and the model trains again. Round wins and damage decide whether a round of training worked; validation loss does not.

## What laya-vision actually is (read this first)

- **It is not part of `laya-mlx`.** `laya-mlx` (used in `../smoke_test.py`) is text-only. laya-vision is a separate research fork: [r33drichards/laya-vision](https://github.com/r33drichards/laya-vision). It uses a **SmolVLM-256M** image backbone with Laya's typed-decision head and the same `predict(state, questions)` API. It is **PyTorch** and runs on Apple **MPS**, not MLX.
- Checkpoint: [`thaitea/laya-vision-smolvlm-256m`](https://huggingface.co/thaitea/laya-vision-smolvlm-256m). It was trained on photo questions and knows no games.
- Its trainer freezes whole layers (`head` / `last_n` / `full`). **It has no LoRA**, so `sf2/lora.py` adds one: rank-r adapters on the text layers' attention and MLP projections, with the decision head fully trained. The adapters are merged back before saving, so a trained run is an ordinary laya-vision checkpoint.
- Upstream has already tried this loop on Atari ([docs/game-training.md](https://github.com/r33drichards/laya-vision/blob/main/docs/game-training.md)):
  - **two frames beat one**: median normalised score 0.201 vs 0.131
  - **one DAgger round lifted the median from 0.201 to 0.310**
  - sampling from the probabilities was worse than taking the top action
  - frame accuracy did **not** predict play strength

  So this project feeds **two frames** (`[4 frames ago, now]`) plus the text note, and keeps the gate on real play.

## Layout

| Plan piece | Here |
|---|---|
| Emulator / fight env | `sf2/env.py`: stable-retro wrapper. One call = one decision, and specials run as macros. It also tracks rounds and matches |
| Action set (12) | `sf2/actions.py`: `idle forward back jump crouch lp hp lk hk block hadouken shoryuken` |
| Text state | `sf2/ram.py`: `me=ryu opp=guile dist=mid my_hp=80 opp_hp=45 last=hadouken airborne=0 opp_airborne=1`, read from RAM |
| Teacher: you | `scripts/record_human.py` records a pad log, then `scripts/label_human.py` → `sf2/labeler.py` recognises fireball and DP motions |
| Teacher: scripted dummy | `sf2/teacher.py`: RAM rules that return a *distribution*, which becomes a soft target |
| Dataset | `sf2/dataset.py`: the exact JSONL layout laya-vision's `load_jsonl_examples` reads |
| LoRA | `scripts/train.py` + `sf2/lora.py`. Early stopping uses held-out frames from the same teacher |
| Student plays | `scripts/play_student.py` logs each frame: student action, teacher action, damage in the next 0.5 s, round result and distance |
| Relabel | `scripts/relabel.py`: `dagger` (the teacher labels student frames, with extra weight on hits taken and whiffed specials) or `filter` (keeps actions followed by damage_for > damage_against) |
| Gate | `scripts/play_teacher.py` (reference lines) and `scripts/gate.py` (win rate, damage per round, whiff rates) |
| One loop turn | `scripts/dagger_round.sh N` |

The plan's `left/right` became **`forward/back`**, relative to the opponent, so the model never has to work out which side it is on. `block` is down-back, the crouching guard. Specials use fierce punch.

## Setup (Mac Studio, Apple silicon)

```bash
cd laya_vision-vs-SF2
uv venv -p 3.12 && source .venv/bin/activate        # Python 3.11–3.13; separate from the laya-mlx venv
uv pip install -e '.[model,dev]'
uv pip install "laya @ git+https://github.com/r33drichards/laya-vision@568feeeada793f70f736756b0f3a7643d1e75910"
pytest -q                                            # 13 tests, no ROM or model download needed
```

**Your ROM.** stable-retro only knows the **Genesis** *Street Fighter II': Special Champion Edition* dump, and it checks the file's SHA-1: `a5aad1d108046d9388e33247610dafb4c6516e0b`. Import your own copy:

```bash
shasum ~/roms/sf2/*.md                                 # or .bin / .gen
python -m stable_retro.import ~/roms/sf2               # copies matching ROMs into stable-retro's data dir
python -c "import stable_retro as r; r.make('StreetFighterIISpecialChampionEdition-Genesis-v0').close(); print('ROM ok')"
```

If your copy is the SNES or arcade version, it will not import. stable-retro has no SF2 integration for those, so you would need a custom one (RAM map, states), and `sf2/ram.py` would change. Genesis is the fastest route.

The only savestate that ships is `Champion.Level1.RyuVsGuile` (Ryu vs the CPU's Guile, arcade level 1). Every script defaults to it. To fight Ken instead, go to vs mode or a later arcade stage in the recorder, press F5 at the fight start, and pass `--state states/<file>.state --opp ken` from then on.

## The week

```bash
# Day 1: hook the emulator, send each of the 12 actions, check the RAM map, record a pad log
python scripts/check_env.py --watch            # prints x/y/life per action, writes out/check/*.png
python scripts/record_human.py --session s1    # optional: 10–20 min of clean fireball / anti-air / block
python scripts/label_human.py --session human/s1 --name human_s1

# Day 2: seed set from the scripted teacher (epsilon-expert, soft targets); reference lines for the gate
python scripts/collect_teacher.py --name seed_teacher --decisions 30000 --eps 0.25
python scripts/play_teacher.py --name teacher --matches 10
python scripts/play_teacher.py --name random --policy random --matches 10

# Day 3: first LoRA, student vs the CPU, log wins and damage
python scripts/train.py --data data/seed_teacher --out runs/r0          # add --data data/human_s1 if you recorded
python scripts/play_student.py --model runs/r0/best --name r0 --matches 10 --watch
python scripts/gate.py rollouts/random rollouts/teacher rollouts/r0

# Days 4–6: DAgger. Student plays, teacher labels its frames (the hot ones in their own group), retrain, gate
scripts/dagger_round.sh 1
scripts/dagger_round.sh 2

# Cheap alternative to DAgger (no teacher): keep only student actions that won the next 0.5 s
python scripts/relabel.py --rollout rollouts/r0 --name filter_r1 --mode filter
```

On day 7, compare `scripts/gate.py rollouts/teacher rollouts/r0 rollouts/r1 rollouts/r2`. If win rate and damage per round have not moved, the labels are too coarse. Fix the teacher or add a macro such as `anti_air`; don't collect more frames.

## Check these on day 1 (they are the likely breakpoints)

1. **Position addresses** (`sf2/ram.py`). Life comes from the stable-retro integration. The x/y addresses come from another project on the same ROM and have not been checked here. In `check_env.py`, x should move during `forward` and `back`, and y should move during `jump`. If they don't, fix them before collecting any data: `dist`, `airborne`, facing and the whole teacher depend on them.
2. **Pad mapping** (`sf2/config.py: PAD`). X/Y/Z are punches and A/B/C are kicks in 6-button mode. The `lp`/`hk` screenshots should show a jab and a roundhouse.
3. **Macro timing** (`sf2/actions.py: MACROS`). `out/check/11_hadouken.png` should show a fireball. If it doesn't, lengthen each motion step from 3 frames to 4.
4. **`INTRO_SKIP`** (`sf2/env.py`): how long the "ROUND 2… FIGHT!" intro lasts, and **`CLOSE`/`MID`** (`sf2/ram.py`): the distance bins. Both are estimates.

## Speed and size on an M3 Ultra (rough)

- Teacher collection needs no model. It is limited by PNG writes, so 30k decisions take minutes.
- Student play is one `predict` per decision: two images plus a short prompt through a 256M model on MPS. Expect tens of ms per decision. The emulator waits for the model, so slow inference doesn't hurt play, only wall-clock time.
- Training runs in fp32 on MPS. LoRA keeps optimizer state small. A best checkpoint is a full ~1 GB save (merged weights).

## Status

The code has been tested with the unit tests plus a fake emulator run through collect → relabel → laya-vision's own loader. It has **not** been run against the real ROM or the real checkpoint, because the build machine had neither. Day 1's `check_env.py` is the first real test.
