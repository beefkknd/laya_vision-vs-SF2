# Runbook for coding agents (Codex, Claude Code) on the Mac

The project and its loop are in README.md; the plan is in PLAN.md. This file covers running things **headless and
in parallel** without a person at the Mesen window.

## One-time setup (a person does this once)
- Mesen 2 installed (default `/Applications/Mesen.app/Contents/MacOS/Mesen`). In Mesen: Debug > Script Window >
  Settings > Restrictions > tick **Allow network access**. The headless test runner uses the same settings.
- `export SF2_ROM=/path/to/Street\ Fighter\ II\ (USA).sfc` (and `SF2_MESEN=...` if Mesen is elsewhere).
- A fight-start savestate in `states/` and a verified `ram_maps/sf2_snes.txt` (README, day 1).
- `source .venv/bin/activate && pytest -q` passes.

## Headless, one worker
Add `--headless` to any emulator script. It starts a windowless Mesen (`--testrunner`) on its own port. Screenshots
switch to the raw screen buffer automatically if PNGs come back blank:

    python scripts/check_env.py --headless --savestate states/<fight>.state --me chunli --opp <opp>

## Headless, N workers (faster collection / evaluation)
`scripts/parallel.py` runs N copies on ports 47810.. with different seeds and start jitter, splits
--decisions / --matches between them, and merges into the usual `data/<name>` or `rollouts/<name>`:

    python scripts/parallel.py --workers 4 collect_teacher --name seed_teacher --decisions 40000 --eps 0.25 \
        --savestate states/<fight>.state --me chunli --opp <opp>
    python scripts/parallel.py --workers 4 play_teacher --name teacher --matches 12 --savestate states/<fight>.state --me chunli
    python scripts/train.py --data data/seed_teacher --out runs/r0
    python scripts/parallel.py --workers 4 play_student --model runs/r0/best --name r0 --matches 12 \
        --savestate states/<fight>.state --me chunli
    WORKERS=4 SAVESTATE=states/<fight>.state EXTRA="--me chunli --opp <opp>" scripts/dagger_round.sh 1

Worker logs: `out/parallel/<name>_w<i>.log`. On failure, read the log first. A worker that cannot connect usually
means network access is not ticked in Mesen's script settings, or SF2_ROM is wrong.

## Sizing on a Mac mini M4
- Collection (`collect_teacher`, `play_teacher`) is CPU-bound: start with **workers = performance cores**
  (4 on the base M4), then check `top` and scale up while throughput still grows.
- `play_student` loads the model once per worker (~1-2 GB each, sharing the one GPU): 2-3 workers on 16 GB,
  4 on 24-32 GB. More workers than that just queue on the GPU.
- Training (`train.py`) is one process on the GPU. It is not parallelised; don't run two at once. Collection for
  the next round can run beside it.

## Rules
- The harness must pass against the real ROM before any collection or training:
  `SF2_ROM=... pytest -q tests/test_rom_harness.py` (headless Mesen on port 47960, under a minute). A full
  passing run writes `out/harness_ok.json`; collect_teacher / play_teacher / play_student refuse to start without
  a stamp matching the ROM, the RAM map and the harness code (`SF2_UNVERIFIED=1` overrides, loudly).
- Judge a round by `scripts/gate.py` (round win rate, damage per round), never by val loss or frame accuracy alone.
- Don't reuse a `--name`: writers refuse to overwrite an existing dataset or rollout. Pick a new name or delete the old one.
- `pytest -q` before every commit (22 tests, no ROM needed; the Lua tests need `lua5.4` + LuaSocket).
