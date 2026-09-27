# Rules for coding agents

This is a hobby project. See AGENTS.md for how to run things and README.md for the loop.

1. **Red/green for every bug fix.** Write a test that fails on the buggy code, show it failing, then fix it and
   show it passing. Run `pytest -q` before each commit.
2. **Don't over-engineer; keep commits small.** Do the simplest thing that fixes the problem or answers the question.
   One concern per commit. Don't add scripts, options or abstractions nobody asked for.
3. **Emulator work runs in parallel.** Mesen runs on one core, so size the workers to the machine (table below).
   Never run emulator jobs one after another when they are independent:
   - **Split the work:** ROM tests, RAM searches, macro checks, data collection and play are split across headless
     workers, each with its own port (`scripts/parallel.py` style, ports 47810 and up). Use about 8–12 at once.
   - **Parallel is the default:** a new emulator script or test must be able to run as N copies at once, with no
     fixed port, no shared output file, and no shared savestate path it writes to.
   - **Per character, in parallel:** work that is per character (for example the 8 characters' ROM checks) runs one
     worker per character at the same time.
   - **Check it:** a serial run needs a reason written in the commit or report. Before a long run, check the load
     (`top`); an idle machine during a long serial run is a bug in the plan.

## Machines (checked 2026-09-27, macOS 27.0)

| Machine | Chip | CPU cores (perf + eff) | GPU cores | RAM | Reach | Role |
|---|---|---|---|---|---|---|
| **Mac Pro** (this one) | M4 Max | 16 (12 + 4) | 40 | 64 GB | local | Main dev; ROM tests; parallel play and collection (8–12 Mesen workers) |
| **Mac Studio** | M3 Ultra | 32 (24 + 8) | 80 | 256 GB | `ssh studio` (192.168.1.216); repo `~/work/hobby/laya_vision_vs_SF2` | Fine-tuning (fastest GPU); up to about 20 Mesen workers; Qwen System 2 on omlx `:8000` |
| **claw** (Mac mini) | M4 Pro | 12 (8 + 4) | 16 | 64 GB | the `claw` shell function (192.168.1.199) | Qwen fallback on omlx `:8800`; codex (GPT-6) blind reviewer; light Mesen work (about 6 workers) |

Measured speeds:
- **Play:** GPU-bound at about 27 decisions/s on the Mac Pro with 4 workers. The Studio does 36.2 decisions/s on 4
  workers and 11.0 on 1.
- **Training:** a short reference run took 163 s on the Mac Pro and 138 s on the Studio.
- **The GPU sets the worker count.** Workers without a model (ROM tests, RAM search, macro checks) are limited only by
  CPU cores, so use most of the performance cores. When omlx is serving Qwen on a machine, leave its GPU and about
  20 GB of RAM to Qwen.
