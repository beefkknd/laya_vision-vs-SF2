# Rules for coding agents

This is a hobby project. See AGENTS.md for how to run things and README.md for the loop.

1. **Red/green for every bug fix.** Write a test that fails on the buggy code, show it failing, then fix it and
   show it passing. Run `pytest -q` before each commit.
2. **Don't over-engineer; keep commits small.** Do the simplest thing that fixes the problem or answers the question.
   One concern per commit. Don't add scripts, options or abstractions nobody asked for.
3. **Emulator work runs in parallel.** Mesen runs on one core, and this Mac Pro has 16 (12 performance). Never run
   emulator jobs one after another when they are independent:
   - **Split the work:** ROM tests, RAM searches, macro checks, data collection and play are split across headless
     workers, each with its own port (`scripts/parallel.py` style, ports 47810 and up). Use about 8–12 at once.
   - **Parallel is the default:** a new emulator script or test must be able to run as N copies at once, with no
     fixed port, no shared output file, and no shared savestate path it writes to.
   - **Per character, in parallel:** work that is per character (for example the 8 characters' ROM checks) runs one
     worker per character at the same time.
   - **Check it:** a serial run needs a reason written in the commit or report. Before a long run, check the load
     (`top`); an idle machine during a long serial run is a bug in the plan.
