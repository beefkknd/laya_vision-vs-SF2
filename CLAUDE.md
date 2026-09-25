# Rules for coding agents

This is a hobby project. See AGENTS.md for how to run things and README.md for the loop.

1. **Red/green for every bug fix.** Write a test that fails on the buggy code, show it failing, then fix it and
   show it passing. Run `pytest -q` before each commit.
2. **Don't over-engineer; keep commits small.** Do the simplest thing that fixes the problem or answers the question.
   One concern per commit. Don't add scripts, options or abstractions nobody asked for.
