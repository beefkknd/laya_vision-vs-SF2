# Two-System Fighter: plan (v4, reset 2026-09-27)

v3 is in git history. It was reset because it drifted: it trained System 1 to copy a scripted player, with basic
moves only (no specials, no combos), and then judged it on win rates against Ryu. Both are out of scope.

## Who does what

- **System 1 (laya, the hands):** reads the screen and the RAM note, and picks one move every 4 frames from **its
  character's move list**. Its job is **mechanics**: play each character correctly, with every move and combo assigned
  to it, facing the right way, at a sensible spacing.
- **System 2 (Qwen, the coach):** after each batch, reads what went wrong and writes situation rules into a memory.
  System 1 consults the memory before every move. **System 2 is the only coach.** The playbook starts empty.
- **The game is the judge:** whether a move came out, hit, whiffed or faced the wrong way is read from RAM, never
  from another player's opinion.
- **Not laya's job:** how character A does against character B. There are no matchup gates.

End goal: start weak, System 1 plays continuously, System 2 refines the memory in batches, and a live view shows
System 1 playing, System 2's rules, and each character's mechanics improving.

## What we keep (built and verified)

- Harness: Mesen lockstep bridge, headless runner, ROM acceptance suite and harness stamp, on 3 machines.
- Boot and pick any of the 8 characters (`sf2/boot.py`), and fight-start savestates for all 8 (`make_savestate.py`).
- Relative directions: macros use forward/back, resolved from the fighters' x positions at decision time.
- The memory and System 2 plumbing: `contract.py`, `memory.py`, `moments.py`, `system2.py`, `learn.py` (batched).
- Qwen `qwen38-27b-oq4e-mtp` on omlx: the Studio `:8000` and claw `:8800`. Thinking off works.
- Parallel headless play, `out/live.log`, the `out/results.jsonl` ledger, and the balance gate (per character).

## Retired (not used again)

- The scripted teacher: its code, its labels, and anything trained on them. That means r0/r1/r2, v0 (`base_all_v0`),
  the B3a and B3b data, and the Chun-Li-vs-Ryu playbooks.
- All matchup checkpoints and the paired win-rate gates against Ryu.

## Priority list

Testability: **PARTIAL**. The ROM tests can already prove that a macro produces a move, as they did for Lightning
Legs. Measuring hits, whiffs and combos in play needs RAM we have not mapped yet (P2), so P2 comes before any gate.

- [ ] **P0 Reset.**
  - [x] Plan rewritten without the teacher (this file); teacher playbooks and teacher facts removed from `memories/`.
  - [ ] Remove the teacher from the code: `sf2/teacher.py`, `collect_teacher`, `play_teacher`, `relabel`,
    `dagger_round.sh`, the DAgger path in `loop.py`, their tests, and `TEACHER.md`. The label field becomes neutral
    (`label_probs`). `pytest -q` is green after.
- [ ] **P1 Move sheet (owner approves).** For each of the 8 characters, a sheet listing:
  - the shared basics;
  - its specials, with their input type (motion, charge or mash);
  - the combos assigned to it.

  I draft it from the game, and you approve or edit it. It is the spec for P3 and P4.
- [ ] **P2 Observability (RAM).** For each character, find and test in RAM:
  - the state or animation id of each special;
  - hit, blocked or whiffed for each attack;
  - hit-stun or the combo count, so a combo can be confirmed.

  Each one is checked on the ROM before it is used.
- [ ] **P3 Macros, verified on the ROM.** Every move on the sheet is a macro. Each one gets a ROM test run from both
  sides (facing left and facing right). Red first: the test must fail with a wrong input, then pass with the right
  one. Charge moves hold back or down for their charge time. The question lists that character's moves, so one
  checkpoint works for every character.
- [ ] **P4 The mechanics report (the gate) [SCRIPT].** For each character and each move, from any rollout:
  - how often it was chosen;
  - how often it came out (RAM);
  - how often it hit, was blocked or whiffed, by distance band;
  - whether it faced the right way;
  - for each assigned combo, how often it was started and how often every hit connected.

  The pass bars are set before any run, and you approve them. The report must be seen failing on a seeded bad macro
  and on random play before it is admitted.
- [ ] **P5 Labels for System 1, with no teacher.** Recommended: exploration judged by the game. Play each character
  with random choices from its full move list. Keep the decisions whose move came out and connected, and was not
  punished within 0.5 s. Train on those. Use the same number of rows per character (balance gate), and opponents
  only for variety. Prototype on one character, check the P4 report, then do all 8.
- [ ] **P6 Train System 1 from the base model** (`thaitea/laya-vision-smolvlm-256m`) on the P5 labels, with each
  character's move list in the question. Gate: the P4 report, per character, at the bars set in P4, and better
  than random play on every character.
- [ ] **P7 Wire the loop.** Point `learn.py` at the P6 checkpoint, one character per session:
  - System 2's feedback becomes the P4 report;
  - its moments become failed executions, whiffs, wrong facing, dropped combos, and hits taken;
  - Qwen gets that character's move sheet and no tactics, and the memory starts empty.

  Verified when a session's memory measurably raises that character's P4 numbers on unseen openings.
- [ ] **P8 Live view.** The arcade console plays continuously with the memory on. Qwen writes rules between batches. A
  local page shows the rules timeline and a mechanics curve per character, read from the ledger.

## Decisions

- [x] System 2 learns in batches; live comes later.
- [x] The playbook starts empty; the owner's rules outrank Qwen's; each rule is a τ nudge (see `sf2/memory.py`).
- [x] Train System 1 from the base model only; no character gets more data than another.
- [ ] P1: which combos each character is assigned (owner).
- [ ] P4: the pass bars (owner approves).
- [ ] P5: the label source; exploration judged by the game is recommended.
