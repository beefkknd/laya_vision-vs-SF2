# Two-System Fighter: plan (v5, 2026-09-27)

v3 and earlier are in git history. They trained System 1 to copy a scripted player, with basic moves only, and judged
it on win rates. Both are out of scope, and the scripted player is gone.

## The goal of this phase

You launch one command, and the learning loop runs:

1. **System 1 (laya-vision) plays a round.** Every 4 frames it picks a move from its character's move list. It
   carries a **short memory** of the round (its recent moves and what each one did), so it *sees* the effect of its
   play as it goes.
2. **System 2 (Qwen) reviews the round.** It reads that round's short memory next to the **playbook**, checks
   whether the play was consistent with the playbook and whether the playbook's moves did what they should, and
   updates the playbook.
3. The next round plays with the updated playbook. Repeat.

**laya's job is mechanics:** play each character correctly, with every move and combo on its sheet, facing the
right way, at a sensible spacing. **How well it does against a given opponent is not laya's job,** and nothing in
this phase measures it. The game is the judge of what a move did: the RAM says whether it came out, hit, was
blocked, or whiffed.

## The three memories

| | What it is | Written by | Read by |
|---|---|---|---|
| **Checkpoint** | laya's weights: reading the screen, and the character's moves as defined | training (P5) | System 1 |
| **Short memory** | this round's log: each decision (note, move, and the move's effect: came out / hit / blocked / whiffed, damage) | the harness, live from RAM | System 1 (its last move's effect is in the note) and System 2 (the whole round) |
| **Playbook** | situation rules (`field=value ... -> move # reason`), starting empty | System 2 (and you) | System 1, before every move (the τ nudge) |

## What we keep (built and verified)

- **Harness:** the Mesen lockstep bridge, the headless runner, and the ROM acceptance suite with its stamp, on 3
  machines.
- **Characters:** boot and pick any of the 8 (`sf2/boot.py`), with fight-start savestates for all 8.
- **Directions:** relative (forward and back), resolved from the fighters' x positions at decision time.
- **Playbook plumbing:** the rule format, the τ nudge, fired/changed logging, and the Qwen writer. That is
  `contract.py`, `memory.py`, `moments.py`, `system2.py` and `learn.py`.
- **Qwen:** `qwen38-27b-oq4e-mtp` on omlx, at the Studio `:8000` and claw `:8800`. Thinking off works.
- **Runs and logs:** parallel headless play, `out/live.log`, the `out/results.jsonl` ledger, and the balance gate.

## Priority list

**Can we test it?** Partly. The ROM tests already prove that a macro produces a move. The short memory also needs
hit, blocked and whiffed from RAM (P2), so P2 comes before P3–P5.

- [ ] **P0 Strip the teacher from the code.** Remove:
  - `sf2/teacher.py`, `collect_teacher`, `play_teacher`, `relabel` and `dagger_round.sh`;
  - the DAgger path in `loop.py`;
  - their tests, and `TEACHER.md`.

  The label field becomes neutral (`label_probs`). `pytest -q` is green. README, JOURNAL and PROGRESS stay as the
  history of r0 to r2.
- [ ] **P1 Move sheet:** `docs/MOVES.md`, compiled from sources and cited. For each of the 8 characters it lists:
  - the shared basics;
  - its specials, with their exact input (motion, charge with its time, or mash);
  - a few standard combos.

  For each move it also says what "working as defined" looks like in the game. For example: a fireball puts a
  projectile on screen; an uppercut rises; a combo's second hit lands during the hit-stun of the first.
- [ ] **P2 RAM for effects.** For every character, from both sides:
  - each special's state id;
  - hit, blocked or whiffed;
  - hit-stun, so a combo can be confirmed.

  Each one is found and tested on the ROM before it is used.
- [ ] **P3 Macros, verified on the ROM.** Each move on the sheet is a macro. A ROM test checks it from both sides:
  it must fail with a wrong input and pass with the right one. The move list shown to laya is per character.
- [ ] **P4 Short memory.**
  - The harness logs each decision's effect as the round plays.
  - The note gains `last_result=` (came out, hit, blocked or whiffed), so System 1 sees the effect of its last move.
  - Each round writes its short memory to a file.
  - Tests: a known macro against a known state gives the expected effect.
- [ ] **P5 Train System 1 from the base model** (`thaitea/laya-vision-smolvlm-256m`), so that each character's moves
  work as defined:
  - **Data:** each character plays every move on its sheet, at mixed distances and from both sides. A decision is
    kept as a label only when its effect matched the definition from P1 and P2.
  - **Balance:** the same number of rows per character.
  - **The note:** it includes `last_result`, so laya learns to read the effects of its own moves.
  - **Pass:** per character, the moves laya chooses come out as defined. There is no effectiveness test and no
    opponent comparison.
- [ ] **P6 The loop, one round at a time.** Change `learn.py` from batches to rounds:
  - laya plays a round with the playbook;
  - Qwen gets that round's short memory, the playbook, and the character's move sheet;
  - Qwen checks the play against the playbook and returns edits to it (add, change, or drop rules, each with a
    reason);
  - the playbook is validated, saved as a snapshot, and used for the next round.

  Every round is logged: the playbook version, the short-memory file, Qwen's reply and its edits. Verified when one
  launch runs N rounds end to end, and every playbook edit can be traced back to a round.
- [ ] **P7 Live view.** It shows the Mesen window, the console, and a local page with the playbook timeline and each
  round's short-memory summary, all read from the logs.

## Decisions

- [x] The teacher is gone. System 2 is the only coach, and the playbook starts empty.
- [x] System 2 reviews every round (it used to review a batch).
- [x] System 1 is trained only to make each move work as defined. Effectiveness is not its job.
- [x] System 1 is trained from the base model only, with no character favoured.
- [ ] Whether the next round waits for Qwen's review: yes by default (in lockstep, the game waits).
