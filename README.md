# laya text vs SF2: a self-learning loop (Qwen + text laya, screen only)

## What this project is
A self-learning loop. Qwen (System 2) writes the knowledge; text laya (System 1) turns it into moves; the game is seen
only through the screen. The question: **game after game, does Chun-Li play better as Qwen's rules are admitted and
rotated?** If not, find out why.

laya-vision is cut out for now (its 512 px action eye is parked on threebody: `runs/eye3_q3_512`). The screen is read
by the sprite reader instead. Swapping laya-vision back in later = plugging it into the same slot.

> Status (2026-10-02): the SCREEN READER is built and measured. The LOOP is NOT built yet - see "What does not exist
> yet". This README is the plan; it was blind-reviewed by Fable (docs/review_readme_fable_2026-10-02.md), and the
> claims below are corrected to match the code.

## Two hard rules (enforced by tests, not by care)
1. **No RAM in real play.** During a game the emulator gives frames and takes buttons, nothing else. RAM is read only
   OUTSIDE play: to score a game afterwards (offline replay) and to CHECK the screen reader. The play bridge has no RAM
   handle; the play emulator raises on any RAM read.
2. **The table is not a player, it is a ruler.** `lessons/value_oracle_v1.json` is stats of what beats what (built
   from RAM-labelled games). It may ONLY score how good a Qwen rule is, offline. It must never choose a move in play.
   (Testing text laya + table measures the table, not the loop - that is why it is useless for this project.)

Both are a HARD GATE: `scripts/hard_gate.py` scans the real play path (import closure from the play entry points +
`sf2/screen/`) and fails if any TABLE or RAM symbol is reachable. Allowed only in tests, the replay scorer, and
collection/gate tooling. See "Known debt" - the gate is RED today on purpose.

## The pieces and their roles

| Piece | Role | What it is NOT |
|---|---|---|
| Screen reader (`sf2/screen/`) | image -> facts: who, where, doing what, in the air, facing, health bars, fireball, round over. No RAM. | not a decider |
| Words (`sf2/system1/screen_words.py`) | facts -> the sentence text laya reads | |
| Qwen (System 2) | starts from web-research rules (owner's preference) or blank; watches games; writes, admits and retires rules (short memory -> advice lines) | not run every frame |
| Text laya (System 1, `runs/text_laya/advice_v1`) | reads the sentence + Qwen's advice + the move options; picks one move by FOLLOWING the advice. Trained on reading advice (a fixed rule over words), never on game outcomes. | has no game knowledge of its own; an UNGUIDED score (no Qwen advice) means nothing - it just follows whatever ratings it is handed |
| The table (`lessons/value_oracle_v1.json`) | a RULER only: scores how good a Qwen rule is, offline. Cells = (my char, range, opp attacking, opp airborne) - no opponent axis, no fireball cell. | NOT a player. Never in the play path. |
| Replay (`scripts/replay_score.py`) | after a game, replays it with RAM to score it (hp, wins, reader vs truth). Refuses to score if the replay drifts. | never fed back to Qwen or play |

## One game, step by step (the TARGET loop - not all wired yet)
1. Screen frame -> reader -> facts -> words ("He is at mid range and jumping. My bar is full, his bar is half.").
2. Text laya gets the words + Qwen's advice lines + the move options -> picks one move by following the advice ->
   buttons.
3. The run logs every decision: facts, words, advice in force, the pick, whether the pick followed the advice
   (`follows_rule`).
4. After the game: Qwen reads a SCREEN-BASED record (damage from health-bar drops, his action from the reader, round
   results) and updates its rules: admit, keep, retire. (Today this record is RAM-built - G3/M4.)
5. Next game with the new rules.

## What we measure
- **Does she get better?** hp per round and rounds won, game after game, per opponent (from the replay).
- **Are Qwen's rules good?** Each admitted rule scored by the table OFFLINE - generically (the table has no opponent
  axis, so "vs Honda" is scored only by range/attacking/airborne, not per opponent), "not scorable" for rules the
  table has no cell for (e.g. fireball).
- **Does text laya follow?** `follows_rule` per decision, per rule.

## When it does not improve: the diagnosis
1. Qwen writes bad rules -> the table's offline score of each rule shows it.
2. Text laya does not follow good rules -> `follows_rule` low. Then:
   a. the question / wording it is asked is wrong (e.g. a word it was never trained on), or
   b. the checkpoint is skewed (check `advice_v1` on held-out wordings: its own test set, test.json).
3. The reader feeds wrong facts -> the replay's reader-vs-RAM agreement per decision shows it.

## What IS built and measured (2026-10-02)
- **Screen reader** (`sf2/screen/`), checked against RAM on held-out games (out/screen_gate/s303/gate.json):
  identity 100%, position within 4 px .98-.99, action .89/.83 (at the catalog's own ceiling), **health as drawn 1.0**,
  7 ms/frame. CAVEAT: the gate script still exits FAIL on one bar (round-over on time-over rounds, 1 frame early) -
  accepted as a wording defect (see Known debt). Health vs RAM *life* is only .94-.97 (the bar shows drawn hp, which
  drains); the health WORD at decisions matched RAM .63-.74 in the one smoke game (M3).
- **No-RAM play plumbing**: a RAM-free bridge (`mesen/sf2_bridge_screen.lua`) and a handle that raises on RAM reads
  (`sf2/system1/screen_emu.py`); offline replay scoring with a pixel-exact drift check (`scripts/replay_score.py`);
  unknown sprite -> "block" + logged.
- **Sprite catalog** (game-captured) + the full ROM pose set (847 poses, out/sprite_rom/).

## What does NOT exist yet (the loop itself)
- **M1. A Qwen-in-loop screen runner.** `scripts/play_screen.py` is NOT the loop: it hardcodes "Advice: none", Qwen
  off, and it builds `System1(oracle=table)` - i.e. it is the table-in-play arm the owner rejected, usable only as a
  reader harness. The real runner (screen facts -> Qwen lessons -> text laya, no table, no RAM) must be written. This
  is why the hard gate is RED today.
- Qwen's evidence is RAM-built end to end (G3/M4), so admission and retirement must be rebuilt on the screen record,
  not re-pointed.

## Gaps to fix before the loop runs - checklist
Each: what breaks if not fixed, and the fix.

- [ ] **G1. Text laya's options without the table.** Today text laya is handed the table's shortlist + rating words
      ("likely works"); `advice.answers` needs every option to carry a rating. Breaks: with the table out of play
      there are no ratings. Fix: options = Chun-Li's move list (plain); text laya follows a lesson if one applies,
      else the default. This is a RETRAIN of text laya on a new question shape (unrated options), not just rewiring.
      Decide the default when no rule applies (walk in? block?).
- [ ] **G2. Starting rules.** Breaks: with a blank start, every decision is the default until Qwen has written
      something. Fix (owner's preference): web research -> starting rules, written in text laya's grammar, admitted
      through the normal checks (`advice.read`), tagged "web" so they rotate like any rule. No "web" source exists yet.
- [ ] **G3. What Qwen observes, with no RAM (UNDERSTATED before).** Qwen's whole evidence path is RAM: damage
      (dealt/taken), his move naming (fireball/uppercut/throw via `opp_moves.py`), hit/whiff/blocked outcome, round
      results (`game_log.py`). Breaks the no-RAM rule AND feeds lesson `cause`/threat views. Fix: a screen record -
      damage from health-bar drops, his action (7 labels + fireball) from the reader, round result from the reader -
      and REBUILD the lesson evidence/threat/`cause` paths on it (not a re-feed: the screen gives no hit/whiff/blocked
      or named special yet). Test: screen record vs replay RAM, at DECISION granularity (M3).
- [ ] **G4. Text laya's grammar is too small.** It knows "when he jumps / crouches / attacks / stands / is stunned" +
      a range; no fireball, no specific move of his. The reader DOES see fireballs but the words never say so. Fix:
      extend the advice grammar + text laya's training data (reading rules only), for the fireball now and more when
      Qwen needs it.
- [ ] **G5. The table as the ruler (offline scorer).** Today the table ranks moves IN PLAY (`system1._by_table`); that
      must leave play. Fix: an offline scorer - for each Qwen rule look up its cell and score its move; report per
      rule, per game; "not scorable" where the table has no cell (fireball, and per-opponent, which the table cannot
      do).
- [ ] **G6. Qwen on.** OFF on threebody by the owner's order. The loop needs it on: threebody (llama.cpp, NOTES.md) or
      local omlx. Owner decides.
- [ ] **G7. Diagnostics wired.** `follows_rule` per rule, the table's offline score per rule, reader-vs-RAM per game -
      one per-game report. No code yet.
- [ ] **G8. The loop test pre-registered** before any game: opponents, games per opponent, starting rules, what counts
      as "better" (hp/round trend over games, per opponent), stop rules.
- [ ] **M2. Gate round-over waiver.** The reader gate FAILS on time-over rounds ending 1 frame early; state the waiver
      or fix the wording so the gate exits 0.
- [ ] **M4. Lesson retirement on the screen record.** `lessons.review/stop` keys on per-game RAM hp; move it to the
      screen/replay record.

## Decisions for the owner
1. Qwen on: threebody or local omlx (G6).
2. Default move when no rule applies (G1).
3. Starting rules: web research (preferred) - which sources / how many rules (G2).
4. Extend text laya's grammar now (fireball) or when first needed (G4).
