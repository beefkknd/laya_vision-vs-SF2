# The self-learning loop: Qwen + text laya, screen only (plan, 2026-10-02, for the owner's review)

## What this project is
A self-learning loop. Qwen (System 2) writes the knowledge; text laya (System 1) turns it into moves; the game is seen
only through the screen. The question: **game after game, does Chun-Li play better as Qwen's rules are admitted and
rotated?** If not, find out why.

laya-vision is cut out for now (its 512 px action eye is parked on threebody: `runs/eye3_q3_512`). The screen is read
by the sprite reader instead. Swapping laya-vision back in later = plugging it into the same slot.

## The pieces and their roles

| Piece | Role | What it is NOT |
|---|---|---|
| Screen reader (`sf2/screen/`) | image -> facts: who, where, doing what, in the air, facing, health bars, fireball, round over. No RAM. | not a decider |
| Words (`sf2/system1/screen_words.py`) | facts -> the sentence text laya reads | |
| Qwen (System 2) | starts from web-research rules (owner's preference) or blank; watches games; writes, admits and retires rules (short memory -> advice lines) | not run every frame |
| Text laya (System 1, `runs/text_laya/advice_v1`) | reads the sentence + Qwen's advice + the move options; picks one move by FOLLOWING the advice. Trained on reading advice (a fixed rule over words), never on game outcomes. | has no game knowledge of its own; without advice it decides nothing useful |
| The table (`lessons/value_oracle_v1.json`) | stats of what beats what (built earlier from RAM-labelled games). A RULER: scores how good a Qwen rule is, offline. | NOT in play. Testing text laya + table measures the table, not the loop - dropped. |
| Replay (`scripts/replay_score.py`) | after a game, replays it with RAM to score it (hp, wins, reader vs truth). Refuses to score if the replay drifts. | never fed back to Qwen or play |

## One game, step by step
1. Screen frame -> reader -> facts -> words ("He is at mid range and jumping. My bar is full, his bar is half.").
2. Text laya gets the words + Qwen's advice lines + Chun-Li's move options -> picks one move -> buttons.
3. The run logs every decision: facts, words, advice in force, the pick, whether the pick followed the advice
   (`follows_rule`).
4. After the game: Qwen reads the screen-based record (damage from health-bar drops, his action from the reader, round
   results) and updates its rules: admit, keep, retire.
5. Next game with the new rules.

## What we measure
- **Does she get better?** hp per round and rounds won, game after game, per opponent (from the replay).
- **Are Qwen's rules good?** Each admitted rule scored by the table: is its move good in the situation it names?
- **Does text laya follow?** `follows_rule` per decision, per rule.

## When it does not improve: the diagnosis
1. Qwen writes bad rules -> the table's score of each rule shows it.
2. Text laya does not follow good rules -> `follows_rule` low. Then:
   a. the question / wording it is asked is wrong (e.g. a word it was never trained on), or
   b. the checkpoint is skewed (check `advice_v1` on held-out wordings: its own test set).
3. The reader feeds wrong facts -> the replay's reader-vs-RAM agreement per decision shows it.

## Gaps to fix before the loop runs - checklist
Each: what breaks if not fixed, and the fix.

- [ ] **G1. Text laya's options without the table.** Today the table builds the shortlist and the rating words
      ("likely works"). Breaks: with the table out of play there are no options / ratings. Fix: options = Chun-Li's
      move list (plain, no ratings); text laya follows a lesson if one applies, else its rule says "walk in". Decide
      the default when no rule applies (walk in? block?) and test that the rule still holds with unrated options.
- [ ] **G2. Starting rules.** Breaks: with a blank start, every decision is the default until Qwen has written
      something. Fix (owner's preference): web research -> starting rules for Chun-Li and per opponent, written in
      text laya's grammar, admitted through the normal checks (`advice.read`), logged as "web" so they can be retired
      like any rule.
- [ ] **G3. What Qwen observes, with no RAM.** Today Qwen's evidence (damage per move, his move, outcome, round
      results) comes from RAM logs. Breaks the no-RAM rule. Fix: a screen-based record per decision: damage dealt /
      taken from the health-bar drops after the move, his action from the reader, round result from the reader;
      feed THAT to Qwen. The replay stays referee only. Test: screen record vs replay RAM agreement.
- [ ] **G4. Text laya's grammar is too small for what Qwen will want to say.** It knows "when he jumps / crouches /
      attacks / stands / is stunned" + a range; no fireball, no specific move of his (docs/qwen_learning.md). Breaks:
      "when a fireball comes, jump" cannot be expressed. Fix: extend the advice grammar + text laya's training data
      (reading rules only, not game play), when Qwen first needs it - or now for the fireball, which the reader
      already sees.
- [ ] **G5. The table as the ruler.** Today the table ranks moves in play. Fix: an offline scorer: for each Qwen rule,
      look up its situation (range, he attacking / in the air) and score the rule's move against the table's values;
      report per rule, per game. The table's cells are coarser than some rules (no fireball cell) - report "not
      scorable" for those rather than guess.
- [ ] **G6. Qwen on.** Qwen is OFF on threebody by the owner's order. The loop needs it on: threebody
      (llama.cpp, NOTES.md commands) or the local omlx server. Owner decides.
- [ ] **G7. Diagnostics wired.** `follows_rule` per rule, the table's score per rule, the reader-vs-RAM agreement
      per game - in one per-game report, so "why didn't it improve" is answered from data.
- [ ] **G8. The loop test pre-registered** before any game: opponents, games per opponent, the starting rules, what
      counts as "better" (e.g. hp/round trend over games vs the first games, per opponent), and stop rules.

## Already done (2026-10-02)
- Screen reader, gated against RAM (position .98-.99, action at the catalog limit, health as drawn 1.0, 7 ms/frame).
- Screen-only play mode: no RAM in play (bridge without RAM, a handle that raises on RAM reads), offline replay
  scoring with a pixel-exact check, unknown sprite -> "block" + logged.
- Sprite catalog (game-captured) + the full ROM pose set (847 poses).

## Decisions for the owner
1. Qwen on: threebody or local omlx (G6).
2. Default move when no rule applies (G1).
3. Starting rules: web research (preferred) - which sources / how many rules (G2).
4. Extend text laya's grammar now (fireball) or when first needed (G4).
