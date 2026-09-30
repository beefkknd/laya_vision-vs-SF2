# Does laya-vision need a fine-tune? Synthesis of two independent analyses (2026-09-30)

Dr Fable (Fable model) and GPT-6 (gpt-6-astra via codex), same brief (2026-09-30_finetune_decision_brief.md), blind to
each other. Full texts: 2026-09-30_dr_fable_finetune_decision.md, 2026-09-30_gpt6_finetune_decision.md.

## Both agree
- No laya-vision fine-tune now. The ranking TARGET was the defect; a learned value head was one possible repair, and
  the table is a working one with no model.
- Neither value run tested the idea: both stopped before the value head trained (run 2 kept step 3000 = 6.4% of the
  schedule, ~4.5% of Chun-Li's rows seen once; its value head learned less than a lookup over its own note).
- Random full games give one noisy label per frame (net sd ~16.6 hp; the note explains ~7% of it). Fine for a table
  that averages ~48 samples per cell; too noisy for a per-frame model at this data size.
- Next: the table as System 1's ranking, its expected net mapped to text laya's three rating words through a common
  advice interface, and a pre-registered 2x2: System 1 ranking {runs/all8, table} x advice {none, book + Qwen loop},
  same text laya in every arm, 6 opponents x 8 seeds, seed as unit. It answers the owner's question "has Qwen already
  learned this?" (book +47 and table +81 are not comparable today: different controls).
- The owner's controlled mode is right for any later fine-tune: one situation, every response from the SAME saved
  frame, a fixed window, no sequences (not "BA, CA, DA"); split whole episodes; gate on regret (the net lost by the
  picked move vs the best move on held-out states), anchored to the table; then in play vs the table.
- Ownership: table = general value prior (System 1); laya-vision = perception (outcomes, general opponent flags);
  Qwen + book = who the opponent is (per-opponent deviations: e.g. lightning legs vs Ryu/Zangief where the throw is the
  general answer); a later fine-tune would own timing inside a situation.

## Where they differ
- The offline calibration gate: GPT-6 found it would reject the table itself (verified: monotone in 2 of 9 files);
  Dr Fable called the evaluation fine. GPT-6 is right: harness ledger #26; regret on branched states replaces it.
- How to pick situations for controlled collection: GPT-6 scripts 5 general families (idle up close, ground attack,
  approach, jump-in, projectile) x 8 characters (12.8k-38.4k branches); Dr Fable branches at natural decision points
  of normal play (every move from each saved state, stratified by the note's cell, 50-80k rows, ~1-1.5 h on 12
  processes), which needs no scripted opponent behaviour and no names.
- Size of the prize: Dr Fable estimates what frames could add (his move right now) at +1 to +9 hp/decision by proxies,
  smaller than what knowing the opponent adds (Qwen's job); GPT-6 asks for a richer note-only baseline (exact dx etc.)
  before crediting vision.
- Weak control: Dr Fable notes part of +81 may be "stop walking into him" (the old rule's walk-in fallback); the 2x2
  separates it.
