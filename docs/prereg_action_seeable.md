# Pre-registration: which actions can be SEEN in two frames, and does laya-vision use the change? (2026-10-02)

Owner: "yes [run the blind audit], and also: what is a true set of distinct 'actions' that can be 'seen' by these two
frames. laya-vision is the 'see' part and should be able to capture the change. If NOT, we are building something it
is not supposed to solve."

Data: test_data_eye_q3v2b_act, test split only (2,419 per answer, 17 kinds). Checkpoint: runs/eye2_q3/best. No
training, no new data, no game.

## Fact already on hand (existing eval rows joined to each row's kind, before this pre-registration)
Per kind, share answered right by eye2_q3: walk toward .55 / away .57; hit in air .57, down .44, on ground .26;
jump in air .44, on ground .19; special ground .37, air .18; attack .29-.36 (all four kinds); stand crouch .31,
**standing still .07 (-> walk .49)**; block crouching .16, **standing .11 (-> walk .43)**.
Reading: still vs moving is where it fails - the one distinction that needs the CHANGE between the frames.

## A. Blind audit (codex on claw, a different engine) - owner approved
- Sample: test rows, seed 73101, 13 per kind x 17 kinds = 221 rows, stratified by kind, at most 1 row per match per
  kind.
- Each item shown as one picture: frame n-4 (left, "A") and frame n (right, "B", 4 frames = 1/15 s later), each 3x
  upscaled, the HUD visible. The auditor is told which fighter (left / right of the screen) and nothing else - no
  label, no kind, no RAM.
- Two passes, separate codex runs (the second must not anchor the first):
  - A1 open: "In a few words, what is that fighter doing from A to B? Then answer: moving or still? in the air or on
    the ground? a limb / attack out? being hit? If you cannot tell from these two pictures, say so."
  - A2 forced: the same 7 answers and descriptions the eye is asked, plus "cannot tell".
- Measured (script, from codex's answers vs the label): A2 agreement per answer and per kind; "cannot tell" rate per
  kind; A1 coded into the four yes/no facts and compared with RAM (moved >= 3 px between the frames; y != 192;
  attack box out; hit state).
- Reading, fixed now: a kind is SEEABLE if A2 agreement >= 0.70; SEEABLE-WITH-CHANGE if it needs the moving/still fact
  (walk vs stand, standing block vs walk away); NOT SEEABLE if < 0.50. 0.50-0.70 = unclear, reported as such. These
  bars are judgment ([LLM] advisory); the per-kind numbers are the result.

## B. Does laya-vision use the change? (script, mechanical)
Re-run eval_eye with eye2_q3 on the same test set, three inputs (one change each):
- B0 as trained (n-4, n) - the existing eval (0.351).
- B1 duplicated: (n, n) - no change between frames left.
- B2 swapped: (n, n-4) - time reversed.
Measured: balanced accuracy and per-kind recall for each; the walk / stand-still / standing-block confusion.
Reading, fixed now: if B1 is within 0.02 of B0 overall AND walk recall drops < 0.10, the eye does NOT use the change
(it reads one pose). If walk recall drops >= 0.10 under B1, it uses the change at least for walking.
Plus: how the model takes the two images (encoded together or separately) from the code - reported with file:line.

## Output
docs/eye_questions_v1.md gets a results section; the per-row audit and eval files in runs/eye2_q3/seeable/. Then a
proposal to the owner: the action answer set that is actually seeable (and what would need a change to the input,
e.g. a wider gap or a difference image) - nothing changed without the owner.

## Results (2026-10-02)
### A. Blind codex audit (221 rows, 13 per kind; files runs/eye2_q3/seeable/audit/, score.json)
A2 agreement overall 0.697 (laya eye2_q3 on the full test: 0.351); codex never said "cannot tell".
- SEEABLE (>= 0.70): down 1.00, standing attack .92, jump in air .92, crouch .92, crouching block .85, hit on ground .85,
  jump on ground .85, standing still .77.
- unclear: jump attack .69 (-> jump), air special .69 (-> jump), walk away .61, walk toward .54 (-> stand), ground
  special .54 (-> attack).
- NOT SEEABLE (< 0.50): hit in air .46 (-> jump), throw .46 (-> stand), crouching attack .39 (-> stand), standing block
  .39 (-> stand).
- A1 open facts vs RAM: in the air .906, being hit .906, moving .738; codex called real walks "not moving" ~40-45%.
- Deviation: all 15 A1 batches re-run once (first launch used an old codex CLI, every call failed with 400; no answers).
### B. Does laya use the change? (eval_dup.json, eval_swap.json)
- Balanced acc: B0 (n-4, n) 0.351; B1 duplicated (n, n) 0.331 (-0.020); B2 swapped (n, n-4) 0.346 (-0.005).
- Same answer as B0: B1 71.3%, B2 85.7%. Walk recall 0.56 -> 0.47 under B1 (-0.09).
- Per the pre-registered reading (B1 within 0.02 AND walk drop < 0.10): the eye does NOT use the change - it reads one
  pose. With no change at all it still says "walk" for 47% of walks and 41% of standing-still rows (a walking pose).
  Time reversal changes almost nothing: it does not know which frame is first.
### Reading
- Labels are right to RAM, but four kinds are not visible in two frames even to codex; the answer set must shrink.
- laya's gap to codex is mostly NOT motion: on single-pose kinds codex sees well and laya does not (standing attack
  .92 vs .31, jump in air .92 vs .44, down 1.00 vs .44). The likely limit is the input: 16 tokens per 256 px frame
  (a 4x4 grid of 64 px patches; subagent report, laya preprocess.py:103-105).

## C. Pre-registration: action at 512 px (owner 2026-10-02: "yes, lift that rule, and test out upscaled 512 px for action first")
- Change ONE thing vs runs/eye2_q3: laya's image size 256 -> 512 by nearest-neighbour 2x (every pixel an exact 2x2
  block; checked by a test that laya's own prep output equals numpy repeat). 64 image tokens per frame instead of 16.
- Same everything else as eye2_q3: data test_data_eye_q3v2b_act (7 answers, same rows and splits), base
  thaitea/laya-vision-smolvlm-256m, rank 16, alpha 32, epochs 1.584 (6,000 steps), batch 8, lr head 1e-4 / LoRA 2e-4,
  eval every 250, patience 10, --select acc, --balance sampling, --min-sampled-train 1000, val limit 4000, seed 0.
  Time limit raised 180 -> 360 min (512 is slower; the aim is the same 6,000 steps). Machine: threebody (RTX 4090),
  Qwen stays off. Output runs/eye3_q3_512.
- Measured on the held-out test (scripts/eval_eye.py, the same as eye2_q3): balanced accuracy with its 2.5% lower
  bound, per answer and per kind; then the frame-duplication probe B1 at 512.
- Reading, fixed now: 512 HELPS if its balanced accuracy minus eye2_q3's (0.351) is >= +0.03 and its lower bound is
  above 0.351; NO CHANGE within +/-0.03; also report the codex-seeable kinds (standing attack, jump in air, down,
  crouch, crouching block, hit on ground) separately - that is where more detail should show first. If it stops
  before 6,000 steps, both are compared at the same step too (eye2_q3's eval log).
- Not in this run: answer-set shrink, difference image (each its own run afterwards, owner's choice). Crop: dropped by the owner ("no crop", 2026-10-02).

## C. Results (2026-10-02): 512 HELPS
- runs/eye3_q3_512 (threebody, 6,000 steps, ~35 min - no slower than 256; best val acc 0.636 at the last step, still
  rising). Test balanced accuracy **0.644** (lower bound 0.635) vs eye2_q3 0.351 (+0.293; bar was +0.03) - HELPS.
- Recall 256 -> 512: attack .30 -> .42, special .34 -> .57, block .14 -> .65, walk .56 -> .84, jump .40 -> .71,
  stand .24 -> .68, hit .48 -> .63. Precision now .60-.74 for every answer (was .31-.46).
- Codex-seeable kinds: down .44 -> .88, crouch .31 -> .85, jump in air .44 -> .75, crouching block .16 -> .62, hit on
  ground .26 -> .31, standing attack .31 -> .39. Kinds codex could NOT see: standing block .11 -> .68 (codex .39),
  hit in air .57 -> .64, throw .36 -> .47, crouching attack .29 -> .36. Overall 0.644 vs codex 0.697 on its sample.
- B1 at 512 (frames duplicated): 0.644 -> 0.582 (-0.062), walk .84 -> .62 (-0.22), same answer 76.3%: by the
  pre-registered reading the 512 eye DOES use the change between the frames (at 256 it did not).
