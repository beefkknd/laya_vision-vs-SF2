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
