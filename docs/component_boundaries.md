# Component boundaries: what fails where (2026-09-30)

Owner: "learn the boundary of each component ... understand the root cause, then decide which is caused by vision
laya, which is text laya, which is Qwen." Evidence: three pre-registered fixed-advice batches (players' tips vs Qwen's
own lessons vs no advice; 8 seeds x 15 rounds per arm; docs/prereg_{ryu,ken,honda}_expert.md), traced stage by stage
with `scripts/boundary.py` (per-line traces: docs/boundary/*.json), plus the harness ledger (docs/harness_ledger.md).

## The results the traces explain

| hp per round vs no advice (seed as unit, 95%) | Ryu | Ken | Honda |
|---|---|---|---|
| use more throw up close (one line) | **+42.4 [+27, +59]** | **+64.0 [+42, +85]** | **+49.5 [+29, +70]** |
| players' 5-line set | **+37.9** | **+57.9** | **+17.0** |
| Qwen's 5 lines most in play in round 5 | **+24.4** | +13.8 (not shown) | **-29.7 [-46, -14] hurts** |
| players' set - Qwen's set | +13.5 (not shown) | **+44.1** | **+46.8** |
| rounds won of 120: none / throw | 32 / 54 | 13 / 50 | 10 / 26 |

## Six opponents (2026-09-30): the three new ones (docs/prereg_new_opponents.md)

| hp per round vs no advice (seed as unit) | Ryu | Ken | Honda | Zangief | Guile | Dhalsim |
|---|---|---|---|---|---|---|
| use more throw up close | **+42.4** | **+64.0** | **+49.5** | **+19.9** | **+59.8** | **+37.5** |
| always throw up close | - | - | - | +19.9 (same decisions) | +59.8 (same) | **+46.7** (+9.2 over "use more") |
| rounds won of 120: none -> throw | 32 -> 54 | 13 -> 50 | 10 -> 26 | 25 -> 45 | 41 -> 72 | 25 -> 58 |
| players' set | +37.9 | +57.9 | +17.0 | **-27.3 hurts** | +29.7 | +8.2 |
| Qwen's loop (character_fgc, own runs) | +14.1 (8) | +31.2 (8) | -7.6 (8) | -5.8 (4) | **+14.6 (4)** | -22.1 (4) |

The throw helps against all six, including Zangief and Dhalsim whose throws outreach hers: the claim that laya-vision's
P(hit) ranking hides a high-value move is general. Up close the throw is in laya-vision's top 3 in 0-19% of decisions
against every opponent; named by a lesson it is rarely vetoed ("use more" and "always" played the same decisions vs
Zangief and Guile).

New boundary - substitution: "avoid spinning_bird_kick" alone vs Zangief -44.7 [-64.8, -25.5] (0 of 120 rounds won).
Without advice she uses it in 36% of decisions and it nets better than her other options there; ruled out, every
other option reads "likely fails", so the rule falls back to walking in (2,646 walks) - into Zangief. The players'
advice (arcade) was wrong for this bot for that reason; Qwen proposed the same "avoid" 29 times vs Zangief, and the
verifier's per-situation yardstick agreed (history: Bird Kick worse than her other moves there). What replaces a
ruled-out move is decided by laya-vision's ratings and the rule's walk fallback, which neither Qwen nor the verifier
sees.

Qwen vs the new opponents: never proposed the throw vs Zangief or Guile; proposed it twice vs Dhalsim (still on test at
the end). Registered "use more forward" (walking in) in every Zangief and Dhalsim run. Against Guile (no history) the
verifier registered almost nothing; the gain came from lessons on test.

## laya-vision (LV): it ranks by chance to hit, so it hides high-value moves - general

- The throw: up close it is in laya-vision's top 3 in 0% (Ken), 2% (Honda), 19% (Ryu) of decisions, yet when a lesson
  names it laya-vision rates it "likely works" / "may work" in 90-100%. Its score is fine; other moves score higher on
  P(hit). A throw's value is its damage, which a P(hit) ranking does not see. Same for mk / hp as anti-airs at mid range
  (top 3: 0%). General: the same for all three opponents.
- The block scale (fixed in code, 7d85eb7): a block's P(blocked) had been read on the P(hit) scale.
- Consequence downstream: she almost never tries what laya-vision does not rank (throws up close: 1 / 4 / 6 in her whole
  history vs Ken / Honda / Ryu), so nothing downstream can learn it.

## Label rule / System 1 glue (R): a soft lesson cannot beat "likely fails" - by design, often decisive

- "use more c.mk at mid range" (Ryu), "use more c.hp / c.mk at mid range when he attacks" (Honda, Qwen's): the rule says
  the move only 14-46% of the time, the rest are "likely fails". "use more block_low far away when he attacks" and
  "use more c.mk far away when he jumps": 0%, the lesson changed nothing. The rule passes laya-vision's veto through.
- Action set: no walk back, so "keep away" (the players' core advice vs Honda) cannot be given (ledger #22).

## Text laya (TL): follows its rule 98-100% where it was trained; two untrained cases - general

- Its training set (test_data/advice/train.jsonl, 16,000 examples) has no lesson naming forward and no "nothing left"
  case. In play:
  - "avoid forward" + every other option "likely fails": it walks forward anyway, 1,165 of 1,165 decisions (Honda).
  - "use more forward ... when he jumps": followed 61% (225 / 367); it picks a "may work" move instead.
- Everywhere else: follows its rule 100% in every arm of the three batches (0 mismatches with the recomputed rule).

## Qwen (Q): proposes from what she did; cannot find what she never tries

- Never proposed "use more throw up close" against any opponent (its only throw claims were "avoid"): her history had
  1-6 close-range throws, "too few" for the verifier. The what-if slot never tried it.
- Against Honda its most-played set includes walking in ("use more forward at mid range when he jumps"): -29.7 hp/round;
  the verifier had accepted it (walking in is better than her other options at that moment; ledger #11).
- About 2x a table-aware chance at proposing lessons that hold; re-proposes known lessons 20-45% of the time.

## Verifier and harness (V, H, S)

- The verifier judges a move within one situation; it cannot see where a lesson takes her (ledger #11) - Qwen's Honda
  set is the worked example.
- 22 harness / analysis defects found and fixed or opened (docs/harness_ledger.md); none changes a result above.

## What this says about the decisions ahead

1. laya-vision's ranking (P(hit), not value) is the root of the biggest miss, and it is general. Two ways to act: in
   code (rank the shortlist by expected damage, like the block-scale fix), or a laya-vision fine-tune that predicts
   value. The owner decides.
2. Text laya's two untrained cases (lessons naming forward; nothing left) are general. Cheapest: the verifier refuses
   lessons naming forward (System 1 cannot follow them reliably) and the rule never ends with nothing to pick; or a
   text-laya fine-tune with those cases in its data. The owner decides.
3. Qwen needs exploration it cannot get from her history: verified players' tips as a starting book (a verified-tip
   status judged by its round record), or a what-if that tries moves laya-vision never ranks.
