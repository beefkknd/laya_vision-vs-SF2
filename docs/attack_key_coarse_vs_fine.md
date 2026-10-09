# Coarse `attack` vs the fine limb×zone split — the verdict and why

**Owner decision (2026-10-09): the value-table key uses ONE coarse `attack` behaviour label by default.
The fine limb×zone attack split (`SF2_LIMB_KEY`) stays OFF and is not used in any go-forward training.**
This note is the evidence behind that decision and must travel with it.

## What the two things are

The value table is keyed by situation: `range | opp-posture | fireball [ | behaviour-label ]`. Two *optional*
refinements can subdivide a cell further. They are independent and have **opposite** verdicts:

| mechanism | what it splits on | where | verdict |
|---|---|---|---|
| **`_maybe_split`** (`value_table.py`, `depth[base]="his_label"`) | opponent **behaviour** (stand/walk/jump/block/attack/special) | adaptive, data-driven (Welch bar) | **KEEP — on** |
| **limb×zone** (`SF2_LIMB_KEY`, `opp_limb.py`, `his_class`) | opponent's **attacking limb & zone** (hand/leg × high/mid/low) | the `attack`/`jumping` cells | **DROP — off by default** |

The behavioural split is a *prediction* key ("in this situation she tends to do X"). The limb×zone split is a
*reaction* key ("she is currently throwing a high kick"). In this game the first works and the second does not.

## The one test that decides it

A finer key earns its keep **only if, inside the finer cells, the best ACTION flips.** If every sub-cell still
wants the same move, the extra resolution is pure cost. We measured the flip rate for both candidate splits:

| split the key on … | best-action flip rate | leverage |
|---|---|---|
| **opponent (style)** — Ryu table (B3) vs Chun-Li table (C1) | **84 %** (21 of 25 shared decided cells) | huge → separate tables |
| opponent's **limb×zone** | **~1 %** (detector: only `close\|attacking`, z = 5.04) | negligible → collapse to coarse |

Same principle, two opposite answers, both earned by the same measurement. **Block is "least-bad" across almost
every limb and zone**, so knowing it is a high kick vs a low punch almost never changes Zangief's correct move.

## The single cell where the fine value looked large — and why it still collapses

`close|jumping|0|hand_high` in B3 shows `double_lariat` at **+30** net-HP vs Ryu's jump-in — the strongest-looking
case for the fine key. But merge that base cell coarsely (base + all children) and the argmax is **still
`double_lariat`** (+1 diluted mean). The coarse cell makes the **same decision**; the fine key only sharpened the
*value estimate*, not the *choice*. Since play is driven by argmax, nothing behavioural is lost by going coarse —
so **no whitelist is needed.** (Reproduce: `study/`-level merge of `B3_newkey_ryu94.json`; see session log 2026-10-08/09.)

## Why the fine attack key could never have helped this matchup

1. **It failed its original mission.** It was built to resolve the Ryu↔Chun-Li interference. That interference
   lives in the **neutral game, keyed by opponent**, not by limb — so an attack-pose feature cannot separate it.
   The fix that works is the opponent/style split (separate tables), not limb×zone.
2. **Reaction features need reaction time Zangief does not have.** Chun-Li's light pokes are sub-reaction-window:
   they start and recover faster than any policy can see-and-respond, even at 1-frame sampling. A limb×zone key is a
   reaction key; it is the wrong tool against speed. Prediction (behaviour label) is the only lever that bites.
3. **C1 confirmed it adds resolution without leverage.** Chun-Li's fine cells (`close|jumping|hand_high` −35,
   `leg_high` −35, `special attack` −52) merely subdivided a *losing* situation into finer losing situations; block
   lost in all of them. Bees that tried to exploit per-limb anti-air (double_lariat) regressed the table −11.

## Supporting record (B3 vs C1 study, 2026-10-08/09)

- **B3** = Zangief vs Ryu table, **94.5 %** match win, 39 cells (the accepted Ryu baseline).
- **C1** = Zangief vs Chun-Li, trained from blank, no bees: 50.8 % (r1) → **64.1 % (r2, high-water)** → 53.1 % (r3,
  3 Ryu-style bees, regressed). r2 = `study/zangief/zangief_c1_r2.json` is the locked Chun-Li table.
- **Structural divergence, readable before training:** Ryu populates the `fireball=1` branch (4 815 decisions);
  Chun-Li 0. A whole dimension of Ryu's table is inapplicable to Chun-Li. The key is already a partial opponent model.
- **Thesis (quantified):** fighting *style* is a latent variable the situation key does not encode; the same observed
  situation demands opposite actions against opponents who play differently, so one shared table is pulled apart and
  regresses. Resolution: **one table per style-cluster**, opponent known at runtime. Ryu≈Ken (near-identical kits)
  are a share candidate (test: flip rate well under 84 %); Chun-Li is a different archetype and needs her own.

## What changed in code

- `opp_limb.limb_key_on()` now **warns once on stderr when `SF2_LIMB_KEY` is ON**, so a run can never enter the
  experimental split silently. Default (unset) stays coarse and byte-identical to the old `his_label` path.
- The limb×zone code (`opp_limb.py`, the `his_class` wiring in `screen_words`/`value_table`/`screen_evidence`) is
  **retained but gated off** — kept only so this analysis stays reproducible. Do **not** set `SF2_LIMB_KEY` forward.
- `_maybe_split` (behavioural his_label split) is **unchanged and stays on** — it is a different, useful mechanism.

## Forward default

**Key = `range | posture | behaviour-label` (coarse `attack`), one table per style-cluster, opponent read at
runtime.** The limb×zone experiment was worth running precisely because it is what let us *prove* it isn't needed.
