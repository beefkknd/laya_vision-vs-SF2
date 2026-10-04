# Bee quorum: integration guide

Branch `feat/laya-quorum`, based on `feat/table-system1` @ `54cdfd4`. Design: [plan_bee_quorum.md](plan_bee_quorum.md).

Everything below was built and tested without the Mac, Mesen or Qwen: fake advisors, a fake `play_round` and a fake
Qwen. The first real run is yours. The "Check on the Mac" section lists what could not be verified here.

## What landed

| Path | What |
|---|---|
| `sf2/quorum/config.py` | `QuorumConfig`: the mode switch and every tunable number (JSON load / save) |
| `sf2/quorum/tally.py` | `Proposal`, the table voter, scoring (votes, recruitment, inhibition), vote share |
| `sf2/quorum/voters.py` | the `laya` voter and the three flavour voters over the two text-laya checkpoints |
| `sf2/quorum/reliability.py` | per-(when, voter) weights: credit from outcomes, merge across workers |
| `sf2/quorum/decider.py` | `quorum_decider(...)`: a drop-in `decide(moment)` for `loop_runner.play_round` |
| `sf2/quorum/escalate.py` | `make_qwen_pick(ask_qwen)`: the optional System-2 hook for split votes |
| `sf2/quorum/genome.py` | genes, bounds, mutation, fitness for the evolution loop |
| `scripts/quorum_report.py` | reads runs back: agreement vs outcome, sources, per-voter stats |
| `scripts/quorum_evolve.py` | (1 + lambda) evolution of the config, scored on real play |
| `tests/test_quorum.py` | 24 tests, all fakes |

Two existing files changed, both additive:

- `sf2/system1/loop_runner.py`: `"quorum"` added to `DECISION_KEYS`, so each decision's quorum record reaches
  `decisions.jsonl`. No other change; the rules, table and hybrid paths are untouched.
- `scripts/play_loop_screen.py`: `--policy quorum` and its flags; `run_loop` takes `quorum`, `quorum_state`,
  `save_quorum`, `qwen_pick`. The quorum policy credits the table and voter reliability every round, and like
  `hybrid` it still runs the short-memory Qwen update between rounds, because the `laya` voter follows those lines.

Checks run here: the full suite gives the same 113 failures with and without this branch (all pre-existing, missing
data and checkpoints on this machine); `scripts/hard_gate.py` stays at its existing 42 violations, none new.

## The switch

```
--policy quorum                       the bee quorum (needs the two text-laya advisors, like rules / hybrid)
--quorum-mode shadow|candidates|vote  override the config's mode
--quorum-config path.json             a QuorumConfig (any subset of keys; the rest keep defaults)
--quorum-qwen                         vote mode: send split votes to Qwen instead of falling back to text-laya
--carry-quorum / --save-quorum path   voter reliability state, carried across blocks like --carry-table / --save-table
--carry-table / --save-table path     the value table, shared with table / hybrid
```

Defaults: mode `shadow`, theta 0.5, epsilon 0.05, beta 0.5, gamma 0.5, k 8, eta 0.1, net_scale 10, table_min_n 3,
every voter prior 1.0, Qwen off. A config file can be as small as `{"mode": "vote", "theta": 0.6}`.

## Run it, phase by phase

Use your usual `play_loop_screen.py` arguments (`--me`, `--opp`, `--games`, `--rounds`, `--no-score`,
`--shared-text-laya`, ...). Carry the same table forward through the phases.

**Phase 0: shadow.** Play is identical to `--policy rules`, but the table is credited too.

```bash
python scripts/play_loop_screen.py --policy quorum --quorum-mode shadow --opp dhalsim --games 10 --rounds 3 \
    --carry-table runs/quorum/table.json --save-table runs/quorum/table.json --save-quorum runs/quorum/rel.json
python scripts/quorum_report.py rollouts/loop_screen/<that run>
```

Gate: in "agreement vs outcome", the low-share bands should have a lower mean net than the high-share bands. If they
don't, a split vote doesn't predict trouble, so the quorum has nothing to work with. Stop and rethink the voters
before Phase 1. Also read the per-voter lines: a flavour that never proposes anything playable is broken.

**Phase 1: candidates.**

```bash
python scripts/play_loop_screen.py --policy quorum --quorum-mode candidates ... \
    --carry-table runs/quorum/table.json --save-table runs/quorum/table.json \
    --carry-quorum runs/quorum/rel.json --save-quorum runs/quorum/rel.json
```

Gate: table cells fill faster than under `--policy hybrid` for the same games, and the round results are no worse.

**Phase 2: vote.** Run it against `--policy hybrid` on the same seeds and opponent: that's the A/B that decides
whether the quorum stays.

```bash
python scripts/play_loop_screen.py --policy quorum --quorum-mode vote --seed 7 ... (carry both files)
python scripts/play_loop_screen.py --policy hybrid --seed 7 ... (carry the table)
```

Gate: net hp per round better than hybrid by 2 standard errors. `verdict.json` → `quorum.escalation_rate` is the
share of decisions with no quorum.

**Phase 3: Qwen on split votes.** Add `--quorum-qwen`. Qwen gets the prompt text, every vote and the candidate list,
and may only answer with a candidate. Anything else falls back to text-laya. The bridge is lockstep, so Qwen's
latency costs wall-clock time only. Gate: escalation near 10%, and Qwen's picks beat the fallback on split cells
(compare `source == "qwen"` rows with `source == "fallback"` rows in the report across runs).

**Phase 4: evolution.** Every candidate config starts from the same frozen table and reliability snapshot:

```bash
python scripts/quorum_evolve.py --out runs/evolve1 --opp dhalsim --holdout-opp ryu --generations 15 --lam 6 \
    --workers 3 --games 2 --rounds 3 --carry-table runs/quorum/table.json --carry-quorum runs/quorum/rel.json \
    --extra "--me chunli --no-score --shared-text-laya"
```

Worker i uses ports `PORTS["system1"][0] + i` and `PORTS["replay"][0] + i`. The best config lands in
`runs/evolve1/best_config.json`; feed it back with `--quorum-config`. `--dry-run` checks the loop without an
emulator. Before choosing `--lam` and `--workers`, measure minutes per run.

## Parallel collection

Tables and reliability states from parallel workers pool cleanly:

```python
from sf2.system1 import value_table as VT
from sf2.quorum import reliability as QR
table = VT.merge([json.load(open(p)) for p in table_paths])
rel = QR.merge([json.load(open(p)) for p in rel_paths])
```

## Check on the Mac (not verifiable here)

1. **Flavour prompts.** The `attack` flavour asks `cat_v3` with a 5-category menu instead of 7, and `defend` / `move`
   ask `move_v2` directly over one category. That is the same question shape on a narrower menu, not something the
   checkpoints were trained on. Phase 0's per-voter lines show whether their picks are sensible.
2. **Decision time.** A decision now costs about 6 laya calls instead of 2 (fewer in the air, where flavours
   abstain). `round.json` → `decide_ms_mean` shows the real cost; `--shared-text-laya` keeps one server per checkpoint.
3. **Qwen per decision.** `sf2.system2.qwen.chat` writes one log file per call, so escalations add a file each.
   Check the latency and the reply format once with `--quorum-qwen` on a short run.
4. **Other characters.** Flavours use the character's own category map (`char_categories`); a flavour whose
   categories that character lacks just abstains.

## Not built yet

- Phase 5, nightly consolidation: turning proven (when, action) cells into text-laya fine-tune rows.
- Racing in the evolution loop (dropping clearly worse candidates early); it re-scores the parent every generation instead.
