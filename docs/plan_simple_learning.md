# Simple early-game learning strategy

Owner-set 2026-10-03. Replaces the tangle of overlapping timers that fought each other and deadlocked
(see the diagnosis at the end). The whole policy is: **while losing, change one line; while winning,
freeze.** Two states, one counter, one threshold.

## The two jobs, separated

The old system tangled *admission* (what to try) with *retention* (what to keep) into one pile of
timers, and the retention timers ended up blocking admission — the deadlock. Split them:

1. **Survival loop — fast, coarse, sovereign. The ONLY thing that edits the short memory.**
   Evaluated after every round on one signal: *trailing lost rounds*.
   - **streak < 2 → do nothing. Freeze.**  (= "stop churning once winning")
   - **streak ≥ 2 → swap exactly one line:** drop the weakest `trying` line, admit one fresh claim
     (the Coach's; if the Coach is silent, the explore pool), reset the streak.
     (= "while losing, the memory MUST change" — guaranteed, because admission is **never** gated by
     measurement, so it can never deadlock.)

2. **Retention — slow, advisory. Only decides `trying` → `kept`.**
   Reuses the existing outcome scorer, but it is **advisory**: it may promote a `trying` line to
   `kept`, and nothing else. It can never block an admission or a swap. `kept` lines are immune to the
   swap — the stable set once she is winning.

## The model

- **States: `trying`, `kept`.** That's all. (No testing/registered/sticky/verified/rejected/retired.)
- **One per-entry counter: `rounds-in-play`** (rounds this line has been in the short memory). Carries
  across blocks and Mesen restarts trivially — it lives in the entry. No `since`/`idx` coordinate, no
  re-basing.
- **One threshold: `SWAP_AFTER = 2`** trailing lost rounds. (Single number, not stage-adaptive — owner
  2026-10-03: fewest knobs.)
- **`short_memory.MAX_LINES = 10`** — the LIVE short memory holds up to 10 lines (owner 2026-10-03). Only the
  *situation-applicable* subset is shown to text-laya per decision (`loop_runner.py` filters by range/stance/
  fireball), so this is cheap on the prompt. The offline `lessons.MAX_LINES` stays 5 (book/scoring untouched).
- **Three tones, adjustable between rounds:** a claim's `kind` is `use_more` (soft), `always` (hard — overrides
  text-laya's "likely fails" rating) or `avoid` (negative). A Coach claim naming the SAME move+situation as a
  line in play but a different kind *retones that line in place* (keeps its state and rounds) — one such change
  per round. So the Coach can dial a rule up/down (or reverse it) across rounds, not only add "more".
- **Weakest `trying` line** = lowest advisory score among `trying` lines that have fired; if none has
  fired, the one with the largest `rounds-in-play` (most chance, nothing shown). `kept` lines never
  dropped.
- **Nothing is permanent.** A dropped line is just gone from the short memory; the Coach or the pool
  may propose it again later. Revive is automatic — no special window.

## Maps to the owner's principles

- *Winning is the objective, not margin* → we key on won/lost rounds; hp-margin logic (`STOP_DROP`) is
  deleted.
- *Stop churning once winning* → freeze when `streak < 2`; `kept` lines immune.
- *While losing, change* → the swap is unconditional at the threshold.
- *Measurement grades outcomes* → it still does, as the graduation scorer — just advisory.
- *Revive dismissed-good moves* → automatic (dropped ≠ dead).

## What is deleted

| Today | Simple |
|---|---|
| 6 states | `trying`, `kept` |
| `TEST_GAMES` `PROMOTE_GAMES` `STICK_WINDOW` `STOP_DROP` `MIN_TRIES` `MAX_TESTS` `REVIVE_EARLY/LATE/STAGE` | `SWAP_AFTER = 2` |
| `since`/`idx` + carry re-basing | `rounds-in-play` per entry |
| career `--loss-trigger` `--loss-trigger-early` `--stage-rules` + between-block injection as `verified` | one policy, in `lessons`, one path |
| rejected lines appended forever (16 dupes) | dropped line simply leaves; no rejected pile |

## Implementation shape (test-first, when approved)

Owned by `sf2/system2` (pure) + `scripts/play_loop_screen.py:update()` (wiring). `scripts/play_career.py`
stops editing the registry between blocks.

- `sf2/system2/lessons.py` — collapse to `trying`/`kept`; `swap(reg, round_wl, claim, scorer)`:
  the survival loop (streak, drop-weakest, admit-one); `graduate(reg, rows, scorer)`: advisory
  trying→kept. Delete the old timer constants + states.
- `sf2/system2/explore.py` — move `EXPLORE_POOL` + a `pick(in_play)` that skips *covered* lines
  (`_covers`, not exact-string) and returns a *claim* (enters as `trying`, never `verified`).
- `scripts/play_loop_screen.py:update()` — compute streak from `round_wl`; one swap per round when
  losing; graduate; trace `added`/`removed`/`streak`.
- `scripts/play_career.py` — delete the between-block injection and the registry-size stage logic;
  import pool helpers from `explore.py`.

**Tests (red first):**
- `trying` line is dropped and a fresh one admitted after 2 lost rounds, whatever its history
  (the deadlock-breaker; fails today).
- No swap while winning; `kept` lines never dropped.
- Invariant: over any run, no two consecutive identical in-play sets while a loss streak persists.
- Carry: counter survives a block boundary; no re-basing needed.
- **DONE gate** — replay `playbooks/chun/round_03_ryu` (12 straight losses, 12 refused Coach claims):
  assert the in-play set changes within 2 rounds of the streak; the pre-change driver FAILS this.

## The deadlock this fixes (diagnosis, verified 2026-10-03)

Real run `playbooks/chun/round_03_ryu`: 12 straight losses, the Coach proposed a fresh grounded move
on all 12 reflections, **every one refused "already 2 claims in test," 0 changes.** Cause: the two
`testing` slots held rules carried from a prior block with `since=10/11`; the per-round counter `idx`
resets to 0 each block, so `idx - 10 >= TEST_GAMES` is never true → the tests are never resolved →
both slots locked → every new claim refused. The same mismatch broke `stop()`/`losing_since()` (empty
window slices). The career driver made it worse by injecting forced rules as `verified`, which are
never measured and permanently consume the test `room`. The simple model has no such coordinate and no
such slot cap, so the class of bug cannot recur.
