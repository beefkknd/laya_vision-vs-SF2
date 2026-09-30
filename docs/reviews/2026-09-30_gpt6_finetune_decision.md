# GPT-6 (gpt-6-astra via codex, high reasoning): does laya-vision need a fine-tune? (2026-09-30)

Independent analyst, different engine, blind to Dr Fable's answer; brief: docs/reviews/2026-09-30_finetune_decision_brief.md.

**Decision: no third fine-tune yet.** The evidence supports replacing System 1’s ranking with the table and retaining Qwen’s verified book. A future vision fine-tune should earn its place by improving decisions *within* situations the table treats identically.

1. **Why run 2 failed—and what remains uncertain**

   Run 2 learned something: value cross-entropy beat its class prior for 7/8 characters, and Chun-Li’s outcome accuracy rose from **0.607 to 0.672**. But value prediction did not transfer usefully: explored-decision spreads were **−2.44 hp** on Chun-Li’s test games and **−1.86** against held-out Guile. Throw entered the favorable situation’s top three **0/37** and **0/66** times, versus **100%** for the table. These are different capabilities: predicting an outcome better does not imply choosing a better move. [Training log](review bundle: results/run2_train_log.json), [evaluation](review bundle: results/run2_eval_v2.json).

   The strongest explanation is **weak action-comparison supervision plus a poorly matched selection objective**, rather than proven failure of vision. Live collection observes one response per state; differences between states, timing and opponent reactions obscure differences between responses. Random exploration removes policy selection of the current action, but states still come from the mixed behavior policy. The table averages those observations into coarse, shrunk estimates; the model must discover that structure while fitting images and other tasks.

   Validation holds out positions, not whole games, so neighboring game states remain across the split. Pooled NLL also mixes outcome and value tasks. Its improvement cannot establish useful move ordering; **0.672 versus run 1’s 0.738 is additionally a comparison across changed validation datasets**. [Training/split code](review bundle: code/train_data.py), [trainer](review bundle: code/train.py).

   Target design is a plausible contributor, not an established cause. Five buckets discard damage magnitude; inference substitutes global bucket means measured under an older policy. “Until next decision” gives different horizons to different moves and misses later consequences. The external score-loss implementation is absent, so we cannot establish whether training actually exploits ordinal structure. Nor does the table’s success prove that buckets, window or recipe are individually sound. [Value target](review bundle: code/value.py), [action execution](review bundle: code/system1.py).

   **The evaluation also needs correction.** Recomputing the table’s curves in run 2’s report gives monotonicity in only **2/9 groups**, with spread below **10 hp in 8/9**. Thus that offline gate would reject the successful table itself. Global calibration of historically chosen moves is not a direct test of choosing among responses from one frame. This weakens “no ordering at all,” but does not excuse the model’s concrete throw failure.

2. **What System 1 should use now**

   Use the frozen table as the general ranking baseline. Its measured gain is **+81.4 hp/round [42.7, 121.9]**, with **369 versus 70 wins out of 720 rounds**. That establishes practical value without another model run. However, the in-play evidence covers **Chun-Li against six opponents**, not all eight playable characters. [Report](review bundle: results/lv_inplay_oracle_report.json), [protocol](review bundle: docs/prereg_value_oracle.md).

   Keep laya-vision’s outcome predictions as a separately measured capability; do not let the old P(hit) ranking silently veto the table. Refresh table estimates only from designated training exploration, retaining counts and uncertainty and excluding evaluation runs. Missing moves currently default to zero, which can beat observed negative values; sparse-cell behavior deserves an explicit fallback. [Table code](review bundle: code/value_oracle.py).

   This revises the earlier claim that a fine-tune was the “principled fix”: **the ranking objective needed repair; a learned visual value head was only one possible repair**. The table supplies no evidence that the eyes were defective.

3. **When controlled collection could justify a fine-tune**

   I read the owner’s proposal as **one situation A, restore the same state, try B/C/D/E separately**. That is the right next diagnostic—not response sequences. Existing collectors already restore states for static actions and high/low/jump-in defense probes, but expanding them to moving threats and every response requires work. [Collector](review bundle: code/collect.py), [defense probes](review bundle: code/vs_defense.py).

   Cover five general families: idle/guarding up close; ground attack startup/recovery; approaching opponent; incoming jump; incoming projectile. Vary distance, attack phase, side and independent boots across all eight own characters and multiple attackers. Keep opponent identities and move names out of model inputs; use them only to organize collection and holdouts.

   A bounded pilot could use **8 characters × 5 families × 20 states × 16 responses = 12,800 branches**. Expand to 60 states per family only if useful, giving **38,400 branches**. Save the identical two input frames for every response. Split entire source episodes/boots—including sibling branches, timings and mirrors—together.

   Register one exchange protocol before collection: for example, a common 120-frame observation horizon, one response followed by neutral input. Record raw dealt/taken damage, next-decision net, duration and truncation separately. This makes comparisons explicit, although neutral continuation is artificial and must be checked in live play.

   Proceed to training only if held-out branches show worthwhile response changes within the same note-defined situation. Compare against a richer **note-only** baseline using existing fields such as exact dx; otherwise improvement over the coarse table could be mistaken for visual understanding.

   If warranted, train **one run from BASE**, balancing characters and situations. Learn raw exchange net with a within-state ranking objective; preserve outcome prediction as an auxiliary task. Lock architecture, loss weights, seed and stopping rule beforehand. Select on whole-episode validation ranking regret—the damage lost by choosing the wrong response—not pooled classification accuracy.

   Required gates: lower held-out regret than both table and runs/all8, evidence that matched-note image substitutions degrade ranking, outcome accuracy within **2 percentage points** of all8 per character, then fresh paired in-play superiority over the table and all8. Pre-register a practical gain threshold, such as **+5 hp/round**, alongside confidence intervals. Include all eight own characters before claiming general improvement.

   Cost is uncertain: at an assumed **0.25–1 second per restored branch**, collection is roughly **1–4 hours** for the pilot or **3–11 hours** expanded, excluding engineering. Benchmark throughput first. Training and evaluation add hours; run stages sequentially with a measured memory budget.

4. **Has Qwen already benefited from searched guides? Yes—but distinguish knowledge use from model learning.**

   With unchanged laya-vision, the verified book delivered **+47.1 hp/round [30.4, 64.0] versus no advice**, and **+40.5 [30.4, 50.7] versus Qwen’s loop without the book**. The no-book loop’s pooled gain was only **+6.6 [−8.3, 21.4]**. This is strong evidence that verified external knowledge helps. It does not demonstrate Qwen weight updates, independent discovery of those tips, or quantify its additional contribution beyond executing them. Higher lesson acceptance is encouraging, not equivalent to additional playing strength. [Book results](review bundle: docs/component_boundaries.md), [registration](review bundle: docs/prereg_book.md).

   Ownership should remain clear: **table—general action value; vision—visual timing; Qwen—opponent knowledge, hypotheses and lessons; text laya—lesson execution; harness/verifier—collection, tests and attribution**. The throw overlaps both successful routes, so their gains must not be added. Their controls also differ: book uses text laya; table uses the plain threshold policy.

5. **One fair joint experiment**

   Run a fresh paired **2×2 experiment: old ranking/table × Qwen loop without/with book**. Keep text laya, Qwen prompt, action set, lesson rules, opponents, seeds and round budgets identical. Use at least **8 seeds per opponent**; freeze the book and table.

   First wire a common advice interface: currently the table returns before advisor execution, and value checkpoints explicitly reject an advisor. Freeze score-to-rating mappings using training data and verify lesson execution before testing. [System 1](review bundle: code/system1.py).

   Measure table benefit with and without the book, book benefit under each ranking, and their interaction. A negative interaction would show overlap; a remaining book benefit would establish complementary value. Analyze paired seed/run means, with opponent-level pooling—not independent rounds.

6. **Risks that could still mislead**

   Controlled scripts may teach recognition of collection artifacts; identical-state replay controls randomness but does not create independent samples. Opponent sprites still reveal identity despite omitted names. Exploration and deployment visit different states. Short exchange value misses positioning and future punishment. One training seed cannot establish recipe reliability. Existing reports support the headline effects, but the supplied subset lacks full rollout data for independently rebuilding their confidence intervals. Finally, unavailable actions and harmful advice substitutions remain harness problems that no vision fine-tune automatically solves.

**Recommendation:** keep the verified Qwen book, establish the table as System 1’s ranking baseline, and measure their combination in the paired factorial experiment. Do not authorize another value fine-tune merely because run 2 failed. Use a small same-state response pilot to determine whether visual timing offers measurable value beyond note-only ranking; train once from BASE only if that evidence exists, and retain the model only if it improves fresh in-play results over the table.