# Bee quorum System 1: design (summary of record)

Full design with diagrams: [Swarm System 1: Quorum Voting, Shared Outcome Table and GA Tuning](https://claude.ai/code/artifact/3000308a-519f-440d-b3f4-7e21906090af).
How to run it: [docs/quorum_integration.md](quorum_integration.md). Code: `sf2/quorum/`.

System 1 becomes a small swarm. Several cheap voters propose a move, a weighted tally decides, and every outcome is
credited to one shared table. Text-laya's weights never change in play: the table holds experience.

## The bee mapping

| Bee | Here |
|---|---|
| Scouts explore different sites | three flavours of the same text-laya checkpoints (defend / attack / move), each over its own categories |
| Dance length grows with quality | vote = confidence x the voter's reliability in this `when` (multiplicative weights) |
| Recruitment to good sites | + beta x shrink(n) x the table's positive mean for the move |
| Cross-inhibition | - gamma x shrink(n) x the table's negative mean for the move |
| Quorum, no leader | act when the top move's share of the positive score >= theta |
| Site quality checked on arrival | the OUTCOME (delayed-hit-corrected net hp) is credited, whoever chose |

## Voters

| Voter | What it is | Cost per decision |
|---|---|---|
| `laya` | today's two-stage pick (`two_stage_decide`), also the fallback | 2 laya calls (already paid today) |
| `defend` | the move model over the `block` category | 1 call |
| `attack` | the category model over punch / kick / throw / special / combo, then the move model | 2 calls |
| `move` | the move model over the `move` category | 1 call |
| `table` | the value table's best followable move with n >= `table_min_n` and mean > 0 (else abstains) | none |

## Modes (the switch) = rollout phases 0-3

| Mode | What plays | Gate before the next phase |
|---|---|---|
| `shadow` | text-laya, exactly as today; the swarm is only logged | split votes predict worse exchanges (`scripts/quorum_report.py`) |
| `candidates` | the table's confident pick among the swarm's candidates, else text-laya; exploration only among candidates | table coverage grows faster, gate no worse |
| `vote` | the quorum; a split falls back to text-laya | beats `hybrid` by 2 SE on paired matches |
| `vote` + `--quorum-qwen` | as `vote`, but a split goes to Qwen (System 2), which may only pick a candidate | escalation near 10%, Qwen beats the fallback on split cells |

Phase 4 (evolution, `scripts/quorum_evolve.py`) tunes theta, epsilon, beta, gamma, k, eta, net_scale and the five
voter priors. Fitness = mean net per round + win weight x win rate - Qwen weight x max(0, escalation - 10%).
Phase 5 (nightly consolidation into a text-laya fine-tune) is not built yet.

Consistent with the plan of record for `feat/table-system1`: Qwen stays out until the quorum has passed its own gate.
